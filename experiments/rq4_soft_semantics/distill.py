"""RQ4 diagnostic and OOF pointwise adequacy distillation with an MRC feature.

This runner never calls Qwen.  It uses already-cached Qwen adequacy scores only as
development targets, computes local NLI/MRC features for each BM25 candidate, and
predicts held-out queries with a fixed ridge regressor.  Qrels are used only after
the OOF scores have been produced, for the final effectiveness comparison.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from axiomrank import paths, ranking
from axiomrank.config import dump_config, load_config
from axiomrank.pipeline import stages
from axiomrank.provenance import write_run_manifest
from axiomrank.ranking.soft_semantics import (
    DEFAULT_MRC_MODEL,
    DEFAULT_NLI_MODEL,
    HuggingFaceCrossEncoder,
    HuggingFaceMRC,
    HuggingFaceNLI,
    clean_passage,
    prepare_query,
)
from axiomrank.ranking.surrogate import assign_query_folds


FEATURE_VERSION = "v1"
RIDGE_ALPHA = 1.0
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP = frozenset(
    "a an and are be by do does for from how in is it of on or the to was what when where which who why with".split()
)


def _feature_path(collection: str, depth: int) -> Path:
    return paths.DATA_DIR / "soft_semantics_features" / FEATURE_VERSION / f"{collection}_top{depth}.parquet"


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _content_coverage(query: str, passage: str) -> float:
    query_terms = {token.lower() for token in _TOKEN.findall(query) if token.lower() not in _STOP}
    if not query_terms:
        return 0.0
    passage_terms = {token.lower() for token in _TOKEN.findall(passage)}
    return len(query_terms & passage_terms) / len(query_terms)


def _load_adequacy() -> pd.DataFrame:
    root = paths.DATA_DIR / "adequacy" / "models__qwen3.6-35B-A3B-AWQ"
    parts = sorted(glob.glob(str(root / "part-*.parquet")))
    if not parts:
        raise SystemExit("no cached Qwen adequacy labels under data/adequacy")
    frame = pd.concat(
        [pd.read_parquet(part, columns=["collection", "qid", "docno", "adequacy", "argmax"]) for part in parts],
        ignore_index=True,
    ).dropna(subset=["adequacy"])
    frame["qid"] = frame["qid"].astype(str)
    frame["docno"] = frame["docno"].astype(str)
    return frame.drop_duplicates(["collection", "qid", "docno"], keep="last")


def _load_features(path: Path, *, refresh: bool) -> pd.DataFrame:
    if refresh or not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    required = {"qid", "docno", "nli_entailment", "mrc_margin"}
    if not required <= set(frame.columns):
        raise ValueError(f"feature cache has the wrong schema: {path}")
    frame["qid"] = frame["qid"].astype(str)
    frame["docno"] = frame["docno"].astype(str)
    return frame.drop_duplicates(["qid", "docno"], keep="last")


def _compute_features(
    pool: pd.DataFrame,
    cache_path: Path,
    *,
    depth: int,
    nli: HuggingFaceNLI,
    mrc: HuggingFaceMRC,
    relevance: HuggingFaceCrossEncoder,
    refresh: bool,
    flush_every: int,
) -> pd.DataFrame:
    scoped = pool.loc[pool["rank"] < depth, ["qid", "docno", "query", "text", "rank", "score"]].copy()
    scoped["qid"] = scoped["qid"].astype(str)
    scoped["docno"] = scoped["docno"].astype(str)
    cached = _load_features(cache_path, refresh=refresh)
    known = set(zip(cached.get("qid", ()), cached.get("docno", ())))
    todo = [row for row in scoped.itertuples(index=False) if (row.qid, row.docno) not in known]
    additions: list[dict] = []
    for row in tqdm(todo, desc=f"MRC/NLI {cache_path.stem}", unit="document"):
        prepared = prepare_query(row.query)
        passage = clean_passage(row.text)
        hypothesis = f"This text provides an answer or explanation to the question: {prepared.text}"
        additions.append(
            {
                "qid": row.qid,
                "docno": row.docno,
                "relevance_probability": relevance.score(prepared.text, passage),
                "nli_entailment": nli.entailment_probability(passage, hypothesis),
                "mrc_margin": mrc.answerability_margin(prepared.text, passage),
                "bm25_score": float(row.score),
                "bm25_rank": int(row.rank),
                "log_length": float(np.log1p(len(_TOKEN.findall(passage)))),
                "query_coverage": _content_coverage(prepared.text, passage),
            }
        )
        if len(additions) >= flush_every:
            cached = pd.concat([cached, pd.DataFrame(additions)], ignore_index=True)
            _atomic_parquet(cache_path, cached)
            additions.clear()
    if additions:
        cached = pd.concat([cached, pd.DataFrame(additions)], ignore_index=True)
        _atomic_parquet(cache_path, cached)
    features = cached.merge(scoped[["qid", "docno"]], on=["qid", "docno"], how="inner")
    if len(features) != len(scoped):
        raise ValueError(f"incomplete feature cache: {cache_path}")
    return features


def _existing_relevance(collection: str, depth: int) -> pd.DataFrame:
    path = paths.DATA_DIR / "soft_semantics" / "v1" / f"{collection}_top{depth}.parquet"
    frame = pd.read_parquet(path, columns=["qid", "docno", "relevance_probability"])
    frame["qid"] = frame["qid"].astype(str)
    frame["docno"] = frame["docno"].astype(str)
    return frame.drop_duplicates(["qid", "docno"], keep="last")


def _oof_ridge(frame: pd.DataFrame, feature_names: list[str], *, seed: int) -> np.ndarray:
    del seed  # GroupKFold is deterministic; retained for a uniform experiment surface.
    groups = frame["group_id"].to_numpy()
    folds = assign_query_folds(groups, n_folds=5)
    predictions = np.full(len(frame), np.nan)
    labelled = frame["adequacy"].notna().to_numpy()
    for fold in np.unique(folds):
        test = folds == fold
        train = ~test & labelled
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(frame.loc[train, feature_names], frame.loc[train, "adequacy"])
        predictions[test] = model.predict(frame.loc[test, feature_names])
    if np.isnan(predictions).any():
        raise RuntimeError("missing OOF distiller predictions")
    return np.clip(predictions, 0.0, 3.0)


def _fidelity(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    labelled = frame["adequacy"].notna().to_numpy()
    target = frame.loc[labelled, "adequacy"].to_numpy(dtype=float)
    predicted = prediction[labelled]
    return {
        "n_labelled_documents": int(labelled.sum()),
        "spearman": float(spearmanr(target, predicted).statistic),
        "mae": float(mean_absolute_error(target, predicted)),
        "r2": float(r2_score(target, predicted)),
    }


def _gate_diagnostics(frame: pd.DataFrame) -> dict:
    labelled = frame.dropna(subset=["adequacy", "argmax"])
    high = labelled["adequacy"] >= 2.5
    return {
        "n_labelled_documents": int(len(labelled)),
        "nli_false_negative_rate_at_0_65_for_qwen_ge_2_5": float(
            (labelled.loc[high, "nli_entailment"] < 0.65).mean()
        ),
        "relevance_false_negative_rate_at_0_40_for_qwen_ge_2_5": float(
            (labelled.loc[high, "relevance_probability"] < 0.40).mean()
        ),
        "mean_features_by_qwen_argmax": {
            str(label): {
                name: float(values[name].mean())
                for name in ("relevance_probability", "nli_entailment", "mrc_margin")
            }
            for label, values in labelled.groupby("argmax")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--mrc-model", default=DEFAULT_MRC_MODEL)
    parser.add_argument(
        "--transfer-config",
        help="unlocked exploratory target config; the fitted DL19/DL20 model is applied unchanged",
    )
    parser.add_argument(
        "--transfer-depth", type=int,
        help="BM25 depth to score on the exploratory target (defaults to its full pool)",
    )
    args = parser.parse_args()
    if min(args.depth, args.threads, args.flush_every, args.transfer_depth or 1) <= 0:
        raise SystemExit("--depth, --threads, --flush-every and --transfer-depth must be positive")

    paths.configure_caches()
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cfg = load_config(args.config)
    out_dir = paths.results_dir("rq4_soft_semantics") / f"distill_top{args.depth}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")
    adequacy = _load_adequacy()
    nli = HuggingFaceNLI(args.nli_model, device="cpu")
    mrc = HuggingFaceMRC(args.mrc_model, device="cpu")
    relevance = HuggingFaceCrossEncoder(device="cpu")
    frames: list[pd.DataFrame] = []
    pools: dict[str, tuple[object, pd.DataFrame]] = {}
    for source in cfg.sources:
        source_cfg = load_config(paths.PROJECT_ROOT / source)
        collection = source_cfg.variant or source_cfg.dataset.irds_id.replace("/", "_")
        pool = stages.build_pool(source_cfg, refresh=False)
        local = _compute_features(
            pool, _feature_path(collection, args.depth), depth=args.depth, nli=nli, mrc=mrc,
            relevance=relevance, refresh=args.refresh_features, flush_every=args.flush_every,
        ).merge(_existing_relevance(collection, args.depth), on=["qid", "docno"], validate="one_to_one")
        local = local.merge(
            adequacy.loc[adequacy["collection"] == collection, ["qid", "docno", "adequacy", "argmax"]],
            on=["qid", "docno"], how="left", validate="one_to_one",
        ).assign(collection=collection)
        local["group_id"] = collection + ":" + local["qid"]
        frames.append(local)
        pools[collection] = source_cfg, pool
    frame = pd.concat(frames, ignore_index=True)
    base_features = ["bm25_score", "bm25_rank", "log_length", "query_coverage"]
    semantic_features = [*base_features, "relevance_probability", "nli_entailment", "mrc_margin"]
    predictions = {
        "bm25_controls": _oof_ridge(frame, base_features, seed=cfg.seed),
        "semantic_mrc_distiller": _oof_ridge(frame, semantic_features, seed=cfg.seed),
    }
    reports = {
        "feature_models": {
            name: {"features": features, "ridge_alpha": RIDGE_ALPHA, "fidelity": _fidelity(frame, predictions[name])}
            for name, features in (("bm25_controls", base_features), ("semantic_mrc_distiller", semantic_features))
        },
        "gate_diagnostics": _gate_diagnostics(frame),
        "collections": {},
    }
    for collection, (source_cfg, pool) in pools.items():
        scoped = frame["collection"] == collection
        baseline = ranking.evaluate_run(pool, source_cfg.dataset.irds_id)
        collection_report = {}
        for name, prediction in predictions.items():
            score_map = {
                (str(row.qid), str(row.docno)): float(score)
                for row, score in zip(frame.loc[scoped].itertuples(index=False), prediction[scoped])
            }
            reranked = ranking.rerank_scored_pool(pool, score_map, depth=args.depth)
            measured = ranking.evaluate_run(reranked, source_cfg.dataset.irds_id)
            _, comparison = ranking.compare_runs(baseline, measured, seed=cfg.seed)
            collection_report[name] = comparison
        reports["collections"][collection] = collection_report
    if args.transfer_config:
        target_cfg = load_config(paths.PROJECT_ROOT / args.transfer_config)
        target_collection = target_cfg.variant or target_cfg.dataset.irds_id.replace("/", "_")
        target_pool = stages.build_pool(target_cfg, refresh=False)
        target_depth = args.transfer_depth or int(target_pool.groupby("qid")["rank"].max().max() + 1)
        target = _compute_features(
            target_pool,
            _feature_path(target_collection, target_depth),
            depth=target_depth,
            nli=nli,
            mrc=mrc,
            relevance=relevance,
            refresh=args.refresh_features,
            flush_every=args.flush_every,
        )
        if "relevance_probability" not in target:
            raise ValueError("transfer feature cache lacks relevance probabilities; refresh it")
        labelled = frame["adequacy"].notna()
        full_model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        full_model.fit(frame.loc[labelled, semantic_features], frame.loc[labelled, "adequacy"])
        target_prediction = np.clip(full_model.predict(target[semantic_features]), 0.0, 3.0)
        score_map = {
            (str(row.qid), str(row.docno)): float(score)
            for row, score in zip(target.itertuples(index=False), target_prediction)
        }
        reranked = ranking.rerank_scored_pool(target_pool, score_map, depth=target_depth)
        baseline = ranking.evaluate_run(target_pool, target_cfg.dataset.irds_id)
        measured = ranking.evaluate_run(reranked, target_cfg.dataset.irds_id)
        _, comparison = ranking.compare_runs(baseline, measured, seed=cfg.seed)
        reports["exploratory_transfer"] = {
            "collection": target_collection,
            "dataset": target_cfg.dataset.irds_id,
            "depth": target_depth,
            "n_documents": int(len(target)),
            "training": "full DL19/DL20 cached-Qwen-adequacy labels only; no target fitting",
            "metrics": comparison,
        }
    output = out_dir / "distill_report.json"
    output.write_text(json.dumps(reports, indent=2) + "\n")
    input_paths = (
        [paths.PROJECT_ROOT / source for source in cfg.sources]
        + [stages.processed_dir(source_cfg) for source_cfg, _ in pools.values()]
        + [
            paths.DATA_DIR / "adequacy" / "models__qwen3.6-35B-A3B-AWQ",
            paths.DATA_DIR / "soft_semantics_features" / FEATURE_VERSION,
        ]
    )
    if args.transfer_config:
        input_paths.append(paths.PROJECT_ROOT / args.transfer_config)
        input_paths.append(stages.processed_dir(target_cfg))
    write_run_manifest(
        out_dir / "run_manifest.json",
        cfg,
        config_source=args.config,
        source_paths=[Path(__file__), paths.PROJECT_ROOT / "src" / "axiomrank"],
        input_paths=input_paths,
        output_paths=[out_dir],
        extra={
            "runner": "experiments/rq4_soft_semantics/distill.py",
            "depth": args.depth,
            "nli_model": args.nli_model,
            "mrc_model": args.mrc_model,
            "ridge_alpha": RIDGE_ALPHA,
            "threads": args.threads,
            "refresh_features": args.refresh_features,
            "transfer_config": args.transfer_config,
            "transfer_depth": args.transfer_depth,
            "cache_policy": "development-config-guard-with-explicit-refresh",
        },
    )
    for collection, report in reports["collections"].items():
        for model, values in report.items():
            ndcg = values["nDCG@10"]
            print(f"{collection} {model}: {ndcg['mean_reranked']:.4f} ({ndcg['mean_delta']:+.4f})")
    if "exploratory_transfer" in reports:
        ndcg = reports["exploratory_transfer"]["metrics"]["nDCG@10"]
        print(
            f"{reports['exploratory_transfer']['collection']} transfer: "
            f"{ndcg['mean_reranked']:.4f} ({ndcg['mean_delta']:+.4f})"
        )
    print(f"[rq4-soft-semantics-distill] -> {output}")


if __name__ == "__main__":
    main()
