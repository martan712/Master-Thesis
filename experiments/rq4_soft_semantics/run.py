"""RQ4 / Phase 3: score BM25's top-N documents with soft-semantic axioms.

Unlike the cached RQ4 pairwise experiments, this evaluates each ``(query, document)``
exactly once.  Scores are resumably cached under ``data/soft_semantics``; a completed
run is reordered by the fixed 0--3 matrix and evaluated against the source collection
qrels at nDCG@10 (and AP as a secondary metric).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from axiomrank import paths, ranking
from axiomrank.config import dump_config, load_config
from axiomrank.pipeline import stages
from axiomrank.ranking.soft_semantics import AxiomThresholds, SoftSemanticAxiomaticReranker


def _cache_path(collection: str, depth: int) -> Path:
    return paths.DATA_DIR / "soft_semantics" / "v1" / f"{collection}_top{depth}.parquet"


def _load_cached(path: Path, *, refresh: bool) -> pd.DataFrame:
    if refresh or not path.exists():
        return pd.DataFrame()
    required = {"qid", "docno", "axiom_score"}
    cached = pd.read_parquet(path)
    if not required <= set(cached):
        raise ValueError(f"soft-semantic cache has wrong schema: {path}")
    return cached.drop_duplicates(["qid", "docno"], keep="last")


def _write_cached(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def score_pool(
    pool: pd.DataFrame,
    reranker: SoftSemanticAxiomaticReranker,
    cache_path: Path,
    *,
    depth: int,
    refresh: bool,
    flush_every: int,
) -> pd.DataFrame:
    """Return one persisted pointwise evaluation per document in the BM25 block."""
    scoped = pool.loc[pool["rank"] < depth, ["qid", "docno", "query", "text"]]
    cached = _load_cached(cache_path, refresh=refresh)
    known = {(str(row.qid), str(row.docno)) for row in cached.itertuples(index=False)}
    pending = [
        row for row in scoped.itertuples(index=False)
        if (str(row.qid), str(row.docno)) not in known
    ]
    additions: list[dict] = []
    for row in tqdm(pending, desc=f"soft semantics {cache_path.stem}", unit="document"):
        evaluation = reranker.evaluate(row.query, row.text)
        additions.append(
            {
                "qid": str(row.qid),
                "docno": str(row.docno),
                "axiom_score": evaluation.score,
                "axiom_1_relevant": evaluation.axiom_1_relevant,
                "axiom_2_answers": evaluation.axiom_2_answers,
                "axiom_3_constraints": evaluation.axiom_3_constraints,
                "relevance_probability": evaluation.relevance_probability,
                "entailment_probability": evaluation.entailment_probability,
                "constraint_similarity": evaluation.constraint_similarity,
            }
        )
        if len(additions) >= flush_every:
            cached = pd.concat([cached, pd.DataFrame(additions)], ignore_index=True)
            _write_cached(cache_path, cached)
            additions.clear()
    if additions:
        cached = pd.concat([cached, pd.DataFrame(additions)], ignore_index=True)
        _write_cached(cache_path, cached)

    scored = cached.merge(scoped[["qid", "docno"]].astype(str), on=["qid", "docno"], how="inner")
    if len(scored) != len(scoped):
        raise ValueError(f"incomplete soft-semantic score cache: {cache_path}")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="multi-source config listing DL19/DL20 pools")
    parser.add_argument("--depth", type=int, default=100, help="BM25 depth to score and reorder")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--threads", type=int, default=2,
        help="CPU inference threads (default: 2, to keep the workstation responsive)",
    )
    parser.add_argument("--refresh-scores", action="store_true")
    parser.add_argument("--flush-every", type=int, default=25)
    args = parser.parse_args()
    if args.depth <= 0 or args.flush_every <= 0 or args.threads <= 0:
        raise SystemExit("--depth, --flush-every and --threads must be positive")

    paths.configure_caches()
    # The three gates are local CPU models.  Bound their thread pools so a long
    # development rerank does not monopolise an interactive workstation.
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cfg = load_config(args.config)
    out_dir = paths.results_dir("rq4_soft_semantics") / f"top{args.depth}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")
    reranker = SoftSemanticAxiomaticReranker.from_huggingface(
        device=args.device, thresholds=AxiomThresholds()
    )

    reports = []
    for source in cfg.sources:
        source_cfg = load_config(paths.PROJECT_ROOT / source)
        collection = source_cfg.variant or source_cfg.dataset.irds_id.replace("/", "_")
        pool = stages.build_pool(source_cfg, refresh=False)
        started = perf_counter()
        scores = score_pool(
            pool, reranker, _cache_path(collection, args.depth), depth=args.depth,
            refresh=args.refresh_scores, flush_every=args.flush_every,
        )
        score_map = {
            (str(row.qid), str(row.docno)): int(row.axiom_score)
            for row in scores.itertuples(index=False)
        }
        reranked = ranking.rerank_scored_pool(pool, score_map, depth=args.depth)
        baseline = ranking.evaluate_run(pool, source_cfg.dataset.irds_id)
        measured = ranking.evaluate_run(reranked, source_cfg.dataset.irds_id)
        _, comparison = ranking.compare_runs(baseline, measured, seed=cfg.seed)
        reports.append(
            {
                "collection": collection,
                "depth": args.depth,
                "n_documents": len(scores),
                "elapsed_seconds": perf_counter() - started,
                "score_distribution": {
                    str(score): int(count)
                    for score, count in scores["axiom_score"].value_counts().sort_index().items()
                },
                "metrics": comparison,
            }
        )

    output = out_dir / "pointwise_effectiveness.json"
    output.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        ndcg = report["metrics"]["nDCG@10"]
        print(
            f"{report['collection']}: nDCG@10 BM25={ndcg['mean_baseline']:.4f} "
            f"soft-semantics={ndcg['mean_reranked']:.4f} "
            f"delta={ndcg['mean_delta']:+.4f} "
            f"[{ndcg['delta_ci_lo']:+.4f}, {ndcg['delta_ci_hi']:+.4f}]"
        )
    print(f"[rq4-soft-semantics] top-{args.depth} results -> {output}")


if __name__ == "__main__":
    main()
