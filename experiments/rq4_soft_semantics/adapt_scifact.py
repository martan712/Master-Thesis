"""Small-label exploratory SciFact adaptation of the RQ4 pointwise distiller.

Thirty query groups are selected before any SciFact qrels are read.  Qwen adequacy labels
only their top-10 documents; the ridge model fits only those labels and effectiveness is
reported exclusively on the remaining query groups.  This is exploratory adaptation, not
the locked NFCorpus confirmation protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from axiomrank import paths, ranking
from axiomrank.config import load_config
from axiomrank.pipeline import stages


FEATURES = [
    "bm25_score", "bm25_rank", "log_length", "query_coverage",
    "relevance_probability", "nli_entailment", "mrc_margin",
]
RIDGE_ALPHA = 1.0
LABELS = ("0", "1", "2", "3")
PROMPT = (
    "You are rating a single passage against a question.\n"
    "Rate how completely the passage ANSWERS the question — not merely whether it is on "
    "the same topic.\n\n"
    "Scale:\n"
    "0 = does not answer at all (off-topic, or only boilerplate / navigation / metadata)\n"
    "1 = on the topic but does not contain the answer\n"
    "2 = partially answers the question\n"
    "3 = directly and completely answers the question\n\n"
    'Question: "{query}"\n'
    "Passage: {passage}\n\n"
    "Reply with a single digit 0, 1, 2, or 3:"
)


def _label_path() -> Path:
    return paths.DATA_DIR / "adequacy" / "scifact_adapt_v1" / "labels.parquet"


def _score(client, model: str, extra_body: dict, query: str, passage: str, max_chars: int) -> float:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(query=query, passage=passage[:max_chars])}],
        temperature=0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=20,
        extra_body=extra_body or None,
    )
    content = response.choices[0].logprobs.content
    if not content:
        return float("nan")
    best: dict[str, float] = {}
    for token in content[0].top_logprobs:
        label = token.token.strip()
        if label in LABELS:
            best[label] = max(best.get(label, float("-inf")), token.logprob)
    if not best:
        return float("nan")
    maximum = max(best.values())
    weights = {label: math.exp(value - maximum) for label, value in best.items()}
    total = sum(weights.values())
    return sum(int(label) * weights.get(label, 0.0) / total for label in LABELS)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-config", default="configs/rq4_soft_semantics_scifact_top10.yaml")
    parser.add_argument("--label-source-config", default="configs/rq2_dl19_top10.yaml")
    parser.add_argument("--train-queries", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-labels", action="store_true")
    args = parser.parse_args()
    if args.train_queries <= 0:
        raise SystemExit("--train-queries must be positive")

    paths.configure_caches()
    target_cfg = load_config(paths.PROJECT_ROOT / args.target_config)
    pool = stages.build_pool(target_cfg, refresh=False)
    collection = target_cfg.variant or target_cfg.dataset.irds_id.replace("/", "_")
    feature_path = paths.DATA_DIR / "soft_semantics_features" / "v1" / f"{collection}_top10.parquet"
    features = pd.read_parquet(feature_path)
    for frame in (pool, features):
        frame["qid"] = frame["qid"].astype(str)
        frame["docno"] = frame["docno"].astype(str)
    frame = pool.merge(features, on=["qid", "docno"], validate="one_to_one")
    qids = np.array(sorted(frame["qid"].unique()))
    if args.train_queries >= len(qids):
        raise SystemExit("--train-queries must leave at least one held-out SciFact query")
    rng = np.random.default_rng(args.seed)
    train_qids = set(rng.choice(qids, size=args.train_queries, replace=False).tolist())
    test_qids = set(qids) - train_qids

    out_dir = paths.results_dir("rq4_soft_semantics") / "scifact_adapt_30q"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split.json").write_text(
        json.dumps({"seed": args.seed, "train_qids": sorted(train_qids), "test_qids": sorted(test_qids)}, indent=2)
        + "\n"
    )
    label_docs = frame[frame["qid"].isin(train_qids)][["qid", "docno", "query", "text"]]
    label_path = _label_path()
    cached = pd.DataFrame(columns=["qid", "docno", "adequacy"])
    if label_path.exists() and not args.refresh_labels:
        cached = pd.read_parquet(label_path)
        cached["qid"] = cached["qid"].astype(str)
        cached["docno"] = cached["docno"].astype(str)
        cached = cached.drop_duplicates(["qid", "docno"], keep="last")
    known = set(zip(cached["qid"], cached["docno"]))
    todo = [row for row in label_docs.itertuples(index=False) if (row.qid, row.docno) not in known]
    source_cfg = load_config(paths.PROJECT_ROOT / args.label_source_config)
    ranker = next(rank for rank in source_cfg.all_rankers if "qwen" in (rank.model or ""))
    from openai import OpenAI

    client = OpenAI(base_url=ranker.base_url, api_key=os.environ.get("OPENAI_API_KEY", "unused"), max_retries=3)
    additions = []
    for row in tqdm(todo, desc="Qwen SciFact adaptation labels", unit="document"):
        additions.append(
            {
                "qid": row.qid,
                "docno": row.docno,
                "adequacy": _score(client, ranker.model, ranker.extra_body, row.query, row.text, ranker.max_chars),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(additions) >= 25:
            cached = pd.concat([cached, pd.DataFrame(additions)], ignore_index=True)
            _atomic_parquet(label_path, cached)
            additions.clear()
    if additions:
        cached = pd.concat([cached, pd.DataFrame(additions)], ignore_index=True)
        _atomic_parquet(label_path, cached)
    labels = cached.merge(label_docs[["qid", "docno"]], on=["qid", "docno"], how="inner")
    if len(labels) != len(label_docs) or labels["adequacy"].isna().any():
        raise RuntimeError("incomplete Qwen adaptation labels")
    train = frame.merge(labels[["qid", "docno", "adequacy"]], on=["qid", "docno"], validate="one_to_one")
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train[FEATURES], train["adequacy"])
    test = frame[frame["qid"].isin(test_qids)].copy()
    test["prediction"] = np.clip(model.predict(test[FEATURES]), 0.0, 3.0)
    score_map = {(row.qid, row.docno): float(row.prediction) for row in test.itertuples(index=False)}
    test_pool = pool[pool["qid"].isin(test_qids)].copy()
    reranked = ranking.rerank_scored_pool(test_pool, score_map, depth=10)
    baseline = ranking.evaluate_run(test_pool, target_cfg.dataset.irds_id)
    measured = ranking.evaluate_run(reranked, target_cfg.dataset.irds_id)
    # ir_measures includes qrels-only queries as zero rows.  Remove the labelled
    # adaptation queries explicitly so this is a strictly held-out evaluation.
    held_out = pd.Series(sorted(test_qids), dtype=str)
    baseline = baseline[baseline["query_id"].astype(str).isin(held_out)].reset_index(drop=True)
    measured = measured[measured["query_id"].astype(str).isin(held_out)].reset_index(drop=True)
    _, comparison = ranking.compare_runs(baseline, measured, seed=args.seed)
    report = {
        "status": "exploratory SciFact adaptation; no held-out target Qwen labels or qrels used in fitting",
        "n_train_queries": len(train_qids),
        "n_train_documents": len(train),
        "n_test_queries": len(test_qids),
        "n_test_documents": len(test),
        "features": FEATURES,
        "ridge_alpha": RIDGE_ALPHA,
        "metrics": comparison,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    ndcg = comparison["nDCG@10"]
    print(f"SciFact adaptation: nDCG@10 {ndcg['mean_baseline']:.4f} -> {ndcg['mean_reranked']:.4f} ({ndcg['mean_delta']:+.4f})")


if __name__ == "__main__":
    main()
