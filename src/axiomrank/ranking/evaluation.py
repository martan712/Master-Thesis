"""Effectiveness evaluation of runs against qrels, via ir_measures.

The effectiveness reference (phase1-design.md §4.4) uses this module to check that the LLM's
cached verdicts actually make a ranker that beats BM25 on nDCG@10 before Phase 1's
top-10 residual is treated as skill rather than noise. The check costs zero new model
calls: it reuses the verdicts already in the preference store.
"""

import numpy as np
import pandas as pd
from ir_measures import AP, nDCG

from axiomrank import paths
from axiomrank.confirmation import assert_dataset_access_allowed

# nDCG@10 primary, AP (aggregates to MAP) secondary — the effectiveness-reference metrics
# (phase1-design.md §4). Literature anchors: BM25 ~0.50 nDCG@10 on DL19/DL20, competent
# PRP-allpair setups ~0.65-0.70.
DEFAULT_METRICS = (nDCG @ 10, AP)


def _load_qrels(dataset_id: str) -> pd.DataFrame:
    """Qrels of an ir_datasets dataset as an ir_measures-shaped frame."""
    assert_dataset_access_allowed(dataset_id)
    paths.configure_caches()  # point ir_datasets at data/cache before import
    import ir_datasets

    dataset = ir_datasets.load(dataset_id)
    rows = [
        (q.query_id, q.doc_id, q.relevance) for q in dataset.qrels_iter()
    ]
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "relevance"])


def metric_names(metrics=DEFAULT_METRICS) -> list[str]:
    """Canonical string names of the measures, e.g. ['nDCG@10', 'AP']."""
    return [str(m) for m in metrics]


def evaluate_run(
    run: pd.DataFrame,
    dataset_id: str | None = None,
    metrics=DEFAULT_METRICS,
    qrels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score a run against qrels with ir_measures; return one row per query.

    `run` is a (`qid`/`query_id`, `docno`/`doc_id`, `score`) frame. Qrels come from
    `dataset_id` (an ir_datasets id) by default, or from an explicit `qrels` frame
    (`query_id`, `doc_id`, `relevance`) — the latter keeps the metric plumbing testable
    without a dataset download. Output: `query_id` plus one column per metric name.
    """
    import ir_measures

    if qrels is None:
        if dataset_id is None:
            raise ValueError("evaluate_run needs either dataset_id or an explicit qrels frame")
        qrels = _load_qrels(dataset_id)

    run_df = run.rename(columns={"qid": "query_id", "docno": "doc_id"})
    run_df = run_df[["query_id", "doc_id", "score"]]

    metrics = list(metrics)
    per_query: dict[str, dict[str, float]] = {}
    for m in ir_measures.iter_calc(metrics, qrels, run_df):
        per_query.setdefault(m.query_id, {})[str(m.measure)] = m.value

    rows = [{"query_id": qid, **values} for qid, values in per_query.items()]
    columns = ["query_id", *metric_names(metrics)]
    return pd.DataFrame(rows, columns=columns).sort_values("query_id").reset_index(drop=True)


def compare_runs(
    baseline: pd.DataFrame,
    reranked: pd.DataFrame,
    metrics=DEFAULT_METRICS,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Paired per-query comparison of two `evaluate_run` frames.

    Joins on `query_id` and, per metric, adds a `<metric>_delta` column
    (reranked - baseline). Returns the per-query frame and a summary dict with the
    baseline/reranked means, the mean delta, its paired query-bootstrap 95% interval,
    and win/tie/loss counts per metric — an honest paired comparison, not two
    disembodied means. Queries are the independent resampling unit.
    """
    names = metric_names(metrics)
    merged = baseline.merge(
        reranked,
        on="query_id",
        suffixes=("_baseline", "_reranked"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"baseline/reranked query sets do not match exactly: {counts}")
    merged = merged.drop(columns="_merge")

    summary: dict[str, dict] = {}
    for name in names:
        base = merged[f"{name}_baseline"]
        rerank = merged[f"{name}_reranked"]
        delta = rerank - base
        merged[f"{name}_delta"] = delta
        values = delta.to_numpy(dtype=float)
        if len(values) and n_bootstrap > 0:
            rng = np.random.default_rng(seed)
            draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
            boot = draws.mean(axis=1)
            ci_lo, ci_hi = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
        else:
            ci_lo = ci_hi = float("nan")
        summary[name] = {
            "mean_baseline": float(base.mean()),
            "mean_reranked": float(rerank.mean()),
            "mean_delta": float(delta.mean()),
            "delta_ci_lo": ci_lo,
            "delta_ci_hi": ci_hi,
            "n_bootstrap": int(n_bootstrap),
            "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()),
            "n_queries": int(len(merged)),
        }
    return merged, summary
