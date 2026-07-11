"""Pairwise verdicts -> a ranking, and effectiveness evaluation against qrels.

This is the generic pairwise->ranking path. `copeland_ranking` aggregates *any*
collapsed pair-verdict frame (the LLM's cached verdicts today, axiom majority votes
later) into a ranked run compatible with ir_measures/pyterrier; nothing here is
LLM-specific. Copeland scoring over a *complete* top-k tournament (every pair scored)
is exactly PRP-allpair from the listwise-reranking literature (Qin et al., "Large
Language Models are Effective Text Rankers with Pairwise Ranking Prompting"), so runs
built this way are directly comparable to published PRP numbers.

The effectiveness gate (phase1-design.md §4) uses this module to check that the LLM's
cached verdicts actually make a ranker that beats BM25 on nDCG@10 before Phase 1's
top-10 residual is treated as skill rather than noise. The check costs zero new model
calls: it reuses the verdicts already in the preference store.
"""

import pandas as pd
from ir_measures import AP, nDCG

from axiomrank import paths

# nDCG@10 primary, AP (aggregates to MAP) secondary — the effectiveness gate metrics
# (phase1-design.md §4). Literature anchors: BM25 ~0.50 nDCG@10 on DL19/DL20, competent
# PRP-allpair setups ~0.65-0.70.
DEFAULT_METRICS = (nDCG @ 10, AP)


def copeland_ranking(verdicts: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Aggregate collapsed pair verdicts into a run via Copeland scoring.

    `verdicts` is any `agreement.model_pair_verdicts`-shaped frame: columns
    `query_id`, `doc_id_1`, `doc_id_2`, `model_pref` in {-1, 0, +1} (+1 prefers
    doc_id_1; position-inconsistent pairs already collapse to 0). `pool` is the
    first-stage frame (`qid`, `docno`, `rank`, `score`; `rank` is 0-based).

    Per query, a document's Copeland score is (wins - losses) over the pairs it appears
    in; ties (model_pref 0) contribute nothing. Documents that appear in at least one
    pair form the reranked block, ordered by Copeland score descending; equal scores are
    broken by first-stage rank (stable, deterministic). Pool documents that appear in no
    pair — the tail below the reranked top-k — keep their first-stage order strictly
    below every reranked document.

    Returns a run frame (`qid`, `docno`, `rank`, `score`) with `rank` the new 0-based
    rank and `score` = -rank, monotone with rank so ir_measures/pyterrier order it
    exactly as ranked.
    """
    rows = []
    for qid, pool_group in pool.groupby("qid", sort=False):
        query_verdicts = verdicts[verdicts["query_id"] == qid]

        copeland: dict[str, int] = {}
        reranked: set[str] = set()
        for v in query_verdicts.itertuples():
            reranked.add(v.doc_id_1)
            reranked.add(v.doc_id_2)
            if v.model_pref > 0:  # prefers doc_id_1
                copeland[v.doc_id_1] = copeland.get(v.doc_id_1, 0) + 1
                copeland[v.doc_id_2] = copeland.get(v.doc_id_2, 0) - 1
            elif v.model_pref < 0:  # prefers doc_id_2
                copeland[v.doc_id_1] = copeland.get(v.doc_id_1, 0) - 1
                copeland[v.doc_id_2] = copeland.get(v.doc_id_2, 0) + 1
            # model_pref == 0: a tie contributes nothing to either document's score.

        # Sort key: reranked block first (0) then the tail (1); within the reranked
        # block by Copeland descending; first-stage rank breaks every remaining tie and
        # is unique per query, so the order is fully deterministic regardless of sort
        # stability.
        ordered = sorted(
            pool_group.itertuples(),
            key=lambda d: (
                0 if d.docno in reranked else 1,
                -copeland.get(d.docno, 0),
                d.rank,
            ),
        )
        for new_rank, doc in enumerate(ordered):
            rows.append((qid, doc.docno, new_rank, float(-new_rank)))

    return pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])


def _load_qrels(dataset_id: str) -> pd.DataFrame:
    """Qrels of an ir_datasets dataset as an ir_measures-shaped frame."""
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
) -> tuple[pd.DataFrame, dict]:
    """Paired per-query comparison of two `evaluate_run` frames.

    Joins on `query_id` and, per metric, adds a `<metric>_delta` column
    (reranked - baseline). Returns the per-query frame and a summary dict with the
    baseline/reranked means, the mean delta and win/tie/loss counts per metric — an
    honest paired comparison, not two disembodied means.
    """
    names = metric_names(metrics)
    merged = baseline.merge(reranked, on="query_id", suffixes=("_baseline", "_reranked"))

    summary: dict[str, dict] = {}
    for name in names:
        base = merged[f"{name}_baseline"]
        rerank = merged[f"{name}_reranked"]
        delta = rerank - base
        merged[f"{name}_delta"] = delta
        summary[name] = {
            "mean_baseline": float(base.mean()),
            "mean_reranked": float(rerank.mean()),
            "mean_delta": float(delta.mean()),
            "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "losses": int((delta < 0).sum()),
            "n_queries": int(len(merged)),
        }
    return merged, summary
