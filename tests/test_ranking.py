"""Hand-computable cases for the pairwise->ranking effectiveness path (ranking.py).

Synthetic frames only — no JVM, no network, no shared-state reads. The Copeland cases
are worked out by hand in the comments so the expected orders are auditable.
"""

import numpy as np
import pandas as pd
import pytest

from axiomrank import ranking


def _order(run: pd.DataFrame, qid: str) -> list[str]:
    """The docnos of one query in ranked (rank-ascending) order."""
    q = run[run["qid"] == qid].sort_values("rank")
    return q["docno"].tolist()


def test_copeland_tournament_hand_computed():
    # One query, 4 docs, the complete top-4 tournament (6 canonical pairs). model_pref
    # +1 prefers doc_id_1. Pair (b,d) collapsed to a tie (position inconsistency).
    #   a beats b, c, d          -> Copeland +3
    #   b beats c, ties d, lost a -> +1 -1 = 0
    #   c loses to a, b, d        -> Copeland -3
    #   d beats c, ties b, lost a -> +1 -1 = 0
    # Order by Copeland desc, ties broken by first-stage rank: a(+3), b(0,rank1),
    # d(0,rank3), c(-3)  ->  a, b, d, c
    pool = pd.DataFrame(
        {
            "qid": ["q1"] * 4,
            "docno": ["a", "b", "c", "d"],
            "rank": [0, 1, 2, 3],
            "score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    verdicts = pd.DataFrame(
        {
            "query_id": ["q1"] * 6,
            "doc_id_1": ["a", "a", "a", "b", "b", "c"],
            "doc_id_2": ["b", "c", "d", "c", "d", "d"],
            "model_pref": [1, 1, 1, 1, 0, -1],  # (b,d) tie; (c,d) prefers d
        }
    )
    run = ranking.copeland_ranking(verdicts, pool)
    assert _order(run, "q1") == ["a", "b", "d", "c"]
    # score is monotone with rank (descending), so ir_measures orders it as ranked.
    ranked = run[run["qid"] == "q1"].sort_values("rank")
    assert ranked["score"].is_monotonic_decreasing
    assert ranked["rank"].tolist() == [0, 1, 2, 3]


def test_copeland_pool_tail_preserved():
    # a, b, c form the tournament; d, e are pool tail (in no pair) and must stay below the
    # reranked block in first-stage order. a beats b, a beats c, b beats c -> a, b, c.
    pool = pd.DataFrame(
        {
            "qid": ["q1"] * 5,
            "docno": ["a", "b", "c", "d", "e"],
            "rank": [0, 1, 2, 3, 4],
            "score": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    verdicts = pd.DataFrame(
        {
            "query_id": ["q1"] * 3,
            "doc_id_1": ["a", "a", "b"],
            "doc_id_2": ["b", "c", "c"],
            "model_pref": [1, 1, 1],
        }
    )
    run = ranking.copeland_ranking(verdicts, pool)
    assert _order(run, "q1") == ["a", "b", "c", "d", "e"]


def test_copeland_tail_below_even_when_reranked_loses():
    # A reranked doc with a negative Copeland score still ranks above every tail doc:
    # b loses to a but appeared in a pair, so it precedes the untouched tail c, d.
    pool = pd.DataFrame(
        {
            "qid": ["q1"] * 4,
            "docno": ["a", "b", "c", "d"],
            "rank": [0, 1, 2, 3],
            "score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    verdicts = pd.DataFrame(
        {
            "query_id": ["q1"],
            "doc_id_1": ["a"],
            "doc_id_2": ["b"],
            "model_pref": [1],  # a beats b
        }
    )
    run = ranking.copeland_ranking(verdicts, pool)
    assert _order(run, "q1") == ["a", "b", "c", "d"]


def test_copeland_multi_query_independent():
    # q1's tournament keeps first-stage order; q2's flips it. Neither leaks into the other.
    pool = pd.DataFrame(
        {
            "qid": ["q1", "q1", "q2", "q2"],
            "docno": ["a", "b", "x", "y"],
            "rank": [0, 1, 0, 1],
            "score": [2.0, 1.0, 2.0, 1.0],
        }
    )
    verdicts = pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "doc_id_1": ["a", "x"],
            "doc_id_2": ["b", "y"],
            "model_pref": [1, -1],  # q1 keeps a>b; q2 prefers y over x
        }
    )
    run = ranking.copeland_ranking(verdicts, pool)
    assert _order(run, "q1") == ["a", "b"]
    assert _order(run, "q2") == ["y", "x"]


def test_compare_runs_hand_computed():
    baseline = pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3"],
            "nDCG@10": [0.5, 0.5, 0.5],
            "AP": [0.4, 0.4, 0.4],
        }
    )
    reranked = pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3"],
            "nDCG@10": [0.7, 0.5, 0.3],  # +0.2, 0, -0.2  -> W/T/L = 1/1/1
            "AP": [0.5, 0.4, 0.3],  # +0.1, 0, -0.1     -> W/T/L = 1/1/1
        }
    )
    per_query, summary = ranking.compare_runs(baseline, reranked)

    ndcg = summary["nDCG@10"]
    assert ndcg["mean_baseline"] == 0.5
    assert ndcg["mean_reranked"] == 0.5
    assert abs(ndcg["mean_delta"]) < 1e-12
    assert (ndcg["wins"], ndcg["ties"], ndcg["losses"]) == (1, 1, 1)
    assert ndcg["n_queries"] == 3
    assert ndcg["delta_ci_lo"] <= 0 <= ndcg["delta_ci_hi"]
    assert ndcg["n_bootstrap"] == 10_000
    assert summary["AP"]["wins"] == 1 and summary["AP"]["losses"] == 1

    row = per_query[per_query["query_id"] == "q1"].iloc[0]
    assert abs(row["nDCG@10_delta"] - 0.2) < 1e-12


def test_compare_runs_bootstrap_ci_is_paired_over_queries():
    baseline = pd.DataFrame({"query_id": ["q1", "q2"], "nDCG@10": [0.2, 0.8], "AP": [0.1, 0.7]})
    reranked = pd.DataFrame({"query_id": ["q1", "q2"], "nDCG@10": [0.3, 0.9], "AP": [0.2, 0.8]})
    _, summary = ranking.compare_runs(baseline, reranked, n_bootstrap=500, seed=3)
    assert np.isclose(summary["nDCG@10"]["delta_ci_lo"], 0.1)
    assert np.isclose(summary["nDCG@10"]["delta_ci_hi"], 0.1)


def test_compare_runs_rejects_mismatched_query_sets():
    baseline = pd.DataFrame({"query_id": ["q1"], "nDCG@10": [0.2], "AP": [0.1]})
    reranked = pd.DataFrame({"query_id": ["q2"], "nDCG@10": [0.3], "AP": [0.2]})
    with pytest.raises(ValueError, match="do not match exactly"):
        ranking.compare_runs(baseline, reranked)


def test_evaluate_run_metric_plumbing():
    # Injected in-memory qrels keep this fast and dataset-free. A ranking that puts the
    # single relevant doc first scores a perfect nDCG@10 and AP of 1.0.
    qrels = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2"],
            "doc_id": ["a", "b", "c", "d"],
            "relevance": [1, 0, 1, 0],
        }
    )
    run = pd.DataFrame(
        {
            "qid": ["q1", "q1", "q2", "q2"],
            "docno": ["a", "b", "c", "d"],
            "rank": [0, 1, 0, 1],
            "score": [2.0, 1.0, 2.0, 1.0],  # relevant doc first in both queries
        }
    )
    per_query = ranking.evaluate_run(run, qrels=qrels)
    assert list(per_query.columns) == ["query_id", "nDCG@10", "AP"]
    assert per_query["query_id"].tolist() == ["q1", "q2"]
    assert (per_query["nDCG@10"] == 1.0).all()
    assert (per_query["AP"] == 1.0).all()


def test_evaluate_run_penalises_bad_order():
    # Sanity: an irrelevant doc ranked first must drop nDCG@10 below the perfect case.
    qrels = pd.DataFrame(
        {"query_id": ["q1", "q1"], "doc_id": ["a", "b"], "relevance": [1, 0]}
    )
    run = pd.DataFrame(
        {"qid": ["q1", "q1"], "docno": ["a", "b"], "score": [1.0, 2.0]}  # b (irrelevant) first
    )
    per_query = ranking.evaluate_run(run, qrels=qrels)
    assert per_query["nDCG@10"].iloc[0] < 1.0
