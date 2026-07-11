"""Hand-computable cases for the Phase 1 analyses (phase1-design.md §4.3)."""

import numpy as np
import pandas as pd

from axiomrank import analysis


def test_agreement_point_estimates_match_definitions():
    merged = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q2"],
            "doc_id_1": ["a", "a", "b", "a"],
            "doc_id_2": ["b", "c", "c", "b"],
            "model_pref": [1, -1, 0, 1],
            "AX": [1, 1, 1, 0],
        }
    )
    table = analysis.agreement_with_ci(merged, ["AX"], n_bootstrap=200)
    row = table.iloc[0]
    assert row["n_pairs"] == 4
    assert row["coverage"] == 0.75  # AX non-neutral on 3 of 4 pairs
    assert row["n_evaluable"] == 2  # model decisive on 2 of those
    assert row["agreement"] == 0.5  # agrees on the q1 a-b pair, not on a-c
    assert 0.0 <= row["ci_lo"] <= 0.5 <= row["ci_hi"] <= 1.0


def test_agreement_ci_degenerates_to_point_on_identical_queries():
    # Both queries contribute the same counts, so every bootstrap draw gives 1.0.
    merged = pd.DataFrame(
        {
            "query_id": ["q1", "q2"],
            "doc_id_1": ["a", "a"],
            "doc_id_2": ["b", "b"],
            "model_pref": [1, 1],
            "AX": [1, 1],
        }
    )
    row = analysis.agreement_with_ci(merged, ["AX"], n_bootstrap=100).iloc[0]
    assert row["agreement"] == row["ci_lo"] == row["ci_hi"] == 1.0


def test_joint_fit_perfect_single_feature():
    rows = []
    for qid in ["q1", "q2", "q3", "q4"]:
        for i, pref in enumerate([1, 1, -1, -1]):
            rows.append(
                {
                    "query_id": qid,
                    "doc_id_1": f"d{i}",
                    "doc_id_2": f"e{i}",
                    "model_pref": pref,
                    "AX": pref,
                }
            )
    # A tied pair must be excluded from the fit entirely.
    rows.append({"query_id": "q1", "doc_id_1": "t", "doc_id_2": "u", "model_pref": 0, "AX": 1})
    stats, oof = analysis.joint_fit(pd.DataFrame(rows), ["AX"])

    assert stats["n_decisive_pairs"] == 16
    assert stats["base_rate"] == 0.5
    assert stats["majority_vote_accuracy"] == 1.0
    assert stats["cv_accuracy"] == 1.0
    assert stats["cv_auc"] == 1.0
    assert stats["n_folds"] == 4  # capped at the number of queries
    assert stats["coefficients"]["AX"] > 0
    assert len(oof) == 16
    assert oof["oof_correct"].all()


def test_attach_rank_gap_from_pool():
    pool = pd.DataFrame({"qid": ["q1"] * 3, "docno": ["a", "b", "c"], "rank": [0, 4, 9]})
    pairs = pd.DataFrame({"qid": ["q1", "q1"], "doc_id_1": ["a", "b"], "doc_id_2": ["b", "c"]})
    out = analysis.attach_rank_gap(pairs, pool)
    assert out["rank_gap"].tolist() == [4, 5]


def test_gap_bins_integer_for_small_ranges():
    gaps = pd.Series([1, 3, 9, 15])
    assert analysis._gap_bins(gaps).tolist() == [1, 3, 9, 15]


def test_gap_bins_quantile_for_wide_ranges():
    gaps = pd.Series(range(1, 101))
    bins = analysis._gap_bins(gaps)
    assert bins.nunique() == 10
    assert bins.value_counts().eq(10).all()  # deciles of a uniform range
    assert bins.max() == 100  # labelled by the bin's upper gap edge


def test_gap_gradient_hand_computed():
    merged = pd.DataFrame(
        {
            "query_id": ["q1"] * 4,
            "doc_id_1": list("abcd"),
            "doc_id_2": list("wxyz"),
            "model_pref": [1, 1, 0, -1],
            "AX": [1, -1, 1, -1],
            "rank_gap": [1, 1, 2, 2],
            "n_presentations": [2] * 4,
            "position_consistent": [True, True, False, True],
        }
    )
    oof = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "doc_id_1": ["a", "b"],
            "doc_id_2": ["w", "x"],
            "oof_correct": [True, False],
        }
    )
    gradient = analysis.gap_gradient(merged, ["AX"], oof=oof)
    assert len(gradient) == 2  # two gap bins x one axiom

    bin1 = gradient[gradient["gap_bin"] == 1].iloc[0]
    assert bin1["n_pairs"] == 2
    assert bin1["decisive_rate"] == 1.0
    assert bin1["position_consistency"] == 1.0
    assert bin1["n_evaluable"] == 2
    assert bin1["agreement"] == 0.5
    assert bin1["joint_cv_accuracy"] == 0.5

    bin2 = gradient[gradient["gap_bin"] == 2].iloc[0]
    assert bin2["n_pairs"] == 2
    assert bin2["decisive_rate"] == 0.5  # the tied pair is not decisive
    assert bin2["position_consistency"] == 0.5
    assert bin2["n_evaluable"] == 1  # tied pair is covered by AX but not evaluable
    assert bin2["agreement"] == 1.0
    assert np.isnan(bin2["joint_cv_accuracy"])  # no OOF predictions in this bin
