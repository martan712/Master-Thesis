"""Hand-computable tests for query-disjoint fitted axiom surrogates."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from axiomrank import ranking


def _rq4_runner():
    path = Path(__file__).parent.parent / "experiments" / "rq4_axioms" / "run.py"
    spec = importlib.util.spec_from_file_location("rq4_runner_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _frame(n_queries=6):
    rows = []
    for query in range(n_queries):
        # The tied target row is intentionally given AX=+1. It is excluded from fitting
        # but must still receive an OOF prediction for complete-tournament reranking.
        for pair, (target, axiom) in enumerate(((1, 1.0), (-1, -1.0), (0, 1.0))):
            rows.append(
                {
                    "query_id": f"q{query}",
                    "doc_id_1": f"a{query}-{pair}",
                    "doc_id_2": f"b{query}-{pair}",
                    "model_pref": target,
                    "AX": axiom,
                }
            )
    return pd.DataFrame(rows)


def test_oof_surrogate_is_query_disjoint_deterministic_and_predicts_ties():
    frame = _frame()
    folds = ranking.assign_query_folds(frame["query_id"], n_folds=3)
    first, metadata = ranking.fit_oof_surrogate(
        frame, ["AX"], group_col="query_id", folds=folds, seed=7
    )
    second, metadata_2 = ranking.fit_oof_surrogate(
        frame, ["AX"], group_col="query_id", folds=folds, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert metadata == metadata_2
    assert first.groupby("group_id")["fold"].nunique().eq(1).all()
    assert first["surrogate_prob"].notna().all()
    assert first.loc[first["target_pref"] == 0, "surrogate_pref"].eq(1).all()

    test_groups = [set(model["test_groups"]) for model in metadata["fold_models"]]
    assert set.union(*test_groups) == set(frame["query_id"])
    assert all(a.isdisjoint(b) for i, a in enumerate(test_groups) for b in test_groups[i + 1 :])
    assert metadata["full_model"]["coefficients"]["AX"] > 0

    fidelity = metadata["fidelity"]
    assert fidelity["n_target_ties"] == 6
    assert fidelity["n_forced_decisions_on_target_ties"] == 6
    assert fidelity["decisive_pair_accuracy"] == 1.0


def test_fold_assignment_rejects_split_query_and_single_class_training():
    frame = _frame(n_queries=2)
    split_query = np.arange(len(frame)) % 2
    with pytest.raises(ValueError, match="exactly one fold"):
        ranking.fit_oof_surrogate(
            frame, ["AX"], group_col="query_id", folds=split_query
        )

    one_class = frame.copy()
    one_class["model_pref"] = np.where(one_class["model_pref"] == 0, 0, 1)
    with pytest.raises(ValueError, match="single-class"):
        ranking.fit_oof_surrogate(one_class, ["AX"], group_col="query_id")


def test_surrogate_fidelity_counts_exact_probability_ties_without_forcing():
    predictions = pd.DataFrame(
        {
            "group_id": ["q1", "q1", "q2"],
            "fold": [0, 0, 1],
            "target_pref": [1, 0, -1],
            "surrogate_prob": [0.9, 0.5, 0.1],
            "surrogate_pref": [1, 0, -1],
            "correct_on_decisive": pd.array([True, pd.NA, True], dtype="boolean"),
        }
    )
    stats = ranking.surrogate_fidelity(predictions)
    assert stats["n_target_ties"] == 1
    assert stats["n_surrogate_ties"] == 1
    assert stats["n_forced_decisions_on_target_ties"] == 0
    assert stats["decisive_pair_accuracy"] == 1.0


def test_variant_residual_profiles_use_only_decisive_oof_targets():
    pooled = pd.DataFrame(
        {
            "d_len": [1, 2, 3, 4, 5, 6],
            "d_qcov": [-1, 0, 1, -1, 0, 1],
            "d_rank": [-3, -2, -1, 1, 2, 3],
            "rank_gap": [3, 2, 1, 1, 2, 3],
        }
    )
    predictions = pd.DataFrame(
        {
            "group_id": ["q1", "q1", "q1", "q2", "q2", "q2"],
            "target_pref": [1, -1, 0, 1, -1, 0],
            "classical_correct": pd.array([True, False, pd.NA, False, True, pd.NA], dtype="boolean"),
            "plus_both_correct": pd.array([True, True, pd.NA, True, True, pd.NA], dtype="boolean"),
        }
    )
    profile = _rq4_runner()._variant_residual_profiles(
        pooled, predictions, ["classical", "plus_both"]
    )
    assert set(profile["variant"]) == {"classical", "plus_both"}
    assert set(profile["covariate"]) == {"d_len", "d_qcov", "d_rank", "rank_gap"}
    # Within each variant/covariate, bins partition exactly the four decisive rows.
    totals = profile.groupby(["variant", "covariate"])["n_pairs"].sum()
    assert totals.eq(4).all()
    assert profile.loc[profile["variant"] == "plus_both", "residual_rate"].eq(0).all()
