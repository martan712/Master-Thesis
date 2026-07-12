"""Hand-computable cases for the Phase 2 decomposition analyses (phase2-design.md §3)."""

import numpy as np
import pandas as pd

from axiomrank import analysis
from axiomrank.analysis.covariates import attach_covariates
from axiomrank.analysis.decomposition import (
    decompose,
    information_decomposition,
    reliability_ceiling,
)
from axiomrank.analysis.residual import residual_clusters, residual_model, residual_profiles
from axiomrank.config import load_config
from axiomrank.paths import CONFIGS_DIR


# --- covariates ------------------------------------------------------------------

def test_attach_covariates_values():
    merged = pd.DataFrame(
        {"query_id": ["q1"], "doc_id_1": ["a"], "doc_id_2": ["b"], "model_pref": [1]}
    )
    pool = pd.DataFrame(
        {"qid": ["q1", "q1"], "docno": ["a", "b"], "rank": [0, 3], "score": [10.0, 4.0]}
    )
    pairs = pd.DataFrame(
        {
            "qid": ["q1"], "query": ["apple banana"], "doc_id_1": ["a"], "doc_id_2": ["b"],
            "text_1": ["apple banana cat"], "text_2": ["apple dog"],
        }
    )
    store = pd.DataFrame(
        {
            "query_id": ["q1", "q1"], "doc_id_a": ["a", "b"], "doc_id_b": ["b", "a"],
            "prob_a": [0.8, 0.3], "score_a": [2.0, 0.5], "score_b": [1.0, 1.5],
        }
    )
    out = attach_covariates(merged, pool, pairs, store).iloc[0]
    assert out["d_rank"] == -3 and out["rank_gap"] == 3 and out["rank_max"] == 3
    assert out["d_score"] == 6.0 and out["score_gap"] == 6.0 and out["score_max"] == 10.0
    assert out["d_len"] == 1.0 and out["len_max"] == 3.0 and out["len_ratio"] == 1.5
    assert out["d_qcov"] == 0.5  # cov_1 = 2/2, cov_2 = 1/2
    assert out["query_len"] == 2.0 and out["query_is_question"] == 0.0
    assert np.isclose(out["conf_margin_prob"], 0.25)  # mean(|0.8-.5|, |0.3-.5|)
    assert np.isclose(out["conf_margin_score"], 1.0)  # mean(|2-1|, |0.5-1.5|)


def test_attach_covariates_missing_confidence_is_nan():
    merged = pd.DataFrame(
        {"query_id": ["q1"], "doc_id_1": ["a"], "doc_id_2": ["b"], "model_pref": [1]}
    )
    pool = pd.DataFrame(
        {"qid": ["q1", "q1"], "docno": ["a", "b"], "rank": [0, 1], "score": [5.0, 1.0]}
    )
    pairs = pd.DataFrame(
        {"qid": ["q1"], "query": ["q?"], "doc_id_1": ["a"], "doc_id_2": ["b"],
         "text_1": ["x y"], "text_2": ["z"]}
    )
    out = attach_covariates(merged, pool, pairs, store_df=None).iloc[0]
    assert np.isnan(out["conf_margin_prob"]) and np.isnan(out["conf_margin_score"])
    assert out["query_is_question"] == 1.0  # contains '?'


# --- decomposition ---------------------------------------------------------------

def test_reliability_ceiling_boundaries():
    assert reliability_ceiling(1.0) == 1.0
    assert reliability_ceiling(0.5) == 0.5
    assert reliability_ceiling(0.4) == 0.5  # below chance clamps
    assert reliability_ceiling(None) is None
    assert np.isclose(reliability_ceiling(0.75), 0.5 * (1 + np.sqrt(0.5)))
    assert reliability_ceiling(0.9) > reliability_ceiling(0.7)  # monotone


def test_information_decomposition_perfect_vs_null():
    y = np.array([1.0, 1.0, 0.0, 0.0])
    info = information_decomposition(y, np.array([0.99, 0.99, 0.01, 0.01]))
    assert np.isclose(info["ce_null"], np.log(2), atol=1e-6)  # prior 0.5
    assert info["pseudo_r2"] > 0.98 and info["log_loss_gain"] > 0.6
    # A null-quality predictor removes ~no entropy.
    flat = information_decomposition(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert np.isclose(flat["pseudo_r2"], 0.0, atol=1e-6)


def test_decompose_perfect_feature():
    rows = []
    for qid in ["q1", "q2", "q3", "q4", "q5"]:
        for i, pref in enumerate([1, 1, -1, -1]):
            rows.append({"query_id": qid, "doc_id_1": f"d{i}", "doc_id_2": f"e{i}",
                         "model_pref": pref, "AX": pref})
    result, oof = decompose(pd.DataFrame(rows), ["AX"], position_consistency=0.75, nonlinear=True)
    assert result["cv_accuracy"] == 1.0
    assert result["information"]["pseudo_r2"] > 0.5
    assert np.isclose(result["reliability_ceiling"], 0.5 * (1 + np.sqrt(0.5)))
    assert result["reducible_residual_upper"] == 0.0  # ceiling below the achieved accuracy
    assert not oof["is_residual"].any()
    assert "nonlinear_headroom" in result


# --- residual --------------------------------------------------------------------

def _oof_frame(d_len, y, oof_prob, n_q=10, per_q=4, seed=0):
    rows = []
    k = 0
    for qi in range(n_q):
        for j in range(per_q):
            rows.append({
                "query_id": f"q{qi}", "doc_id_1": f"a{k}", "doc_id_2": f"b{k}",
                "y_true": bool(y[k]), "oof_prob": float(oof_prob[k]),
                "oof_correct": bool((oof_prob[k] >= 0.5) == y[k]),
                "is_residual": bool((oof_prob[k] >= 0.5) != y[k]),
                "d_len": float(d_len[k]),
            })
            k += 1
    return pd.DataFrame(rows)


def test_residual_model_detects_planted_signal():
    n = 40
    y = np.tile([1, 1, 0, 0], 10)
    d_len = np.where(y == 1, 1.0, -1.0)          # content covariate perfectly aligned
    oof_prob = np.full(n, 0.5)                     # axioms know nothing
    frame = _oof_frame(d_len, y, oof_prob)
    res = residual_model(frame, ["d_len"], n_boot=300)
    assert res["lift"] > 0.3 and res["lift_ci_lo"] > 0.0  # signal, CI above zero


def test_residual_model_null_on_noise():
    rng = np.random.default_rng(1)
    n = 40
    y = np.tile([1, 1, 0, 0], 10)
    d_len = rng.normal(size=n)                     # uncorrelated with y
    oof_prob = np.full(n, 0.5)
    frame = _oof_frame(d_len, y, oof_prob)
    res = residual_model(frame, ["d_len"], n_boot=300)
    assert res["lift_ci_lo"] <= 0.0 <= res["lift_ci_hi"]  # CI spans zero


def test_residual_profiles_and_clusters():
    n = 40
    y = np.tile([1, 1, 0, 0], 10)
    d_len = np.where(y == 1, 1.0, -1.0)
    oof_prob = np.where(y == 1, 0.4, 0.6)          # axiom wrong on everything -> all residual
    frame = _oof_frame(d_len, y, oof_prob)
    prof = residual_profiles(frame, ["d_len"], n_bins=2)
    assert set(prof.columns) >= {"covariate", "bin", "oof_accuracy", "residual_rate"}
    assert (prof["residual_rate"] == 1.0).all()    # every pair is a residual here
    clusters = residual_clusters(frame, ["d_len"], n_clusters=2)
    assert len(clusters) == 2 and clusters["size"].sum() == n


# --- config ----------------------------------------------------------------------

def test_sources_config_parses():
    cfg = load_config(CONFIGS_DIR / "rq3_smoke.yaml")
    assert cfg.sources == ["configs/p1_smoke.yaml"]
    default = load_config(CONFIGS_DIR / "rq2_dl19_top10.yaml")
    assert default.sources == []  # absent -> empty, older configs stay valid
