"""RQ3 out-of-fold prediction and error summaries (phase2-design.md §3).

Given the combined axiom model (the Phase 1 L2 logistic, :func:`analysis.joint_fit`), report
out-of-fold correct/error partitions and an **information** summary: normalised log-loss gain
relative to the fold-local null. These are predictive quantities, not mechanistic fractions.
Order-swap agreement is retained only as an explicitly assumption-dependent
sensitivity calculation; it is not an identified reliability ceiling or noise decomposition.
"""

import numpy as np
import pandas as pd

from axiomrank.analysis.joint import grouped_splits, joint_fit

_EPS = 1e-12


def single_order_accuracy_sensitivity(position_consistency: float | None) -> float | None:
    """Independent-error sensitivity implied by order-swap agreement.

    If two presentation orders were exchangeable, conditionally independent noisy
    measurements of one latent preference, and if they had equal error rates, then if they
    agree with probability ``c``, and each measurement matches the latent label with
    probability ``a``, then ``c = a² + (1−a)²`` (both right or both wrong). Solving gives
    ``a = ½(1 + √(2c−1))``. Order swaps are interventions and need not satisfy those
    assumptions, so this is a sensitivity diagnostic for a *single presentation*, not an
    identified reliability ceiling for the collapsed two-order label. Returns None if c
    is missing; values below chance map to 0.5 under the hypothetical model.
    """
    if position_consistency is None or not np.isfinite(position_consistency):
        return None
    c = float(position_consistency)
    if c <= 0.5:
        return 0.5
    return float(0.5 * (1.0 + np.sqrt(2.0 * c - 1.0)))


# Historical API alias. The quantity is not an identified reliability ceiling; new code
# should use the assumption-explicit name above.
reliability_ceiling = single_order_accuracy_sensitivity


def _cross_entropy(y: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(prob, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def information_decomposition(
    y: np.ndarray,
    oof_prob: np.ndarray,
    null_prob: np.ndarray | None = None,
) -> dict:
    """Fraction of label entropy removed relative to a class-prior null.

    ``pseudo_r2`` is McFadden's 1 − CE(model)/CE(null) on the out-of-fold probabilities;
    ``log_loss_gain`` is the absolute cross-entropy reduction in nats. In decomposition,
    ``null_prob`` contains training-fold class priors, keeping the null fully out of fold.
    The full-data prior remains available only for small standalone calculations.
    """
    if null_prob is None:
        null_prob = np.full_like(y, float(y.mean()), dtype=float)
    ce_null = _cross_entropy(y, np.asarray(null_prob, dtype=float))
    ce_model = _cross_entropy(y, oof_prob)
    return {
        "ce_null": ce_null,
        "ce_model": ce_model,
        "log_loss_gain": ce_null - ce_model,
        "pseudo_r2": (1.0 - ce_model / ce_null) if ce_null > _EPS else float("nan"),
    }


def _nonlinear_accuracy(merged: pd.DataFrame, feature_names, seed: int, n_folds: int) -> float:
    """Grouped-CV accuracy of a depth-limited gradient-boosted complement (design §3.1)."""
    from sklearn.ensemble import GradientBoostingClassifier

    decisive = merged[merged["model_pref"] != 0]
    X = decisive[feature_names].to_numpy(dtype=float)
    y = (decisive["model_pref"] > 0).to_numpy()
    groups = decisive["query_id"].to_numpy()
    splits = grouped_splits(X, y, groups, n_folds)
    oof = np.full(len(y), np.nan)
    for train, test in splits:
        clf = GradientBoostingClassifier(max_depth=2, n_estimators=100, random_state=seed)
        clf.fit(X[train], y[train])
        oof[test] = clf.predict(X[test])
    return float((oof.astype(bool) == y).mean())


def decompose(
    merged: pd.DataFrame,
    feature_names: list[str],
    position_consistency: float | None = None,
    seed: int = 42,
    n_folds: int = 5,
    nonlinear: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Out-of-fold prediction and error summary for decisive model verdicts.

    Returns (result, oof). `result` carries joint-fit statistics, the accuracy/error
    partition, the information summary, the order-swap sensitivity diagnostic and — when `nonlinear` — the
    gradient-boosted headroom. `oof` is the per-pair out-of-fold frame (from
    :func:`joint_fit`), with an
    `is_residual` column marking the misclassified pairs for the residual analysis.
    """
    stats, oof = joint_fit(merged, feature_names, seed=seed, n_folds=n_folds)
    y = oof["y_true"].to_numpy(dtype=float)
    p = oof["oof_prob"].to_numpy(dtype=float)
    oof = oof.copy()
    oof["is_residual"] = ~oof["oof_correct"].to_numpy()

    info = information_decomposition(
        y, p, oof["oof_null_prob"].to_numpy(dtype=float)
    )
    ceiling = single_order_accuracy_sensitivity(position_consistency)
    result = {
        "features": feature_names,
        "n_decisive_pairs": stats["n_decisive_pairs"],
        "base_rate": stats["base_rate"],
        "cv_null_accuracy": stats["cv_null_accuracy"],
        "cv_accuracy": stats["cv_accuracy"],
        "cv_auc": stats["cv_auc"],
        # Accuracy partitions observations into correct/incorrect predictions; it is not
        # variance explained and should not be labelled an "explained fraction".
        "oof_correct_fraction": stats["cv_accuracy"],
        "oof_error_fraction": 1.0 - stats["cv_accuracy"],
        "accuracy_gain_over_base": stats["cv_accuracy"] - stats["cv_null_accuracy"],
        "information": info,
        "reliability_ceiling": ceiling,
        "single_order_accuracy_sensitivity": ceiling,
        "reliability_ceiling_identified": False,
        "reliability_ceiling_assumptions": (
            "sensitivity only: exchangeable conditionally independent order errors "
            "with equal accuracy; applies to one presentation, not the collapsed label"
        ),
        # The accuracy model predicts collapsed two-order labels, whereas the sensitivity
        # quantity concerns one order. Their difference is not an identified residual.
        "reducible_residual_upper": None,
        "coefficients": stats["coefficients"],
        "intercept": stats["intercept"],
    }
    if nonlinear:
        gbm_acc = _nonlinear_accuracy(merged, feature_names, seed, n_folds)
        result["nonlinear_cv_accuracy"] = gbm_acc
        result["nonlinear_headroom"] = gbm_acc - stats["cv_accuracy"]
    return result, oof
