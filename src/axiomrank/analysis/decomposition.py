"""The RQ3 explained/residual decomposition (phase2-design.md §3).

Given the combined axiom model (the Phase 1 L2 logistic, :func:`analysis.joint_fit`), split
the model's decisive verdicts into an axiom-explained part and a residual, and report the
split two ways: an **accuracy** decomposition (out-of-fold correct/incorrect) and an
**information** decomposition (the fraction of the label entropy the axiom model removes,
McFadden-style), plus a **reliability ceiling** estimating how much of the gap to perfect
prediction is the model's own noise rather than missed structure (design §3.3).
"""

import numpy as np
import pandas as pd

from axiomrank.analysis.joint import joint_fit

_EPS = 1e-12


def reliability_ceiling(position_consistency: float | None) -> float | None:
    """Single-measurement accuracy ceiling implied by order-swap reproducibility.

    Two presentation orders are two noisy measurements of a latent preference; if they
    agree with probability ``c``, and each measurement matches the latent label with
    probability ``a``, then ``c = a² + (1−a)²`` (both right or both wrong). Solving gives
    ``a = ½(1 + √(2c−1))`` — the Bayes-optimal accuracy any predictor of a single
    collapsed verdict can reach. A **lower bound on noise / upper bound on reducible
    structure**: c=1 → 1.0 (noiseless), c=0.5 → 0.5 (chance). Returns None if c is missing
    or below chance.
    """
    if position_consistency is None or not np.isfinite(position_consistency):
        return None
    c = float(position_consistency)
    if c <= 0.5:
        return 0.5
    return float(0.5 * (1.0 + np.sqrt(2.0 * c - 1.0)))


def _cross_entropy(y: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(prob, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def information_decomposition(y: np.ndarray, oof_prob: np.ndarray) -> dict:
    """Fraction of the label entropy the axiom model removes, vs a class-prior null.

    ``pseudo_r2`` is McFadden's 1 − CE(model)/CE(null) on the out-of-fold probabilities;
    ``log_loss_gain`` is the absolute cross-entropy reduction in nats. The null predicts
    the constant class prior for every pair.
    """
    prior = float(y.mean())
    ce_null = _cross_entropy(y, np.full_like(y, prior, dtype=float))
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
    from sklearn.model_selection import GroupKFold

    decisive = merged[merged["model_pref"] != 0]
    X = decisive[feature_names].to_numpy(dtype=float)
    y = (decisive["model_pref"] > 0).to_numpy()
    groups = decisive["query_id"].to_numpy()
    n_folds = min(n_folds, len(np.unique(groups)))
    oof = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=n_folds).split(X, y, groups):
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
    """Explained/residual decomposition of the model's decisive verdicts.

    Returns (result, oof). `result` carries the joint-fit stats, the accuracy split, the
    information split, the reliability ceiling and — when `nonlinear` — the gradient-boosted
    headroom. `oof` is the per-pair out-of-fold frame (from :func:`joint_fit`), with an
    `is_residual` column marking the misclassified pairs for the residual analysis.
    """
    stats, oof = joint_fit(merged, feature_names, seed=seed, n_folds=n_folds)
    y = oof["y_true"].to_numpy(dtype=float)
    p = oof["oof_prob"].to_numpy(dtype=float)
    oof = oof.copy()
    oof["is_residual"] = ~oof["oof_correct"].to_numpy()

    info = information_decomposition(y, p)
    ceiling = reliability_ceiling(position_consistency)
    result = {
        "features": feature_names,
        "n_decisive_pairs": stats["n_decisive_pairs"],
        "base_rate": stats["base_rate"],
        "cv_accuracy": stats["cv_accuracy"],
        "cv_auc": stats["cv_auc"],
        "explained_fraction": stats["cv_accuracy"],
        "residual_fraction": 1.0 - stats["cv_accuracy"],
        "accuracy_gain_over_base": stats["cv_accuracy"] - stats["base_rate"],
        "information": info,
        "reliability_ceiling": ceiling,
        "reducible_residual_upper": (
            None if ceiling is None else max(ceiling - stats["cv_accuracy"], 0.0)
        ),
        "coefficients": stats["coefficients"],
        "intercept": stats["intercept"],
    }
    if nonlinear:
        gbm_acc = _nonlinear_accuracy(merged, feature_names, seed, n_folds)
        result["nonlinear_cv_accuracy"] = gbm_acc
        result["nonlinear_headroom"] = gbm_acc - stats["cv_accuracy"]
    return result, oof
