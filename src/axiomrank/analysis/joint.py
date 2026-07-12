"""Joint axiom fits: predicting the model's verdicts from all axioms at once
(phase1-design.md §4.3)."""

import numpy as np
import pandas as pd

from axiomrank.analysis.verdicts import PAIR_KEY

# The Phase 0 battery at ir_axioms defaults, as column names — the comparability
# feature set for joint fits (phase1-design.md §4.1 tier 1).
STRICT_CORE = [
    "TFC1", "TFC3", "M_TDC", "LNC1", "TF_LNC",
    "PROX1", "PROX2", "PROX3", "PROX4", "PROX5",
]

N_FOLDS = 5


def grouped_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return query-disjoint folds and reject statistically undefined fits."""
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(groups))
    n_folds = min(n_folds, n_groups)
    if n_folds < 2:
        raise ValueError("grouped cross-validation needs at least two distinct queries")
    splits = list(GroupKFold(n_splits=n_folds).split(X, y, groups))
    bad = [i for i, (train, _) in enumerate(splits) if len(np.unique(y[train])) < 2]
    if bad:
        raise ValueError(
            "grouped cross-validation has a single-class training fold; "
            "use more queries or revise the fold design"
        )
    return splits


def joint_fit(
    merged: pd.DataFrame,
    feature_names: list[str],
    seed: int = 42,
    n_folds: int = N_FOLDS,
) -> tuple[dict, pd.DataFrame]:
    """Predict the model's decisive verdicts from all axiom columns at once.

    Reports the majority-class base rate, an axiom majority vote, and a query-grouped
    cross-validated L2 logistic regression (accuracy + ROC-AUC), plus full-data
    coefficients. Returns (stats, out_of_fold predictions per pair) so the gap
    analysis can bin the joint accuracy.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    decisive = merged[merged["model_pref"] != 0].reset_index(drop=True)
    if not feature_names:
        raise ValueError("joint_fit needs at least one feature")
    if decisive.empty:
        raise ValueError("joint_fit needs at least one decisive pair")
    X = decisive[feature_names].to_numpy(dtype=float)
    y = (decisive["model_pref"] > 0).to_numpy()
    groups = decisive["query_id"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("joint_fit needs both preference classes")

    base_rate = float(max(y.mean(), 1 - y.mean()))

    splits = grouped_splits(X, y, groups, n_folds)
    n_folds = len(splits)
    oof_prob = np.full(len(y), np.nan)
    oof_null_prob = np.full(len(y), np.nan)
    for train, test in splits:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X[train], y[train])
        oof_prob[test] = clf.predict_proba(X[test])[:, 1]
        # The null must be estimated without seeing the held-out queries too. Using the
        # full-data class prior here makes both accuracy gain and pseudo-R2 optimistic.
        oof_null_prob[test] = y[train].mean()
    oof_pred = oof_prob >= 0.5
    oof_null_pred = oof_null_prob >= 0.5
    accuracy = float((oof_pred == y).mean())
    null_accuracy = float((oof_null_pred == y).mean())
    auc = float(roc_auc_score(y, oof_prob)) if len(np.unique(y)) == 2 else float("nan")

    # A zero-sum axiom vote abstains. Resolve that abstention with the training-fold
    # majority, not the full-data majority, so the reported comparator is out-of-fold.
    votes = X.sum(axis=1)
    vote_pred = np.where(votes == 0, oof_null_pred, votes > 0)
    vote_accuracy = float((vote_pred == y).mean())

    full = LogisticRegression(max_iter=1000, random_state=seed).fit(X, y)
    coefficients = dict(zip(feature_names, (float(c) for c in full.coef_[0])))

    stats = {
        "features": feature_names,
        "n_decisive_pairs": int(len(y)),
        "base_rate": base_rate,
        "cv_null_accuracy": null_accuracy,
        "majority_vote_accuracy": vote_accuracy,
        "cv_accuracy": accuracy,
        "cv_auc": auc,
        "n_folds": n_folds,
        "coefficients": coefficients,
        "intercept": float(full.intercept_[0]),
    }
    oof = decisive[PAIR_KEY].copy()
    oof["y_true"] = y
    oof["oof_prob"] = oof_prob
    oof["oof_correct"] = oof_pred == y
    oof["oof_null_prob"] = oof_null_prob
    oof["oof_null_correct"] = oof_null_pred == y
    return stats, oof
