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
    from sklearn.model_selection import GroupKFold

    decisive = merged[merged["model_pref"] != 0].reset_index(drop=True)
    X = decisive[feature_names].to_numpy(dtype=float)
    y = (decisive["model_pref"] > 0).to_numpy()
    groups = decisive["query_id"].to_numpy()

    majority_class = bool(y.mean() >= 0.5)
    base_rate = float(max(y.mean(), 1 - y.mean()))

    votes = X.sum(axis=1)
    vote_pred = np.where(votes == 0, majority_class, votes > 0)
    vote_accuracy = float((vote_pred == y).mean())

    n_folds = min(n_folds, len(np.unique(groups)))
    oof_prob = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=n_folds).split(X, y, groups):
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X[train], y[train])
        oof_prob[test] = clf.predict_proba(X[test])[:, 1]
    oof_pred = oof_prob >= 0.5
    accuracy = float((oof_pred == y).mean())
    auc = float(roc_auc_score(y, oof_prob)) if len(np.unique(y)) == 2 else float("nan")

    full = LogisticRegression(max_iter=1000, random_state=seed).fit(X, y)
    coefficients = dict(zip(feature_names, (float(c) for c in full.coef_[0])))

    stats = {
        "features": feature_names,
        "n_decisive_pairs": int(len(y)),
        "base_rate": base_rate,
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
    return stats, oof
