"""Query-disjoint fitted axiom surrogates for pairwise reranking.

The surrogate is trained only on decisive LLM labels from training queries, but predicts
every pair of each held-out query. Qrels never enter this module. Keeping fold assignment,
pair predictions, and coefficients explicit makes the resulting effectiveness evaluation
auditable and prevents an in-sample fitted axiom model from masquerading as a reranker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_query_folds(groups, n_folds: int = 5) -> np.ndarray:
    """Assign each query group to one deterministic GroupKFold test fold."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    n_folds = min(n_folds, n_groups)
    if n_folds < 2:
        raise ValueError("surrogate OOF fitting needs at least two distinct queries")
    fold = np.full(len(groups), -1, dtype=int)
    dummy = np.zeros((len(groups), 1))
    for fold_id, (_, test) in enumerate(
        GroupKFold(n_splits=n_folds).split(dummy, groups=groups)
    ):
        fold[test] = fold_id
    if (fold < 0).any():
        raise RuntimeError("not every pair received an OOF fold")
    check = pd.DataFrame({"group": groups, "fold": fold}).groupby("group")["fold"].nunique()
    if not check.eq(1).all():
        raise RuntimeError("a query was split across surrogate folds")
    return fold


def _hard_preference(probability: np.ndarray) -> np.ndarray:
    """Convert P(doc_1 preferred) to {-1,0,+1}; exact 0.5 remains a tie."""
    return np.sign(np.asarray(probability) - 0.5).astype(int)


def surrogate_fidelity(predictions: pd.DataFrame) -> dict:
    """Fidelity and tie/forced-decision diagnostics for OOF pair predictions."""
    from sklearn.metrics import log_loss, roc_auc_score

    decisive = predictions["target_pref"] != 0
    target_positive = predictions.loc[decisive, "target_pref"].to_numpy() > 0
    probability = predictions.loc[decisive, "surrogate_prob"].to_numpy(dtype=float)
    correct = (
        predictions.loc[decisive, "surrogate_pref"].to_numpy()
        == predictions.loc[decisive, "target_pref"].to_numpy()
    )
    per_query = predictions.loc[decisive, ["group_id"]].assign(correct=correct)
    query_accuracy = per_query.groupby("group_id")["correct"].mean()
    target_tie = ~decisive
    surrogate_tie = predictions["surrogate_pref"] == 0
    return {
        "n_pairs": int(len(predictions)),
        "n_queries": int(predictions["group_id"].nunique()),
        "n_target_decisive": int(decisive.sum()),
        "n_target_ties": int(target_tie.sum()),
        "n_surrogate_ties": int(surrogate_tie.sum()),
        "n_forced_decisions_on_target_ties": int((target_tie & ~surrogate_tie).sum()),
        "decisive_pair_accuracy": float(correct.mean()) if len(correct) else float("nan"),
        "decisive_query_macro_accuracy": (
            float(query_accuracy.mean()) if len(query_accuracy) else float("nan")
        ),
        "decisive_auc": (
            float(roc_auc_score(target_positive, probability))
            if len(np.unique(target_positive)) == 2
            else float("nan")
        ),
        "decisive_log_loss": (
            float(log_loss(target_positive, probability, labels=[False, True]))
            if len(probability)
            else float("nan")
        ),
    }


def fit_oof_surrogate(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    group_col: str,
    target_col: str = "model_pref",
    folds: np.ndarray | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Fit an L2-logistic axiom surrogate and predict every pair out of fold.

    Training uses only decisive target pairs. Testing covers every pair in held-out
    queries, including target ties, so the downstream ranking does not peek at where the
    LLM abstained. Returns a row-aligned prediction frame and JSON-serialisable model
    metadata containing the common fold map and fold/full-data coefficients.
    """
    from sklearn.linear_model import LogisticRegression

    if not feature_names:
        raise ValueError("surrogate fitting needs at least one axiom feature")
    required = {group_col, target_col, *feature_names}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"surrogate frame missing columns: {sorted(missing)}")

    X = frame[feature_names].to_numpy(dtype=float)
    target = frame[target_col].to_numpy(dtype=int)
    groups = frame[group_col].astype(str).to_numpy()
    folds = assign_query_folds(groups, n_folds) if folds is None else np.asarray(folds)
    if len(folds) != len(frame):
        raise ValueError("fold vector length does not match the pair frame")
    if (folds < 0).any():
        raise ValueError("fold ids must be non-negative")
    if not pd.DataFrame({"g": groups, "f": folds}).groupby("g")["f"].nunique().eq(1).all():
        raise ValueError("each query must belong to exactly one fold")

    probability = np.full(len(frame), np.nan)
    fold_models = []
    for fold_id in sorted(np.unique(folds)):
        test = folds == fold_id
        train = ~test
        train_decisive = train & (target != 0)
        y_train = target[train_decisive] > 0
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"surrogate fold {fold_id} has a single-class training set")
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X[train_decisive], y_train)
        probability[test] = clf.predict_proba(X[test])[:, 1]
        fold_models.append(
            {
                "fold": int(fold_id),
                "n_train_queries": int(len(np.unique(groups[train]))),
                "n_test_queries": int(len(np.unique(groups[test]))),
                "n_train_decisive_pairs": int(train_decisive.sum()),
                "test_groups": sorted(np.unique(groups[test]).tolist()),
                "coefficients": {
                    name: float(value) for name, value in zip(feature_names, clf.coef_[0])
                },
                "intercept": float(clf.intercept_[0]),
            }
        )
    if np.isnan(probability).any():
        raise RuntimeError("surrogate produced missing OOF probabilities")

    all_decisive = target != 0
    y_all = target[all_decisive] > 0
    if len(np.unique(y_all)) < 2:
        raise ValueError("surrogate full fit needs both decisive preference classes")
    full = LogisticRegression(max_iter=1000, random_state=seed).fit(X[all_decisive], y_all)

    predictions = pd.DataFrame(
        {
            "group_id": groups,
            "fold": folds.astype(int),
            "target_pref": target,
            "surrogate_prob": probability,
            "surrogate_pref": _hard_preference(probability),
        },
        index=frame.index,
    )
    predictions["correct_on_decisive"] = pd.array(
        np.where(
            predictions["target_pref"] != 0,
            predictions["surrogate_pref"] == predictions["target_pref"],
            pd.NA,
        ),
        dtype="boolean",
    )
    metadata = {
        "features": list(feature_names),
        "estimator": {
            "class": "sklearn.linear_model.LogisticRegression",
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 1000,
            "random_state": seed,
        },
        "n_folds": int(len(np.unique(folds))),
        "fold_models": fold_models,
        "full_model": {
            "n_decisive_pairs": int(all_decisive.sum()),
            "coefficients": {
                name: float(value) for name, value in zip(feature_names, full.coef_[0])
            },
            "intercept": float(full.intercept_[0]),
        },
        "fidelity": surrogate_fidelity(predictions),
    }
    return predictions, metadata
