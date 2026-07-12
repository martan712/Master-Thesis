"""Exploratory characterisation of axiom-model errors (phase2-design.md §3).

Three complementary development-data views of pairs the combined axiom model gets wrong:

1. :func:`residual_profiles` — out-of-fold accuracy and the signed residual, stratified by
   each non-axiom covariate (the gap gradient of §4 generalised).
2. :func:`residual_model` — a query-grouped cross-validated model that predicts the LLM's
   verdict from the non-axiom covariates *conditioned on the axiom prediction*; its **lift**
   over the axiom-only baseline, with a query-bootstrap CI, is the operational test of
   whether prespecified covariates add held-out predictive value on development folds.
3. :func:`residual_clusters` — exploratory clusters with representative exemplars, used only
   as one source of RQ4 hypotheses alongside theory and literature.
"""

import numpy as np
import pandas as pd

from axiomrank.analysis.verdicts import PAIR_KEY


def _bin_labels(values: pd.Series, n_bins: int) -> pd.Series:
    """Quantile bins (labelled by upper edge) for many-valued covariates, raw value else."""
    if values.nunique() <= n_bins:
        return values
    binned = pd.qcut(values, q=n_bins, duplicates="drop")
    return binned.apply(lambda iv: round(float(iv.right), 3))


def residual_profiles(
    oof_with_covariates: pd.DataFrame, covariate_names: list[str], n_bins: int = 5
) -> pd.DataFrame:
    """Per (covariate, bin): n, OOF accuracy, residual rate, mean signed LLM verdict.

    Long format, one row per (covariate, bin). `oof_with_covariates` is the decomposition
    OOF frame joined to the covariates; it must carry `oof_correct`, `y_true` and each
    covariate column.
    """
    df = oof_with_covariates
    rows = []
    for cov in covariate_names:
        col = df[cov]
        valid = col.notna()
        if not valid.any():
            continue
        sub = df[valid].copy()
        sub["_bin"] = _bin_labels(sub[cov], n_bins)
        for bin_label, g in sub.groupby("_bin", sort=True, observed=True):
            rows.append(
                {
                    "covariate": cov,
                    "bin": bin_label,
                    "n_pairs": int(len(g)),
                    "oof_accuracy": float(g["oof_correct"].mean()),
                    "residual_rate": float((~g["oof_correct"]).mean()),
                    "mean_signed_verdict": float((2 * g["y_true"] - 1).mean()),
                }
            )
    return pd.DataFrame(rows)


def _grouped_oof_correct(
    axiom_X: np.ndarray,
    cov_X: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_folds: int,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from axiomrank.analysis.joint import grouped_splits

    pred = np.full(len(y), np.nan)
    for train, test in grouped_splits(axiom_X, y, groups, n_folds):
        train_X, test_X = axiom_X[train], axiom_X[test]
        if cov_X is not None:
            # Covariates have heterogeneous units (BM25 score, tokens, proportions).
            # Fit scaling on training queries only, while leaving ternary axioms in their
            # Phase-1 representation so the baseline is exactly comparable to joint_fit.
            scaler = StandardScaler().fit(cov_X[train])
            train_X = np.hstack([train_X, scaler.transform(cov_X[train])])
            test_X = np.hstack([test_X, scaler.transform(cov_X[test])])
        clf = LogisticRegression(max_iter=1000, random_state=seed).fit(train_X, y[train])
        pred[test] = clf.predict(test_X)
    return pred.astype(bool) == y


def residual_model(
    oof_with_covariates: pd.DataFrame,
    covariate_names: list[str],
    axiom_names: list[str],
    seed: int = 42,
    n_folds: int = 5,
    n_boot: int = 2000,
) -> dict:
    """Query-grouped CV lift from adding covariates to the axiom features.

    Both models are refit inside the same query-disjoint folds: baseline = raw ternary axiom
    columns; full = axioms plus training-fold-standardised covariates. This direct nested
    comparison avoids a subtle stacked-CV leak: first-stage OOF logits for meta-training
    queries may have been fitted using labels from the meta-test queries. A positive
    query-macro lift whose query-bootstrap CI is above zero means the covariates carry
    structure the axioms miss.
    """
    if not axiom_names:
        raise ValueError("residual_model needs the axiom feature columns")
    df = oof_with_covariates.dropna(subset=[*axiom_names, *covariate_names]).reset_index(drop=True)
    y = df["y_true"].to_numpy(dtype=bool)
    groups = df["query_id"].to_numpy()
    axiom_X = df[axiom_names].to_numpy(dtype=float)
    cov = df[covariate_names].to_numpy(dtype=float)

    base_correct = _grouped_oof_correct(axiom_X, None, y, groups, seed, n_folds)
    full_correct = _grouped_oof_correct(axiom_X, cov, y, groups, seed, n_folds)

    lift_per_pair = full_correct.astype(float) - base_correct.astype(float)
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in unique_groups}
    base_by_query = np.array([base_correct[idx_by_group[g]].mean() for g in unique_groups])
    full_by_query = np.array([full_correct[idx_by_group[g]].mean() for g in unique_groups])
    lift_by_query = full_by_query - base_by_query
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        boot[b] = lift_by_query[picked].mean()

    return {
        "axiom_features": axiom_names,
        "features": covariate_names,
        "n_pairs": int(len(y)),
        "n_queries": int(len(unique_groups)),
        "base_accuracy": float(base_by_query.mean()),
        "cov_accuracy": float(full_by_query.mean()),
        "lift": float(lift_by_query.mean()),
        "pair_micro_base_accuracy": float(base_correct.mean()),
        "pair_micro_cov_accuracy": float(full_correct.mean()),
        "pair_micro_lift": float(lift_per_pair.mean()),
        "lift_ci_lo": float(np.quantile(boot, 0.025)),
        "lift_ci_hi": float(np.quantile(boot, 0.975)),
    }


def residual_clusters(
    oof_with_covariates: pd.DataFrame,
    covariate_names: list[str],
    n_clusters: int = 4,
    seed: int = 42,
    n_exemplars: int = 3,
) -> pd.DataFrame:
    """KMeans over the residual pairs' covariates: size, centroid means, exemplar keys."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    res = oof_with_covariates[oof_with_covariates["is_residual"]].dropna(
        subset=covariate_names
    )
    if len(res) < n_clusters:
        return pd.DataFrame()
    X = StandardScaler().fit_transform(res[covariate_names].to_numpy(dtype=float))
    fit = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
    labels = fit.labels_
    distances = np.linalg.norm(X - fit.cluster_centers_[labels], axis=1)
    res = res.assign(_cluster=labels, _centroid_distance=distances)

    rows = []
    for cid, g in res.groupby("_cluster", sort=True):
        # Choose representative observations, not whichever cache rows appear first.
        exemplars = (
            g.sort_values("_centroid_distance")[PAIR_KEY]
            .head(n_exemplars)
            .to_dict("records")
        )
        rows.append(
            {
                "cluster": int(cid),
                "size": int(len(g)),
                **{f"mean_{c}": float(g[c].mean()) for c in covariate_names},
                "exemplars": "; ".join(
                    f"{e['query_id']}:{e['doc_id_1']}|{e['doc_id_2']}" for e in exemplars
                ),
            }
        )
    return pd.DataFrame(rows)
