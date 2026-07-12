"""Characterising the axiom residual — the analysis that decides RQ4 (phase2-design.md §3.4).

Three converging views of the pairs the combined axiom model gets wrong:

1. :func:`residual_profiles` — out-of-fold accuracy and the signed residual, stratified by
   each non-axiom covariate (the gap gradient of §4 generalised).
2. :func:`residual_model` — a query-grouped cross-validated model that predicts the LLM's
   verdict from the non-axiom covariates *conditioned on the axiom prediction*; its **lift**
   over the axiom-only baseline, with a query-bootstrap CI, is the operational test of
   whether the residual is systematic (the §6.1 gate reads the content-only lift).
3. :func:`residual_clusters` — clusters of the residual pairs with exemplars, turning a
   statistical residual into formalisable RQ4 hypotheses.
"""

import numpy as np
import pandas as pd

from axiomrank.analysis.verdicts import PAIR_KEY

_EPS = 1e-12


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


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


def _grouped_oof_correct(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                         seed: int, n_folds: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    n_folds = min(n_folds, len(np.unique(groups)))
    pred = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=n_folds).split(X, y, groups):
        clf = LogisticRegression(max_iter=1000, random_state=seed).fit(X[train], y[train])
        pred[test] = clf.predict(X[test])
    return pred.astype(bool) == y


def residual_model(
    oof_with_covariates: pd.DataFrame,
    covariate_names: list[str],
    seed: int = 42,
    n_folds: int = 5,
    n_boot: int = 2000,
) -> dict:
    """Grouped-CV lift of the covariates over the axiom-only baseline at predicting the LLM.

    The baseline predicts the verdict from the axiom model's OOF logit alone (i.e. the
    axiom prediction); the covariate model adds the signed non-axiom covariates. A positive
    lift whose query-bootstrap CI is above zero means the covariates carry structure the
    axioms miss — the systematic residual. Returns accuracies, the lift and its 95% CI.
    """
    df = oof_with_covariates.dropna(subset=covariate_names).reset_index(drop=True)
    y = df["y_true"].to_numpy(dtype=bool)
    groups = df["query_id"].to_numpy()
    axiom_logit = _logit(df["oof_prob"].to_numpy(dtype=float)).reshape(-1, 1)
    cov = df[covariate_names].to_numpy(dtype=float)

    base_correct = _grouped_oof_correct(axiom_logit, y, groups, seed, n_folds)
    full_correct = _grouped_oof_correct(np.hstack([axiom_logit, cov]), y, groups, seed, n_folds)

    lift_per_pair = full_correct.astype(float) - base_correct.astype(float)
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in unique_groups}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in picked])
        boot[b] = lift_per_pair[rows].mean()

    return {
        "features": covariate_names,
        "n_pairs": int(len(y)),
        "base_accuracy": float(base_correct.mean()),
        "cov_accuracy": float(full_correct.mean()),
        "lift": float(lift_per_pair.mean()),
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
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)
    res = res.assign(_cluster=labels)

    rows = []
    for cid, g in res.groupby("_cluster", sort=True):
        exemplars = g[PAIR_KEY].head(n_exemplars).to_dict("records")
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
