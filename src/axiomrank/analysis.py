"""Phase 1 analyses: agreement CIs, joint axiom fits, and the gap gradient.

Everything operates on the *merged pair frame*: one row per canonical pair with the
model's collapsed verdict (`model_pref` in {-1, 0, +1}, from
`agreement.model_pair_verdicts`) and one {-1, 0, +1} column per axiom (from
`axioms.axiom_preferences`). Definitions follow docs/phase0-design.md §3.5; the analyses
are specified in docs/phase1-design.md §4.3.
"""

import numpy as np
import pandas as pd

from axiomrank.agreement import PAIR_KEY

# The Phase 0 battery at ir_axioms defaults, as column names — the comparability
# feature set for joint fits (phase1-design.md §4.1 tier 1).
STRICT_CORE = [
    "TFC1", "TFC3", "M_TDC", "LNC1", "TF_LNC",
    "PROX1", "PROX2", "PROX3", "PROX4", "PROX5",
]

N_BOOTSTRAP = 2000
N_FOLDS = 5


def merge_pairs(axiom_prefs: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Join axiom preferences and collapsed model verdicts on the canonical pair key."""
    return axiom_prefs.rename(columns={"qid": "query_id"}).merge(verdicts, on=PAIR_KEY)


def attach_rank_gap(pairs_or_merged: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Add `rank_gap` = |BM25 rank of doc 1 − rank of doc 2| from the first-stage pool."""
    df = pairs_or_merged.copy()
    qid_col = "query_id" if "query_id" in df.columns else "qid"
    ranks = pool[["qid", "docno", "rank"]].rename(columns={"qid": qid_col})
    for side in ("1", "2"):
        side_ranks = ranks.rename(columns={"docno": f"doc_id_{side}", "rank": f"rank_{side}"})
        df = df.merge(side_ranks, on=[qid_col, f"doc_id_{side}"], how="left")
    df["rank_gap"] = (df["rank_1"] - df["rank_2"]).abs()
    return df


def agreement_with_ci(
    merged: pd.DataFrame,
    axiom_names: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-axiom coverage and agreement with 95% query-bootstrap CIs.

    Point estimates match `agreement.agreement_table` (pair-pooled); the CI resamples
    queries with replacement, the natural sampling unit of a TREC collection.
    """
    rng = np.random.default_rng(seed)
    decisive = merged["model_pref"] != 0
    n_pairs = len(merged)

    # Per-query evaluable/agreeing counts per axiom, so the bootstrap is a cheap
    # resample of count rows rather than of the pair frame.
    per_query = {}
    rows = []
    for name in axiom_names:
        active = merged[name] != 0
        evaluable = active & decisive
        agree_mask = evaluable & (merged[name] == merged["model_pref"])
        counts = pd.DataFrame(
            {
                "n_eval": evaluable.groupby(merged["query_id"]).sum(),
                "n_agree": agree_mask.groupby(merged["query_id"]).sum(),
            }
        )
        per_query[name] = counts
        n_eval = int(evaluable.sum())
        rows.append(
            {
                "axiom": name,
                "n_pairs": n_pairs,
                "coverage": float(active.mean()) if n_pairs else float("nan"),
                "n_evaluable": n_eval,
                "agreement": float(agree_mask.sum() / n_eval) if n_eval else float("nan"),
            }
        )
    table = pd.DataFrame(rows)

    qids = merged["query_id"].unique()
    draws = rng.choice(len(qids), size=(n_bootstrap, len(qids)), replace=True)
    ci_lo, ci_hi = [], []
    for name in axiom_names:
        counts = per_query[name].reindex(qids).fillna(0).to_numpy()
        n_eval = counts[draws, 0].sum(axis=1)
        n_agree = counts[draws, 1].sum(axis=1)
        with np.errstate(invalid="ignore"):
            stats = n_agree / n_eval
        stats = stats[n_eval > 0]
        if len(stats):
            ci_lo.append(float(np.percentile(stats, 2.5)))
            ci_hi.append(float(np.percentile(stats, 97.5)))
        else:
            ci_lo.append(float("nan"))
            ci_hi.append(float("nan"))
    table["ci_lo"] = ci_lo
    table["ci_hi"] = ci_hi
    return table


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


def _gap_bins(gaps: pd.Series, max_integer_gap: int = 15, n_bins: int = 10) -> pd.Series:
    """Integer bins when the gap range is small (top-10), quantile deciles otherwise."""
    if gaps.max() <= max_integer_gap:
        return gaps.astype(int)
    binned = pd.qcut(gaps, q=n_bins, duplicates="drop")
    return binned.apply(lambda iv: int(iv.right))  # label = bin's upper gap edge


def gap_gradient(
    merged: pd.DataFrame,
    axiom_names: list[str],
    oof: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Agreement (and joint OOF accuracy) as a function of the BM25 rank gap.

    Long format: one row per (gap_bin, axiom) plus per-bin context columns repeated —
    n_pairs, decisive_rate, position_consistency and, when `oof` is given,
    joint_cv_accuracy. This is the data behind the RQ1 signature figure.
    """
    df = merged.copy()
    df["gap_bin"] = _gap_bins(df["rank_gap"])
    if oof is not None:
        df = df.merge(oof[[*PAIR_KEY, "oof_correct"]], on=PAIR_KEY, how="left")

    rows = []
    for gap_bin, group in df.groupby("gap_bin", sort=True):
        decisive = group["model_pref"] != 0
        both = group[group["n_presentations"] >= 2]
        context = {
            "gap_bin": gap_bin,
            "n_pairs": int(len(group)),
            "decisive_rate": float(decisive.mean()),
            "position_consistency": (
                float(both["position_consistent"].mean()) if len(both) else float("nan")
            ),
        }
        if oof is not None:
            correct = group["oof_correct"].dropna()
            context["joint_cv_accuracy"] = (
                float(correct.mean()) if len(correct) else float("nan")
            )
        for name in axiom_names:
            evaluable = (group[name] != 0) & decisive
            n_eval = int(evaluable.sum())
            rows.append(
                {
                    **context,
                    "axiom": name,
                    "n_evaluable": n_eval,
                    "agreement": (
                        float((group.loc[evaluable, name] == group.loc[evaluable, "model_pref"]).mean())
                        if n_eval
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def gap_figure(gradient: pd.DataFrame, path, title: str) -> None:
    """Draft of the RQ1 signature figure: agreement vs. rank gap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    context = gradient.drop_duplicates("gap_bin").sort_values("gap_bin")
    for axiom, group in gradient.groupby("axiom"):
        group = group.sort_values("gap_bin")
        axis.plot(group["gap_bin"], group["agreement"], color="0.75", lw=1, zorder=1)
    if "joint_cv_accuracy" in gradient.columns:
        axis.plot(
            context["gap_bin"], context["joint_cv_accuracy"],
            color="C0", lw=2.5, label="joint logistic (CV acc.)", zorder=3,
        )
    axis.plot(
        context["gap_bin"], context["position_consistency"],
        color="C3", lw=1.5, ls="--", label="position consistency", zorder=2,
    )
    axis.axhline(0.5, color="0.4", lw=0.8, ls=":")
    axis.set_xlabel("BM25 rank gap (bin)")
    axis.set_ylabel("agreement / accuracy")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
