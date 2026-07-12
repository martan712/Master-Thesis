"""Agreement as a function of the BM25 rank gap — the RQ1 signature analysis
(phase1-design.md §4.3)."""

import numpy as np
import pandas as pd

from axiomrank.analysis.verdicts import PAIR_KEY


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
    bm25: bool = False,
) -> pd.DataFrame:
    """Agreement (and joint OOF accuracy) as a function of the BM25 rank gap.

    Long format: one row per (gap_bin, axiom) plus per-bin context columns repeated —
    n_pairs, decisive_rate, position_consistency and, when `oof` is given,
    joint_cv_accuracy. This is the data behind the RQ1 signature figure.

    When `bm25` is set, each bin also carries `bm25_accuracy`: the accuracy of the
    first-stage BM25 ranking used *as a pairwise predictor* — it "prefers" the
    better-ranked document (smaller rank number) — scored against the LLM's decisive
    verdict on the same decisive pairs the combined model is scored on. This is the
    baseline the combined axiom model is read against (opt-in so Phase 1's `measure_cell`
    gradient stays byte-for-byte unchanged).
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
        if bm25:
            dec = group[decisive]
            bm25_pref = np.sign(dec["rank_2"] - dec["rank_1"])  # +1 == doc_1 better ranked
            context["bm25_accuracy"] = (
                float((bm25_pref == np.sign(dec["model_pref"])).mean())
                if len(dec)
                else float("nan")
            )
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
