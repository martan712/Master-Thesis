"""Per-axiom agreement with the model's verdicts, with and without bootstrap CIs.

Everything operates on the *merged pair frame*: one row per canonical pair with the
model's collapsed verdict (`model_pref` in {-1, 0, +1}, from
:func:`axiomrank.analysis.verdicts.model_pair_verdicts`) and one {-1, 0, +1} column per
axiom (from :func:`axiomrank.axioms.axiom_preferences`). Definitions follow
docs/phase0-design.md §3.5: agreement of an axiom is, among pairs where the axiom is
non-neutral AND the model verdict is decisive, the fraction with matching sign;
coverage is the fraction of all pairs where the axiom is non-neutral.
"""

import numpy as np
import pandas as pd

from axiomrank.analysis.verdicts import PAIR_KEY

N_BOOTSTRAP = 2000


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


def agreement_table(
    axiom_prefs: pd.DataFrame, verdicts: pd.DataFrame, axiom_names: list[str]
) -> pd.DataFrame:
    """Per-axiom coverage and agreement against the model's decisive verdicts."""
    merged = merge_pairs(axiom_prefs, verdicts)
    n_pairs = len(merged)
    decisive = merged["model_pref"] != 0
    rows = []
    for name in axiom_names:
        active = merged[name] != 0
        evaluable = active & decisive
        n_eval = int(evaluable.sum())
        agree = (
            float((merged.loc[evaluable, name] == merged.loc[evaluable, "model_pref"]).mean())
            if n_eval
            else float("nan")
        )
        rows.append(
            {
                "axiom": name,
                "n_pairs": n_pairs,
                "coverage": float(active.mean()) if n_pairs else float("nan"),
                "n_evaluable": n_eval,
                "agreement": agree,
            }
        )
    return pd.DataFrame(rows)


def agreement_with_ci(
    merged: pd.DataFrame,
    axiom_names: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-axiom coverage and agreement with 95% query-bootstrap CIs.

    Point estimates match `agreement_table` (pair-pooled); the CI resamples
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
