"""Collapsing per-presentation model verdicts into canonical pair verdicts.

Definitions (fixed in docs/phase0-design.md §3.5 so all phases stay comparable):
the model's verdict per canonical pair is +1 if it prefers doc_id_1, -1 for doc_id_2,
0 for ties. When both presentation orders were queried and disagree, the pair is
position-inconsistent and its verdict is 0.
"""

import pandas as pd

PAIR_KEY = ["query_id", "doc_id_1", "doc_id_2"]


def model_pair_verdicts(store_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-presentation verdicts into one row per canonical pair.

    Input: preference-store rows (one per presentation). Output columns: query_id,
    doc_id_1, doc_id_2, model_pref (+1/-1/0), n_presentations, position_consistent
    (True/False, or pd.NA when only one order was queried).
    """
    df = store_df.copy()
    swapped = df["doc_id_a"] > df["doc_id_b"]
    df["doc_id_1"] = df["doc_id_a"].where(~swapped, df["doc_id_b"])
    df["doc_id_2"] = df["doc_id_b"].where(~swapped, df["doc_id_a"])
    sign = df["verdict"].map({"a": 1, "b": -1, "tie": 0})
    df["pref"] = sign.where(~swapped, -sign)

    def collapse(group: pd.DataFrame) -> pd.Series:
        prefs = group["pref"].tolist()
        if len(prefs) == 1:
            return pd.Series(
                {"model_pref": prefs[0], "n_presentations": 1, "position_consistent": pd.NA}
            )
        consistent = len(set(prefs)) == 1
        return pd.Series(
            {
                "model_pref": prefs[0] if consistent else 0,
                "n_presentations": len(prefs),
                "position_consistent": consistent,
            }
        )

    out = df.groupby(PAIR_KEY, sort=True).apply(collapse, include_groups=False).reset_index()
    out["model_pref"] = out["model_pref"].astype(int)
    return out


def consistency_stats(verdicts: pd.DataFrame) -> dict:
    both = verdicts[verdicts["n_presentations"] >= 2]
    return {
        "n_pairs": int(len(verdicts)),
        "n_pairs_both_orders": int(len(both)),
        "position_consistency": (
            float(both["position_consistent"].mean()) if len(both) else None
        ),
        "decisive_rate": float((verdicts["model_pref"] != 0).mean()) if len(verdicts) else None,
    }
