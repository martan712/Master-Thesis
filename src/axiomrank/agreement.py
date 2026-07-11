"""Agreement, position-consistency and transitivity analysis.

Definitions (fixed in docs/phase0-design.md §3.5 so all phases stay comparable):

- Model verdict per canonical pair: +1 if the model prefers doc_id_1, -1 for doc_id_2,
  0 for ties. When both presentation orders were queried and disagree, the pair is
  position-inconsistent and its verdict is 0.
- Agreement of an axiom: among pairs where the axiom is non-neutral AND the model verdict
  is decisive, the fraction with matching sign. Coverage: fraction of all pairs where the
  axiom is non-neutral.
- Non-transitivity rate: cyclic fraction of complete decisive triangles.
"""

from itertools import combinations

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


def agreement_table(
    axiom_prefs: pd.DataFrame, verdicts: pd.DataFrame, axiom_names: list[str]
) -> pd.DataFrame:
    """Per-axiom coverage and agreement against the model's decisive verdicts."""
    merged = axiom_prefs.rename(columns={"qid": "query_id"}).merge(verdicts, on=PAIR_KEY)
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


def nontransitivity_rate(verdicts: pd.DataFrame) -> dict:
    """Cyclic fraction of complete decisive triangles, per query then pooled.

    A triangle is *sampled* when all three of its pairs were scored (a property of the
    pair sample, identical across models); it is *complete* — evaluable for cyclicity —
    only when the model was additionally decisive on all three edges. A pair whose two
    presentation orders disagree collapses to a tie, so `n_complete_triangles` shrinks
    with position inconsistency; `triangle_survival` reports that shrinkage explicitly
    so complete-triangle counts are comparable across models.
    """
    sampled = 0
    complete = 0
    cyclic = 0
    for _, group in verdicts.groupby("query_id"):
        scored = set()
        pref = {}
        docs = set()
        for row in group.itertuples():
            scored.add((row.doc_id_1, row.doc_id_2))
            if row.model_pref != 0:
                pref[(row.doc_id_1, row.doc_id_2)] = row.model_pref
            docs.update((row.doc_id_1, row.doc_id_2))
        for x, y, z in combinations(sorted(docs), 3):
            if not {(x, y), (x, z), (y, z)} <= scored:
                continue
            sampled += 1
            edges = [pref.get((x, y)), pref.get((x, z)), pref.get((y, z))]
            if None in edges:
                continue
            complete += 1
            # Wins per node; a 3-tournament is cyclic iff every node has one win.
            wins = {x: 0, y: 0, z: 0}
            wins[x if edges[0] > 0 else y] += 1
            wins[x if edges[1] > 0 else z] += 1
            wins[y if edges[2] > 0 else z] += 1
            if set(wins.values()) == {1}:
                cyclic += 1
    return {
        "n_triangles_sampled": sampled,
        "n_complete_triangles": complete,
        "triangle_survival": (complete / sampled) if sampled else None,
        "n_cyclic": cyclic,
        "nontransitivity_rate": (cyclic / complete) if complete else None,
    }
