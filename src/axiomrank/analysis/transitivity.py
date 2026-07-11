"""Non-transitivity of the model's pairwise verdicts (phase0-design.md §3.5)."""

from itertools import combinations

import pandas as pd


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
