import pandas as pd

from axiomrank.analysis import (
    agreement_table,
    consistency_stats,
    model_pair_verdicts,
    nontransitivity_rate,
)
from axiomrank.data.preferences import new_row


def presentation(query_id, doc_a, doc_b, verdict):
    return new_row(
        dataset="ds",
        query_id=query_id,
        doc_id_a=doc_a,
        doc_id_b=doc_b,
        model="m",
        prompt_version="v0",
        verdict=verdict,
        prob_a=0.8,
        score_a=-1.0,
        score_b=-2.0,
        latency_ms=1.0,
    )


def test_verdict_collapse_consistent_and_inconsistent():
    store = pd.DataFrame(
        [
            # consistent pair: d1 preferred in both orders
            presentation("q1", "d1", "d2", "a"),
            presentation("q1", "d2", "d1", "b"),
            # inconsistent pair: whichever is shown first wins
            presentation("q1", "d1", "d3", "a"),
            presentation("q1", "d3", "d1", "a"),
            # single-order pair
            presentation("q1", "d4", "d2", "b"),
        ]
    )
    v = model_pair_verdicts(store).set_index(["doc_id_1", "doc_id_2"])
    assert v.loc[("d1", "d2"), "model_pref"] == 1
    assert v.loc[("d1", "d2"), "position_consistent"] == True  # noqa: E712
    assert v.loc[("d1", "d3"), "model_pref"] == 0
    assert v.loc[("d1", "d3"), "position_consistent"] == False  # noqa: E712
    # (d4, d2) canonicalises to (d2, d4); "b" with d4 shown first means d2 preferred -> ...
    # d2 < d4, presentation was (a=d4, b=d2), verdict "b" = d2 preferred = doc_id_1 -> +1
    assert v.loc[("d2", "d4"), "model_pref"] == 1
    assert pd.isna(v.loc[("d2", "d4"), "position_consistent"])

    stats = consistency_stats(model_pair_verdicts(store))
    assert stats["n_pairs"] == 3
    assert stats["n_pairs_both_orders"] == 2
    assert stats["position_consistency"] == 0.5


def test_agreement_table_counts_only_active_and_decisive():
    verdicts = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1"],
            "doc_id_1": ["d1", "d1", "d2"],
            "doc_id_2": ["d2", "d3", "d3"],
            "model_pref": [1, -1, 0],
        }
    )
    axiom_prefs = pd.DataFrame(
        {
            "qid": ["q1", "q1", "q1"],
            "doc_id_1": ["d1", "d1", "d2"],
            "doc_id_2": ["d2", "d3", "d3"],
            "AX1": [1, 1, 1],   # agrees on pair 1, disagrees on pair 2, pair 3 not decisive
            "AX2": [0, -1, 0],  # neutral on pair 1, agrees on pair 2
        }
    )
    table = agreement_table(axiom_prefs, verdicts, ["AX1", "AX2"]).set_index("axiom")
    assert table.loc["AX1", "coverage"] == 1.0
    assert table.loc["AX1", "n_evaluable"] == 2
    assert table.loc["AX1", "agreement"] == 0.5
    assert table.loc["AX2", "coverage"] == 1 / 3
    assert table.loc["AX2", "n_evaluable"] == 1
    assert table.loc["AX2", "agreement"] == 1.0


def test_nontransitivity_detects_cycle():
    verdicts = pd.DataFrame(
        {
            "query_id": ["q1"] * 3 + ["q2"] * 3,
            "doc_id_1": ["d1", "d1", "d2"] * 2,
            "doc_id_2": ["d2", "d3", "d3"] * 2,
            # q1: d1>d2, d2>d3, d1>d3  -> transitive
            # q2: d1>d2, d2>d3, d3>d1  -> cycle
            "model_pref": [1, 1, 1, 1, -1, 1],
        }
    )
    stats = nontransitivity_rate(verdicts)
    assert stats["n_triangles_sampled"] == 2
    assert stats["n_complete_triangles"] == 2
    assert stats["triangle_survival"] == 1.0
    assert stats["n_cyclic"] == 1
    assert stats["nontransitivity_rate"] == 0.5


def test_tied_edge_drops_triangle_from_complete_but_not_sampled():
    verdicts = pd.DataFrame(
        {
            "query_id": ["q1"] * 3,
            "doc_id_1": ["d1", "d1", "d2"],
            "doc_id_2": ["d2", "d3", "d3"],
            # d2-d3 was scored but position-inconsistent -> tie -> triangle incomplete
            "model_pref": [1, 1, 0],
        }
    )
    stats = nontransitivity_rate(verdicts)
    assert stats["n_triangles_sampled"] == 1
    assert stats["n_complete_triangles"] == 0
    assert stats["triangle_survival"] == 0.0
    assert stats["nontransitivity_rate"] is None
