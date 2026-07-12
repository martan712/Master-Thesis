"""Synthetic sanity checks that ir_axioms' defaults behave as the axiom definitions say.

Marked slow: importing ir_axioms starts the Terrier JVM. Skip with -m "not slow".
"""

import pandas as pd
import pytest

from axiomrank.axioms import axiom_preferences

pytestmark = pytest.mark.slow


def make_frames(query: str, text_1: str, text_2: str):
    pool = pd.DataFrame(
        {
            "qid": ["q1", "q1"],
            "query": [query, query],
            "docno": ["d1", "d2"],
            "rank": [0, 1],
            "score": [2.0, 1.0],
            "text": [text_1, text_2],
        }
    )
    pairs = pd.DataFrame(
        {
            "qid": ["q1"],
            "query": [query],
            "doc_id_1": ["d1"],
            "doc_id_2": ["d2"],
            "text_1": [text_1],
            "text_2": [text_2],
        }
    )
    return pool, pairs


def test_tfc1_prefers_higher_term_frequency_at_equal_length():
    pool, pairs = make_frames(
        "cat",
        "cat cat dog bird",  # TF(cat)=2
        "cat dog bird fish",  # TF(cat)=1, same length
    )
    result = axiom_preferences(pool, pairs, ["TFC1"])
    assert result["TFC1"].iloc[0] == 1


def test_tfc1_neutral_on_equal_term_frequency():
    pool, pairs = make_frames("cat", "cat dog bird fish", "cat fish bird dog")
    result = axiom_preferences(pool, pairs, ["TFC1"])
    assert result["TFC1"].iloc[0] == 0


def test_prox2_prefers_earlier_query_terms():
    # Guards the sign convention: ir_axioms 1.1.2's batch preferences() override has
    # its strictly_greater arguments swapped vs its own pairwise preference(), so the
    # old transformer-based stage produced sign-flipped PROX2 values.
    pool, pairs = make_frames(
        "cat dog",
        "cat dog house tree fish bird",  # query terms at positions 0, 1
        "house tree fish bird cat dog",  # query terms at positions 4, 5
    )
    result = axiom_preferences(pool, pairs, ["PROX2"])
    assert result["PROX2"].iloc[0] == 1


def test_prox1_prefers_query_terms_closer_together():
    pool, pairs = make_frames(
        "cat dog",
        "cat dog house tree fish bird",  # adjacent query terms
        "cat house tree fish bird dog",  # query terms far apart
    )
    result = axiom_preferences(pool, pairs, ["PROX1"])
    assert result["PROX1"].iloc[0] == 1


def test_prox1_is_neutral_for_adjacent_terms_in_opposite_orders():
    # Distance must be symmetric in term order. The upstream 1.1.2 expression
    # abs(pos1-pos2-1) assigns distances 2 and 0 to these equally adjacent cases.
    pool, pairs = make_frames(
        "ant zebra",
        "ant zebra house tree",
        "zebra ant house tree",
    )
    result = axiom_preferences(pool, pairs, ["PROX1"])
    assert result["PROX1"].iloc[0] == 0
