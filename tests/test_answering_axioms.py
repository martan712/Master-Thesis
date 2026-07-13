"""Synthetic direction, precondition and neutrality tests for D0 answer axioms."""

import pandas as pd
import pytest

from axiomrank.axioms import axiom_preferences
from axiomrank.axioms.answering import (
    _boilerplate_score,
    _numeric_score,
    candidate_query_precondition,
    parse_query,
)


pytestmark = pytest.mark.slow


def _frames(query: str, preferred: str, alternative: str):
    pool = pd.DataFrame(
        {
            "qid": ["q1", "q1"],
            "query": [query, query],
            "docno": ["d1", "d2"],
            "rank": [0, 1],
            "score": [2.0, 1.0],
            "text": [preferred, alternative],
        }
    )
    pairs = pd.DataFrame(
        {
            "qid": ["q1"],
            "query": [query],
            "doc_id_1": ["d1"],
            "doc_id_2": ["d2"],
            "text_1": [preferred],
            "text_2": [alternative],
        }
    )
    return pool, pairs


@pytest.mark.parametrize("axiom", ["DEFANS", "NUMANS", "COMPARE", "CBP"])
def test_answer_axiom_direction_reverses_with_document_order(axiom):
    cases = {
        "DEFANS": (
            "define bmt",
            "BMT: Bone marrow transplantation replaces diseased marrow.",
            "BMT is listed in the medical acronym directory.",
        ),
        "NUMANS": (
            "how much money do speakers make",
            "Speakers make about $80,000 per year.",
            "Speakers make money from books and public events.",
        ),
        "COMPARE": (
            "difference between strategy and business model",
            "A strategy seeks advantage, whereas a business model explains value creation.",
            "This article is about strategy and business model topics.",
        ),
        "CBP": (
            "who sings monk theme song",
            "The Monk theme song is sung by Randy Newman.",
            "Incoming search terms: 1 who sings monk theme song 2 monk theme song.",
        ),
    }
    query, preferred, alternative = cases[axiom]
    pool, pairs = _frames(query, preferred, alternative)
    assert axiom_preferences(pool, pairs, [axiom])[axiom].iloc[0] == 1

    pool, pairs = _frames(query, alternative, preferred)
    assert axiom_preferences(pool, pairs, [axiom])[axiom].iloc[0] == -1


def test_definition_axiom_is_neutral_outside_definition_queries():
    pool, pairs = _frames(
        "what is bmt prescribed for",
        "BMT: Bone marrow transplantation replaces diseased marrow.",
        "BMT is prescribed only in specialist settings.",
    )
    assert axiom_preferences(pool, pairs, ["DEFANS"])["DEFANS"].iloc[0] == 0


def test_numeric_axiom_rejects_incompatible_or_indirect_numbers():
    pool, pairs = _frames(
        "how old is the actor",
        "The actor was born in 1937.",
        "The actor appeared in 20 films.",
    )
    assert axiom_preferences(pool, pairs, ["NUMANS"])["NUMANS"].iloc[0] == 0


def test_count_axiom_binds_number_to_requested_unit():
    pool, pairs = _frames(
        "how many sons robert kraft has",
        "Robert Kraft has a net worth of $6.2 billion and owns 2 teams.",
        "Robert Kraft has four sons.",
    )
    assert axiom_preferences(pool, pairs, ["NUMANS"])["NUMANS"].iloc[0] == -1


def test_count_binding_rejects_unrelated_number_in_same_sentence():
    frame = parse_query("how many sons does he have")
    # The cardinal is bound to "teams", not to the requested "sons" in the same sentence.
    assert _numeric_score(frame, "He has sons and owns 2 teams.") == 0


def test_count_binding_accepts_local_number_noun_pairs():
    frame = parse_query("how many sons does he have")
    assert _numeric_score(frame, "He has four sons.") == 1
    assert _numeric_score(frame, "Sons: four, all adults.") == 1
    # Singular surface form of the requested plural noun still binds.
    assert _numeric_score(frame, "He has one son.") == 1


def test_comparison_axiom_requires_both_sides_and_contrast():
    pool, pairs = _frames(
        "difference between cats and dogs",
        "Cats are independent whereas dogs are often social.",
        "Cats are independent and agile animals.",
    )
    assert axiom_preferences(pool, pairs, ["COMPARE"])["COMPARE"].iloc[0] == 0


def test_boilerplate_score_ignores_prose_years_and_quantities():
    prose = "The firm has 3 offices and raised 40 million dollars in 1999."
    assert _boilerplate_score(prose) == 0


def test_boilerplate_score_counts_true_numbered_lists():
    listed = "Here is the plan. 1. Preheat the oven. 2. Mix the flour. 3. Bake it."
    plain = "Here is the plan: preheat the oven, mix the flour, then bake it."
    assert _boilerplate_score(plain) == 0
    assert _boilerplate_score(listed) >= 2
    assert _boilerplate_score(listed) > _boilerplate_score(plain)


def test_cbp_is_neutral_when_both_are_normal_prose():
    pool, pairs = _frames(
        "what is a solar eclipse",
        "A solar eclipse occurs when the Moon blocks the Sun.",
        "A solar eclipse happens when sunlight is obscured by the Moon.",
    )
    assert axiom_preferences(pool, pairs, ["CBP"])["CBP"].iloc[0] == 0


def test_cbp_abstains_on_explicit_list_requests():
    pool, pairs = _frames(
        "list types of solar eclipses",
        "Solar eclipses include total, partial, annular, and hybrid eclipses.",
        "1. Total eclipse 2. Partial eclipse 3. Annular eclipse",
    )
    assert not candidate_query_precondition("CBP_d0v0", "list types of solar eclipses")
    assert axiom_preferences(pool, pairs, ["CBP"])["CBP"].iloc[0] == 0


def test_cbp_abstains_on_unqualified_person_identity_queries():
    pool, pairs = _frames(
        "who is robert gray",
        "Robert Gray was an American sea captain and explorer.",
        "Incoming search terms: Robert Gray governor profile.",
    )
    assert not candidate_query_precondition("CBP_d0v1", "who is robert gray")
    assert axiom_preferences(pool, pairs, ["CBP"])["CBP"].iloc[0] == 0
