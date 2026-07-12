"""Unit tests for the RQ4 residual-seed axioms VERB, VERB_R, QCOV (phase3-design.md §2).

Hand-built cases exercise the direction, the VERB same-covered-terms precondition gate, the
VERB_R coverage-fraction margin widening, the QCOV graded direction, and the tie /
empty-query neutral cases. A pass through `axiom_preferences` (the real spaCy tokenizer)
checks name resolution through the registry and determinism (two runs, identical output).

Marked slow: importing ir_axioms (also via axiomrank.axioms) starts the Terrier JVM, so all
ir_axioms imports stay inside the tests.
"""

import pandas as pd
import pytest

from axiomrank.config import AxiomSpec

pytestmark = pytest.mark.slow


class _TextContents:
    def contents(self, obj):
        return obj.text


class _Tokenizer:
    """Whitespace tokenizer with the TermTokenizer surface the axioms use."""

    def terms(self, text):
        return text.split()

    def unique_terms(self, text):
        return set(text.split())


def _pref(axiom_cls, query: str, text_1: str, text_2: str, **kwargs):
    from ir_axioms.model import Document, Query

    axiom = axiom_cls(text_contents=_TextContents(), term_tokenizer=_Tokenizer(), **kwargs)
    return axiom.preference(
        Query(id="q", text=query),
        Document(id="d1", text=text_1),
        Document(id="d2", text=text_2),
    )


# --- VERB -----------------------------------------------------------------------------

def test_verb_prefers_longer_when_same_query_terms_covered():
    from axiomrank.axioms.rq4 import VerbosityAxiom

    # Both cover exactly {cat}; doc1 is the longer document.
    assert _pref(VerbosityAxiom, "cat", "cat dog bird fish tree", "cat dog") == 1
    assert _pref(VerbosityAxiom, "cat", "cat dog", "cat dog bird fish tree") == -1


def test_verb_precondition_gates_on_different_covered_terms():
    from axiomrank.axioms.rq4 import VerbosityAxiom

    # doc1 is longer but covers {a} while doc2 covers {a, b}: precondition fails -> 0.
    assert _pref(VerbosityAxiom, "a b", "a x y z w", "a b") == 0


def test_verb_tie_on_equal_length_is_neutral():
    from axiomrank.axioms.rq4 import VerbosityAxiom

    # Same covered set {cat}, equal word length -> tie -> 0.
    assert _pref(VerbosityAxiom, "cat", "cat dog bird", "cat fish tree") == 0


# --- VERB_R ---------------------------------------------------------------------------

def test_verb_r_fires_where_strict_verb_gate_rejects_on_equal_fraction():
    from axiomrank.axioms.rq4 import RelaxedVerbosityAxiom, VerbosityAxiom

    # doc1 covers {a} (cov 0.5), doc2 covers {b} (cov 0.5): different SET (VERB neutral)
    # but equal coverage FRACTION, so VERB_R's isclose gate opens and prefers the longer.
    assert _pref(VerbosityAxiom, "a b", "a x y z", "b w") == 0
    assert _pref(RelaxedVerbosityAxiom, "a b", "a x y z", "b w", margin_fraction=0.0) == 1


def test_verb_r_margin_widens_admissible_coverage_gap():
    from axiomrank.axioms.rq4 import RelaxedVerbosityAxiom

    # doc1 covers a,b,c (cov 0.6, longer); doc2 covers a,b (cov 0.4). |0.6-0.4|=0.2.
    q, t1, t2 = "a b c d e", "a b c x y z", "a b"
    # rel_tol 0.1: 0.2 > 0.1*0.6 -> gate closed -> neutral.
    assert _pref(RelaxedVerbosityAxiom, q, t1, t2, margin_fraction=0.1) == 0
    # rel_tol 0.5: 0.2 <= 0.5*0.6 -> gate open -> prefer the longer doc1.
    assert _pref(RelaxedVerbosityAxiom, q, t1, t2, margin_fraction=0.5) == 1


# --- QCOV -----------------------------------------------------------------------------

def test_qcov_prefers_higher_distinct_coverage():
    from axiomrank.axioms.rq4 import QueryCoverageAxiom

    # doc1 covers a,b,c (3/3); doc2 covers a (1/3) -> prefer doc1. Graded, not all-or-none.
    assert _pref(QueryCoverageAxiom, "a b c", "a b c z", "a z") == 1
    # doc1 covers a,b (2/3); doc2 covers a,b,c (3/3) -> prefer doc2, even though neither
    # contains *all* terms in doc1's case (distinct from AND).
    assert _pref(QueryCoverageAxiom, "a b c", "a b", "a b c") == -1


def test_qcov_tie_is_neutral():
    from axiomrank.axioms.rq4 import QueryCoverageAxiom

    assert _pref(QueryCoverageAxiom, "a b c", "a b z z z", "a b w") == 0


def test_qcov_empty_query_is_neutral():
    from axiomrank.axioms.rq4 import QueryCoverageAxiom

    assert _pref(QueryCoverageAxiom, "", "some long document", "short") == 0


# --- registry resolution + determinism through axiom_preferences ----------------------

RQ4_SPECS = [
    AxiomSpec(name="VERB"),
    AxiomSpec(name="QCOV"),
    AxiomSpec(name="VERB_R", alias="VERB@m0.2", params={"margin_fraction": 0.2}),
]


def _frames(query, text_1, text_2):
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


def test_rq4_axioms_resolve_by_name_and_produce_columns():
    from axiomrank.axioms import axiom_preferences

    pool, pairs = _frames("machine learning", "machine learning models are useful tools",
                          "machine only")
    result = axiom_preferences(pool, pairs, RQ4_SPECS)
    assert {"VERB", "QCOV", "VERB@m0.2"} <= set(result.columns)
    # doc1 covers both query terms and is longer; QCOV prefers doc1.
    assert result["QCOV"].iloc[0] == 1


def test_rq4_axioms_are_deterministic():
    from axiomrank.axioms import axiom_preferences

    pool, pairs = _frames("machine learning", "machine learning models are useful tools",
                          "machine only")
    a = axiom_preferences(pool, pairs, RQ4_SPECS)
    b = axiom_preferences(pool, pairs, RQ4_SPECS)
    pd.testing.assert_frame_equal(a, b)
