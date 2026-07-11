"""Synthetic strict-vs-relaxed sanity pairs for the relaxed axioms (phase1-implementation.md §5).

Each relaxed variant gets a constructed pair where the strict axiom is neutral and the
relaxed one fires, one where both agree, and a check that margin 0 reproduces the
strict axiom. TF-LNC runs through axiom_preferences like the Phase 0 tests; M-TDC
needs index statistics, so it is exercised directly with stub tools instead of a
Terrier index.

Marked slow: importing ir_axioms (also via axiomrank.axioms/relaxed) starts the
Terrier JVM, so all ir_axioms imports stay inside the tests.
"""

import pandas as pd
import pytest

from axiomrank.config import AxiomSpec

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


TF_LNC_SPECS = [
    AxiomSpec(name="TF-LNC"),
    AxiomSpec(name="TF-LNC-R", alias="TF-LNC@r0.5", params={"margin_fraction": 0.5}),
    AxiomSpec(name="TF-LNC-R", alias="TF-LNC@r0", params={"margin_fraction": 0.0}),
]


def test_tf_lnc_relaxed_fires_where_strict_is_neutral():
    from axiomrank.axioms import axiom_preferences

    # Non-query lengths 3 vs. 2: unequal (strict neutral) but within a 0.5 rel. margin.
    pool, pairs = make_frames("cat", "cat cat dog bird fish", "cat dog bird")
    result = axiom_preferences(pool, pairs, TF_LNC_SPECS)
    assert result["TF_LNC"].iloc[0] == 0
    assert result["TF_LNC@r0.5"].iloc[0] == 1
    assert result["TF_LNC@r0"].iloc[0] == 0  # margin 0 reproduces the strict axiom


def test_tf_lnc_strict_and_relaxed_agree_at_equal_length():
    from axiomrank.axioms import axiom_preferences

    pool, pairs = make_frames("cat", "cat cat dog bird", "cat dog bird")
    result = axiom_preferences(pool, pairs, TF_LNC_SPECS)
    assert result["TF_LNC"].iloc[0] == 1
    assert result["TF_LNC@r0.5"].iloc[0] == 1
    assert result["TF_LNC@r0"].iloc[0] == 1


class _TextContents:
    def contents(self, obj):
        return obj.text


class _Tokenizer:
    def unique_terms(self, text):
        return set(text.split())


class _TextStatistics:
    def __init__(self, term_frequencies):
        self._tf = term_frequencies  # {query/document id: {term: tf}}

    def term_frequency(self, obj, term):
        return self._tf[obj.id].get(term, 0)


class _IndexStatistics:
    def __init__(self, idf):
        self._idf = idf

    def inverse_document_frequency(self, term):
        return self._idf[term]


def _mtdc_preference(axiom_cls, tf_doc1: dict, tf_doc2: dict, **kwargs):
    """Run an M-TDC variant on a two-term query with hand-set TFs and IDFs."""
    from ir_axioms.model import Document, Query

    axiom = axiom_cls(
        text_contents=_TextContents(),
        term_tokenizer=_Tokenizer(),
        index_statistics=_IndexStatistics({"rare": 5.0, "common": 1.0}),
        text_statistics=_TextStatistics(
            {"q": {"rare": 1, "common": 1}, "d1": tf_doc1, "d2": tf_doc2}
        ),
        **kwargs,
    )
    return axiom.preference(Query(id="q", text="rare common"), Document(id="d1"), Document(id="d2"))


def test_m_tdc_relaxed_fires_where_strict_gate_rejects():
    from ir_axioms.axiom.retrieval.term_frequency import ModifiedTdcAxiom

    from axiomrank.axioms.relaxed import RelaxedMTdcAxiom

    tf_doc1 = {"rare": 2, "common": 1}  # total query-term mass 3
    tf_doc2 = {"rare": 1, "common": 1}  # mass 2: strict equal-mass gate rejects
    assert _mtdc_preference(ModifiedTdcAxiom, tf_doc1, tf_doc2) == 0
    # Mass gap 1 <= 0.4 * 3, and d1 has more of the rarer term.
    assert _mtdc_preference(RelaxedMTdcAxiom, tf_doc1, tf_doc2, margin_fraction=0.4) == 1
    assert _mtdc_preference(RelaxedMTdcAxiom, tf_doc1, tf_doc2, margin_fraction=0.0) == 0


def test_m_tdc_strict_and_relaxed_agree_on_equal_mass():
    from ir_axioms.axiom.retrieval.term_frequency import ModifiedTdcAxiom

    from axiomrank.axioms.relaxed import RelaxedMTdcAxiom

    tf_doc1 = {"rare": 2, "common": 1}  # equal mass, swapped distribution: strict fires
    tf_doc2 = {"rare": 1, "common": 2}
    assert _mtdc_preference(ModifiedTdcAxiom, tf_doc1, tf_doc2) == 1
    assert _mtdc_preference(RelaxedMTdcAxiom, tf_doc1, tf_doc2, margin_fraction=0.4) == 1
    assert _mtdc_preference(RelaxedMTdcAxiom, tf_doc1, tf_doc2, margin_fraction=0.0) == 1
