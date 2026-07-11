"""Relaxed-precondition variants of strict ir_axioms axioms (phase1-implementation.md §4).

TF-LNC and M-TDC have hardcoded preconditions in ir_axioms 1.1.2 (exact non-query
length equality; exactly equal total query-term mass), which leaves them with ~0-5%
coverage on natural pairs (phase0-design.md §7.3). The subclasses here replace those
exact equalities with a relative tolerance, `margin_fraction`, and are otherwise
verbatim ports of the upstream preference logic — pinned to ir_axioms 1.1.2; re-check
on upgrade. Margin 0 reproduces the strict axiom.

These are *our* axioms: their agreement numbers are not comparable to literature
TF-LNC/M-TDC and must be reported as relaxed variants.

Import lazily (via axiomrank.axioms), after cache configuration: importing ir_axioms
starts the Terrier JVM.
"""

from dataclasses import dataclass
from itertools import combinations
from math import isclose
from typing import AbstractSet, Collection, Mapping

from injector import NoInject, inject
from ir_axioms.axiom.retrieval.length_norm import TfLncAxiom
from ir_axioms.axiom.retrieval.term_frequency import ModifiedTdcAxiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Document, Preference, Query
from ir_axioms.tools import TextStatistics
from ir_axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class RelaxedTfLncAxiom(TfLncAxiom):
    """TF-LNC with the per-term non-query length equality relaxed to a rel. margin."""

    margin_fraction: NoInject[float] = 0.1

    def _preference(
        self,
        query_unique_terms: AbstractSet[str],
        document1_terms: Collection[str],
        document2_terms: Collection[str],
        document1_term_frequencies: Mapping[str, float],
        document2_term_frequencies: Mapping[str, float],
    ) -> Preference:
        sum_document1 = 0
        sum_document2 = 0
        for query_term in query_unique_terms:
            tf_d1 = document1_term_frequencies[query_term]
            tf_d2 = document2_term_frequencies[query_term]

            len_d1 = sum(1 for term in document1_terms if term != query_term)
            len_d2 = sum(1 for term in document2_terms if term != query_term)

            # Upstream requires len_d1 == len_d2 exactly.
            if isclose(len_d1, len_d2, rel_tol=self.margin_fraction):
                if tf_d1 > tf_d2:
                    sum_document1 += 1
                elif tf_d2 > tf_d1:
                    sum_document2 += 1

        return strictly_greater(sum_document1, sum_document2)


def _approx_equal_mass_different_distribution(
    query_unique_terms: AbstractSet,
    text_statistics: TextStatistics[Document],
    output1: Document,
    output2: Document,
    margin_fraction: float,
) -> bool:
    """M-TDC's gate with the total-mass equality relaxed to a relative margin."""
    sum_term_frequency1 = 0.0
    sum_term_frequency2 = 0.0
    term_frequency_different = False
    for term in query_unique_terms:
        count1 = text_statistics.term_frequency(output1, term)
        count2 = text_statistics.term_frequency(output2, term)
        if count1 != count2:
            term_frequency_different = True
        sum_term_frequency1 += count1
        sum_term_frequency2 += count2
    return (
        isclose(sum_term_frequency1, sum_term_frequency2, rel_tol=margin_fraction)
        and term_frequency_different
    )


@inject
@dataclass(frozen=True, kw_only=True)
class RelaxedMTdcAxiom(ModifiedTdcAxiom):
    """M-TDC with the equal-total-query-term-mass gate relaxed to a rel. margin.

    The per-term-pair validity logic (swapped-frequency / query-frequency checks) is
    kept verbatim from upstream; only the entry gate is widened.
    """

    margin_fraction: NoInject[float] = 0.1

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )

        if not _approx_equal_mass_different_distribution(
            query_unique_terms=query_unique_terms,
            text_statistics=self.text_statistics,
            output1=output1,
            output2=output2,
            margin_fraction=self.margin_fraction,
        ):
            return 0

        score1 = 0
        score2 = 0

        for query_term1, query_term2 in combinations(query_unique_terms, 2):
            idf_qt1 = self.index_statistics.inverse_document_frequency(query_term1)
            idf_qt2 = self.index_statistics.inverse_document_frequency(query_term2)

            if isclose(idf_qt1, idf_qt2):
                # Equally rare query terms would make the axiom random; skip.
                continue

            if idf_qt1 < idf_qt2:
                # Query term 1 is rarer. Swap query terms.
                query_term1, query_term2 = query_term2, query_term1

            tf_d1_qt1 = self.text_statistics.term_frequency(output1, query_term1)
            tf_d1_qt2 = self.text_statistics.term_frequency(output1, query_term2)
            tf_d2_qt1 = self.text_statistics.term_frequency(output2, query_term1)
            tf_d2_qt2 = self.text_statistics.term_frequency(output2, query_term2)
            tf_q_qt1 = self.text_statistics.term_frequency(input, query_term1)
            tf_q_qt2 = self.text_statistics.term_frequency(input, query_term2)

            if not (
                (isclose(tf_d1_qt1, tf_d2_qt2) and isclose(tf_d1_qt2, tf_d2_qt1))
                or tf_q_qt1 >= tf_q_qt2
            ):
                continue

            if tf_q_qt1 < tf_q_qt2 and (
                tf_d1_qt1 != tf_d2_qt2 or tf_d1_qt2 != tf_d2_qt1
            ):
                continue

            # Document with more occurrences of the rarer query term gets a point.
            if tf_d1_qt1 > tf_d2_qt1:
                score1 += 1
            elif tf_d1_qt1 < tf_d2_qt1:
                score2 += 1

        return strictly_greater(score1, score2)


# Factories mirroring ir_axioms' naming convention, resolvable from axiom configs.
TF_LNC_R = lazy_inject(RelaxedTfLncAxiom)
M_TDC_R = lazy_inject(RelaxedMTdcAxiom)
