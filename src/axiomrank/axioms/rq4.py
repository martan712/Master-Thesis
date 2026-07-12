"""RQ4 residual-seed axioms: verbosity/length (VERB) and query-coverage (QCOV).

Phase 2 decomposed a competent LLM pairwise ranker into a thin axiom-explained part and a
large, content-shaped residual whose systematic part clusters into two shapes
(`thesis/phase2-writeup.md` §3.4-3.5): a **verbosity/length** cluster (`d_len`) and a
**query-coverage** cluster (`d_qcov`). This module formalises those two residual seeds as
new retrieval axioms, following `docs/phase3-design.md` §1.

Both are text-only pairwise axioms of the same cheap shape as PROX1 — computed from the
term tokenizer alone, no collection statistics, no downloads. They are the discrete
preference form of the residual covariates in `analysis/covariates.py` (`_pair_features`:
word-level length, distinct query-term coverage). Direction is fixed to the *observed*
residual direction (the model preferred longer / higher-coverage documents); whether that
direction earns weight is decided by the decomposition, not tuned here.

The injection plumbing (`@inject` / `term_tokenizer` / `text_contents`) mirrors
`DeterministicProx1Axiom` in `axioms/relaxed.py`; `strictly_greater` fixes the sign. The
axioms are deterministic: they read only set sizes and token counts, never iteration order
of a set. Pinned to ir_axioms 1.1.2.
"""

from dataclasses import dataclass
from math import isclose
from typing import AbstractSet, Sequence, Union

from injector import NoInject, inject
from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Document, Preference, Query
from ir_axioms.tools import TermTokenizer, TextContents
from ir_axioms.utils.lazy import lazy_inject


def _coverage_fraction(
    query_unique_terms: AbstractSet[str], document_terms: AbstractSet[str]
) -> float:
    """Fraction of the query's distinct terms the document covers; 0 for an empty query."""
    if not query_unique_terms:
        return 0.0
    return len(query_unique_terms & document_terms) / len(query_unique_terms)


@inject
@dataclass(frozen=True, kw_only=True)
class VerbosityAxiom(Axiom[Query, Document]):
    """VERB — among documents giving comparable query evidence, prefer the longer one.

    Precondition (the "other things equal" gate that stops VERB re-expressing TF/coverage):
    the two documents cover the **same set of distinct query terms**. When it holds,
    ``preference = strictly_greater(len_1, len_2)`` on document word length; 0 when lengths
    tie or the precondition fails. Formalises the long-document residual cluster
    (`phase2-writeup.md` §3.5) that subsumes the Phase 1 DIV-with-gap reversal.
    """

    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )

        covered1 = query_unique_terms & set(document1_terms)
        covered2 = query_unique_terms & set(document2_terms)
        # Precondition: the two documents cover the same set of distinct query terms.
        if covered1 != covered2:
            return 0

        return strictly_greater(len(document1_terms), len(document2_terms))


@inject
@dataclass(frozen=True, kw_only=True)
class RelaxedVerbosityAxiom(Axiom[Query, Document]):
    """VERB_R(margin_fraction) — VERB with the same-covered-terms gate relaxed.

    The exact same-covered-set precondition of VERB is widened to
    ``isclose(cov_1, cov_2, rel_tol=margin_fraction)`` on the query-coverage *fraction*,
    mirroring the relaxed TF-LNC / M-TDC pattern in `axioms/relaxed.py`. Margin 0 requires
    exactly equal coverage fractions (looser than VERB's set equality: same count, possibly
    different terms). Given the gate holds, ``preference = strictly_greater(len_1, len_2)``.
    """

    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
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
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )

        cov1 = _coverage_fraction(query_unique_terms, set(document1_terms))
        cov2 = _coverage_fraction(query_unique_terms, set(document2_terms))
        if not isclose(cov1, cov2, rel_tol=self.margin_fraction):
            return 0

        return strictly_greater(len(document1_terms), len(document2_terms))


@inject
@dataclass(frozen=True, kw_only=True)
class QueryCoverageAxiom(Axiom[Query, Document]):
    """QCOV — prefer the document covering more distinct query terms.

    ``cov_i = |query_terms ∩ doc_terms| / |query_terms|``,
    ``preference = strictly_greater(cov_1, cov_2)``; 0 on a tie or empty query. The *graded*
    relaxation of AND (which fires only when one document contains **all** query terms), and
    distinct from the TF axioms (which count frequency, not distinct coverage). Formalises
    the query-coverage residual cluster (`phase2-writeup.md` §3.5).
    """

    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        if not query_unique_terms:
            return 0

        document1_terms: Sequence[str] = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document2_terms: Sequence[str] = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )
        cov1 = _coverage_fraction(query_unique_terms, set(document1_terms))
        cov2 = _coverage_fraction(query_unique_terms, set(document2_terms))

        return strictly_greater(cov1, cov2)


# Factories mirroring ir_axioms' naming convention, resolvable from axiom configs by name
# (axiomrank.axioms.registry resolves rq4/relaxed before falling back to ir_axioms).
VERB = lazy_inject(VerbosityAxiom)
VERB_R = lazy_inject(RelaxedVerbosityAxiom)
QCOV = lazy_inject(QueryCoverageAxiom)
