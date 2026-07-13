"""Hermetic contract tests for the Phase 4 soft-semantic pointwise reranker."""

from collections.abc import Sequence

import pytest

from axiomrank.ranking.soft_semantics import (
    AxiomThresholds,
    SoftSemanticAxiomaticReranker,
    clean_passage,
    derive_pointwise_score,
    prepare_query,
    rerank_scored_pool,
)


class FakeRelevance:
    def __init__(self, value: float):
        self.value = value
        self.calls: list[tuple[str, str]] = []

    def score(self, query: str, passage: str) -> float:
        self.calls.append((query, passage))
        return self.value


class FakeNLI:
    def __init__(self, value: float):
        self.value = value
        self.calls: list[tuple[str, str]] = []

    def entailment_probability(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        return self.value


class FakeSimilarity:
    def __init__(self, values: dict[str, float] | None = None):
        self.values = values or {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def max_similarity(self, constraint: str, passage_tokens: Sequence[str]) -> float:
        self.calls.append((constraint, tuple(passage_tokens)))
        return self.values.get(constraint, 1.0)


def _reranker(relevance=0.9, entailment=0.9, similarities=None):
    return SoftSemanticAxiomaticReranker(
        FakeRelevance(relevance), FakeNLI(entailment), FakeSimilarity(similarities)
    )


@pytest.mark.parametrize(
    ("gates", "expected"),
    [
        ((False, False, False), 0),
        ((False, True, True), 0),
        ((True, False, True), 1),
        ((True, True, False), 2),
        ((True, True, True), 3),
    ],
)
def test_scoring_matrix(gates, expected):
    assert derive_pointwise_score(*gates) == expected


def test_off_topic_gate_short_circuits_expensive_models():
    relevance, nli, similarity = FakeRelevance(0.39), FakeNLI(0.99), FakeSimilarity()
    result = SoftSemanticAxiomaticReranker(relevance, nli, similarity).evaluate(
        "how to sort a list in Python", "A recipe about baking bread."
    )
    assert result.score == 0
    assert result.gates == (False, False, False)
    assert result.entailment_probability is None
    assert not nli.calls
    assert not similarity.calls


def test_relevant_but_non_answer_is_one_and_skips_constraint_gate():
    relevance, nli, similarity = FakeRelevance(0.40), FakeNLI(0.64), FakeSimilarity()
    result = SoftSemanticAxiomaticReranker(relevance, nli, similarity).evaluate(
        "how to sort a list lang:python", "Python discusses lists."
    )
    assert result.score == 1
    assert result.gates == (True, False, False)
    assert not similarity.calls


def test_constrained_answer_scores_two_when_one_constraint_fails():
    reranker = _reranker(similarities={"python": 0.9, "docs.python.org": 0.74})
    result = reranker.evaluate(
        "sort a list lang:python site:docs.python.org", "<p>Python list sorting explained.</p>"
    )
    assert result.score == 2
    assert result.gates == (True, True, False)
    assert result.constraint_similarity == pytest.approx(0.74)
    assert result.constraints == ("python", "docs.python.org")


def test_unconstrained_answer_can_reach_three_without_embedding_model_call():
    relevance, nli, similarity = FakeRelevance(0.9), FakeNLI(0.65), FakeSimilarity()
    result = SoftSemanticAxiomaticReranker(relevance, nli, similarity).evaluate(
        "Why does rain form?", "Rain forms when water vapour condenses."
    )
    assert result.score == 3
    assert result.gates == (True, True, True)
    assert result.constraint_similarity is None
    assert not similarity.calls


def test_preprocessing_removes_structural_html_and_metadata_before_model_calls():
    relevance, nli, similarity = FakeRelevance(0.9), FakeNLI(0.9), FakeSimilarity({"python": 0.8})
    reranker = SoftSemanticAxiomaticReranker(relevance, nli, similarity)
    result = reranker.evaluate(
        "sort a list lang:python",
        "<header>home | login</header><nav>menu</nav><p>Use <b>sorted</b> in Python.</p><footer>cookies</footer>",
    )
    assert result.score == 3
    assert relevance.calls == [("sort a list", "Use sorted in Python.")]
    assert nli.calls[0][0] == "Use sorted in Python."
    assert "lang:python" not in nli.calls[0][1]
    assert similarity.calls == [("python", ("use", "sorted", "in", "python"))]


def test_prepare_query_handles_language_phrases_and_stable_deduplication():
    prepared = prepare_query("parse JSON in Python lang:python site:docs.example")
    assert prepared.text == "parse JSON in Python"
    assert prepared.constraints == ("python", "docs.example")


def test_clean_passage_plain_text_is_idempotent():
    assert clean_passage("  A plain\n passage. ") == "A plain passage."


def test_invalid_model_probability_fails_loudly():
    with pytest.raises(ValueError, match="relevance scorer"):
        _reranker(relevance=1.01).score("query", "passage")


def test_negative_cosine_similarity_is_valid_but_fails_the_constraint_gate():
    result = _reranker(similarities={"python": -0.1}).evaluate(
        "sort lang:python", "A relevant answer about another language."
    )
    assert result.score == 2


def test_threshold_validation_and_inclusive_boundaries():
    with pytest.raises(ValueError, match="relevance threshold"):
        AxiomThresholds(relevance=-0.01)
    result = _reranker(0.40, 0.65, {"python": 0.75}).evaluate(
        "sort lang:python", "Python sort answer"
    )
    assert result.score == 3


def test_rerank_scored_pool_orders_the_scored_block_and_uses_bm25_for_ties():
    import pandas as pd

    pool = pd.DataFrame(
        {
            "qid": ["q", "q", "q", "q"],
            "docno": ["a", "b", "c", "d"],
            "rank": [0, 1, 2, 3],
        }
    )
    reranked = rerank_scored_pool(
        pool,
        {("q", "a"): 2, ("q", "b"): 3, ("q", "c"): 3},
        depth=3,
    )
    assert reranked["docno"].tolist() == ["b", "c", "a", "d"]
    assert reranked["score"].tolist() == [0.0, -1.0, -2.0, -3.0]


def test_rerank_scored_pool_rejects_missing_score_within_depth():
    import pandas as pd

    pool = pd.DataFrame({"qid": ["q"], "docno": ["a"], "rank": [0]})
    with pytest.raises(ValueError, match="missing soft-semantic scores"):
        rerank_scored_pool(pool, {}, depth=1)
