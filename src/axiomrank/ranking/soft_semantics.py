"""Pointwise soft-semantic answer reranker for the Phase 3 / RQ4 path.

The component deliberately separates probabilistic semantic measurements from the
final decision: models return calibrated-ish continuous signals, while the public
``score`` is always the documented integer in ``{0, 1, 2, 3}``.  Model adapters are
lazy, optional dependencies.  Supplying the three small protocol implementations is
therefore enough to use this module in tests or with a non-Hugging-Face inference
service, without an implicit model download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from math import isfinite
import re
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

from axiomrank import paths


DEFAULT_RELEVANCE_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
# The originally proposed ``MoritzLaurer/DeBERTa-v3-xsmall-mnli`` identifier is
# not a public Hugging Face repository.  This public xsmall DeBERTa NLI cross
# encoder has the same three-way NLI output required by Gate 2.
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MRC_MODEL = "deepset/roberta-base-squad2"

_TOKEN = re.compile(r"[a-z0-9]+(?:[._+-][a-z0-9]+)*", re.IGNORECASE)
_METADATA = re.compile(r"(?<!\S)(?:lang|site):([^\s]+)", re.IGNORECASE)
_LANGUAGE = re.compile(
    r"\b(?:in|using|with|for)\s+"
    r"(python|java(?:script)?|typescript|rust|go|c\+\+|c#|ruby|php|kotlin|swift|r)\b",
    re.IGNORECASE,
)


def derive_pointwise_score(a1: bool, a2: bool, a3: bool) -> int:
    """Return the fixed 0--3 score from the three named axioms."""
    if not a1:
        return 0
    if not a2:
        return 1
    if not a3:
        return 2
    return 3


class _ContentParser(HTMLParser):
    """Small dependency-free prose extractor for passages containing HTML."""

    _SKIP = frozenset({"footer", "header", "nav", "noscript", "script", "style", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[no-untyped-def]
        if tag.lower() in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def clean_passage(passage: str) -> str:
    """Remove common HTML structural noise and normalise whitespace.

    This is intentionally stdlib-only.  Deployments that already clean documents can
    pass their cleaned text directly; the operation is idempotent for plain text.
    """
    parser = _ContentParser()
    parser.feed(passage)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


def _normalise_constraint(value: str) -> str:
    return " ".join(_TOKEN.findall(value.lower()))


@dataclass(frozen=True)
class PreparedQuery:
    """Search text and independently checkable hard/soft query constraints."""

    text: str
    constraints: tuple[str, ...]


def prepare_query(query: str) -> PreparedQuery:
    """Extract supported inline constraints without any LLM parsing.

    ``lang:python`` and ``site:docs.python.org`` are preserved as constraints but
    removed from the text passed to relevance/NLI models.  Language phrases such as
    ``"sort a list in Python"`` add the ``python`` constraint while retaining the
    natural-language query.
    """
    constraints: list[str] = []

    def take_metadata(match: re.Match[str]) -> str:
        value = _normalise_constraint(match.group(1))
        if value:
            constraints.append(value)
        return " "

    text = _METADATA.sub(take_metadata, query)
    for match in _LANGUAGE.finditer(text):
        value = _normalise_constraint(match.group(1))
        if value:
            constraints.append(value)
    # Stable de-duplication matters for logs and for reproducible batch results.
    unique = tuple(dict.fromkeys(constraints))
    return PreparedQuery(text=" ".join(text.split()), constraints=unique)


@runtime_checkable
class RelevanceScorer(Protocol):
    """Returns a relevance probability for one query/passage pair."""

    def score(self, query: str, passage: str) -> float: ...


@runtime_checkable
class EntailmentScorer(Protocol):
    """Returns P(hypothesis is entailed by premise)."""

    def entailment_probability(self, premise: str, hypothesis: str) -> float: ...


@runtime_checkable
class TokenSimilarity(Protocol):
    """Returns max cosine-like similarity between a constraint and passage tokens."""

    def max_similarity(self, constraint: str, passage_tokens: Sequence[str]) -> float: ...


@dataclass(frozen=True)
class AxiomThresholds:
    relevance: float = 0.40
    entailment: float = 0.65
    constraint_similarity: float = 0.75

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} threshold must be in [0, 1], got {value}")


@dataclass(frozen=True)
class AxiomEvaluation:
    """Auditable result of a single pointwise reranking decision."""

    score: int
    axiom_1_relevant: bool
    axiom_2_answers: bool
    axiom_3_constraints: bool
    relevance_probability: float
    entailment_probability: float | None
    constraint_similarity: float | None
    query: str
    constraints: tuple[str, ...]
    passage: str

    @property
    def gates(self) -> tuple[bool, bool, bool]:
        return self.axiom_1_relevant, self.axiom_2_answers, self.axiom_3_constraints


def _probability(value: float, source: str) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{source} must return a finite probability in [0, 1], got {value}")
    return value


def _similarity(value: float) -> float:
    """Validate a cosine-like signal without incorrectly assuming it is a probability."""
    value = float(value)
    if not isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"token similarity must be finite and in [-1, 1], got {value}")
    return value


@dataclass
class SoftSemanticAxiomaticReranker:
    """Evaluate relevance, answer alignment and constraints in cost order.

    Gate 2 is skipped for off-topic documents and Gate 3 is skipped unless the first
    two gates pass.  A query with no extractable constraint satisfies Axiom 3 by
    construction; otherwise an unconstrained question could never receive score 3.
    """

    relevance_scorer: RelevanceScorer
    entailment_scorer: EntailmentScorer
    token_similarity: TokenSimilarity
    thresholds: AxiomThresholds = field(default_factory=AxiomThresholds)

    def evaluate(self, query: str, passage: str) -> AxiomEvaluation:
        prepared = prepare_query(query)
        cleaned = clean_passage(passage)
        relevance = _probability(self.relevance_scorer.score(prepared.text, cleaned), "relevance scorer")
        a1 = relevance >= self.thresholds.relevance
        if not a1:
            return AxiomEvaluation(0, False, False, False, relevance, None, None,
                                   prepared.text, prepared.constraints, cleaned)

        hypothesis = f"This text provides an answer or explanation to the question: {prepared.text}"
        entailment = _probability(
            self.entailment_scorer.entailment_probability(cleaned, hypothesis), "entailment scorer"
        )
        a2 = entailment >= self.thresholds.entailment
        if not a2:
            return AxiomEvaluation(1, True, False, False, relevance, entailment, None,
                                   prepared.text, prepared.constraints, cleaned)

        if not prepared.constraints:
            return AxiomEvaluation(3, True, True, True, relevance, entailment, None,
                                   prepared.text, prepared.constraints, cleaned)
        tokens = tuple(dict.fromkeys(token.lower() for token in _TOKEN.findall(cleaned)))
        similarities = [
            _similarity(self.token_similarity.max_similarity(constraint, tokens))
            for constraint in prepared.constraints
        ]
        # All explicit constraints must be present; the weakest one is the gate value.
        similarity = min(similarities)
        a3 = similarity >= self.thresholds.constraint_similarity
        return AxiomEvaluation(derive_pointwise_score(True, True, a3), True, True, a3,
                               relevance, entailment, similarity, prepared.text,
                               prepared.constraints, cleaned)

    def score(self, query: str, passage: str) -> int:
        """Return only the deterministic pointwise score; use ``evaluate`` for logs."""
        return self.evaluate(query, passage).score

    @classmethod
    def from_huggingface(
        cls,
        *,
        relevance_model: str = DEFAULT_RELEVANCE_MODEL,
        nli_model: str = DEFAULT_NLI_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
        thresholds: AxiomThresholds | None = None,
    ) -> "SoftSemanticAxiomaticReranker":
        """Build lazy local-transformer adapters (no models load until first score)."""
        return cls(
            HuggingFaceCrossEncoder(relevance_model, device=device),
            HuggingFaceNLI(nli_model, device=device),
            HuggingFaceTokenSimilarity(embedding_model, device=device),
            thresholds or AxiomThresholds(),
        )


def rerank_scored_pool(
    pool: pd.DataFrame,
    scores: dict[tuple[str, str], int],
    *,
    depth: int,
) -> pd.DataFrame:
    """Rerank a BM25 block from precomputed pointwise 0--3 axiom scores.

    Every document in the requested BM25 depth must have a score.  Documents beyond
    the depth remain below the reranked block in their original BM25 order.  The
    output's strictly decreasing ``score`` encodes the final tie-broken order for
    evaluation tools that do not consume a rank column.
    """
    required = {"qid", "docno", "rank"}
    if missing := required - set(pool.columns):
        raise ValueError(f"pool is missing required columns: {sorted(missing)}")
    if depth <= 0:
        raise ValueError("depth must be positive")

    rows: list[tuple[object, object, int, float]] = []
    for qid, group in pool.groupby("qid", sort=False):
        scoped = group.loc[group["rank"] < depth]
        missing = [
            (str(row.qid), str(row.docno))
            for row in scoped[["qid", "docno"]].itertuples(index=False)
            if (str(row.qid), str(row.docno)) not in scores
        ]
        if missing:
            preview = ", ".join(f"{query_id}/{docno}" for query_id, docno in missing[:3])
            raise ValueError(
                f"missing soft-semantic scores for {len(missing)} documents in query {qid} "
                f"(e.g. {preview})"
            )
        ordered = sorted(
            group.itertuples(index=False),
            key=lambda document: (
                0 if document.rank < depth else 1,
                -scores.get((str(document.qid), str(document.docno)), -1),
                document.rank,
            ),
        )
        rows.extend(
            (qid, document.docno, rank, float(-rank))
            for rank, document in enumerate(ordered)
        )
    return pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])


def _transformers_imports():
    paths.configure_caches()
    try:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Soft semantic inference needs `uv sync --extra llm`.") from exc
    return torch, AutoModel, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class HuggingFaceCrossEncoder:
    """Lazy sequence-classification cross encoder returning a sigmoid relevance score."""

    model_name: str = DEFAULT_RELEVANCE_MODEL
    device: str = "cpu"
    max_length: int = 512
    _bundle: tuple = field(init=False, repr=False, default=None)

    def _load(self):
        if self._bundle is None:
            torch, _, sequence_classifier, tokenizer_cls = _transformers_imports()
            tokenizer = tokenizer_cls.from_pretrained(self.model_name)
            model = sequence_classifier.from_pretrained(self.model_name).to(self.device).eval()
            self._bundle = torch, tokenizer, model
        return self._bundle

    def score(self, query: str, passage: str) -> float:
        torch, tokenizer, model = self._load()
        encoded = tokenizer(query, passage, return_tensors="pt", truncation=True, max_length=self.max_length)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        if len(logits) != 1:
            raise ValueError(f"Cross encoder {self.model_name!r} must have one output logit")
        return float(torch.sigmoid(logits[0]).item())


@dataclass
class HuggingFaceNLI:
    """Lazy NLI classifier which resolves the entailment label from model metadata."""

    model_name: str = DEFAULT_NLI_MODEL
    device: str = "cpu"
    max_length: int = 512
    _bundle: tuple = field(init=False, repr=False, default=None)

    def _load(self):
        if self._bundle is None:
            torch, _, sequence_classifier, tokenizer_cls = _transformers_imports()
            tokenizer = tokenizer_cls.from_pretrained(self.model_name)
            model = sequence_classifier.from_pretrained(self.model_name).to(self.device).eval()
            labels = {str(label).lower(): int(index) for index, label in model.config.id2label.items()}
            entailment = next((index for label, index in labels.items() if "entail" in label), None)
            if entailment is None:
                raise ValueError(f"NLI model {self.model_name!r} has no entailment label: {labels}")
            self._bundle = torch, tokenizer, model, entailment
        return self._bundle

    def entailment_probability(self, premise: str, hypothesis: str) -> float:
        torch, tokenizer, model, entailment = self._load()
        encoded = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True,
                            max_length=self.max_length)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            probabilities = torch.softmax(model(**encoded).logits[0], dim=-1)
        return float(probabilities[entailment].item())


@dataclass
class HuggingFaceTokenSimilarity:
    """Mean-pooled MiniLM embeddings for the constraint-to-token matching gate."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str = "cpu"
    max_length: int = 32
    max_passage_tokens: int = 512
    _bundle: tuple = field(init=False, repr=False, default=None)

    def _load(self):
        if self._bundle is None:
            torch, model_cls, _, tokenizer_cls = _transformers_imports()
            tokenizer = tokenizer_cls.from_pretrained(self.model_name)
            model = model_cls.from_pretrained(self.model_name).to(self.device).eval()
            self._bundle = torch, tokenizer, model
        return self._bundle

    def _encode(self, texts: Sequence[str]):
        torch, tokenizer, model = self._load()
        encoded = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True,
                            max_length=self.max_length)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            states = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            vectors = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return torch.nn.functional.normalize(vectors, p=2, dim=1)

    def max_similarity(self, constraint: str, passage_tokens: Sequence[str]) -> float:
        if not passage_tokens:
            return 0.0
        vectors = self._encode([constraint, *passage_tokens[: self.max_passage_tokens]])
        return float((vectors[1:] @ vectors[0]).max().item())


@dataclass
class HuggingFaceMRC:
    """Lazy extractive-QA adapter returning a null-adjusted answer-span margin.

    Positive values favour a span in the passage over the SQuAD2 no-answer ``[CLS]``
    position.  It is intentionally a raw continuous feature, not an answer-correctness
    claim or a thresholded gate.
    """

    model_name: str = DEFAULT_MRC_MODEL
    device: str = "cpu"
    max_length: int = 512
    max_answer_tokens: int = 30
    _bundle: tuple = field(init=False, repr=False, default=None)

    def _load(self):
        if self._bundle is None:
            torch, _, _, tokenizer_cls = _transformers_imports()
            from transformers import AutoModelForQuestionAnswering

            tokenizer = tokenizer_cls.from_pretrained(self.model_name, use_fast=True)
            model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(self.device).eval()
            self._bundle = torch, tokenizer, model
        return self._bundle

    def answerability_margin(self, query: str, passage: str) -> float:
        torch, tokenizer, model = self._load()
        encoded = tokenizer(
            query,
            passage,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_length,
        )
        sequence_ids = encoded.sequence_ids(0)
        context = [index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
        if not context:
            return float("-inf")
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**model_inputs)
        start = outputs.start_logits[0].detach().cpu().tolist()
        end = outputs.end_logits[0].detach().cpu().tolist()
        best = float("-inf")
        for offset, start_index in enumerate(context):
            for end_index in context[offset : offset + self.max_answer_tokens]:
                best = max(best, float(start[start_index] + end[end_index]))
        null = float(start[0] + end[0])
        return best - null
