"""Cheap deterministic answer-shape axioms for Phase 3 development v0.

These rules implement the D0 candidates specified in
``docs/phase3-candidate-axiom-specs.md``. They deliberately trade recall for explicit
preconditions and auditable neutrality. No rule estimates factual correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from re import compile as re_compile
from typing import Union

from injector import NoInject, inject
from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Document, Preference, Query
from ir_axioms.tools import TextContents
from ir_axioms.utils.lazy import lazy_inject


_TOKEN = re_compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE = re_compile(r"(?<=[.!?])\s+|[\r\n]+")
_NUMBER = (
    r"(?:\d+(?:[.,]\d+)*|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b)"
)
_MONEY = re_compile(
    rf"(?:[$€£]\s*{_NUMBER}|{_NUMBER}\s*(?:dollars?|euros?|pounds?)|"
    rf"{_NUMBER}\s*(?:per|a)\s*(?:year|month|week|day|hour)|"
    rf"(?:salary|income|earn(?:s|ed|ing)?|make(?:s|ing)?)\D{{0,30}}{_NUMBER})"
)
_DURATION = re_compile(
    rf"{_NUMBER}\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)"
)
_AGE = re_compile(rf"{_NUMBER}\s*(?:years?\s*old|year-old)")
_NUMBER_WORDS = frozenset(
    ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
)
# Maximum token distance between a cardinal and the requested noun for a count to bind.
_COUNT_WINDOW = 2

_STOP = {
    "a", "an", "and", "are", "be", "between", "do", "does", "for", "how", "in",
    "is", "of", "or", "the", "to", "what", "when", "where", "which", "who", "why",
}
_DEFINITION_CUES = {
    "define", "definition", "meaning", "mean", "means", "stand", "stands", "medical",
}
_NUMERIC_CUES = {
    "much", "many", "long", "old", "money", "make", "makes", "earn", "earns", "take",
    "takes",
}
_FINITE_VERBS = re_compile(
    r"\b(?:am|is|are|was|were|be|been|being|has|have|had|can|could|will|would|"
    r"means|refers|contains|uses|used|treats|prevents|makes|costs|takes|provides)\b"
)
_BOILERPLATE_CUES = (
    "incoming search terms",
    "available from:",
    "all acronyms",
    "answered by a verified",
    "article about",
    "click here",
    "contact us",
    "accessed ",
    "add your answer",
)
_LIST_REQUEST = re_compile(
    r"^(?:list|name|give|show)\b|\b(?:examples|types|kinds|list)\s+of\b|^top\s+\d+\b"
)
# A numbered-list item requires actual list punctuation after the numeral, so ordinary
# prose years and quantities ("In 1999 Smith became ...") are not counted as boilerplate.
_LIST_ITEM = re_compile(r"(?:^|\s)\d+[.)]\s+")
_NON_DEFINITIONAL_COPULAS = (
    "associated", "available", "found", "included", "listed", "located", "mentioned",
    "prescribed", "shown", "used",
)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text.lower()))


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(sentence.strip() for sentence in _SENTENCE.split(text) if sentence.strip())


def _coverage(anchors: frozenset[str], text: str) -> float:
    if not anchors:
        return 0.0
    return len(anchors & set(_tokens(text))) / len(anchors)


@dataclass(frozen=True)
class QueryFrame:
    intent: str
    anchors: frozenset[str]
    numeric_type: str | None = None
    numeric_unit: str | None = None
    comparison_left: frozenset[str] = frozenset()
    comparison_right: frozenset[str] = frozenset()


def _content_terms(text: str, extra_stop: set[str] | frozenset[str] = frozenset()) -> frozenset[str]:
    return frozenset(
        token for token in _tokens(text) if token not in _STOP and token not in extra_stop
    )


@lru_cache(maxsize=1 << 15)
def parse_query(query: str) -> QueryFrame:
    """Parse only high-precision D0 intents; ambiguous queries remain UNKNOWN."""
    lower = " ".join(_tokens(query))
    comparison = re_compile(r"^difference between (.+?) and (.+?)(?: is)?$").match(lower)
    if comparison:
        left = _content_terms(comparison.group(1))
        right = _content_terms(comparison.group(2))
        return QueryFrame("comparison", left | right, comparison_left=left, comparison_right=right)

    definition = (
        lower.startswith("define ")
        or lower.startswith("definition of ")
        or lower.startswith("meaning of ")
        or (lower.startswith("what does ") and lower.endswith(" mean"))
        or (lower.startswith("what does ") and lower.endswith(" stand for"))
    )
    if definition:
        return QueryFrame("definition", _content_terms(lower, _DEFINITION_CUES))

    numeric_type = None
    if lower.startswith("how much "):
        numeric_type = "money" if any(cue in lower for cue in ("money", "cost", "earn", "make")) else "count"
    elif lower.startswith("how many "):
        numeric_type = "count"
    elif lower.startswith("how long "):
        numeric_type = "duration"
    elif lower.startswith("how old "):
        numeric_type = "age"
    if numeric_type:
        numeric_unit = None
        if numeric_type == "count":
            remainder = lower.removeprefix("how many ")
            terms = _content_terms(remainder, _NUMERIC_CUES)
            numeric_unit = next(
                (token for token in _tokens(remainder) if token in terms),
                None,
            )
        return QueryFrame(
            "numeric",
            _content_terms(lower, _NUMERIC_CUES),
            numeric_type=numeric_type,
            numeric_unit=numeric_unit,
        )

    return QueryFrame("unknown", frozenset())


def candidate_query_precondition(candidate: str, query: str) -> bool:
    """Return the explicit D0-v0 query-side eligibility decision.

    Pair-side topical comparability and evidence margins are evaluated separately by the
    axioms. Version suffixes used in experiment aliases are ignored here.
    """
    name = candidate.upper().split("@")[0].split("_D0")[0]
    frame = parse_query(query)
    if name == "DEFANS":
        return frame.intent == "definition"
    if name == "NUMANS":
        return frame.intent == "numeric"
    if name == "COMPARE":
        return frame.intent == "comparison"
    if name == "CBP":
        lower = " ".join(_tokens(query))
        return (
            bool(_content_terms(lower))
            and not _LIST_REQUEST.search(lower)
            and not _unqualified_person_query(lower)
        )
    raise ValueError(f"Unknown D0 candidate: {candidate}")


def _comparable(
    frame: QueryFrame,
    text1: str,
    text2: str,
    min_anchor_coverage: float,
    max_anchor_gap: float,
) -> bool:
    if not frame.anchors:
        return False
    cov1 = _coverage(frame.anchors, text1)
    cov2 = _coverage(frame.anchors, text2)
    return min(cov1, cov2) >= min_anchor_coverage and abs(cov1 - cov2) <= max_anchor_gap


def _definition_score(frame: QueryFrame, text: str) -> int:
    if frame.intent != "definition":
        return 0
    score = 0
    for sentence in _sentences(text):
        lower = sentence.lower()
        present = [anchor for anchor in frame.anchors if anchor in set(_tokens(lower))]
        if not present:
            continue
        relation = False
        for anchor in present:
            explicit_relation = re_compile(
                rf"\b{anchor}\b\s*(?::|-|means\b|refers\s+to\b|stands\s+for\b)"
            )
            blocked = "|".join(_NON_DEFINITIONAL_COPULAS)
            copular_relation = re_compile(
                rf"\b{anchor}\b\s+(?:is|are)\s+(?!(?:{blocked})\b)"
            )
            if explicit_relation.search(lower) or copular_relation.search(lower):
                relation = True
                break
        if relation:
            sentence_score = 1
            non_query = set(_tokens(lower)) - frame.anchors - _STOP - _DEFINITION_CUES
            if len(non_query) >= 2:
                sentence_score += 1
            score = max(score, sentence_score)
    return score


def _count_unit_forms(unit: str) -> frozenset[str]:
    """Deterministic singular/plural surface forms of the requested count noun."""
    forms = {unit}
    forms.add(unit[:-1] if unit.endswith("s") else unit + "s")
    return frozenset(forms)


def _count_bound(sentence: str, unit: str) -> bool:
    """True when a cardinal sits within `_COUNT_WINDOW` tokens of the requested noun."""
    tokens = _tokens(sentence)
    forms = _count_unit_forms(unit)
    number_positions = [
        index
        for index, token in enumerate(tokens)
        if token.isdigit() or token in _NUMBER_WORDS
    ]
    noun_positions = [index for index, token in enumerate(tokens) if token in forms]
    return any(
        abs(number - noun) <= _COUNT_WINDOW
        for number in number_positions
        for noun in noun_positions
    )


def _numeric_score(frame: QueryFrame, text: str) -> int:
    if frame.intent != "numeric" or frame.numeric_type is None:
        return 0
    pattern = {
        "money": _MONEY,
        "duration": _DURATION,
        "age": _AGE,
    }.get(frame.numeric_type)
    score = 0
    for sentence in _sentences(text):
        sentence_tokens = set(_tokens(sentence))
        if not (frame.anchors & sentence_tokens):
            continue
        if frame.numeric_type == "count":
            if frame.numeric_unit is None or not _count_bound(sentence, frame.numeric_unit):
                continue
            score += 1
        elif pattern.search(sentence.lower()):
            score += 1
    return min(score, 2)


def _unqualified_person_query(query: str) -> bool:
    """Conservative D0 ambiguity screen for bare `who is FIRST LAST` queries."""
    match = re_compile(r"^who is (.+)$").match(query)
    if not match:
        return False
    terms = _tokens(match.group(1))
    role_markers = {"president", "governor", "actor", "author", "representative", "rep"}
    return 1 <= len(terms) <= 4 and not (set(terms) & role_markers)


def _comparison_score(frame: QueryFrame, text: str) -> int:
    if frame.intent != "comparison":
        return 0
    contrast = re_compile(
        r"\b(?:differ(?:s|ent|ence)?|whereas|while|unlike|versus|compared|but)\b|\bis that\b"
    )
    score = 0
    sentences = _sentences(text)
    for index, sentence in enumerate(sentences):
        window = sentence
        if index + 1 < len(sentences):
            window += " " + sentences[index + 1]
        tokens = set(_tokens(window))
        has_left = bool(frame.comparison_left) and frame.comparison_left <= tokens
        has_right = bool(frame.comparison_right) and frame.comparison_right <= tokens
        if has_left and has_right and contrast.search(window.lower()):
            score = max(score, 2 if _FINITE_VERBS.search(window.lower()) else 1)
    return score


def _boilerplate_score(text: str) -> int:
    lower = text.lower()
    score = 2 * sum(cue in lower for cue in _BOILERPLATE_CUES)
    score += min(lower.count("http://") + lower.count("https://") + lower.count("www."), 2)
    score += min(len(_LIST_ITEM.findall(text)), 2)
    sentences = _sentences(text)
    normalised = [" ".join(_tokens(sentence)) for sentence in sentences]
    score += min(len(normalised) - len(set(normalised)), 2)
    if sentences and not any(_FINITE_VERBS.search(sentence.lower()) for sentence in sentences):
        score += 1
    return score


@inject
@dataclass(frozen=True, kw_only=True)
class DefinitionAnswerAxiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    min_anchor_coverage: NoInject[float] = 0.50
    max_anchor_gap: NoInject[float] = 0.25

    def preference(self, input: Query, output1: Document, output2: Document) -> Preference:
        query = self.text_contents.contents(input)
        text1 = self.text_contents.contents(output1)
        text2 = self.text_contents.contents(output2)
        frame = parse_query(query)
        if frame.intent != "definition" or not _comparable(
            frame, text1, text2, self.min_anchor_coverage, self.max_anchor_gap
        ):
            return 0
        return strictly_greater(_definition_score(frame, text1), _definition_score(frame, text2))


@inject
@dataclass(frozen=True, kw_only=True)
class NumericAnswerAxiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    min_anchor_coverage: NoInject[float] = 0.50
    max_anchor_gap: NoInject[float] = 0.25

    def preference(self, input: Query, output1: Document, output2: Document) -> Preference:
        query = self.text_contents.contents(input)
        text1 = self.text_contents.contents(output1)
        text2 = self.text_contents.contents(output2)
        frame = parse_query(query)
        if frame.intent != "numeric" or not _comparable(
            frame, text1, text2, self.min_anchor_coverage, self.max_anchor_gap
        ):
            return 0
        return strictly_greater(_numeric_score(frame, text1), _numeric_score(frame, text2))


@inject
@dataclass(frozen=True, kw_only=True)
class ComparisonAnswerAxiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    min_anchor_coverage: NoInject[float] = 0.50
    max_anchor_gap: NoInject[float] = 0.25

    def preference(self, input: Query, output1: Document, output2: Document) -> Preference:
        query = self.text_contents.contents(input)
        text1 = self.text_contents.contents(output1)
        text2 = self.text_contents.contents(output2)
        frame = parse_query(query)
        if frame.intent != "comparison" or not _comparable(
            frame, text1, text2, self.min_anchor_coverage, self.max_anchor_gap
        ):
            return 0
        return strictly_greater(_comparison_score(frame, text1), _comparison_score(frame, text2))


@inject
@dataclass(frozen=True, kw_only=True)
class ContentBearingPassageAxiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    min_anchor_coverage: NoInject[float] = 0.50
    max_anchor_gap: NoInject[float] = 0.25
    boilerplate_margin: NoInject[int] = 2

    def preference(self, input: Query, output1: Document, output2: Document) -> Preference:
        query = self.text_contents.contents(input)
        text1 = self.text_contents.contents(output1)
        text2 = self.text_contents.contents(output2)
        frame = parse_query(query)
        if not candidate_query_precondition("CBP", query):
            return 0
        if frame.intent == "unknown":
            anchors = _content_terms(query)
            frame = QueryFrame("topical", anchors)
        if not _comparable(
            frame, text1, text2, self.min_anchor_coverage, self.max_anchor_gap
        ):
            return 0
        score1 = _boilerplate_score(text1)
        score2 = _boilerplate_score(text2)
        if max(score1, score2) < self.boilerplate_margin:
            return 0
        # Lower boilerplate is better; require the explicit margin.
        if abs(score1 - score2) < self.boilerplate_margin:
            return 0
        return strictly_greater(score2, score1)


DEFANS = lazy_inject(DefinitionAnswerAxiom)
NUMANS = lazy_inject(NumericAnswerAxiom)
COMPARE = lazy_inject(ComparisonAnswerAxiom)
CBP = lazy_inject(ContentBearingPassageAxiom)
