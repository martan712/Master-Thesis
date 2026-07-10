"""The pairwise ranker interface every backend implements."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PairVerdict:
    """Outcome of one presentation: passage A was shown first, B second."""

    verdict: str  # "a" | "b" | "tie"
    prob_a: float  # P(A preferred) under the two-way choice
    score_a: float  # raw label score for A (e.g. log-likelihood)
    score_b: float


class PairwiseRanker(ABC):
    """A model that answers: which of two passages is more relevant to the query?

    Implementations must be deterministic given (query, passage_a, passage_b) so that
    cached verdicts remain valid, and must not reorder the passages internally —
    presentation order is an experimental variable.
    """

    name: str
    prompt_version: str

    @abstractmethod
    def compare(self, query: str, passage_a: str, passage_b: str) -> PairVerdict: ...
