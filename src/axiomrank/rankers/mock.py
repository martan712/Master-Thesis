"""Deterministic mock ranker for pipeline tests.

Scores each (query, passage) with a hash-derived pseudo-random pointwise score, so its
pairwise verdicts are transitive and position-consistent by construction. Useful to
validate the pipeline (a correct agreement analysis must find ~0 inconsistency here) and
cheap enough for CI.
"""

import hashlib

from axiomrank.rankers.base import PairVerdict, PairwiseRanker


def _score(query: str, passage: str) -> float:
    digest = hashlib.sha256(f"{query}\x00{passage}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class MockRanker(PairwiseRanker):
    name = "mock"
    prompt_version = "v0"

    def compare(self, query: str, passage_a: str, passage_b: str) -> PairVerdict:
        score_a = _score(query, passage_a)
        score_b = _score(query, passage_b)
        if score_a == score_b:
            return PairVerdict("tie", 0.5, score_a, score_b)
        prob_a = score_a / (score_a + score_b)
        return PairVerdict("a" if score_a > score_b else "b", prob_a, score_a, score_b)
