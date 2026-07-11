import math

import pytest

from axiomrank.rankers.openai_api import MISSING, verdict_from_top_logprobs


def test_prefers_higher_label_logprob():
    v = verdict_from_top_logprobs([("A", -0.1), ("B", -2.5), ("The", -3.0)])
    assert v.verdict == "a"
    assert v.prob_a == pytest.approx(math.exp(-0.1) / (math.exp(-0.1) + math.exp(-2.5)))
    assert (v.score_a, v.score_b) == (-0.1, -2.5)


def test_whitespace_and_case_variants_fold_to_best():
    # tokenisers emit " B", "b", etc.; the best-scoring variant of each label counts
    v = verdict_from_top_logprobs([(" b", -0.2), ("B", -4.0), ("A", -1.0)])
    assert v.verdict == "b"
    assert v.score_b == -0.2


def test_missing_label_gets_zero_probability():
    v = verdict_from_top_logprobs([("A", -0.5), ("C", -1.0)])
    assert v.verdict == "a"
    assert v.prob_a == 1.0
    assert v.score_b == MISSING


def test_no_labels_is_a_tie():
    # e.g. a thinking model whose first token is a reasoning delimiter
    v = verdict_from_top_logprobs([("<think>", -0.01), ("Sure", -5.0)])
    assert v.verdict == "tie"
    assert v.prob_a == 0.5
