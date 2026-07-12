"""Selection-contract tests for the Phase 3 qualitative case resource."""

import importlib.util

from axiomrank.paths import PROJECT_ROOT


def _runner():
    path = PROJECT_ROOT / "experiments" / "rq4_qualitative" / "run.py"
    spec = importlib.util.spec_from_file_location("rq4_qualitative", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_contributory_reversal_requires_every_condition():
    selected = {
        "query_delta": 0.1,
        "bm25_winner": 7,
        "bm25_loser": 1,
        "llm_winner": 0,
        "llm_loser": 5,
        "rel_winner": 3,
        "rel_loser": 0,
    }
    check = _runner()._is_contributory_reversal
    assert check(**selected)

    alternatives = {
        "query_delta": 0.0,
        "bm25_winner": 0,
        "llm_winner": 9,
        "rel_winner": 0,
    }
    for field, invalid in alternatives.items():
        case = {**selected, field: invalid}
        assert not check(**case), field


def test_winner_relative_normalises_pair_orientation():
    normalise = _runner()._winner_relative
    assert normalise(1, True) == 1
    assert normalise(-1, False) == 1
    assert normalise(0, False) == 0
