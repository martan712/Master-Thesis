"""Small invariants for the Phase 3 candidate evaluator."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest


RUNNER = Path(__file__).parents[1] / "experiments" / "rq4_candidates" / "run.py"
SPEC = spec_from_file_location("rq4_candidates_run", RUNNER)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _base(keys):
    return pd.DataFrame(
        {
            "query_id": [k[0] for k in keys],
            "doc_id_1": [k[1] for k in keys],
            "doc_id_2": [k[2] for k in keys],
            "model_pref": [1] * len(keys),
        }
    )


def _candidate(keys):
    return pd.DataFrame(
        {
            "qid": [k[0] for k in keys],
            "query": ["q"] * len(keys),
            "doc_id_1": [k[1] for k in keys],
            "doc_id_2": [k[2] for k in keys],
            "CBP": [1] * len(keys),
        }
    )


def test_join_candidate_matches_exact_pair_keys():
    keys = [("q1", "a", "b"), ("q1", "a", "c")]
    merged = MODULE._join_candidate(_base(keys), _candidate(keys), "dl19")
    assert len(merged) == 2
    assert "_merge" not in merged.columns


def test_join_candidate_rejects_extra_candidate_keys():
    base_keys = [("q1", "a", "b")]
    candidate_keys = [("q1", "a", "b"), ("q1", "a", "c")]
    with pytest.raises(ValueError, match="do not match exactly"):
        MODULE._join_candidate(_base(base_keys), _candidate(candidate_keys), "dl19")


def test_join_candidate_rejects_missing_candidate_keys():
    base_keys = [("q1", "a", "b"), ("q1", "a", "c")]
    candidate_keys = [("q1", "a", "b")]
    with pytest.raises(ValueError, match="do not match exactly"):
        MODULE._join_candidate(_base(base_keys), _candidate(candidate_keys), "dl19")


def test_feature_sets_are_nested_and_ablatable():
    classical = ["A", "B"]
    candidates = ["DEFANS_d0v2", "NUMANS_d0v2"]
    variants = MODULE._feature_sets(classical, candidates)
    assert variants == {
        "classical": ["A", "B"],
        "plus_defans_d0v2": ["A", "B", "DEFANS_d0v2"],
        "plus_numans_d0v2": ["A", "B", "NUMANS_d0v2"],
        "plus_all_d0": ["A", "B", "DEFANS_d0v2", "NUMANS_d0v2"],
    }
    assert len({feature for feature in variants["classical"]}) == 2
