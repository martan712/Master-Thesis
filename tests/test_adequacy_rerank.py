"""Coverage guards for the deeper answer-adequacy reranking sweep."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest

from axiomrank.config import load_config


RUNNER = Path(__file__).parents[1] / "experiments" / "rq4_candidates" / "adequacy_rerank.py"
SPEC = spec_from_file_location("adequacy_rerank", RUNNER)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _pool():
    return pd.DataFrame(
        {"qid": ["q", "q", "q"], "docno": ["a", "b", "c"], "rank": [0, 1, 2]}
    )


def test_coverage_accepts_exact_scored_block_and_ignores_tail():
    MODULE._assert_coverage(_pool(), {("cell", "q", "a"): 1.0, ("cell", "q", "b"): 2.0}, "cell", 2)


def test_coverage_fails_before_partial_depth_evaluation():
    with pytest.raises(SystemExit, match="cache is incomplete.*q/b"):
        MODULE._assert_coverage(_pool(), {("cell", "q", "a"): 1.0}, "cell", 2)


def test_top100_config_parses_and_points_to_the_fixed_development_sources():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "rq4_adequacy_top100.yaml")
    assert cfg.sources == ["configs/rq2_dl19_top10.yaml", "configs/rq2_dl20_top10.yaml"]
