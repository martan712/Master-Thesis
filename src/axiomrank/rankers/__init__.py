from axiomrank.rankers.base import PairVerdict, PairwiseRanker
from axiomrank.rankers.mock import MockRanker

__all__ = ["PairVerdict", "PairwiseRanker", "MockRanker", "make_ranker"]


def make_ranker(cfg) -> PairwiseRanker:
    """Instantiate the ranker backend named in a RankerConfig."""
    if cfg.backend == "mock":
        return MockRanker()
    if cfg.backend == "hf":
        from axiomrank.rankers.hf import HFPairwiseRanker

        if not cfg.model:
            raise ValueError("ranker.model must be set for the hf backend")
        return HFPairwiseRanker(model_name=cfg.model, max_chars=cfg.max_chars)
    raise ValueError(f"Unknown ranker backend: {cfg.backend}")
