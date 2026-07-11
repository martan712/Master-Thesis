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
        return HFPairwiseRanker(
            model_name=cfg.model, prompt_version=cfg.prompt_version, max_chars=cfg.max_chars
        )
    if cfg.backend == "openai":
        from axiomrank.rankers.openai_api import OpenAIPairwiseRanker

        if not cfg.model or not cfg.base_url:
            raise ValueError("ranker.model and ranker.base_url must be set for the openai backend")
        return OpenAIPairwiseRanker(
            model_name=cfg.model,
            base_url=cfg.base_url,
            prompt_version=cfg.prompt_version,
            max_chars=cfg.max_chars,
            extra_body=cfg.extra_body,
        )
    raise ValueError(f"Unknown ranker backend: {cfg.backend}")
