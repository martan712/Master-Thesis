"""Pairwise verdicts -> a ranking (Copeland = PRP-allpair), and run evaluation."""

from axiomrank.ranking.copeland import copeland_ranking
from axiomrank.ranking.evaluation import (
    DEFAULT_METRICS,
    compare_runs,
    evaluate_run,
    metric_names,
)

__all__ = [
    "DEFAULT_METRICS",
    "compare_runs",
    "copeland_ranking",
    "evaluate_run",
    "metric_names",
]
