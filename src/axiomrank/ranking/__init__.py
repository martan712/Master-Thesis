"""Pairwise verdicts -> a ranking (Copeland = PRP-allpair), and run evaluation."""

from axiomrank.ranking.copeland import copeland_ranking
from axiomrank.ranking.evaluation import (
    DEFAULT_METRICS,
    compare_runs,
    evaluate_run,
    metric_names,
)
from axiomrank.ranking.surrogate import (
    assign_query_folds,
    fit_oof_surrogate,
    surrogate_fidelity,
)
from axiomrank.ranking.soft_semantics import (
    AxiomEvaluation,
    AxiomThresholds,
    SoftSemanticAxiomaticReranker,
    clean_passage,
    derive_pointwise_score,
    prepare_query,
    rerank_scored_pool,
)

__all__ = [
    "DEFAULT_METRICS",
    "assign_query_folds",
    "compare_runs",
    "copeland_ranking",
    "evaluate_run",
    "fit_oof_surrogate",
    "metric_names",
    "surrogate_fidelity",
    "AxiomEvaluation",
    "AxiomThresholds",
    "SoftSemanticAxiomaticReranker",
    "clean_passage",
    "derive_pointwise_score",
    "prepare_query",
    "rerank_scored_pool",
]
