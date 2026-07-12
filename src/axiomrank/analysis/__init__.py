"""Analyses of model verdicts against axiom preferences.

- :mod:`verdicts` — collapse per-presentation verdicts, position consistency
- :mod:`agreement` — per-axiom coverage/agreement, query-bootstrap CIs
- :mod:`transitivity` — cyclic-triangle rate
- :mod:`joint` — all-axiom fits (majority vote, grouped-CV logistic)
- :mod:`gap` — agreement binned by BM25 rank gap
- :mod:`figures` — figure drafts

Definitions are fixed in docs/phase0-design.md §3.5 and phase1-design.md §4.3.
"""

from axiomrank.analysis.agreement import (
    agreement_table,
    agreement_with_ci,
    attach_rank_gap,
    merge_pairs,
)
from axiomrank.analysis.covariates import (
    CONTENT_COVARIATES,
    COVARIATE_COLUMNS,
    MAGNITUDE_COVARIATES,
    SIGNED_COVARIATES,
    attach_covariates,
)
from axiomrank.analysis.decomposition import (
    decompose,
    information_decomposition,
    reliability_ceiling,
)
from axiomrank.analysis.figures import gap_figure
from axiomrank.analysis.gap import gap_gradient
from axiomrank.analysis.joint import STRICT_CORE, joint_fit
from axiomrank.analysis.residual import (
    residual_clusters,
    residual_model,
    residual_profiles,
)
from axiomrank.analysis.transitivity import nontransitivity_rate
from axiomrank.analysis.verdicts import PAIR_KEY, consistency_stats, model_pair_verdicts

__all__ = [
    "CONTENT_COVARIATES",
    "COVARIATE_COLUMNS",
    "MAGNITUDE_COVARIATES",
    "PAIR_KEY",
    "SIGNED_COVARIATES",
    "STRICT_CORE",
    "agreement_table",
    "agreement_with_ci",
    "attach_covariates",
    "attach_rank_gap",
    "consistency_stats",
    "decompose",
    "gap_figure",
    "gap_gradient",
    "information_decomposition",
    "joint_fit",
    "merge_pairs",
    "model_pair_verdicts",
    "nontransitivity_rate",
    "reliability_ceiling",
    "residual_clusters",
    "residual_model",
    "residual_profiles",
]
