"""Rebuild one grid cell's analysis frame from cache, for the RQ3 decomposition.

`merged_cell_frame` reproduces exactly the merged pair frame the Phase 1 `measure_cell`
builds internally — collapsed verdicts joined to axiom preferences with the BM25 rank gap —
and adds the non-axiom covariates (phase2-implementation.md §3.1). It reads only cached
stages and the preference store: no model calls, no downloads. Factored here so the rq3
runner (and later phases) reuse it without touching `measure_cell` (whose Phase 1 outputs
must stay bit-for-bit reproducible).
"""

import pandas as pd

from axiomrank import analysis
from axiomrank.config import ExperimentConfig, RankerConfig
from axiomrank.data.preferences import PreferenceStore
from axiomrank.pipeline import collect, stages


def merged_cell_frame(
    cfg: ExperimentConfig, ranker_cfg: RankerConfig, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(merged frame with axiom columns + covariates, per-presentation store rows).

    The returned frame is one row per canonical pair: model verdict, one column per axiom
    spec, the BM25 rank gap, and the `analysis.COVARIATE_COLUMNS`.
    """
    pool = stages.build_pool(cfg, refresh)
    pairs = stages.build_pairs(cfg, pool, refresh)
    axiom_df = stages.build_axiom_prefs(cfg, pool, pairs, refresh)

    store_df = collect.collect_verdicts(
        cfg.dataset.irds_id,
        ranker_cfg,
        pairs,
        PreferenceStore(),
        allow_new=False,
    )
    verdicts = analysis.model_pair_verdicts(store_df)
    merged = analysis.merge_pairs(axiom_df, verdicts)
    merged = analysis.attach_rank_gap(merged, pool)
    merged = analysis.attach_covariates(merged, pool, pairs, store_df)
    return merged, store_df
