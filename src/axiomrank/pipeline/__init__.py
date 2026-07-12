"""Experiment orchestration: cached stages, verdict collection, measurement recipes.

Factored out of experiments/p0_pilot/run.py so every phase's runner is a thin recipe.
"""

from axiomrank.pipeline.collect import collect_verdicts
from axiomrank.pipeline.frames import merged_cell_frame
from axiomrank.pipeline.measurement import measure_cell
from axiomrank.pipeline.stages import (
    build_axiom_prefs,
    build_pairs,
    build_pool,
    cached_frame,
    output_dir,
    processed_dir,
)

__all__ = [
    "build_axiom_prefs",
    "build_pairs",
    "build_pool",
    "cached_frame",
    "collect_verdicts",
    "measure_cell",
    "merged_cell_frame",
    "output_dir",
    "processed_dir",
]
