"""Cached experiment stages: pool, pair sample, axiom preferences.

Stage outputs cache as Parquet under data/processed/<experiment>[/<variant>]/;
delete the directory (or pass refresh) to recompute.
"""

import os
from pathlib import Path

import pandas as pd

from axiomrank import paths
from axiomrank.axioms import axiom_preferences
from axiomrank.config import ExperimentConfig
from axiomrank.data.pairs import sample_pairs
from axiomrank.data.retrieval import bm25_pool, index_ref


def processed_dir(cfg: ExperimentConfig) -> Path:
    base = paths.PROCESSED_DIR / cfg.experiment
    return base / cfg.variant if cfg.variant else base


def output_dir(cfg: ExperimentConfig) -> Path:
    base = paths.results_dir(cfg.experiment)
    out = base / cfg.variant if cfg.variant else base
    out.mkdir(parents=True, exist_ok=True)
    return out


def cached_frame(path: Path, refresh: bool, compute) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a parallel run of the same stage (the qwen and flan
    # runbooks share the model-independent stages) can never leave a torn file.
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return df


def build_pool(cfg: ExperimentConfig, refresh: bool = False) -> pd.DataFrame:
    return cached_frame(
        processed_dir(cfg) / "pool.parquet",
        refresh,
        lambda: bm25_pool(cfg.dataset, cfg.first_stage),
    )


def build_pairs(cfg: ExperimentConfig, pool: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    return cached_frame(
        processed_dir(cfg) / "pairs.parquet",
        refresh,
        lambda: sample_pairs(pool, cfg.pairs, cfg.seed),
    )


def build_axiom_prefs(
    cfg: ExperimentConfig, pool: pd.DataFrame, pairs: pd.DataFrame, refresh: bool = False
) -> pd.DataFrame:
    """Cached axiom preferences, recomputed when the configured battery grew."""
    path = processed_dir(cfg) / "axiom_prefs.parquet"
    names = [s.column for s in cfg.axioms.specs]
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if set(names) <= set(cached.columns):
            return cached
        print("      battery changed; recomputing axiom preferences")
    return cached_frame(
        path,
        True,
        lambda: axiom_preferences(
            pool, pairs, cfg.axioms.specs, index_location=index_ref(cfg.dataset)
        ),
    )
