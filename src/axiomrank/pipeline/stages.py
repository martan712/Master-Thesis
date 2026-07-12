"""Cached experiment stages: pool, pair sample, axiom preferences.

Stage outputs cache as Parquet under data/processed/<experiment>[/<variant>]/;
delete the directory (or pass refresh) to recompute.
"""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from axiomrank import paths
from axiomrank.axioms import axiom_preferences
from axiomrank.config import ExperimentConfig
from axiomrank.data.pairs import sample_pairs
from axiomrank.data.retrieval import bm25_pool, index_ref

# Bump when a stage's implementation changes in a result-affecting way. The per-stage
# config fingerprints prevent a cache produced under one design from being silently read
# after the YAML changes while config.yaml claims the new design was executed.
CACHE_SCHEMA_VERSION = 2


def processed_dir(cfg: ExperimentConfig) -> Path:
    base = paths.PROCESSED_DIR / cfg.experiment
    return base / cfg.variant if cfg.variant else base


def output_dir(cfg: ExperimentConfig) -> Path:
    base = paths.results_dir(cfg.experiment)
    out = base / cfg.variant if cfg.variant else base
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fingerprint(stage: str, payload: dict) -> str:
    serialised = json.dumps(
        {"cache_schema": CACHE_SCHEMA_VERSION, "stage": stage, **payload},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialised.encode()).hexdigest()


def cached_frame(
    path: Path,
    refresh: bool,
    compute,
    fingerprint: str | None = None,
    legacy_validator=None,
) -> pd.DataFrame:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    cache_matches = False
    if path.exists() and not refresh:
        if fingerprint is None:
            cache_matches = True
        elif meta_path.exists():
            try:
                cache_matches = json.loads(meta_path.read_text()).get("fingerprint") == fingerprint
            except (json.JSONDecodeError, OSError):
                cache_matches = False
        if cache_matches:
            return pd.read_parquet(path)
        if fingerprint is not None and not meta_path.exists() and legacy_validator is not None:
            legacy = pd.read_parquet(path)
            if legacy_validator(legacy):
                # One-time migration for caches predating manifests. The stage-specific
                # validator checks every property recoverable from the frame. The marker
                # remains explicit because axiom parameter provenance cannot be recovered
                # perfectly from a legacy Parquet file.
                meta_path.write_text(
                    json.dumps(
                        {"fingerprint": fingerprint, "legacy_migrated": True}, indent=2
                    )
                    + "\n"
                )
                print(f"      validated and registered legacy cache {path.name}")
                return legacy
        print(f"      cache provenance missing/changed; recomputing {path.name}")
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a parallel run of the same stage (the qwen and flan
    # runbooks share the model-independent stages) can never leave a torn file.
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    if fingerprint is not None:
        meta_tmp = meta_path.with_name(f"{meta_path.name}.tmp-{os.getpid()}")
        meta_tmp.write_text(json.dumps({"fingerprint": fingerprint}, indent=2) + "\n")
        os.replace(meta_tmp, meta_path)
    return df


def build_pool(cfg: ExperimentConfig, refresh: bool = False) -> pd.DataFrame:
    fingerprint = _fingerprint(
        "pool",
        {
            "dataset": dataclasses.asdict(cfg.dataset),
            "first_stage": dataclasses.asdict(cfg.first_stage),
        },
    )
    return cached_frame(
        processed_dir(cfg) / "pool.parquet",
        refresh,
        lambda: bm25_pool(cfg.dataset, cfg.first_stage),
        fingerprint,
        lambda df: (
            {"qid", "query", "docno", "rank", "score", "text"} <= set(df.columns)
            and not df.duplicated(["qid", "docno"]).any()
            and (df["rank"] >= 0).all()
            and df.groupby("qid").size().le(cfg.first_stage.pool_depth).all()
        ),
    )


def build_pairs(cfg: ExperimentConfig, pool: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    fingerprint = _fingerprint(
        "pairs",
        {
            "dataset": dataclasses.asdict(cfg.dataset),
            "first_stage": dataclasses.asdict(cfg.first_stage),
            "pairs": dataclasses.asdict(cfg.pairs),
            "seed": cfg.seed,
        },
    )
    def expected():
        return sample_pairs(pool, cfg.pairs, cfg.seed)

    def validate_legacy(df: pd.DataFrame) -> bool:
        try:
            pd.testing.assert_frame_equal(
                df.reset_index(drop=True), expected().reset_index(drop=True),
                check_dtype=False,
            )
            return True
        except AssertionError:
            return False

    return cached_frame(
        processed_dir(cfg) / "pairs.parquet",
        refresh,
        expected,
        fingerprint,
        validate_legacy,
    )


def build_axiom_prefs(
    cfg: ExperimentConfig, pool: pd.DataFrame, pairs: pd.DataFrame, refresh: bool = False
) -> pd.DataFrame:
    """Cached axiom preferences, keyed by the exact sampling design and axiom specs."""
    path = processed_dir(cfg) / "axiom_prefs.parquet"
    fingerprint = _fingerprint(
        "axiom_prefs",
        {
            "dataset": dataclasses.asdict(cfg.dataset),
            "first_stage": dataclasses.asdict(cfg.first_stage),
            "pairs": dataclasses.asdict(cfg.pairs),
            "seed": cfg.seed,
            "axioms": dataclasses.asdict(cfg.axioms),
        },
    )
    names = [s.column for s in cfg.axioms.specs]
    expected_keys = pairs[["qid", "doc_id_1", "doc_id_2"]].reset_index(drop=True)

    def validate_legacy(df: pd.DataFrame) -> bool:
        if not {"qid", "doc_id_1", "doc_id_2", *names} <= set(df.columns):
            return False
        try:
            pd.testing.assert_frame_equal(
                df[["qid", "doc_id_1", "doc_id_2"]].reset_index(drop=True),
                expected_keys,
                check_dtype=False,
            )
            return True
        except AssertionError:
            return False

    return cached_frame(
        path,
        refresh,
        lambda: axiom_preferences(
            pool, pairs, cfg.axioms.specs, index_location=index_ref(cfg.dataset)
        ),
        fingerprint,
        # Legacy PROX1 values used the asymmetric upstream distance and must be rebuilt.
        None if "PROX1" in names else validate_legacy,
    )
