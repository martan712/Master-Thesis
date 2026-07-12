"""Additively compute the RQ4 axiom-pref columns (VERB, QCOV, VERB@m0.2) into cache.

Phase 3 appends three new axioms to the two rq2 top-10 cells' lexical battery. Rather than
a full `build_axiom_prefs --refresh` (which would recompute *every* column and could perturb
the Phase 1/2 baseline — e.g. the PROX determinism/hash-seed history, MEMORY.md), this
computes ONLY the columns absent from the cached `axiom_prefs.parquet` and merges them onto
the existing frame, leaving every existing axiom column bit-for-bit intact (append-only).

Text-only axioms (VERB/QCOV): no index statistics, no model calls, no downloads — pure CPU
over the cached pool/pairs. Idempotent: re-running is a no-op once the columns exist.

Usage:
    uv run --no-sync python scripts/add_rq4_axiom_columns.py
"""

import pandas as pd

from axiomrank.axioms import axiom_preferences
from axiomrank.config import load_config
from axiomrank.pipeline import stages

KEY = ["qid", "doc_id_1", "doc_id_2"]
CONFIGS = ["configs/rq2_dl19_top10.yaml", "configs/rq2_dl20_top10.yaml"]


def main() -> None:
    for config in CONFIGS:
        cfg = load_config(config)
        path = stages.processed_dir(cfg) / "axiom_prefs.parquet"
        cached = pd.read_parquet(path)

        wanted = [s.column for s in cfg.axioms.specs]
        missing = [s for s in cfg.axioms.specs if s.column not in cached.columns]
        if not missing:
            print(f"{config}: nothing to add ({len(cached.columns)} cols already present)")
            continue

        print(f"{config}: adding {[s.column for s in missing]} to {path}")
        pool = stages.build_pool(cfg, refresh=False)
        pairs = stages.build_pairs(cfg, pool, refresh=False)
        # Text-only axioms -> index_location=None (no collection statistics needed).
        new = axiom_preferences(pool, pairs, missing, index_location=None)

        merged = cached.merge(new, on=KEY, how="left", validate="one_to_one")
        assert len(merged) == len(cached), "row count changed on merge"
        for s in missing:
            assert merged[s.column].notna().all(), f"missing values in {s.column}"
        # Preserve column order: existing columns first, then the new ones, matching config.
        ordered = list(cached.columns) + [s.column for s in missing]
        merged = merged[ordered]
        assert set(wanted) <= set(merged.columns)

        tmp = path.with_suffix(".parquet.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(path)
        print(f"  -> wrote {merged.shape[1]} columns ({merged.shape[0]} pairs)")


if __name__ == "__main__":
    main()
