"""RQ1: lexical axiom agreement profiles (phase1-plan.md §3).

Per grid cell (one config = collection × sampling condition, both rankers):
agreement profile with bootstrap CIs, joint fits on the strict Phase 0 core vs. the
full extended battery (additions + relaxed preconditions), and the gap gradient.

Usage:
    uv run python experiments/rq1_lexical_agreement/run.py --config configs/rq1_dl19_top10.yaml
    uv run python experiments/rq1_lexical_agreement/run.py --config configs/rq1_dl20_top10.yaml --only-model qwen

Verdicts are cached in the preference store (never recomputed); --refresh recomputes
the derived stages (pool, pairs, axiom preferences). --only-model substring-filters the
configured rankers, e.g. to run the slow CPU contrast model separately.
"""

import argparse

from axiomrank.analysis import STRICT_CORE
from axiomrank.config import load_config
from axiomrank.measurement import measure_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--refresh", action="store_true", help="recompute cached stages")
    parser.add_argument("--only-model", help="substring filter on the configured rankers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    columns = [s.column for s in cfg.axioms.specs]
    feature_sets = {
        "strict_core": [c for c in STRICT_CORE if c in columns],
        "full_battery": columns,
    }
    measure_cell(
        cfg,
        feature_sets=feature_sets,
        gap_feature_set="full_battery",
        refresh=args.refresh,
        only_model=args.only_model,
    )


if __name__ == "__main__":
    main()
