"""RQ2: do semantic axioms add agreement/predictive power? (phase1-plan.md §4)

Same measurement as rq1 but with the semantic battery (STMC1/2, REG/ANTI-REG at the
configured similarity backend) alongside the lexical one; the joint fit compares the
lexical-only feature set against lexical+semantic, and the deltas decide the fastText
gate (§4.1).

Usage:
    uv run python experiments/rq2_semantic_agreement/run.py --config configs/rq2_dl19_top10.yaml
"""

import argparse

from axiomrank.config import load_config
from axiomrank.measurement import measure_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--refresh", action="store_true", help="recompute cached stages")
    parser.add_argument("--only-model", help="substring filter on the configured rankers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.axioms.semantic:
        raise SystemExit("rq2 config must define semantic axioms (axioms.semantic)")
    feature_sets = {
        "lexical": [s.column for s in cfg.axioms.lexical_specs],
        "combined": [s.column for s in cfg.axioms.specs],
    }
    measure_cell(
        cfg,
        feature_sets=feature_sets,
        gap_feature_set="combined",
        refresh=args.refresh,
        only_model=args.only_model,
    )


if __name__ == "__main__":
    main()
