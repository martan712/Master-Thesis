"""RQ2: do semantic axioms add agreement/predictive power? (phase1-design.md §5)

Same measurement as rq1 but with the semantic battery (STMC1/2, REG/ANTI-REG at the
configured similarity backend) alongside the lexical one; the joint fit compares the
lexical-only feature set against lexical+semantic, and the deltas decide the fastText
gate (§4.1).

Usage:
    uv run python experiments/rq2_semantic_agreement/run.py --config configs/rq2_dl19_top10.yaml
"""

import argparse

from axiomrank.config import load_config
from axiomrank.pipeline import measure_cell

# The rq2 source configs are append-only artefact producers and now also carry Phase-3
# residual-seed columns. They must never enter the historical RQ2 estimand or outputs.
POST_PHASE2_COLUMNS = {"VERB", "QCOV", "VERB@m0.2"}


def _analysis_feature_sets(cfg):
    lexical = [
        s.column for s in cfg.axioms.lexical_specs if s.column not in POST_PHASE2_COLUMNS
    ]
    semantic = [s.column for s in cfg.axioms.semantic_specs]
    return {"lexical": lexical, "combined": lexical + semantic}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--refresh", action="store_true", help="recompute cached stages")
    parser.add_argument("--only-model", help="substring filter on the configured rankers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.axioms.semantic:
        raise SystemExit("rq2 config must define semantic axioms (axioms.semantic)")
    feature_sets = _analysis_feature_sets(cfg)
    lexical = feature_sets["lexical"]
    semantic = [c for c in feature_sets["combined"] if c not in lexical]
    measure_cell(
        cfg,
        feature_sets=feature_sets,
        gap_feature_set="combined",
        refresh=args.refresh,
        only_model=args.only_model,
        analysis_columns=lexical + semantic,
    )


if __name__ == "__main__":
    main()
