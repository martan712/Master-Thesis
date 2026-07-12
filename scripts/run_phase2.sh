#!/usr/bin/env bash
# Phase 2 decomposition (RQ3) runbook (docs/phase2-implementation.md §5) as one script.
#
# Usage: scripts/run_phase2.sh [all|qwen|flan]
#
#   all   (default) smoke + the pooled decomposition for both models
#   qwen  smoke + the pooled decomposition for Qwen only (the primary)
#   flan  the flan-t5-large replication only
#
# RQ3 is an analysis phase: it pools the cached DL19+DL20 top-10 verdicts (already in the
# append-only preference store from Phase 1) and decomposes them. It collects ZERO new
# model verdicts — no vLLM endpoint or local model stack is needed, only local CPU. There
# is no effectiveness gate in this phase; fidelity throughout.
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-all}"
case "$mode" in all|qwen|flan) ;; *) echo "usage: $0 [all|qwen|flan]" >&2; exit 2 ;; esac

step() { printf '\n=== %s ===\n' "$*"; }
rq3() { uv run python experiments/rq3_decomposition/run.py "$@"; }

if [[ "$mode" != flan ]]; then
    step "smoke test: scifact + mock through the rq3 runner (frame -> decomposition -> residual)"
    rq3 --config configs/rq3_smoke.yaml

    step "Qwen: pooled DL19+DL20 top-10 decomposition (primary)"
    rq3 --config configs/rq3_pooled_top10.yaml --only-model qwen --uniform
fi

if [[ "$mode" != qwen ]]; then
    step "flan-t5-large: pooled DL19+DL20 top-10 decomposition (replication)"
    rq3 --config configs/rq3_pooled_top10.yaml --only-model flan --uniform
fi

step "phase 2 decomposition done — results under results/rq3_decomposition/pooled_top10/"
echo "next: the analysis pass — fill docs/phase2-design.md §7 with the decomposition numbers,"
echo "      the noise-floor estimate, the residual-model result and CIs, the gap-gradient"
echo "      resolution, the cross-model/collection stability, and the §6 RQ4-emphasis decision."
echo "      notebooks/p2_overview.ipynb is the working overview."
