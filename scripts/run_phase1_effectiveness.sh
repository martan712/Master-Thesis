#!/usr/bin/env bash
# Phase 1 effectiveness gate runbook (docs/phase1-implementation.md §6) as one script.
# The companion agreement (fidelity) runbook is scripts/run_phase1_agreement.sh.
#
# Copeland-aggregate each ranker's cached pairwise verdicts (= PRP-allpair) into a ranking
# and compare it to the BM25 baseline on nDCG@10 / MAP (phase1-design.md §4). Decision
# rule: Qwen must clearly beat BM25 on nDCG@10 on both DL19 and DL20, else stop-and-fix
# before drawing axiom conclusions; flan-t5-large is reported as contrast, not gated.
#
# Usage: scripts/run_phase1_effectiveness.sh [all|qwen|flan]
#
#   all   (default) smoke + both collections, both models
#   qwen  smoke + both collections, Qwen only
#   flan  both collections, flan-t5-large only (no smoke)
#
# Prerequisites:
#   - Verdicts should already be in the preference store from the rq1 runs
#     (scripts/run_phase1_agreement.sh); then this is a zero-cost re-analysis.
#   - Otherwise this script collects them, needing the vLLM endpoint at localhost:9086
#     (Qwen) or the local flan stack (`uv sync --extra llm`).
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-all}"
case "$mode" in all|qwen|flan) ;; *) echo "usage: $0 [all|qwen|flan]" >&2; exit 2 ;; esac

step() { printf '\n=== %s ===\n' "$*"; }
eff() { uv run python experiments/ranking_effectiveness/run.py "$@"; }

if [[ "$mode" != flan ]]; then
    step "smoke test: scifact + mock through the effectiveness runner (end-to-end path)"
    eff --config configs/eff_smoke.yaml
fi

case "$mode" in
    all)  filter=() ;;
    qwen) filter=(--only-model qwen) ;;
    flan) filter=(--only-model flan) ;;
esac

step "effectiveness gate: DL19 top-10 (verdicts shared with rq1, no new model calls)"
eff --config configs/eff_dl19_top10.yaml "${filter[@]}"

step "effectiveness gate: DL20 top-10 (verdicts shared with rq1, no new model calls)"
eff --config configs/eff_dl20_top10.yaml "${filter[@]}"

step "effectiveness gate done — metrics under results/ranking_effectiveness/*/metrics/"
echo "next: record the nDCG@10 gate verdict in docs/phase1-design.md §9 (pass = residual is skill)"
