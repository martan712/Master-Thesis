#!/usr/bin/env bash
# Phase 1 agreement (fidelity) runbook (docs/phase1-implementation.md §6) as one script.
# The companion effectiveness gate is scripts/run_phase1_effectiveness.sh.
#
# Usage: scripts/run_phase1_agreement.sh [all|qwen|flan]
#
#   all   (default) smoke + gate + every grid cell, both models — Qwen first,
#         then the ~4.6 h flan-t5-large CPU replication
#   qwen  smoke + gate + the DL19 re-analyses + the Qwen collection cells (~1.6 h new)
#   flan  the flan-t5-large replication cells only (background CPU, ~4.6 h)
#
# Everything is resumable: verdicts are cached in the append-only preference store
# (lookup-before-call), derived stages cache as Parquet. The DL19 top-10 cells collect
# nothing new — both models' verdicts are in the store from Phase 0; only the extended
# axiom battery is computed (local CPU).
#
# Prerequisites:
#   - Qwen steps need the vLLM endpoint at localhost:9086 (see the rq*.yaml configs).
#   - flan steps need the local model stack: `uv sync --extra llm` once.
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-all}"
case "$mode" in all|qwen|flan) ;; *) echo "usage: $0 [all|qwen|flan]" >&2; exit 2 ;; esac

step() { printf '\n=== %s ===\n' "$*"; }
rq1() { uv run python experiments/rq1_lexical_agreement/run.py "$@"; }
rq2() { uv run python experiments/rq2_semantic_agreement/run.py "$@"; }

if [[ "$mode" != flan ]]; then
    step "smoke test: scifact + mock through the rq2 runner (relaxed + aliases + WordNet)"
    rq2 --config configs/p1_smoke.yaml

    step "sanity gate: Qwen endpoint (re-run before any new collection, plan §7)"
    uv run python scripts/sanity_gate.py --config configs/pilot_qwen.yaml

    step "DL19 top-10 re-analysis, both models (verdicts cached from Phase 0, no new calls)"
    rq1 --config configs/rq1_dl19_top10.yaml
    rq2 --config configs/rq2_dl19_top10.yaml

    step "Qwen: DL20 top-10 (~31 min) — the replication headline"
    rq1 --config configs/rq1_dl20_top10.yaml --only-model qwen

    step "Qwen: uniform depth-100 cells (~1 h) — the gap gradient"
    rq1 --config configs/rq1_dl19_uniform.yaml --only-model qwen
    rq1 --config configs/rq1_dl20_uniform.yaml --only-model qwen

    step "Qwen: DL20 top-10 semantic battery (verdicts shared with rq1, axioms only)"
    rq2 --config configs/rq2_dl20_top10.yaml --only-model qwen
fi

if [[ "$mode" != qwen ]]; then
    step "flan-t5-large replication: DL20 top-10 (~1.5 h, CPU)"
    rq1 --config configs/rq1_dl20_top10.yaml --only-model flan

    step "flan-t5-large replication: uniform cells (~3 h, CPU)"
    rq1 --config configs/rq1_dl19_uniform.yaml --only-model flan
    rq1 --config configs/rq1_dl20_uniform.yaml --only-model flan

    step "flan-t5-large replication: DL20 top-10 semantic battery"
    rq2 --config configs/rq2_dl20_top10.yaml --only-model flan
fi

step "phase 1 agreement runs done — results under results/rq1_lexical_agreement/ and results/rq2_semantic_agreement/"
echo "next: run the effectiveness gate — scripts/run_phase1_effectiveness.sh (verdicts are"
echo "      already in the store, so it is a zero-cost check that Qwen beats BM25 on nDCG@10)"
