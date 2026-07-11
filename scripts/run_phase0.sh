#!/usr/bin/env bash
# Phase 0 pilot runbook (docs/phase0-plan.md) as one script.
#
# Everything is resumable: pool/pairs/axiom stages cache as Parquet, model verdicts
# live in the append-only preference store (lookup-before-call), so re-running after
# an interruption — or after Phase 0 completion — only fills in what is missing.
#
# Prerequisites:
#   - flan-t5-large runs locally on CPU: `uv sync --extra llm` once.
#   - The Qwen steps need the vLLM endpoint at localhost:9086 (configs/pilot_qwen.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n=== %s ===\n' "$*"; }

step "smoke test: scifact + mock ranker (minutes, no large downloads)"
uv run python experiments/p0_pilot/run.py --config configs/smoke.yaml

step "sanity gate: flan-t5-large (4-way order swap, phase0-plan §8.2)"
uv run python scripts/sanity_gate.py --config configs/pilot.yaml

step "sanity gate: Qwen endpoint"
uv run python scripts/sanity_gate.py --config configs/pilot_qwen.yaml

step "pilot: Qwen, DL19 top-10 all-pairs (primary ranker, ~25 min uncached)"
uv run python experiments/p0_pilot/run.py --config configs/pilot_qwen.yaml

step "pilot: flan-t5-large, DL19 top-10 all-pairs (contrast ranker, ~1.5 h uncached, CPU)"
uv run python experiments/p0_pilot/run.py --config configs/pilot.yaml

step "phase 0 done — results under results/p0_pilot/"
