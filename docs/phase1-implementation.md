# Phase 1 Implementation — Measurement, RQ1–RQ2 (engineering plan)

> Companion document: the scientific design — framing, grid rationale, battery tiers,
> relaxed-precondition rationale, analyses and replication targets — lives in
> `phase1-design.md`. This document holds the engineering plan.

The engineering plan for Phase 1: what it costs to collect, the code that changes, how the
relaxation levers are wired into ir_axioms, the test discipline, the runbook, and the
operational risks.

## 1. Scope

This covers only the software and operations of RQ1–RQ2. The *why* of each design choice
(collections, conditions, battery tiers, margins, the fastText gate) is in
`phase1-design.md`; here we assume those decisions and implement them.

## 2. Cost and latency

Measured latencies from Phase 0: 386 ms / 1,138 ms per presentation (Qwen / flan-t5-large).
The DL19 top-10 verdicts already exist in the store; only the following cells are new:

| new cell | presentations | Qwen | flan-t5-large |
|---|---|---|---|
| DL20 top10 | 54×45×2 = 4,860 | ~31 min | ~1.5 h |
| DL19 uniform50 | 43×50×2 = 4,300 | ~28 min | ~1.4 h |
| DL20 uniform50 | 54×50×2 = 5,400 | ~35 min | ~1.7 h |
| total new | 14,560/model | ~1.6 h | ~4.6 h |

All runs are resumable (lookup-before-call); order swap stays mandatory everywhere. k=20
pools remain a fallback if the gradient analysis wants more mid-gap pairs, not a default.

## 3. Architecture

New and changed code, following the Phase 0 layout (shared logic in `src/axiomrank/`,
experiment scripts as recipes):

```
config.py        + axiom specs with params ({name, alias, params}), variant field,
                   rankers list (one config drives both models); Phase 0 configs stay valid
axioms.py        + spec resolution: margin params, LEN precondition margins,
                   similarity backend binding, alias-aware battery construction
relaxed.py       NEW: RelaxedTfLnc, RelaxedMTdc (ir_axioms subclasses)
analysis.py      NEW: bootstrap CIs, joint fit (majority vote + grouped-CV logistic),
                   gap-gradient binning, out-of-fold predictions
pipeline.py      NEW: cached stages + verdict collection, factored out of
                   experiments/p0_pilot/run.py (which becomes a thin recipe over it)
experiments/rq1_lexical_agreement/run.py   grid cell -> profiles + joint fit + gap CSVs
experiments/rq2_semantic_agreement/run.py  rq1 + semantic battery + lexical-vs-combined delta
```

### 3.1 Config naming scheme

One config per grid cell (`rq1_dl19_top10.yaml`, `rq1_dl19_uniform.yaml`,
`rq1_dl20_top10.yaml`, `rq1_dl20_uniform.yaml`, `rq2_dl19_top10.yaml`, …), each listing
both rankers; outputs land under `results/<experiment>/<variant>/metrics/<model>/` with
the exact config alongside, and intermediates cache under
`data/processed/<experiment>/<variant>/`.

### 3.2 Output inventory per grid cell × model

The runners write:
- `agreement.csv` — axiom, coverage, n_evaluable, agreement, ci_lo, ci_hi
- `consistency.json` — position consistency, decisiveness, transitivity, latency
- `joint_fit.json` — base rate, majority-vote and CV-logistic accuracy/AUC, coefficients,
  per feature set (strict core / full battery / +semantic)
- `gap_agreement.csv` — per-gap-bin agreement, joint accuracy, consistency
- `gap_agreement.png` — the signature-figure draft

## 4. Relaxation levers: ir_axioms mechanics

The scientific rationale for each lever is in `phase1-design.md` §4.2; the mechanics
(verified against ir_axioms 1.1.2) are:

| axiom | strict precondition | relaxation lever |
|---|---|---|
| TFC1 | doc lengths ≈ equal (`LEN`, rel. margin 0.1) | `precondition=LEN(margin_fraction=m)` |
| TFC3 | same `LEN` precondition | same lever |
| LNC1 | equal TF per query term (rel. margin 0.1) | `margin_fraction=m` on the axiom |
| TF-LNC | non-query length *exactly* equal (hardcoded `==`) | custom `RelaxedTfLnc` subclass: `isclose(…, rel_tol=m)` |
| M-TDC | exactly one query term differs in TF (hardcoded) | custom `RelaxedMTdc` subclass: drop the single-difference gate, keep the per-term-pair validity logic |

Aliased columns, e.g. `TFC1@len0.2`, `M-TDC@r0.1`, so strict and relaxed coexist in one
agreement table. Custom subclasses live in `src/axiomrank/relaxed.py`.

For the semantic axioms, similarity is a config field on the axiom spec (`similarity:
wordnet | fasttext`), bound into ir_axioms' injector at battery-build time; axiom columns
are aliased (`STMC1@wn`, `STMC1@ft`) so both similarity tiers coexist in one table.

## 5. Test discipline

Mirrors the Phase 0 test discipline:
- Custom relaxed subclasses get synthetic sanity tests: a constructed pair where the
  strict variant is neutral and the relaxed one fires, and one where both agree.
- `analysis.py` gets a hand-computable test case (bootstrap, joint fit, gap bins).
- `config.py` extensions get spec-parsing tests (axiom specs with params, multi-ranker
  configs).
- `pipeline.py` is factored out of `experiments/p0_pilot/run.py` with **no behaviour
  change** — the Phase 0 outputs must be reproducible bit-for-bit from the cached store.
- A scifact+mock smoke config runs through the rq1 runner end-to-end.

## 6. Work breakdown and runbook

1. `config.py` extensions + parameterised `axioms.py` + `relaxed.py`, with tests (spec
   parsing; synthetic strict-vs-relaxed sanity pairs).
2. `analysis.py` (bootstrap, joint fit, gap bins) with a hand-computable test case.
3. `pipeline.py` factored out; `p0_pilot/run.py` reduced to a recipe (no behaviour change
   — its outputs must be reproducible bit-for-bit from the cached store).
4. rq1/rq2 runners + the six grid configs; scifact+mock smoke config through the rq1
   runner end-to-end.
5. Collection runs, in this order (each resumable, order swap on):
   1. Qwen DL20 top10 (~31 min) — the replication headline.
   2. Qwen DL19 uniform50 + DL20 uniform50 (~1 h) — the gradient.
   3. flan-t5-large, same three cells, background CPU (~4.6 h).
6. Analysis pass over all cells; `phase1-design.md` §8 gets the numbers and the four
   decisions (battery+margins for RQ3, similarity backend, poolability of collections,
   fastText go/no-go).

Commands (from the repo root):

```
uv run python experiments/rq1_lexical_agreement/run.py --config configs/rq1_dl19_top10.yaml
uv run python experiments/rq1_lexical_agreement/run.py --config configs/rq1_dl20_top10.yaml
uv run python experiments/rq1_lexical_agreement/run.py --config configs/rq1_dl19_uniform.yaml
uv run python experiments/rq1_lexical_agreement/run.py --config configs/rq1_dl20_uniform.yaml
uv run python experiments/rq2_semantic_agreement/run.py --config configs/rq2_dl19_top10.yaml
uv run python experiments/rq2_semantic_agreement/run.py --config configs/rq2_dl20_top10.yaml
```

## 7. Operational risks

- **The Qwen endpoint is external state** — it may be down or serve a different model id
  later. Mitigation: verdicts are cached the moment they are collected; runs are
  resumable; the sanity gate re-runs before any new collection (same script,
  `scripts/sanity_gate.py`).
- **flan-t5-large CPU runs are ~5 h total.** Background, resumable, and strictly second
  priority: every analysis lands on Qwen first; flan is replication.
- **Uniform-pool pairs need document text for pairs deep in the pool** — same pooling path
  as Phase 0, no new download, but per-pair text truncation (`max_chars: 2000`) now
  matters more because deep-pool passages vary more in length. Unchanged from Phase 0;
  recorded here as a known constant of the setup.
- **Resumability.** All collection runs are resumable (lookup-before-call), so any of the
  cells above can be interrupted and restarted without recomputing verdicts.
