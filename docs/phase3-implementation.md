# Phase 3 Implementation — RQ4 Increment 1: lexical-adjacent residual axioms (engineering notes)

> **Increment 1 of RQ4.** This document records what was built for the *first* RQ4 probe — the
> two lexical-adjacent residual axioms VERB and QCOV, their capture test, and the reranking /
> fitted-surrogate effectiveness study (§7) — and how: the append-only handling of the axiom
> cache, the runners, and the runbook. Numerical results live in the runners' JSON outputs under
> `results/` (cited inline below); the thesis write-up is deferred until the phase is complete.
> The headline: **QCOV validated in capture, VERB a reported null; untuned axioms ≈ BM25 and even
> a fitted lexical surrogate stays well short of the LLM.** The **main RQ4 line** the probe
> motivated — the two-tier LLM-aligned *semantic* preference-axiom investigation — is designed in
> `phase3-design.md`; its implementation notes are not yet written because that work is not yet
> run.

## 1. Scope and cost

RQ4 formalises the two residual seeds Phase 2 named — a verbosity/length cluster and a
query-coverage cluster (`phase2-writeup.md` §3.4–3.5) — as new retrieval axioms, and tests
whether they capture the residual (decomposition) and help reranking. Like RQ3 it is an
**analysis phase**: **zero model/LLM calls, zero downloads**. Everything computes from the
cached rq2 top-10 stages (pool, pairs, axiom preferences) and the append-only preference
store. New axiom columns are pure CPU over cached text (~5 s per cell).

## 2. The axioms (`src/axiomrank/axioms/rq4.py`)

Three `ir_axioms` `Axiom` subclasses, modelled on `DeterministicProx1Axiom` in
`axioms/relaxed.py` (the `@inject` / `term_tokenizer` / `text_contents` plumbing; sign fixed
with `strictly_greater`). Text-only, no collection statistics — the same cheap shape as
PROX1. Direction is fixed to the *observed* residual direction (prefer longer / higher
coverage); whether that direction earns weight is decided by the decomposition, not tuned.

- **`VerbosityAxiom` (VERB).** Precondition: the two documents cover the **same set of
  distinct query terms** (`query∩doc1 == query∩doc2`). When it holds,
  `preference = strictly_greater(len_1, len_2)` on word-token length; 0 on a length tie or
  when the precondition fails.
- **`RelaxedVerbosityAxiom` (VERB_R, `margin_fraction`).** The exact same-covered-set gate is
  widened to `isclose(cov_1, cov_2, rel_tol=margin_fraction)` on the query-coverage
  *fraction* (`cov_i = |query∩doc_i| / |query|`), mirroring the relaxed TF-LNC / M-TDC
  pattern. Then `strictly_greater(len_1, len_2)`.
- **`QueryCoverageAxiom` (QCOV).** `preference = strictly_greater(cov_1, cov_2)`; 0 on a tie
  or an empty query. The graded relaxation of AND (which fires only on full coverage) and
  distinct from the TF axioms (which count frequency, not distinct coverage).

Factories `VERB`, `VERB_R`, `QCOV` are `lazy_inject(...)` exports, resolvable by name. These
are the discrete-preference form of the `d_len` / `d_qcov` covariates in
`analysis/covariates.py`; they use the ir_axioms spaCy term tokenizer rather than the
covariates' whitespace regex, so column values are close but not identical to `sign(d_len)`
/ `sign(d_qcov)` — the axiom is a faithful tokenizer-consistent re-statement, not a copy.

**Determinism.** The axioms read only set sizes and token counts — no order-dependent
iteration over a set — so they are deterministic without the `sorted()` guard PROX1 needed
(verified by `test_rq4_axioms_are_deterministic`). Pinned to ir_axioms 1.1.2.

## 3. Wiring (`axioms/registry.py`)

`_factory` now resolves `rq4` first, then `relaxed`, then ir_axioms (a three-line change).
Bare names (`VERB`, `QCOV`) and `{name: VERB_R, alias: …, params: {margin_fraction: …}}`
specs both resolve through the existing `_instantiate` path with no other changes.

## 4. Config changes (append-only)

`VERB`, `QCOV`, and `VERB_R` at `margin_fraction: 0.2` (alias `VERB@m0.2`) are appended to
the **lexical** battery of `configs/rq2_dl19_top10.yaml` and `configs/rq2_dl20_top10.yaml`
(the two cells the `rq3_decomposition` runner pools as `sources:`). New columns only — no
existing axiom or ranker entry is touched. Coverage on the cached pairs: VERB ≈ 60 %, QCOV
≈ 31 %, VERB@m0.2 ≈ 68 % non-neutral.

## 5. Additive cache recompute (`scripts/add_rq4_axiom_columns.py`)

The hard constraint is append-only: never overwrite an existing axiom-pref column. A plain
`build_axiom_prefs --refresh` recomputes *every* column and would risk perturbing the Phase
1/2 baseline (e.g. the PROX determinism history, MEMORY.md). Instead the script computes
**only** the columns absent from the cached `axiom_prefs.parquet` and merges them onto the
existing frame (`validate="one_to_one"`, no-NaN assertion), leaving every existing column
bit-for-bit intact. Idempotent. That the classical-only decomposition in §6 reproduces the
Phase 2 numbers to 3 dp (Qwen 0.629/0.057, flan-large 0.666/0.074, flan-xl 0.639/0.059)
confirms the existing columns were untouched.

## 6. Evaluation (`experiments/rq4_axioms/run.py`, `configs/rq4_axioms.yaml`)

A dedicated runner rather than re-running `rq3_decomposition/run.py`. Rationale: the rq3
runner's single "lexical" feature set is now the *augmented* battery, so it cannot produce
the **paired** classical-vs-augmented comparison with a bootstrap CI on the lift that the
design asks for; re-running it would also clobber the Phase 2 baseline artefact under
`results/rq3_decomposition/`. The new runner reuses the same pooling (`merged_cell_frame`,
collection-namespaced query ids) and `analysis.decompose`. Two arms:

1. **Capture.** Per ranker, `decompose` on the classical lexical battery (full rq2 battery
   minus the degenerate columns and minus the RQ4 columns) and on classical + {VERB, QCOV},
   on the same query-grouped folds. Reports CV accuracy, pseudo-R², the reducible-residual
   upper bound and the fitted VERB/QCOV coefficients, plus two paired lifts with 2000-draw
   query-bootstrap CIs: the OOF **accuracy** lift and the per-pair **log-loss** lift (the
   information view Phase 2 §3.1 treated as the honest figure). VERB_R is carried in the
   cache as an auxiliary variant but is not in the {VERB, QCOV} headline pair.
2. **Reranking.** Per collection, the axiom battery is aggregated into a per-pair preference
   by **majority vote** (`sign` of the column sum — the canonical untuned ir_axioms
   aggregate), Copeland-ranked (`ranking.copeland_ranking`) and scored against the qrels.
   nDCG@10 / MAP, classical-only vs classical + {VERB, QCOV}, with a paired query-bootstrap
   CI on the per-query lift. Per §3.8 the deliverable is **lift over the classical battery**,
   not absolute nDCG (the depth-10 oracle ceiling caps the metric).

Outputs: `results/rq4_axioms/pooled_top10/metrics/<model>/capture.json` and
`.../reranking.json`.

## 7. Reranking effectiveness and the fitted-surrogate baseline (`experiments/rq5_surrogate/`)

Two questions beyond capture: do these axioms make a reranker that beats BM25, and how far
toward the LLM can a *weighted* axiom model get? Both are zero-model-call, from the same cached
rq2 top-10 stages + store.

**Effectiveness (untuned aggregate).** Reranking BM25's top-10 by the §6.2 majority vote,
classical + {VERB, QCOV} lands at **BM25 parity** — DL19 +0.008 nDCG@10 (CI spans 0), DL20 −0.013
(CI spans 0). The classical battery alone is significantly *below* BM25 on DL20 (−0.027, CI
excludes 0); VERB+QCOV rescue it to parity. The LLM rerankers on the same top-10 beat BM25 by
+0.038–0.069 (all CIs exclude 0); the perfect-top-10 oracle is +0.089/+0.096. So the untuned
axioms ≈ BM25 (they largely re-express BM25's own ordering on lexically-strong top-10 pairs), the
LLM captures ~70 % of the oracle gain, the axioms ~0–9 %.

*Gotcha (recorded).* Building the LLM reranker by Copeland-aggregating the **raw** preference-store
load double-counts: the store also holds the uniform50 wide-gap cells' verdicts, whose pairs reach
ranks > 10, so Copeland then reranks a deeper pool and inflates nDCG (Qwen appeared to *beat* the
oracle). Cell reranking must restrict verdicts to the cell's top-10 pairs — which
`merged_cell_frame` does by construction (it merges axioms onto the cell's `pairs`), so the
runners are correct; the caveat is for any ad-hoc store-level aggregation.

**Fitted surrogate (RQ5 preview, `run.py` + `configs/rq5_surrogate.yaml`).** The fair "how far can
axioms get" test: per LLM target, an L2 logistic (same estimator as `joint_fit`) is trained to
predict that model's pairwise verdict from the axiom battery, **query-grouped out-of-fold** (a
held-out query's pairs scored by a model trained only on other queries — no leakage), predicting
P(doc₁ preferred) on **all** top-10 pairs (not just the LLM's decisive ones, so it never peeks at
which pairs the LLM tied on); the hard sign is its Copeland preference. Both the classical and the
classical + {VERB, QCOV} feature sets are fitted. Results (nDCG@10, Δ vs BM25):

- **DL19** (headroom): the fitted surrogate reaches +0.025–0.028 (CI excludes 0) — it beats both
  BM25 and the untuned vote — recovering **~37–60 % of the LLM's over-BM25 gain** (~26 % of the
  oracle's).
- **DL20** (saturated, BM25 already near oracle): parity (−0.006, CI spans 0); none of the LLM's
  +0.062.
- **VERB+QCOV add nothing to the fitted surrogate** (marginal Δ within ±0.004) — a weighted
  classical battery already absorbs their small, partly-redundant signal, even though they *did*
  help the equal-weight vote. Honest nuance.
- The surrogate barely depends on which LLM it mimics (±0.003) — the axiom-learnable structure is
  LLM-agnostic and BM25-adjacent.

The picture: **BM25 ≈ untuned axioms < fitted axiom surrogate < LLM ≪ oracle.** The fitted lexical
surrogate is the baseline the semantic (Tier-A) axioms of the main RQ4 line must move toward the
LLM (`phase3-design.md` §4). Output:
`results/rq5_surrogate/pooled_top10/metrics/surrogate_reranking.json`.

## 8. Tests (`tests/test_rq4.py`)

Ten tests (marked `slow`, JVM start): VERB direction; VERB same-covered-terms precondition
gating; VERB length-tie → 0; VERB_R firing where strict VERB's set gate rejects but coverage
fractions match; VERB_R margin widening the admissible coverage gap; QCOV graded direction
(including the not-all-terms case that separates it from AND); QCOV tie → 0; QCOV empty-query
→ 0; registry resolution + column production through `axiom_preferences`; determinism (two
identical runs). Hand cases use a whitespace stub tokenizer; the last two use the real spaCy
tokenizer end-to-end.

## 9. Runbook

```bash
uv run --no-sync python -m pytest tests/test_rq4.py -q
uv run --no-sync python scripts/add_rq4_axiom_columns.py          # additive cache columns
uv run --no-sync python experiments/rq4_axioms/run.py --config configs/rq4_axioms.yaml
uv run --no-sync python experiments/rq5_surrogate/run.py --config configs/rq5_surrogate.yaml
```
