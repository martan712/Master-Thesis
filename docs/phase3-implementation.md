# Phase 3 Implementation — RQ4 Increment 1 Retrospective Engineering Record

> This document records the implemented VERB/QCOV development increment, append-only axiom-cache
> handling, coherent fitted pairwise runner and runbook. VERB/QCOV are retrospective candidates
> because their definitions followed inspection of DL19/DL20 diagnostics. Current scientific
> interpretation and the broader candidate protocol are in `phase3-design.md`; chronology is in
> `research-logbook.md`.

## 1. Scope and cost

Increment 1 encodes two retrospective diagnostic leads—verbosity/length and query coverage—as
explicit same-pair axioms and tests their internal fidelity and fitted reranking effectiveness.
It is a cache-only analysis: **zero model/LLM calls, zero downloads**. Everything computes from the
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
the then-current Phase 2 numbers to 3 dp. After the audit, regenerated classical
accuracy/normalised-log-loss values are Qwen 0.630/0.060, flan-large 0.663/0.076, and
flan-xl 0.639/0.060. This confirms key alignment, while the Phase 2 decision itself has
changed under the corrected residual analysis.

## 6. Evaluation (`experiments/rq4_axioms/run.py`, `configs/rq4_axioms.yaml`)

A dedicated runner rather than re-running `rq3_decomposition/run.py`. Rationale: the rq3
runner's single "lexical" feature set is now the *augmented* battery, so it cannot produce
the **paired** classical-vs-augmented comparison with a bootstrap CI on the lift that the
design asks for; re-running it would also clobber the Phase 2 baseline artefact under
`results/rq3_decomposition/`. The new runner reuses the same pooling (`merged_cell_frame`,
collection-namespaced query ids) and `analysis.decompose`. Two arms:

1. **Capture.** Per ranker, `decompose` on the classical lexical battery (full rq2 battery
   minus the degenerate columns and minus the RQ4 columns) and on classical + {VERB, QCOV},
   on the same query-grouped folds. Reports CV accuracy, normalised log-loss gain and the
   fitted VERB/QCOV coefficients, plus two paired lifts with 2000-draw
   query-bootstrap CIs: the OOF **accuracy** lift and the per-pair **log-loss** lift (the
   information view Phase 2 §3.1 treated as the honest figure). VERB_R is carried in the
   cache as an auxiliary variant but is not in the {VERB, QCOV} headline pair.
2. **Reranking.** Per collection, the axiom battery is aggregated into a per-pair preference
   by **majority vote** (`sign` of the column sum — the canonical untuned ir_axioms
   aggregate), Copeland-ranked (`ranking.copeland_ranking`) and scored against the qrels.
   nDCG@10 / MAP, classical-only vs classical + {VERB, QCOV}, with a paired query-bootstrap
   CI on the per-query lift. The restructured RQ4 report also places these arms in the absolute,
   depth-matched BM25/LLM/classical/extended nDCG@10 and MAP table; the oracle ceiling is context,
   not a reason to omit absolute effectiveness.

Outputs: `results/rq4_axioms/pooled_top10/metrics/<model>/capture.json` and
`.../reranking.json`.

## 7. Reranking effectiveness and the fitted pairwise-axiom baseline

The coherent `rq4_axioms` runner fits classical, +VERB, +QCOV and +both arms on fixed query-
disjoint folds, predicts every top-10 pair, and Copeland-aggregates each arm. It reports BM25,
LLM, untuned and fitted runs with 10,000-draw paired query-bootstrap comparisons.

Fitted nDCG@10, classical vs +both:

| target | query set | classical | +VERB+QCOV |
|---|---|---:|---:|
| Qwen | DL19 | 0.5028 | 0.5024 |
| Qwen | DL20 | 0.4845 | 0.4882 |
| FLAN-large | DL19 | 0.5045 | 0.5036 |
| FLAN-large | DL20 | 0.4900 | 0.4923 |
| FLAN-XL | DL19 | 0.5084 | 0.5063 |
| FLAN-XL | DL20 | 0.4908 | 0.4860 |

No add-one or nested new-axiom effectiveness interval excludes zero. The corresponding OOF
accuracy-lift intervals also include zero for every target. Log-loss lift is positive with an
interval above zero for Qwen (+0.0126 [+0.0003,+0.0247]) and FLAN-large
(+0.0118 [+0.0007,+0.0229]), but not FLAN-XL (+0.0075 [−0.0051,+0.0187]). These are separate
internal fidelity and effectiveness results; none is held-out confirmation.

The model remains pairwise and requires Copeland aggregation. It is RQ4 infrastructure, not the
RQ5 pointwise scorer. Older naive-vote and legacy “surrogate preview” summaries are superseded;
their chronology is retained in `research-logbook.md`.

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
```

## 10. Current coherent runner and remaining work

The current `rq4_axioms` runner now produces fixed query-disjoint folds; classical, +VERB, +QCOV
and +both fitted arms; pair predictions and serialized models; depth-matched BM25, LLM, untuned
and fitted runs; 10,000-draw paired query-bootstrap comparisons; agreement and tie diagnostics;
candidate ablations; and residual profiles. It runs cache-only and makes zero model calls.

The remaining work for the full `phase3-design.md` protocol is:

1. nested inner-C selection rather than a fixed regularisation value;
2. implementation and complete ledgering of the broader candidate families and bounded revisions;
3. family-level leave-one-out ablations for the frozen extended battery;
4. a frozen development manifest and external collection choice before any confirmation result
   (`beir/nfcorpus/test` is the current proposal, not yet locked);
5. separate held-out fidelity and effectiveness claims with multiplicity-adjusted secondary tests.

RQ5 begins only after a separate implementation proves which retained features decompose into
per-document values, scores each document once, and measures latency/call/FLOP savings.
