# Phase 3 Implementation — RQ4 Engineering Record

> This document records the implemented VERB/QCOV and casebook-derived D0 development increments,
> cache handling, coherent fitted pairwise runners and runbooks. All are retrospective candidates
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

A dedicated runner preserves the Phase 2 baseline and makes the RQ4 comparison model-specific.
It reuses `merged_cell_frame`, namespaces queries by query set, assigns one common query-disjoint
fold map, and fits four variants: classical, +VERB, +QCOV and +both. Training uses decisive
preferences from training queries; every held-out top-10 pair receives an OOF probability and
preference before qrels are loaded.

The runner then reconstructs BM25, LLM, fitted-variant and untuned-vote runs. Majority vote is a
transparency diagnostic, not the primary RQ4 test. Paired 10,000-draw query bootstraps compare
each fitted variant with BM25 and the LLM and evaluate the nested VERB/QCOV ablations. Pairwise
fidelity, explicit/forced ties, coefficients, fold assignments, per-query metrics and remaining-
gap profiles are persisted alongside the runs.

Outputs under `results/rq4_axioms/pooled_top10/metrics/<model>/` include
`pair_predictions.parquet`, `folds.csv`, `surrogate_models.json`, `runs.parquet`,
`effectiveness_per_query.csv`, `effectiveness.json`, `new_axiom_agreement.csv`, `capture.json`
and `residual_profiles.csv`.

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

1. either freeze the current `C=1.0` regularisation or add genuinely nested inner-C selection;
2. implement and ledger the remaining candidate families within the bounded revision budget;
3. family-level leave-one-out ablations for the frozen extended battery;
4. a frozen development manifest before the already locked `beir/nfcorpus/test` is opened;
5. separate held-out fidelity and effectiveness claims with multiplicity-adjusted secondary tests.

RQ5 begins only after a separate implementation proves which retained features decompose into
per-document values, scores each document once, and measures latency/call/FLOP savings.

## 11. Qualitative reversal resource

`experiments/rq4_qualitative/run.py` selects development cases where Qwen reverses a BM25 pair in
the qrel-improving direction and the final query nDCG@10 increases. It makes no model calls and
writes the complete candidate table and inspection packets under
`results/rq4_axioms/pooled_top10/qualitative/`.

The manual synthesis is tracked in `phase3-qualitative-casebook.md`, with machine-readable labels
in `resources/phase3-qualitative-case-annotations.csv`. The first run found 583 contributory
reversals; thirteen primary cases and two cautionary cases motivated QARA, CBP, QCS and a typed
specificity refinement. These remain retrospective development hypotheses.

## 12. Casebook-derived D0-v2 pipeline

`src/axiomrank/axioms/answering.py` implements the versioned D0 subset of DEFANS, NUMANS, COMPARE
and CBP with explicit query and pair preconditions. `experiments/rq4_candidates/run.py` computes
candidate preferences fresh from cached DL19/DL20 pool/pair text, joins cached LLM labels with
`allow_new=False`, and fits classical, four add-one and all-D0 arms on a shared query-disjoint fold
map. It does not append to or invalidate the Phase 1/2 axiom caches. Qrels are read only after all
OOF pair predictions exist. The runner also writes `provenance.json` (variant alias, `answering.py`
source digest, git revision) so a code change under an unchanged version alias is detectable.

D0-v0 revealed that count evidence was not bound to the requested noun and that CBP could act on
bare ambiguous person queries; D0-v1 corrected both. An independent review then found two
result-changing flaws still in D0-v1: CBP counted ordinary prose numbers as numbered-list
boilerplate (list punctuation was optional), and NUMANS bound counts only at sentence granularity.
D0-v2 requires genuine list punctuation, binds counts within a local number–noun token window with
deterministic singular/plural matching, suppresses degenerate bootstrap intervals below five
evaluable-query clusters, and uses an exact bidirectional candidate merge; it also adds a
frozen-coefficient feature-zero diagnostic. The `d0v0/` and `d0v1/` result directories are retained;
current outputs are under `d0v2/`. Twenty-one focused tests cover direction reversal, neutrality,
explicit list abstention, person ambiguity, local count binding, prose-number vs true-list
boilerplate, exact candidate merge, registry metadata and evaluator nesting.

Runbook:

```bash
uv run --no-sync pytest -q tests/test_answering_axioms.py tests/test_candidate_registry.py tests/test_rq4_candidates.py
uv run --no-sync python experiments/rq4_candidates/run.py --config configs/rq4_candidates_d0.yaml
```

The corrected D0-v2 rerun (`0 newly collected` for every ranker) weakened the earlier signal.
All-D0 nDCG@10 deltas against classical are Qwen −0.0057/+0.0024, FLAN-large +0.0002/+0.0044 and
FLAN-XL −0.0058/+0.0031 on DL19/DL20; only the negative FLAN-XL DL19 delta is individually
significant. Fidelity lifts all contain zero except a marginal FLAN-large log-loss, and all-D0
remains far below the LLM reranker. These are development diagnostics, not confirmation; the
candidate-by-candidate disposition (retain NUMANS, defer COMPARE, reject DEFANS and CBP in their
current D0 form, do not freeze the union) is recorded in `phase3-candidate-axiom-specs.md` §7.
