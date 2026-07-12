# Phase 2 Implementation — Predictive Diagnostics, RQ3 (engineering record)

> **Status: executed.** This file is the implementation and reproduction record; scientific
> outcomes are recorded in `phase2-design.md` §5; corrections are in `research-logbook.md`.

> Companion document: the current scientific design, estimand and compact results lives in
> `phase2-design.md`. This file records pooling, covariates, outputs, tests and reproduction.

The engineering record for Phase 2 (RQ3) describes its zero-call cost, how the query sets are pooled from cache, how diagnostic covariates
are assembled, the test discipline, the runbook, and the operational risks.

## 1. Scope

RQ3 is a cache-only diagnostic phase: it consumes Phase 1 top-10 verdicts and produces
query-grouped existing-battery predictions plus exploratory error diagnostics. It collects
**zero new model verdicts**. The *why* of every
design choice is in `phase2-design.md`; here we assume those decisions and implement them.

## 2. Cost

- **Model calls: none.** The DL19+DL20 top-10 verdicts for both rankers are already in the
  store (`phase1-design.md` §6). RQ3 re-reads them and pools.
- **Local CPU only:** rebuilding the merged pair frames from the cached stages (pool,
  pairs, axiom preferences — all Parquet under `data/processed/`), assembling covariates,
  and fitting existing-battery/incremental diagnostics plus exploratory clusters. Seconds-to-minutes per cell.
- The uniform50 cells (already cached) are re-read only for the gap-gradient item (§3.4 of
  the design); they add no model calls either.

## 3. Architecture

New and changed code, following the established layout (shared logic in
`src/axiomrank/`, experiment scripts as thin recipes):

```
config.py            + `sources: list[str]` on ExperimentConfig — a decomposition config
                       names the source grid-cell configs it pools, nothing else changes;
                       all Phase 0/1 configs stay valid (default empty list)
analysis/covariates.py  NEW: attach_covariates(merged, pool, pairs, store_df) -> merged +
                       non-axiom covariate columns; COVARIATE_COLUMNS, CONTENT_COVARIATES
analysis/decomposition.py NEW: decompose(merged, feature_names, verdicts, ...) ->
                       accuracy split + information split (McFadden pseudo-R²) +
                       an assumption-dependent single-order sensitivity calculation from
                       position consistency; reuses joint.joint_fit OOF
analysis/residual.py  NEW: residual_profiles (covariate-binned OOF accuracy + signed
                       residual), residual_model (grouped-CV lift over axiom-only, all vs
                       content-only covariates, bootstrap CI), residual_clusters (KMeans +
                       exemplars)
pipeline/frames.py    NEW: merged_cell_frame(cfg, ranker_cfg, refresh) -> (merged+covariates,
                       verdicts, pool); the from-cache rebuild of one cell's analysis frame,
                       factored so the rq3 runner and any later phase reuse it. measure_cell
                       is left untouched (Phase 1 outputs stay bit-for-bit reproducible)
experiments/rq3_decomposition/run.py  NEW: load the rq3 config, pool the source cells per
                       ranker, run decomposition + residual analysis + the joint gap
                       gradient; write per-model outputs
notebooks/p2_overview.ipynb  NEW: thin loader of the rq3 outputs (the working overview,
                       mirroring p1_overview.ipynb)
```

`analysis/decomposition.py` and `analysis/residual.py` build on `analysis.joint_fit`
(unchanged) — the combined model *is* the Phase 1 L2 logistic, so the starting number stays
comparable. The gradient-boosted complement (design §3.1) is an optional feature set inside
`decompose` (a `nonlinear=True` flag fitting a depth-limited `GradientBoostingClassifier`
on the same folds), reported as headroom, never as the headline.

### 3.1 Pooling from cache

`merged_cell_frame(cfg, ranker_cfg)` reproduces, from cache, exactly the frame the Phase 1
`measure_cell` builds internally, plus covariates:

1. `stages.build_pool` / `build_pairs` / `build_axiom_prefs` (cached Parquet; no recompute).
2. `collect.collect_verdicts` (lookup-only against the store — no model call) → `store_df`.
3. `analysis.model_pair_verdicts` → collapsed verdicts; `merge_pairs`; `attach_rank_gap`.
4. `analysis.attach_covariates(merged, pool, pairs, store_df)` → the non-axiom covariates.

The rq3 runner calls this for each source cell × ranker, tags each frame with a
`collection` column, and `pd.concat`s the two top-10 cells into the **pooled** frame. It
runs the decomposition three times per ranker — pooled (headline), DL19-only, DL20-only
(robustness) — and on both feature sets (lexical full battery; lexical+WordNet ablation)
and both covariate sets (all; content-only). Qwen is primary; flan-t5-large replicates.

Pooling reuses `query_id` across collections; because DL19 and DL20 have disjoint query id
spaces, `GroupKFold` on `query_id` keeps folds query-clean across the pool with no extra
work. (Guarded with an assertion that the id spaces are disjoint.)

### 3.2 Covariates (`analysis/covariates.py`)

All derivable from cached frames — no new download, no model call:

| covariate | source | content-based? |
|---|---|---|
| `rank_gap`, `rank_max` | pool ranks (`attach_rank_gap`) | position, not content |
| `score_gap`, `score_max` | pool BM25 `score` | lexical-strength, not new content |
| `d_len`, `len_max`, `len_ratio` | signed length difference, maximum and ratio from `pairs.text_*` | **yes** (verbosity/length) |
| `d_qcov` | signed query-term-coverage difference (tokenizer) | **yes** (matching) |
| `query_len`, `query_is_question` | `pairs.query` | **yes** (query type) |
| `conf_margin_prob`, `conf_margin_score` | store `prob_a`, `score_a−score_b`, per-pair mean | model confidence, **not** content |

`CONTENT_COVARIATES` = [`d_len`, `d_qcov`] for the scoped content-only diagnostic; confidence
and BM25/rank strength are excluded from that arm and retained in the broader comparison.
The confidence margin is aggregated over whichever presentations are available as the mean
of `|prob_a − 0.5|` and `|score_a − score_b|`; it does not require both orders.

### 3.3 Output inventory per pooled cell × model

Under `results/rq3_decomposition/<variant>/metrics/<model>/`:
- `decomposition.json` — per feature set: base rate, CV accuracy/AUC (from `joint_fit`),
  normalised log-loss gain (pseudo-R²), legacy order-consistency-derived fields retained for
  artifact compatibility but not interpreted as a reliability ceiling, and correct/incorrect
  fractions; plus nonlinear-complement headroom.
- `coefficients.csv` — the combined model's fitted axiom coefficients (interpretable
  decomposition), full-data fit.
- `residual_profiles.csv` — long format: covariate, bin, n, OOF accuracy, signed residual.
- `residual_model.json` — per covariate set (all / content-only): grouped-CV accuracy, the
  lift over the axiom-only baseline, and its 95% query-bootstrap CI.
- `residual_clusters.csv` — cluster id, size, covariate centroid means, exemplar pair keys.
- `gap_gradient.csv` — the joint-level gradient on the pooled top-10 (reusing
  `gap.gap_gradient` with the decomposition OOF), for the design-§4 open item.
- `gap_gradient_uniform.csv` — the wide-gap arm of the same (written only under
  `--uniform`), the combined model's OOF accuracy across the full BM25 rank-gap range from
  the `uniform_sources` cells.
- `config.yaml` — the exact rq3 config.

Per-collection robustness fits write under `.../<variant>/metrics/<model>/by_collection/`.

### 3.4 Config naming

- `rq3_pooled_top10.yaml` — the headline cell: `sources: [configs/rq2_dl19_top10.yaml,
  configs/rq2_dl20_top10.yaml]` (the rq2 top-10 configs carry both the full lexical battery
  and the WordNet semantic columns, so the ablation needs no new axiom computation), plus
  `experiment: rq3_decomposition`, `variant: pooled_top10`, `seed: 42`,
  `primary_metric: fidelity`.
- `rq3_smoke.yaml` — `sources: [configs/p1_smoke.yaml]` (scifact + mock, a trivial
  single-cell "pool") to exercise the whole frame → decomposition → residual path end to
  end without touching a real model.

The runner resolves `sources` (and, under `--uniform`, `uniform_sources`) relative to the
repo root, loads each with `load_config`, and pools their cells; `--uniform` adds the
wide-gap gradient from the cached `rq1_*_uniform` cells for the design-§4 comparison.

## 4. Test discipline

Mirrors the Phase 0/1 discipline — hand-computable cases, no network:

- `analysis/covariates.py`: a constructed pool+pairs+store frame with known ranks, scores,
  lengths and confidences → asserted covariate values (gaps, ratios, coverage, aggregated
  confidence), and the NaN path when a pair has only one presentation.
- `analysis/decomposition.py`: known OOF probabilities → hand-computed log-loss, normalised
  log-loss gain and accuracy/error partitions. Order consistency is tested as a separate
  descriptive output, never converted into a prediction ceiling.
- `analysis/residual.py`: a synthetic frame where one content covariate predicts incremental
  structure → corrected same-fold lift > 0 with CI above zero; pure-noise covariates → lift
  ≈ 0 with a CI spanning zero. Clustering returns the
  planted number of groups on separable synthetic covariates.
- `config.py`: `sources` parses and round-trips; a config without it stays valid (default
  empty).
- `rq3_smoke.yaml` runs through `experiments/rq3_decomposition/run.py` end to end on
  scifact+mock, asserting every output file is written and well-formed.

## 5. Work breakdown and runbook

1. `config.py` `sources` field + parsing test.
2. `analysis/covariates.py` + `analysis/decomposition.py` + `analysis/residual.py`, each
   with its hand-computable test; export from `analysis/__init__.py`.
3. `pipeline/frames.py` (`merged_cell_frame`) reusing the cached stages.
4. `experiments/rq3_decomposition/run.py` + `rq3_pooled_top10.yaml` + `rq3_smoke.yaml`;
   smoke through the runner.
5. Real run over the cached store (zero model calls):

   ```
   uv run python experiments/rq3_decomposition/run.py --config configs/rq3_pooled_top10.yaml
   ```

   Qwen first (primary), flan-t5-large as replication (`--only-model` filter reused from the
   Phase 1 runners). Sanity-check the pooled CV accuracy lands in the 0.59–0.67 band from
   Phase 1's fitted range before interpreting downstream diagnostics.
6. Analysis pass: fill `phase2-design.md` §5 with the existing-battery prediction, WordNet
   ablation, corrected incremental diagnostic and exploratory scope analyses. Do not derive a
   reliability/noise ceiling from order swaps or use RQ3 as a gate on RQ4.
   `notebooks/p2_overview.ipynb` is the working overview.

`scripts/run_phase2.sh [all|qwen|flan]` wraps step 5 (fidelity throughout; qrels are not used
in this phase).

## 6. Operational risks

- **Cache staleness.** RQ3 trusts the `data/processed/` stages and the store. All numbers
  postdate the 2026-07-11 ir_axioms PROX fixes (`phase1-design.md` §6); the runner records the store
  path and the axiom battery in `config.yaml` so a stale cache is detectable. `--refresh`
  recomputes the derived stages (never the verdicts).
- **Covariate/axiom overlap.** Length and coverage overlap conceptually with LNC/DIV and AND.
  The corrected comparison refits both arms inside identical folds; any lift remains an
  association rather than proof of an independent mechanism.
- **Confidence-margin availability.** The margin uses whichever presentations are available;
  per-feature NaN handling does not drop unrelated content covariates.
- **Small per-collection folds.** DL19's 43 queries make per-collection fits noisy; the
  pooled fit is the headline and the per-collection ones are robustness, reported with the
  same query-bootstrap CIs the Phase 1 profiles used.

## 7. FLAN-T5-XL collection reproducibility

Supplementary XL verdicts were collected append-only on the bronze host using the same prompt-v0
label-likelihood implementation and top-10 order-swapped pairs as FLAN-T5-large. Collection used
CPU `bfloat16` in the available ROCm container because the co-tenant GPU path was unavailable.
Returned Parquet parts were analysed locally from cache. Scientific comparison depends on matched
prompt/scoring/pool semantics; host and dtype are reproducibility metadata. Depth/scale results are
in `phase2-design.md` §5.4 and the detailed Phase 2 record; chronology is in
`research-logbook.md`.
