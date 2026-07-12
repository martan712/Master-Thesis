# Phase 2 Design — Compact Predictive Diagnostics (RQ3)

Engineering details are in `phase2-implementation.md`; chronology and invalidated analyses are in
`research-logbook.md`. In the final thesis, this material is folded into the compact foundation
chapter rather than presented as a standalone centerpiece.

## 1. Purpose

Phase 2 quantifies the incremental predictive value of the corrected existing battery on the
primary development population and creates an evidence-graded handoff to RQ4. It does not decide
whether RQ4 proceeds, identify stochastic noise, or confirm a residual mechanism.

## 2. Analysis population

- DL19 and DL20 BM25 top-10 all-pairs, pooled with collection-namespaced query ids.
- Qwen and FLAN-T5-large analysed separately.
- Stable, decisive canonical pairs only: 3,158 Qwen and 2,931 FLAN pairs from 4,330 sampled pairs.
- Query-grouped folds for every fitted comparison; per-query-set results as robustness checks.

The excluded order-inconsistent pairs remain a separate outcome. The primary conclusions apply
only to stable, decisive pairs within the BM25 top ten.

## 3. Models and estimands

### 3.1 Existing-battery model

The headline model is L2 logistic regression over the non-degenerate full lexical battery from
Phase 1. WordNet columns are an ablation. An identically folded shallow gradient-boosted model is a
limited nonlinear comparison, not proof of functional completeness.

### 3.2 Predictive measures

Report majority baseline, query-grouped OOF accuracy, ROC-AUC, log loss and normalised log-loss
gain. Normalised gain is relative prediction improvement, not a fraction of ranker behaviour or
mechanism explained. Correctly classified pairs may be called predicted, not mechanistically
explained.

### 3.3 Incremental diagnostic covariates

Compare axiom-only with axiom-plus-covariate models inside identical query-disjoint folds. Content
covariates are signed document-length and query-coverage differences; the broader set also includes
BM25 rank/score differences. Because length and coverage overlap conceptually with existing
axioms, any lift is association, not discovery of an independent mechanism.

### 3.4 Order sensitivity

Order-swap consistency is reported separately. Swapped prompts are systematic interventions, not
independent identical repeats, so they do not identify a reliability ceiling, aleatoric noise or a
reducible residual.

## 4. Outputs

- Pooled and query-set-specific predictive table.
- WordNet ablation.
- Limited nonlinear comparison.
- Corrected incremental-covariate lift with query-macro bootstrap intervals.
- Exploratory rank-gap, covariate and cluster summaries labelled as hypothesis generation.
- Candidate handoff to RQ4 with evidence grade.

## 5. Completed results

### 5.1 Existing-battery baseline

| ranker system | decisive n | majority | OOF accuracy | gain | AUC | normalised log-loss gain |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 3,158 | 0.559 | 0.630 | +0.072 | 0.659 | 0.060 |
| FLAN-T5-large | 2,931 | 0.587 | 0.663 | +0.076 | 0.687 | 0.076 |

The existing battery has modest incremental predictive value. The tested nonlinear complement adds
0.013 accuracy for Qwen and 0.007 for FLAN, without uncertainty sufficient to claim completeness.

Adding WordNet lowers pooled accuracy from 0.630 to 0.623 for Qwen and 0.663 to 0.651 for FLAN,
consistent with the Phase 1 WordNet-specific null.

### 5.2 Corrected incremental diagnostic

| ranker system | content-only accuracy lift [95% CI] | broader-set lift [95% CI] |
|---|---|---|
| Qwen | +0.011 [−0.012,+0.034] | +0.029 [+0.003,+0.056] |
| FLAN-T5-large | +0.006 [−0.009,+0.021] | +0.005 [−0.013,+0.023] |

Content-only intervals include zero. The broader Qwen association includes BM25 rank/score
signals and does not satisfy a content-mechanism claim.

### 5.3 Exploratory leads

Fixed-K cluster summaries suggested length/verbosity and query coverage, but K was fixed and
stored exemplars were not representative medoids. These labels are developmental leads only.
The expected wide-gap axiom-alignment gradient did not appear. DIV's gap behaviour and length are
related observations, not one demonstrated mechanism.

### 5.4 Depth and ranker-scale context

FLAN-T5-XL achieves pairwise qrel-ordering accuracy about 0.83, but depth-10 nDCG@10 is
0.526/0.533 against BM25 0.480/0.494 and oracle 0.575/0.583. It captures about 49%/44% of the
available BM25-to-oracle improvement, not 91% of meaningful headroom. The published PRP anchor
uses a deeper pool. Nominal parameter scale, effectiveness and axiom predictability are not
identified as a causal scale relation by the three heterogeneous systems.

## 6. RQ4 handoff

Phase 2 supplies no confirmed content mechanism. Length and query coverage enter RQ4 as
retrospective developmental probes, alongside a broader theory- and literature-led menu. RQ4 is
the main contribution and requires bounded iteration, fitted pairwise reranking, two-dimensional
fidelity/effectiveness reporting, ablation and locked external confirmation.
