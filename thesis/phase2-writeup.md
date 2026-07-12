# Phase 2 — Predictive Diagnostics for New-Axiom Development

*Detailed analysis record. The final manuscript uses the compressed treatment in
`foundation-writeup.md`; chronology and superseded interpretations are in
`docs/research-logbook.md`. Citation numbers refer to `docs/literature-overview.md`.*

## 1. Research role

Phase 2 measures how much incremental predictive value the corrected existing axiom battery
provides on the primary development population. Its purpose is to establish the baseline and
produce evidence-graded candidate leads for RQ4. It is not a decomposition of mechanism, an
estimate of stochastic noise, or a gate between RQ4 and RQ5. RQ4 remains the main contribution.

RQ3 asks: how well does a combined existing-axiom model predict stable pairwise preferences, and
which remaining errors are useful for candidate generation without being treated as confirmed
structure?

## 2. Population and method

The analysis pools the DL19 and DL20 BM25 top-10 all-pairs cells, using collection-namespaced query
ids. From 4,330 sampled pairs, the stable, decisive subsets contain 3,158 Qwen and 2,931
FLAN-T5-large pairs. Position-inconsistent pairs are excluded from this fidelity estimand and
reported separately.

The headline model is L2 logistic regression over the non-degenerate full lexical battery from
Phase 1. All predictions are query-grouped out of fold; null priors are estimated from training
folds only. Reported measures are majority baseline, accuracy, ROC-AUC, log loss and normalised
log-loss gain. The latter is relative predictive improvement, not a fraction of behaviour or
mechanism explained.

WordNet semantic columns form a scoped ablation. A shallow gradient-boosted model on identical
folds checks one form of nonlinear headroom but cannot establish that the linear model is complete.

The corrected incremental diagnostic refits axiom-only and axiom-plus-covariate models inside the
same query-disjoint folds. Content covariates are signed length and query-coverage differences; a
broader set also includes BM25 rank and score. Query-macro bootstrap intervals preserve the query
as the inferential unit.

Order-swap consistency is not converted into reliability. The swapped prompts differ
systematically, and the fitted population already conditions on their agreement. Run-to-run
randomness would require repeated identical prompts.

## 3. Existing-battery result

| ranker system | decisive n | majority | OOF accuracy | gain | AUC | normalised log-loss gain |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 3,158 | 0.559 | 0.630 | +0.072 | 0.659 | 0.060 |
| FLAN-T5-large | 2,931 | 0.587 | 0.663 | +0.076 | 0.687 | 0.076 |

The classical battery provides modest incremental prediction on stable, decisive top-10 pairs.
The tested gradient-boosted complement adds 0.013 accuracy for Qwen and 0.007 for FLAN; without a
broader model comparison and paired uncertainty, this is only evidence that the tested complement
adds little.

Per-query-set fits move in the same general direction, but DL19 and DL20 are query sets from one
corpus rather than independent domain replications.

## 4. Semantic ablation

Adding WordNet columns lowers pooled accuracy from 0.630 to 0.623 for Qwen and from 0.663 to 0.651
for FLAN-T5-large. This reproduces the Phase 1 WordNet-specific null. Dense semantic embeddings and
ranker representations were not tested, so no general semantic-null claim follows.

## 5. Corrected incremental diagnostic

| ranker system | content-only accuracy lift [95% CI] | broader-set lift [95% CI] |
|---|---|---|
| Qwen | +0.011 [−0.012,+0.034] | +0.029 [+0.003,+0.056] |
| FLAN-T5-large | +0.006 [−0.009,+0.021] | +0.005 [−0.013,+0.023] |

Neither content-only interval excludes zero. Qwen's broader association includes BM25 rank and
score and therefore does not establish a content-specific residual mechanism. Length and query
coverage remain developmental candidate sources only.

Exploratory K=4 clustering suggested length/verbosity and coverage labels. The engineering audit
now selects nearest-centroid exemplars rather than first rows, but cluster interpretation remains
exploratory because K, features and candidate labels were chosen on development data.

## 6. Rank-gap and order diagnostics

Within top ten, joint predictive accuracy and model decisiveness tend to rise with rank gap. On
uniform depth-100 pairs, the strong expected axiom-alignment gradient does not appear. The data do
not validate the proposed explanation that classical axioms work on easy pairs and fail only on
hard pairs. The top-10 predictive baseline remains a replicated measurement within this domain.

Conditional cycle rates remain low among triangles whose three edges survive the canonical-label
filter. This does not establish transitivity of all raw presentations. Tie diagnostics distinguish
explicit model ties from order-disagreement collapse.

## 7. Depth and ranker-scale context

FLAN-T5-XL has pairwise qrel-ordering accuracy about 0.83. At depth ten its nDCG@10 is 0.526/0.533,
against BM25 0.480/0.494 and qrel oracle 0.575/0.583. It captures about 49%/44% of the available
BM25-to-oracle improvement. The older ratio of model nDCG to oracle nDCG (~91%) overstated useful
headroom capture.

The published PRP result reranks a deeper pool, so its absolute score is not a direct competence
target here. FLAN-large, FLAN-XL and Qwen do not form a controlled scale experiment: effectiveness
is not monotonic, and Qwen also changes architecture, prompt and scoring. No causal scale relation
is claimed.

## 8. RQ4 handoff

Phase 2 supplies a fitted classical baseline and two retrospective probes, not a confirmed
mechanism. RQ4 therefore uses a broad candidate registry informed by theory, literature and error
analysis. Every candidate is evaluated on two separate axes:

- fidelity to the target LLM;
- qrel effectiveness of the fitted axiom reranker.

Development uses bounded, fully logged revisions. Final definitions, features and fitting choices
are frozen before external confirmation. A candidate may be explanatory only, effective only,
both, or neither. RQ5 begins later only for features with a genuine per-document decomposition.
