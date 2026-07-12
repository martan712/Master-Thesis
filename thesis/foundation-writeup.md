# Experimental and Diagnostic Foundation for RQ4

*Final-manuscript consolidation of Phases 0–2. The detailed phase write-ups remain analysis
records; this is the compact foundation chapter to carry into the thesis. Citation numbers refer
to `docs/literature-overview.md`.*

## 1. Role in the thesis

Pairwise ranking prompting provides the target LLM protocol [2], while classical axiomatic IR
provides the interpretable constraints against which it is analysed [7, 8]. Phases 0–2 establish that the pairwise-ranking pipeline is reproducible, define the population
and estimands, measure the limits of the existing axiom battery, and generate candidate directions
for RQ4. They are diagnostic groundwork and do not determine whether RQ4 proceeds. RQ4 is
the main study: iterative development, fitted reranking, ablation and held-out confirmation of
new axioms. RQ5 is a later efficiency application and requires a genuinely pointwise scorer.

## 2. Experimental contract

The development setting is English passage reranking on the judged TREC DL19 and DL20 query sets
over MS MARCO. BM25 supplies the candidate pool. The primary condition contains all 45 unordered
pairs among each query's top ten documents; a uniform sample from depth 100 maps scope across rank
gaps. Each pair is presented to the ranker in both orders. A decisive canonical label is retained
only when the swapped presentations agree; order-inconsistent pairs are reported separately.

The query—not the pair—is the inferential unit. Cross-validation is query-grouped, uncertainty is
estimated by resampling queries, and all metrics state their denominator. RQ1–RQ3 primarily measure
fidelity to the ranker's canonical preference. Qrel effectiveness is a separate analysis. RQ4
retains two claim axes: held-out LLM fidelity for an explanatory claim and held-out nDCG@10 for
an effectiveness claim. Improvement on one does not imply improvement on the other.

Two ranker systems supply the core measurements: Qwen with the chat/log-probability protocol and
FLAN-T5-large with the sequence-to-sequence label-likelihood protocol. They differ in model,
prompt and scoring, so agreement across them is within-study robustness, not a clean architecture
effect. DL19 and DL20 are two query sets from one corpus/domain, not independent domain
replications.

## 3. Pipeline validation and corrections

The pipeline caches ordered model verdicts in an append-only store, computes signed axiom
preferences, collapses swapped presentations under a fixed rule, and produces agreement,
query-grouped prediction and reranking outputs from versioned configuration. The Phase 0 PROX1
and PROX2 values are invalid: PROX1 was nondeterministic and PROX2's batch path had a sign error.
All scientific proximity results use the corrected Phase 1 rerun. The baseline implementation is
built around `ir_axioms` [19].

Conditional cycles are rare among triangles whose three edges survive the order-consistency
filter, but this does not establish transitivity of all raw presentations. Position inconsistency
remains a separate outcome and excludes a substantial share of pairs from the primary fidelity
estimand. It is not stochastic reliability: swapped prompts are systematically different
measurements, so their agreement cannot identify a noise ceiling.

The depth-matched effectiveness reference establishes that the aggregated LLM preferences improve
BM25 on DL19 and DL20. It does not imply that every unexplained pairwise decision is skill. The
published PRP absolute scores use a deeper pool and are not direct anchors for a top-10 reranker.

## 4. Baseline diagnostic results

The corrected lexical profile is stable across the tested query sets and ranker systems. AND and
LB1 have the strongest individual agreement; broad term-frequency constraints such as TFC1 and
DIV are near chance in the primary top-10 condition. Relaxing strict preconditions increases
coverage but does not recover the pilot's apparent M-TDC effect. The expected strong rise in
axiom alignment on wide-gap pairs does not materialise, so BM25 redundancy remains a hypothesis
rather than an established cause of the top-10 result.

The tested WordNet operationalisation of semantic term-matching axioms [10] adds no held-out predictive value. Because dense
embedding semantics was not evaluated, this is a WordNet-specific null, not evidence that semantic
criteria are irrelevant.

On stable, decisive pooled top-10 pairs, the fitted classical lexical model reaches accuracy
0.630 for Qwen and 0.663 for FLAN-T5-large, gains of 0.072 and 0.076 over their majority baselines.
Normalised out-of-fold log-loss gains are 0.060 and 0.076. These are modest predictive gains, not
fractions of behaviour mechanistically explained. A tested nonlinear complement adds little, but
does not prove that the linear model is complete.

The corrected content-only incremental model does not resolve an additional effect: Qwen
+0.011 [−0.012, +0.034], FLAN +0.006 [−0.009, +0.021]. Length and query-coverage clusters are
therefore exploratory. Increment 1 of RQ4 likewise treats VERB as null and QCOV as an internal
development signal, not held-out validation.

## 5. Handoff to the main contribution

The diagnostic null determines the burden of evidence, not whether RQ4 proceeds. Candidate
axioms must come from an explicit theory-led menu, retain a complete variant ledger, and pass a
bounded development loop. Each iteration compares BM25, the LLM reranker, a fitted classical
axiom reranker and the fitted extended model on absolute nDCG@10 and MAP; pairwise capture and
gap analyses explain outcomes but do not replace effectiveness.

After no more than two documented revisions per candidate family, the retained Tier-A battery is
frozen. Family ablations separate unique from redundant contributions. The final extended model
has one primary held-out effectiveness test; individual-family confirmation tests are secondary
and multiplicity-adjusted. Confirmation data are not used to redefine thresholds, signs,
regularisation or candidate membership. The final model has separate prespecified confirmation
tests for LLM fidelity and qrel effectiveness, and the four possible joint outcomes are reported
without collapsing them into one success label. Null and rejected candidates remain part of the
RQ4 result and define the tested boundary of the framework.
