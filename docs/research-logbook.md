# Research Logbook and Provenance Record

This file preserves how the research design changed. It is intentionally separate from the clean
active protocols in `research-plan.md` and `phase0-design.md` through `phase3-design.md`.

Repository commits are evidence that text/code existed at a time; they are not a formal
preregistration unless a fixed pre-analysis artifact was archived before results were inspected.

## 1. Repository chronology

All commit identifiers below are dated 2026-07-12.

- Initial Phase 2 development sequence: `b98ad01`, `60894a2`, `08a2ded`, `2238b7a`, `9271eaa`,
  `0dfabf6`.
- Ranker-scale plan: `6e0e866`.
- Subsequent checkpoints: `b337cc0`, `87c7d50`, `ab8e870`, `5a1f42c`.
- The methodological audit and clean restructuring documented below occurred after those commits
  and was initially uncommitted workspace work.

The audited replacement was committed on 2026-07-13 in phase order:

- `801f76e` — add this scientific audit logbook;
- `c4487eb` — correct the Phase 0 measurement and cache contract;
- `f2b29a9` — correct Phase 1 baselines and effectiveness inference;
- `c346168` — correct Phase 2 prediction, residual and scale interpretations;
- `5df8c35` — rebuild RQ4 as the fitted axiom-development pipeline;
- `bd3bf9a` — rebuild the overall thesis architecture around RQ4.

The commit bodies state the superseded error and the replacement method. None carries a
co-author trailer.

These identifiers allow reconstruction of what was written and implemented, but do not by
themselves establish that every criterion preceded every analysis.

## 2. Original architecture and assumptions

The initial project architecture was:

1. build a cached preference/axiom pipeline;
2. measure lexical and semantic agreement;
3. decompose LLM preferences into an “explained” part and residual;
4. use residual covariates/clusters to decide whether RQ4 or RQ5 became the thesis emphasis;
5. interpret a rich residual as support for new axioms and a weak residual as support for an
   efficiency surrogate.

Important initial assumptions included:

- top-10 low agreement would be validated by higher agreement on wide-gap pairs;
- order-swap agreement could estimate a reliability ceiling/noise floor;
- pseudo-R² could be described as a fraction of behaviour explained;
- Qwen and FLAN similarity could support architecture-general claims;
- WordNet failure could gate stronger semantic operationalisations;
- larger FLAN parameter scale could proxy ranker strength;
- retrospective VERB/QCOV probes could be described as validated on DL19/DL20.

The audit found that several assumptions were not identified by the design or were contradicted by
the implemented estimand.

## 3. Phase 0 corrections

### 3.1 Conditional estimand

The canonical fidelity analysis excludes order-inconsistent pairs by turning them into ties and
then analysing decisive labels. Therefore, its population is stable, decisive BM25-top-10 pairs,
not all sampled pairs or all raw model presentations. This conditioning is now stated in every
clean protocol.

### 3.2 Transitivity

Cycle rates were computed only on triangles whose three edges survived the decisive filter. The
very low cycle rate is valid conditionally, but it does not establish that the raw judge is
transitive. Triangle survival and order inconsistency are reported separately.

### 3.3 PROX invalidation

The Phase 0 PROX1 result was nondeterministic and PROX2's batch path had the wrong sign. Their
pilot cells were invalidated rather than retained as scientific evidence. Corrected Phase 1 values
supersede them.

## 4. Phase 1 corrections

### 4.1 Effectiveness gate interpretation

Depth-10 Copeland reranking improves BM25 for the tested LLM systems. This establishes aggregate
relevance signal under the chosen protocol. It does not mean every axiom-model residual is skill
or that the systems match published deeper-pool PRP effectiveness.

### 4.2 Rank-gap expectation

The expected strong increase in axiom alignment on wide-gap pairs did not materialise. Model
decisiveness increased, but most axiom agreement did not. Therefore the wide-gap arm is a scope
analysis, not successful validation of the proposed BM25-redundancy mechanism.

### 4.3 WordNet scope

The completed semantic null uses WordNet synonym sets. FastText, sentence embeddings and internal
ranker representations were not tested. The conclusion was narrowed to the WordNet
operationalisation.

### 4.4 System and dataset labels

DL19 and DL20 are two query sets over one corpus/domain, not independent collections in a domain-
general sense. Qwen and FLAN also differ in prompt and scoring, so they are ranker systems rather
than a controlled architecture contrast.

## 5. Phase 2 corrections

### 5.1 Decisive sample size

The pooled sampled-pair count is 4,330. The actual decisive samples used by the decomposition are
3,158 for Qwen and 2,931 for FLAN-T5-large. A table that reported 4,330 for both fitted models was
corrected.

### 5.2 Pseudo-R² interpretation

Values near 0.06–0.08 are normalised out-of-fold log-loss gains relative to an intercept-only
model. They are not “6–8% of behaviour explained,” and `1−pseudo-R²` is not the residual fraction.
Accuracy is also not an explained fraction because the majority baseline is already 0.56–0.59.

### 5.3 Order swap is not reliability/noise

Swapped presentations are systematically different prompts, not independent repeats of an
identical measurement. The analysis then conditions on agreement between them. Converting overall
order consistency into 0.839/0.797 “reliability ceilings” required unsupported symmetric,
independent error assumptions. The resulting 0.21/0.13 “reducible residual” and claims that the
remainder was model stochasticity were withdrawn.

Estimating run-to-run randomness would require repeated calls to identical prompts under a fixed
sampling/inference regime.

### 5.4 OOF stacking leakage

The original residual model conditioned on out-of-fold axiom predictions in a way that did not
refit both compared pipelines cleanly inside identical folds. The corrected incremental model
refits axiom-only and axiom-plus-covariate arms within the same query-disjoint folds and uses a
query-macro bootstrap.

Old content-only lifts (+0.027 for Qwen pooled, for example) were superseded. Corrected results:

- Qwen content-only +0.011 [−0.012,+0.034]; broader set +0.029 [+0.003,+0.056].
- FLAN content-only +0.006 [−0.009,+0.021]; broader set +0.005 [−0.013,+0.023].

The content-only intervals include zero. Rank/score signals in the broader Qwen set do not prove a
content residual mechanism.

### 5.5 Clusters

Residual clustering used fixed K=4, and the initial exemplars were first stored rows rather than
representative points. The engineering audit later changed selection to nearest-centroid examples
(§8). Length/verbosity and query-coverage names remain exploratory, not replicated mechanisms.

### 5.6 Phase role

The original RQ4-versus-RQ5 branch was removed from the active architecture. A diagnostic null
cannot cancel the planned main contribution. Phase 2 is now compact baseline groundwork; RQ4 uses
a broader theory/literature/error-informed menu and locked confirmation.

## 6. Ranker-scale correction

FLAN-T5-XL depth-10 nDCG@10 is 0.526/0.533, BM25 is 0.480/0.494 and the qrel oracle is
0.575/0.583. Dividing XL by the oracle yields about 91%, but that ignores BM25's starting value.
The relevant fraction of available improvement is approximately 49%/44%. “Near-oracle” wording was
withdrawn.

The published PRP anchor used a deeper candidate pool. Depth mismatch explains why its absolute
nDCG is not a direct target, but does not prove model quality or eliminate all protocol confounds.

FLAN-large, FLAN-XL and Qwen differ in effectiveness and, for Qwen, architecture/prompt/scoring.
Three heterogeneous points do not identify a causal relationship between parameter scale and
axiom predictability. Claims that scaling refuted a weak-ranker explanation were withdrawn.

## 7. RQ4 Increment 1 correction

VERB and QCOV were proposed after inspecting Phase 2 length/coverage diagnostics. Tests on the same
DL19/DL20 data are retrospective development evidence even under query-disjoint CV. “QCOV
validated” was replaced by “internal capture signal”; VERB remains a null probe.

A coherent fitted pairwise RQ4 runner subsequently produced:

- accuracy lift for classical→+VERB+QCOV: Qwen +0.0123 [−0.0059,+0.0299], FLAN-large −0.0027
  [−0.0154,+0.0089], FLAN-XL +0.0051 [−0.0063,+0.0166];
- log-loss lift: Qwen +0.0126 [+0.0003,+0.0247], FLAN-large +0.0118
  [+0.0007,+0.0229], FLAN-XL +0.0075 [−0.0051,+0.0187];
- no add-one or nested nDCG@10 effectiveness interval excluding zero.

Older naive-vote and “RQ5 preview” narratives were superseded. The logistic pairwise model predicts
every pair and aggregates with Copeland; it is an RQ4 fitted pairwise reranker, not a pointwise
linear-time scorer.

## 8. Engineering audit corrections after the checkpoints

The following corrections were first implemented in the post-checkpoint audit workspace and then
committed in the phase-ordered sequence listed in §1:

- **Fold-local null priors:** majority/intercept baselines are estimated from training folds only;
  held-out labels no longer influence the null prediction.
- **Merge integrity:** pair/query joins validate exact key cardinality and fail on duplicate,
  missing or cross-cell matches rather than silently changing the analysis population.
- **Collapse provenance:** canonical verdict records distinguish an explicit model tie from an
  order-disagreement tie, so exclusion reasons are auditable.
- **PROX1 correction and cache safety:** the local PROX1 variant uses the intended symmetric
  distance definition; corrected axiom logic invalidates stale cache entries through fingerprints
  rather than reusing incompatible columns.
- **Cache-only analysis guard:** analysis-only runs fail instead of making an accidental model call
  when an expected cached preference is absent.
- **Effectiveness uncertainty:** paired query-bootstrap effectiveness intervals use 10,000 draws
  on aligned per-query differences.
- **Representative cluster exemplars:** the earlier first-row exemplar selection was corrected to
  nearest-centroid representatives. Cluster interpretation remains exploratory because K and the
  feature space are development choices.

## 9. Current implications

- RQ4 remains the main contribution.
- Candidate development is bounded and fully logged.
- Fidelity and qrel effectiveness are separate claim axes with four possible joint outcomes.
- DL19/DL20 are development data; external confirmation is locked before inspection.
- SciFact cannot be confirmation because it served smoke diagnostics. `beir/nfcorpus/test` was
  locked on 2026-07-13 before local outcome access.
- Tier-B LLM judges are diagnostic operationalisations, not literal oracles/upper bounds.
- RQ5 begins only with features proven to decompose into per-document scores and measured once per
  document.

## 10. Qualitative relevance-improving reversal study

On 2026-07-13, a cache-only selector joined Qwen pair preferences, final Copeland ranks, BM25,
qrels and axiom predictions. It found 583 pair reversals satisfying a deliberately outcome-
conditioned development rule: the query nDCG@10 improved, Qwen decisively preferred the
higher-grade passage, BM25 had ranked that passage lower, and the final Qwen run reversed the
pair. These are labelled *contributory reversals*, not causal edge effects.

A purposive reading sample was stored in `phase3-qualitative-casebook.md`. Repeated distinctions
were requested-relation satisfaction, answer-bearing prose rather than boilerplate, typed
specificity/completeness, and correct binding of query qualifiers and roles. They motivated the
development candidates QARA, CBP and QCS and a typed refinement of Specificity. Ambiguous-entity
and answer-shaped-but-unreliable cases were retained as cautions. Because selection and coding use
DL19/DL20 and one unblinded analyst, no candidate was promoted to validated status.

## 11. External confirmation lock

Before implementing the casebook-derived candidates, `beir/nfcorpus/test` was fixed on
2026-07-13 as the one-shot external confirmation set. Selection used public catalogue metadata
only; no local queries, documents, qrels, BM25 pool or result statistics were loaded. The lock
contract is stored in `phase3-confirmation-lock.yaml`, and shared retrieval/evaluation functions
now reject the dataset until a deliberate unlock manifest exists.

The final candidate registry, preconditions, thresholds, feature versions, ablations and
development-trained coefficients must be hashed and recorded before unlock. Confirmation applies
the frozen models without fitting on holdout LLM labels or qrels. Qwen is the primary target;
FLAN-T5-large is optional replication. The external domain makes transfer part of the claim and
part of the risk.
