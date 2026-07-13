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

## 12. Preconditioned candidate specification

The casebook patterns were converted into a development-v0 candidate registry on 2026-07-13.
QARA was deliberately split into definition, numeric, purpose, location, composition, comparison
and causal constraints rather than implemented as one semantic score. Each candidate records a
query precondition, topical-comparability gate, evidence margin, neutral cases, cost level and
failure risks. CBP, AMBIG, QCS, TSC and answer localisation are specified separately.

The initial implementation order favours deterministic token/regex rules and the already-pinned
spaCy parser. A frozen encoder/NLI model is permitted only after deterministic coverage is
measured inadequate; factual verification remains diagnostic and cannot be claimed from surface
answer shape. The NFCorpus confirmation lock remains active throughout development.

## 13. Casebook-derived D0 implementation and first development test

On 2026-07-13, DEFANS, NUMANS, COMPARE and CBP were implemented programmatically before any new
commit. Synthetic tests fixed their signs, reversed-order behavior, preconditions and neutral
cases. A separate cache-only runner recomputed only these candidate preferences from DL19/DL20,
loaded all three LLM targets with `allow_new=False`, fit common-fold classical/add-one/all-D0
models, and evaluated complete Copeland rankings. No ranker call, download or confirmation-set
access occurred.

The initial D0-v0 inspection exposed a real conceptual bug: NUMANS treated unrelated numerals in
an anchor sentence as count evidence, allowing Robert Kraft net-worth/team numbers to compete with
“four sons.” It also showed CBP acting on the already-documented under-specified `who is Robert
Gray` query. The v0 artifacts were retained locally. D0-v1 binds count numerals to the requested
noun and composes explicit-list plus unqualified-person abstention into CBP. This was the first of
the maximum two allowed revisions, and new regression tests preserve both corrections.

D0-v1 did not establish a successful final battery, and an independent review then found two
result-changing implementation flaws in it before any commit. First, CBP's numbered-list feature
used optional list punctuation, so ordinary prose numbers (years, quantities) were scored as
boilerplate — this had inflated CBP's coverage and its apparent contribution. Second, NUMANS bound
a count to the requested noun only at sentence granularity, so an unrelated number in the same
sentence could still score. The bootstrap could also emit a degenerate one-query interval, and the
candidate/base merge was exact in only one direction.

D0-v2 (2026-07-13, the second and final allowed revision) corrects all four: CBP requires genuine
list punctuation; NUMANS binds a cardinal within a local number–noun token window with deterministic
singular/plural matching; the agreement bootstrap reports `n_evaluable_queries` and suppresses the
interval below five query clusters; and the evaluator uses an exact bidirectional (outer,
one-to-one, indicator) candidate merge. A frozen-coefficient feature-zero diagnostic was added to
separate each candidate's direct pair effect from the add-one refit, and a source digest plus git
revision are written to result provenance. The v0/v1 mistakes are kept here for integrity; their
artifacts remain under `results/rq4_candidates/d0v0/` and `d0v1/`.

The corrected v2 rerun (all three targets, `0 newly collected` for every ranker) weakened rather
than strengthened the earlier signal, which is the honest outcome: removing the spurious CBP
prose-number activations cut CBP's DL19 coverage from 0.176 to 0.051 and DL20 from 0.167 to 0.075,
and the previously significant positive DL20 nDCG deltas are no longer individually significant.
All-D0 minus classical OOF accuracy lifts all include zero (only the FLAN-large log-loss interval
marginally excludes it); the only significant nDCG delta is a negative FLAN-XL DL19 change; and
all-D0 remains far below the LLM reranker everywhere. NUMANS keeps a positive coefficient in every
fold and target and COMPARE is directionally consistent, but both are far too sparse (1–4 evaluable
queries) to freeze; DEFANS is target-dependent (negative/unstable for Qwen) and CBP is
target-inconsistent (helps Qwen, hurts both FLAN targets) and collection-inconsistent — its qrel
agreement is 37:14 on DL19 but 49:37 (chance) on DL20, matching its LLM agreement. Decision: retain
NUMANS and defer COMPARE to D1 as sparse probes, reject DEFANS and CBP in their current D0 form, and
do not freeze the union. DL19/DL20 remain development data, and the NFCorpus lock remains unopened.

## 14. Answer-adequacy oracle: concept validation and a reranking spinout

On 2026-07-13, having established that the four D0 rules are bad detectors but not why, the concept
behind them was tested directly. The four candidates are cheap proxies for one latent property —
*answer-adequacy*, whether a passage delivers the answer rather than merely sharing the topic. A D3
diagnostic (`experiments/rq4_candidates/adequacy.py`, documented in `docs/phase3-adequacy-oracle.md`)
had the local Qwen oracle rate each (query, document) once on an absolute 0–3 adequacy scale, scored
via the project's first-token-logprob convention softmaxed into an expected value. Rating is
per-document and absolute by design: a pairwise "which is better" prompt would just re-run the
reranking task and prove nothing. All 965 top-10 documents scored with zero degenerate outputs, and
mean adequacy rose monotonically with the human qrel grade (Spearman ρ 0.65 DL19 / 0.72 DL20),
confirming the scale tracks relevance rather than a judge artifact.

The concept validated. The adequacy gap Δ = a(q,d1) − a(q,d2) predicts the cached pairwise
preference not only for Qwen itself (the self-consistency upper bound, ~0.88–0.90 sign agreement)
but across model families — 0.81–0.83 on the FLAN-large and FLAN-XL targets, AUC 0.84–0.87, stable
across DL19 and DL20 with no collection interaction. That cross-model transfer is the honest test
and it settles the earlier ambiguity: the answer-adequacy concept is a strong, transferable,
collection-stable driver of the preferences, and the entire D0 failure is operationalisation — the
regexes are poor detectors of a real property, not detectors of nothing.

The scalar then turned out to be a reranker in its own right, independent of the axiom framing.
Sorting each query's block by a(q,d) descending (first-stage rank breaking ties, unscored tail held
below) and scoring against qrels gives a depth-matched comparison, since the oracle scored exactly
the top-10 all-pairs block that Copeland reranks. At depth-10 adequacy beat BM25 decisively on
nDCG@10 (DL19 0.480 → 0.539, +0.059 [+0.038, +0.081]; DL20 0.494 → 0.550, +0.056 [+0.033, +0.080])
and statistically matched the full PRP-allpair tournament (adequacy − PRP −0.010/−0.006, tied on
DL20), recovering ~98–99% of it at one LLM call per document versus ninety per query for the
order-swapped 45-pair tournament — roughly 9× cheaper.

Because nDCG@10 at depth-10 can never rescue a relevant passage BM25 ranked below rank 10 — and BM25
buries 200 (DL19) / 221 (DL20) relevant documents in ranks 10–19 alone — the pool was widened. Each
query's top-50 was scored (3,819 additional Qwen calls, per-document and resumable; 4,784 documents
cached, zero degenerate outputs) and the deeper block reranked. The gains are large and CI-clear at
every step: nDCG@10 climbs 0.539 → 0.630 → 0.672 on DL19 and 0.550 → 0.619 → 0.671 on DL20 for
depths 10/20/50, i.e. +0.193 [+0.136, +0.250] and +0.177 [+0.116, +0.238] over BM25 at depth-50, far
past the depth-10 pairwise tournament (PRP-allpair stays at depth-10; extending it needs a fresh
tournament per depth). AP moves the same way (DL19 0.291 → 0.342, DL20 0.314 → 0.379).

The predicted coarse-scale saturation is visible but does not erase the win. Measured against an
oracle ceiling (the top-N reranked perfectly by qrel grade: 0.575/0.583, 0.719/0.716, 0.824/0.831 at
depths 10/20/50), the fraction of ceiling adequacy captures falls 94% → 88% → 82% as the four-level
scale must order more documents and top-band ties multiply; most of the lift is 10→20, with 20→50
adding a smaller but real increment. The obvious lever for closing the remaining gap is a finer
rating scale or a lexical/first-stage tiebreak within a top adequacy band. This is a D3 diagnostic
and a development-side effectiveness result, not a deployable cheap axiom (the oracle is an LLM, and
Qwen scoring Qwen is partly circular); it defines the ceiling and the training target for a distilled
detector. All numbers are development-only; the NFCorpus confirmation lock remains unopened.

## 15. Continuous soft-semantics distillation

The first local approximation implemented the proposed relevance, NLI and constraint gates as a
hard 0–3 decision matrix. It failed at depth 100: nDCG@10 was 0.480 → 0.478 on DL19 and 0.494 →
0.483 on DL20, with both paired intervals including zero. The failure was structural rather than a
small threshold miss. The output collapsed to score bands 0/1/3 (no score-2 documents): 2,018/2,113/
74 documents on DL19 and 2,637/2,622/70 on DL20. Most ties therefore reverted to BM25. The NLI
gate was especially invalid as a binary answer detector: 97.3% of documents with cached Qwen
adequacy at least 2.5 fell below its 0.65 entailment threshold. The public
`cross-encoder/nli-deberta-v3-xsmall` checkpoint replaced the unavailable originally proposed
MoritzLaurer identifier.

The repaired D2 experiment retains local scores as continuous features, adds a SQuAD2
null-adjusted extractive-QA span margin, and fits a fixed ridge regressor to cached Qwen expected
adequacy. It is evaluated query-grouped out of fold: each held-out query receives a score from a
model trained only on other DL19/DL20 queries, and qrels enter only after those scores are frozen.
On 7,334 labelled documents, semantic+MRC features improve adequacy fidelity over BM25/length/
coverage controls from Spearman 0.377 to 0.665, MAE 0.847 to 0.653 and R² 0.116 to 0.413. The MRC
margin also orders Qwen's argmax bands monotonically (means −4.88, −2.97, −0.22 and +2.28 for
labels 0–3).

Effectiveness follows: DL19 nDCG@10 rises 0.480 → 0.624, Δ +0.145 [+0.087, +0.204], and DL20
0.494 → 0.565, Δ +0.071 [−0.001, +0.145]. AP rises by +0.065 and +0.052 respectively. This is a
successful development-side distillation result, not evidence of independent generalisation: Qwen
adequacy is the target and DL19/DL20 informed every design decision.

The zero-shot full-development fit does not transfer to SciFact: top-10 nDCG@10 changes 0.684 →
0.655, Δ −0.029 [−0.054, −0.004]. SciFact is exploratory only because it was previously used for
smoke testing. A bounded adaptation check then selected 30 SciFact queries deterministically before
reading held-out qrels, collected Qwen adequacy for their 300 top-10 documents, fit only those labels,
and evaluated the other 270 queries. It improves nDCG@10 0.677 → 0.695, Δ +0.018 [+0.004, +0.032],
and AP 0.625 → 0.650, Δ +0.025 [+0.006, +0.044]. This establishes domain adaptation as a viable
exploratory path, not a confirmation result. NFCorpus remains locked and untouched.
