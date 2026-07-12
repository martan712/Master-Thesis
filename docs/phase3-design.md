# Phase 3 Design — RQ4 New-Axiom Development and Confirmation

RQ4 is the main thesis contribution. Engineering status is in `phase3-implementation.md`; dated
development history is in `research-logbook.md`. This document separates the active protocol from
completed internal evidence. Candidate definitions extend classical and semantic axiomatic IR
[7, 8, 10, 11]; the target pairwise protocol follows PRP [2]. Citation numbers refer to
`literature-overview.md`.

## 1. Research objective

RQ4 asks which explicit retrieval constraints add:

1. **explanatory fidelity** to a target LLM pairwise reranker;
2. **qrel effectiveness** to a fitted axiom reranker; or
3. both.

The contribution is the candidate-development and confirmation framework plus its retained,
rejected and null axioms. A null final battery remains a result about the tested axiomatic scope.

## 2. Starting baselines

- BM25 at the same candidate depth.
- Cached depth-matched LLM Copeland rerankings.
- A fitted pairwise classical-axiom model: L2 logistic prediction for every top-10 pair followed
  by Copeland aggregation.
- An untuned axiom majority vote as a transparency baseline only.

The fitted pairwise model belongs to RQ4. It still evaluates every pair and is not the RQ5
pointwise efficiency scorer.

## 3. Candidate sources and registry

Candidates may be motivated by retrieval theory, reviewed literature, qualitative error analysis
or prior diagnostics. Each registry entry fixes:

- name and family;
- mathematical or algorithmic definition;
- same-pair input and signed output `{+1,0,−1}`;
- precondition and neutral case;
- intended direction and failure cases;
- Tier A/Tier B status and computational cost;
- provenance and revision number.

### 3.1 Completed lexical-adjacent probes

- **VERB:** prefer the longer document when distinct query-term coverage is equal.
- **VERB_R:** relaxed coverage precondition for VERB.
- **QCOV:** prefer broader distinct query-term coverage.

These were proposed after inspecting DL19/DL20 and are retrospective development candidates.

### 3.2 Qualitative reversal study

The development-only [`phase3-qualitative-casebook.md`](phase3-qualitative-casebook.md) examines
thirteen purposively sampled pairs from 583 relevance-improving Qwen reversals. It adds three
retrospective candidate families to the registry. They are hypotheses, not evidence that the
corresponding axioms work:

- **QARA — Query–Answer Relation Alignment:** prefer evidence that instantiates the relation or
  answer type requested by the query, rather than merely sharing its topic;
- **CBP — Content-Bearing Passage preference:** prefer complete answer-bearing prose over
  navigation, search-term, citation or advertising boilerplate;
- **QCS — Query-Constraint Satisfaction:** prefer passages that bind important entity,
  geographic, temporal and role qualifiers in one local proposition.

Typed Specificity and Completeness (TSC) is recorded as a refinement of the existing Specificity
and Aspect Completeness family rather than a wholly separate construct.

### 3.3 Candidate menu

| family | candidate | tier | operationalisation |
|---|---|---|---|
| coverage | Semantic Intent Coverage | A | aggregate maximum query-aspect-to-sentence embedding similarity |
| coverage | Aspect Completeness | B | judge-based query-aspect decomposition and coverage comparison |
| relation alignment | QARA | A | query-intent/answer-type classification plus sentence-level evidence for the requested relation |
| content quality | CBP | A | complete-sentence evidence minus navigation, duplicate-fragment and boilerplate signals |
| constraint satisfaction | QCS | A | local binding of query entities, roles, locations, times and other qualifiers |
| focus | Semantic Focus / Distraction | A | distribution of sentence-to-query similarity; penalise off-topic mass |
| focus | Redundancy Penalty | A | intra-document sentence-similarity redundancy |
| specificity | Specificity | A | IDF-weighted content density, named entities and numerals |
| specificity | Typed Specificity and Completeness | A | question-conditioned compatible values, classes, units or enumerated items |
| explanation | Causal/Explanatory Adequacy | B | judge comparison restricted to why/how information needs |
| directness | Answer Localisation | A | normalised position of peak query-relevant sentence |
| grounding | Evidence Support | A | prespecified citation, quotation and numerical-evidence signals |
| grounding | Factual Faithfulness | B | judge-based support comparison |
| consistency | Factual Consistency | B | judge-based internal-contradiction comparison |
| entity alignment | Entity Relevance | A | query/document entity and entity-type alignment |

**Tier A** candidates are deterministic, cheap proxies eligible for the fitted reranker. **Tier B**
candidates are expensive judge-dependent diagnostics and never final deployable axioms. A Tier-B
score is not a literal oracle or guaranteed upper bound. A/B gaps compare operationalisations only
under the assumption that both validly measure the intended construct.

## 4. Data roles

### 4.1 Development

DL19, DL20 and all already inspected Qwen/FLAN systems are development evidence. Query-disjoint
CV estimates internal generalisation but does not make retrospectively proposed candidates
confirmatory.

### 4.2 Locked confirmation

The proposed external dataset is `beir/nfcorpus/test`; the choice remains open until frozen without
result inspection. SciFact is ineligible because it already served smoke diagnostics. Before the
development manifest is frozen, only mechanical access and schema checks are allowed. Candidate membership, direction, thresholds, feature extraction,
regularisation grid and aggregation are then locked. No confirmation result may trigger another
revision.

A not-previously-analysed ranker system is added if affordable. Without it, confirmation supports
cross-dataset but not cross-model generalisation.

## 5. Fitted reranking protocol

### 5.1 Nested query-disjoint fitting

For each target LLM, fit classical and extended L2 pairwise models. Inner query-grouped folds
select regularisation using pairwise log loss; outer query-grouped folds estimate pair predictions
and rankings. Every top-10 pair is scored, then aggregated by the same Copeland rule. The target
LLM's tie locations are not used to choose which pairs the axiom model scores.

### 5.2 Depth-matched comparisons

For every query, write one table containing:

- BM25;
- LLM reranker;
- fitted classical-axiom reranker;
- fitted classical plus each candidate family;
- fitted full extended reranker;
- depth-matched qrel oracle.

Report absolute nDCG@10, MAP, paired query deltas, bootstrap intervals, win/tie/loss counts and
fraction of available BM25-to-oracle improvement. Absolute values are never omitted merely because
the depth-10 ceiling is low.

### 5.3 Fidelity measures

On stable, decisive LLM-labelled pairs, report OOF log-loss lift and accuracy lift relative to the
classical model. At ranking level, use one prespecified correlation measure. Coverage and order-
consistency strata are reported descriptively.

## 6. Bounded development loop

1. Implement and synthetically test sign, neutrality and determinism.
2. Run pairwise fidelity and ranking-effectiveness comparisons.
3. Analyse gaps by query, rank gap, candidate coverage and family.
4. Revise or reject the definition.

At most two documented revisions are allowed per family. Every attempted variant remains in a
machine-readable ledger. Development selection may use outer-fold point estimates, but inferential
claims are reserved for locked confirmation.

## 7. Ablation and multiplicity

After development, evaluate classical only, classical+each retained family, all retained families,
and leave-one-family-out. The final extended battery has one prespecified model-level confirmation
test per claim axis. Individual family tests are secondary and Holm-adjusted. No post-confirmation
family selection is allowed.

## 8. Two-dimensional decision framework

The explanatory claim and effectiveness claim are evaluated separately:

| held-out LLM fidelity | held-out nDCG@10 | interpretation |
|---|---|---|
| improves | improves | explanatory and effective axiom battery |
| improves | does not improve | explanatory; no effectiveness claim |
| does not improve | improves | effective reranking features; no LLM-explanation claim |
| does not improve | does not improve | null/rejected extension; tested boundary |

A Tier-A candidate may enter the frozen confirmation battery if it is deterministic, exceeds the
prespecified coverage threshold, has a directionally coherent coefficient and has a positive
development point estimate on at least one declared axis. Its intended claim axis is frozen before
confirmation; one axis cannot rescue or imply the other.

## 9. Completed internal evidence — Increment 1

The current coherent RQ4 runner used cached data and made zero model calls. These results are
retrospective/internal because VERB and QCOV were derived from the same query sets.

### 9.1 Pairwise fidelity

Classical→classical+VERB+QCOV OOF accuracy lift:

| target | lift [95% query-bootstrap CI] |
|---|---|
| Qwen | +0.0123 [−0.0059,+0.0299] |
| FLAN-T5-large | −0.0027 [−0.0154,+0.0089] |
| FLAN-T5-XL | +0.0051 [−0.0063,+0.0166] |

OOF log-loss lift:

| target | lift [95% query-bootstrap CI] |
|---|---|
| Qwen | +0.0126 [+0.0003,+0.0247] |
| FLAN-T5-large | +0.0118 [+0.0007,+0.0229] |
| FLAN-T5-XL | +0.0075 [−0.0051,+0.0187] |

This is internal explanatory evidence for Qwen and FLAN-large on log loss, not held-out
confirmation and not an effectiveness result.

### 9.2 Fitted nDCG@10

| target | query set | fitted classical | fitted + VERB+QCOV |
|---|---|---:|---:|
| Qwen | DL19 | 0.5028 | 0.5024 |
| Qwen | DL20 | 0.4845 | 0.4882 |
| FLAN-T5-large | DL19 | 0.5045 | 0.5036 |
| FLAN-T5-large | DL20 | 0.4900 | 0.4923 |
| FLAN-T5-XL | DL19 | 0.5084 | 0.5063 |
| FLAN-T5-XL | DL20 | 0.4908 | 0.4860 |

No add-one or nested new-axiom effectiveness interval excludes zero. VERB/QCOV therefore have no
confirmed internal effectiveness gain in the coherent fitted evaluation.

## 10. Required outputs

- candidate registry and complete revision ledger;
- synthetic and determinism tests;
- nested-CV per-query prediction/ranking table;
- BM25/LLM/classical/extended effectiveness and fidelity tables;
- gap and coverage analyses;
- add-family and leave-family-out ablations;
- frozen development manifest;
- untouched external confirmation results;
- explicit four-outcome interpretation.
