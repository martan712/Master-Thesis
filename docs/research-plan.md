# Research Plan
## Axiomatic analysis and extension of generative-LLM pairwise rerankers

This document defines the current thesis architecture. Detailed engineering records live in
`phase*-implementation.md`; dated changes, invalidated interpretations and audit decisions live
in [`research-logbook.md`](research-logbook.md). Historical reasoning is kept out of the active
protocol so that prospective choices and completed results remain distinguishable.

## 1. Aim and scope

The thesis asks:

> Which interpretable retrieval axioms describe the preferences of generative-LLM pairwise
> rerankers, which important preferences do existing axioms miss, and can new axioms improve an
> axiom-based reranker while retaining a clear relation to the target LLM?

The target ranking protocol follows pairwise ranking prompting [2]; the initial constraint families
come from classical and semantic axiomatic IR [7, 8, 10, 11] and are implemented through the
available experimental tooling [19]. Citation numbers refer to `literature-overview.md`.

The empirical scope is English passage reranking. The development domain is MS MARCO passage
ranking with the judged TREC DL19 and DL20 query sets. BM25 supplies candidate pools and the
primary condition reranks its top ten. This conditional population is deliberate: conclusions do
not automatically extend to arbitrary corpus pairs, deeper pools, listwise rankers, languages or
domains.

RQ4—new-axiom development and confirmation—is the main contribution. Phases 0–2 provide the
measurement and diagnostic foundation it requires. RQ5 is a future downstream efficiency study;
it begins only when retained features are shown to decompose into per-document scores.

## 2. Research questions

- **RQ1 — lexical baseline.** Which classical lexical axioms agree with stable pairwise LLM
  preferences, at what coverage, and how consistently across the tested query sets and ranker
  systems?
- **RQ2 — semantic baseline.** Does the tested semantic-axiom operationalisation add held-out
  predictive value beyond the lexical battery?
- **RQ3 — diagnostic model.** How much incremental predictive value does the combined existing
  battery provide, and which errors or gaps motivate candidate axioms without being mistaken for
  confirmed mechanisms?
- **RQ4 — main contribution.** Which theory-, literature- and error-informed candidate axioms
  provide held-out explanatory fidelity, retrieval effectiveness, or both when added to a fitted
  axiom reranker?
- **RQ5 — future pointwise efficiency.** Which retained RQ4 features genuinely decompose into a
  score evaluated once per document, and what effectiveness–efficiency trade-off follows?
- **RQ6 — optional cascade.** Can a calibrated uncertainty signal route only selected comparisons
  to the LLM without unacceptable effectiveness loss?

## 3. Common experimental contract

### 3.1 Unit of inference and dependence

The query is the inferential unit. Pairs within a query are dependent. Model selection and
evaluation use query-disjoint folds; uncertainty is estimated by resampling queries. No pair-level
confidence interval is interpreted as if pairs were independent.

### 3.2 Pairs and canonical labels

The primary development condition contains all unordered pairs among BM25's top ten documents.
Each pair is presented in both orders. The primary fidelity estimand uses stable, decisive pairs:
the two presentations must imply the same document preference after mapping labels back to
canonical document order. Order-inconsistent pairs are a separate outcome, not stochastic noise.

A uniform sample from the BM25 depth-100 pool maps how results vary with first-stage rank gap; it
is a scope analysis, not a pass/fail pipeline validation.

### 3.3 Two claim axes

Fidelity and effectiveness answer different questions:

- **Explanatory fidelity:** does an axiom model better reproduce the target LLM? Primary measures
  are query-grouped out-of-fold pairwise log loss and a prespecified ranking-fidelity measure.
- **Retrieval effectiveness:** does it rank relevant documents better? nDCG@10 is primary; MAP is
  secondary. BM25, the LLM reranker, the fitted classical-axiom reranker and the fitted extended
  reranker are compared at the same depth.

An axiom may be explanatory without improving qrels, effective without explaining the LLM, both,
or neither. These outcomes are never collapsed into a single success label.

### 3.4 Reporting requirements

Every result states the dataset, ranker system, prompt/scoring protocol, pool depth, pair filter,
denominator and whether it is developmental or confirmatory. Agreement is reported with coverage
and evaluable count. Accuracy is reported against its majority baseline. Normalised log-loss gain
is a predictive statistic, not a percentage of mechanism or behaviour explained.

## 4. Phase architecture

### Phase 0 — measurement foundation

Build and validate retrieval, pair construction, order-swapped preference collection, append-only
caching, axiom evaluation and query-level analysis. Pilot the complete chain and record invalid or
unstable measurements explicitly.

### Phase 1 — corrected lexical and semantic baselines

Measure RQ1–RQ2 on DL19/DL20, top-10 and uniform-depth controls, with query-bootstrap intervals.
Verify that aggregated LLM preferences improve BM25 at the chosen rerank depth. Establish the
classical fitted baseline that every RQ4 candidate must beat.

### Phase 2 — compact predictive diagnostics

Fit the existing battery under query-disjoint CV, report its incremental prediction, preserve
order sensitivity as a separate estimand, and produce an evidence-graded candidate handoff. Phase
2 cannot discover a mechanism merely because a residual covariate is predictive. It is a compact
baseline-diagnostics subsection in the final thesis, not a standalone centerpiece or a gate that
can redirect the thesis to RQ5.

### Phase 3 — RQ4 iterative axiom development and confirmation

1. Define candidate families from retrieval theory, reviewed literature and documented error
   analysis.
2. Encode each candidate as a deterministic same-pair preference with explicit preconditions,
   direction, neutral case and cost.
3. Fit classical and extended pairwise axiom models in nested query-disjoint folds.
4. Compare BM25, LLM, fitted classical and fitted extended rankings on fidelity and effectiveness.
5. Analyse residual gaps and permit at most two documented revisions per candidate family.
6. Freeze candidate membership, thresholds, direction, features and fitting choices.
7. Run family ablations and locked external confirmation without further tuning.

All attempted variants remain in a machine-readable ledger. The final extended battery has one
prespecified model-level confirmation test per claim axis; individual family tests are secondary
and multiplicity-adjusted.

### Phase 4 — RQ5, only after RQ4

The fitted pairwise model used in RQ4 predicts every pair and still requires aggregation; it is not
a linear-time surrogate. RQ5 requires an explicit per-document decomposition, one evaluation per
document, and measured latency/model-call/FLOP savings.

## 5. Development and confirmation data

DL19/DL20 and all ranker systems already inspected are development evidence. Query-disjoint CV on
them estimates internal generalisation but does not make retrospectively proposed candidates
confirmatory.

The proposed external dataset is `beir/nfcorpus/test`; the collection choice remains open until it
is frozen without result inspection. SciFact is ineligible because it already served smoke
diagnostics. Before freezing RQ4, only a mechanical access/schema check is allowed—no qrel or
model-result inspection. A new ranker system is a second confirmation axis if affordable;
otherwise claims are explicitly cross-dataset, not cross-model.

## 6. Deliverables

- Reproducible preference and axiom-evaluation pipeline.
- Corrected lexical and WordNet-semantic baseline profiles.
- Compact RQ3 diagnostic baseline with explicit estimand limitations.
- RQ4 candidate registry and complete revision ledger.
- Fitted BM25/LLM/classical/extended comparison tables for nDCG@10, MAP and fidelity.
- Candidate-family ablations and locked external confirmation.
- A thesis result whether candidates succeed, split across the four fidelity/effectiveness
  outcomes, or establish a tested boundary.
- Optional RQ5 pointwise efficiency study after decomposability is demonstrated.

## 7. Principal validity risks

- **Selection and multiplicity:** bounded revisions, complete ledger, frozen confirmation
  manifest and adjusted secondary family tests.
- **Order effects:** reported separately; identical-prompt repeats are required to estimate
  run-to-run randomness.
- **Development reuse:** DL19/DL20 findings are developmental regardless of CV when they informed
  candidate definitions.
- **System confounding:** Qwen and FLAN comparisons change model, prompt and scoring; they are
  robustness checks, not isolated architecture effects.
- **Semantic scope:** the completed semantic null is WordNet-specific.
- **Pool depth:** absolute effectiveness is depth-dependent; every comparison is depth-matched and
  accompanied by BM25 and oracle context.
