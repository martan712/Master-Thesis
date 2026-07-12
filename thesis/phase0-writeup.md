# Phase 0 — Establishing the Pairwise-Axiom Measurement Pipeline

*Detailed analysis record. The final manuscript consolidates this material in
`foundation-writeup.md`; superseded pilot outputs are documented in
`docs/research-logbook.md`. Citation numbers refer to `docs/literature-overview.md`.*

## 1. Purpose

The pilot operationalised pairwise ranking prompting [2] and classical retrieval constraints
[7, 8] using the available axiomatic tooling [19]. It fixed
how BM25 candidates become canonical document pairs, how each pair is presented in both orders,
how pairwise LLM labels are mapped back to document identity, how verdicts are cached, and how
axiom coverage, agreement, order sensitivity and conditional cycles are computed.

The pilot was not designed to answer whether existing axioms explain LLM rankers generally. Its
scientific output is a reproducible protocol, a scoped first estimate and explicit identification
of measurements requiring correction or larger-sample follow-up.

## 2. Experimental setup

The development collection was the judged TREC DL19 passage task over MS MARCO. BM25 supplied a
candidate pool for each of 43 queries. The primary pilot condition took all 45 unordered pairs
among each query's top ten documents, giving 1,900 pairs after available-query filtering. Each
pair was presented in both possible orders.

Document ids, not A/B labels, define canonical pair orientation. FLAN-T5 used prompt-v0 label-
continuation likelihood; Qwen used a chat prompt with single-token A/B log probabilities and
thinking disabled. Every ordered verdict was written to an append-only store keyed by dataset,
query, ordered ids, model and prompt version.

The pilot battery contained TFC1, TFC3, M-TDC, LNC1, TF-LNC and PROX1–PROX5. The semantic tier was
deferred to Phase 1; this was a pilot cost decision rather than a semantic result.

## 3. Estimands

A canonical pair is decisive only when both presentation orders imply the same preferred
document. Disagreement is flagged as order inconsistency and collapsed to a tie. Therefore the
primary agreement estimand is conditional on stable, decisive pairs.

- Coverage is the fraction of sampled pairs on which an axiom is non-neutral.
- Agreement is the matching-sign fraction among axiom-covered, model-decisive pairs.
- Position consistency is the fraction whose preferred document survives order swap.
- Conditional cyclicity is the cycle fraction among triangles decisive on all three edges.

The query is the inferential unit. The conditional cycle rate does not describe all raw
presentations, and position consistency is not a reliability/noise estimate.

## 4. Ranker sanity results

FLAN-T5-small selected the first position regardless of content on the small order-swap sanity
set. FLAN-T5-base remained insufficiently decisive. FLAN-T5-large and the Qwen system passed and
were retained for Phase 1. Passing establishes that the scoring path responds to content under a
small obvious-pair check; it is not a qrel effectiveness claim.

## 5. Order sensitivity and cycles

On DL19 top-10, position consistency was 0.714 for Qwen and 0.671 for FLAN-T5-large. Bias direction
differed across systems, so order swap could not be replaced by one universal positional
correction.

There were 5,050 possible document triangles. Qwen retained 2,096 triangles decisive on all three
edges and FLAN retained 1,699; 2 and 6 were cyclic. Thus cycles are rare conditional on triangle
survival, while the much larger order-inconsistency exclusion remains scientifically important.

## 6. Valid pilot baseline

Strict-precondition axioms fired rarely on natural passage pairs: TFC3 approximately 0.1%, M-TDC
0.8%, TF-LNC 4.9% and LNC1 6.3% coverage. The initial ten-feature logistic model added only about
1–3 accuracy points over its majority baseline. This justified Phase 1's larger corrected study,
relaxed-precondition diagnostics and additional lexical axioms.

No substantive conclusion relies on the original PROX1/PROX2 pilot cells; the corrected Phase 1
rerun is authoritative. Details of the invalidation are kept only in the research logbook.

## 7. Limitations and handoff

The pilot uses one query set, two retained ranker systems and one depth. It conditions fidelity on
order-stable labels and does not evaluate qrel effectiveness. Phase 1 therefore adds DL20,
query-level intervals, a uniform depth-100 scope arm, corrected proximity implementations, relaxed
preconditions, AND/DIV/LB1, WordNet semantic axioms and depth-matched reranking effectiveness.
