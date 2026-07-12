# Phase 0 Design — Measurement Foundation and Pilot

Engineering details and commands are in `phase0-implementation.md`. Historical corrections are
in `research-logbook.md`.

## 1. Purpose

Phase 0 establishes that later axiom claims are measurable and reproducible. It fixes the data
flow, pair identity, presentation-order handling, cache contract and metric definitions before
larger experiments.

## 2. Scope and estimands

- **Pilot collection:** `msmarco-passage/trec-dl-2019/judged` (43 queries).
- **Primary pilot pairs:** all 45 unordered pairs among each BM25 top ten; 1,900 pairs after
  available-query filtering, each presented in both orders.
- **Smoke data:** `beir/scifact/test` with a mock ranker, used only for engineering diagnostics;
  it is not external confirmation for later axioms.
- **Fidelity estimand:** agreement with the canonical LLM preference on stable, decisive pairs.
- **Order estimand:** fraction and direction of preferences that change under presentation swap.
- **Conditional cyclicity:** cycles only among triangles whose three canonical edges are decisive.

The query is the inferential unit. Phase 0 numbers are pilot estimates, not confirmatory answers to
RQ1–RQ4.

## 3. Fixed protocol

### 3.1 Pair identity

Unordered pairs use a canonical document-id ordering. Presentation order is stored separately.
This prevents a label such as “A” from being confused with a document identity.

### 3.2 Pairwise scoring

The ranker compares two labelled passages. FLAN-T5 uses label-continuation likelihood under prompt
v0; the OpenAI-compatible Qwen system uses single-token A/B log probabilities under prompt v1 with
thinking disabled. No free-generation parsing is used. Model, prompt version and ordered document
ids are part of the preference-store key.

### 3.3 Canonical verdict

Both presentations are mapped back to canonical document order. If they imply the same document
preference, that preference is decisive. If they disagree, the canonical verdict is a tie and the
pair is flagged order-inconsistent. Later decisive-pair analyses exclude these ties but must report
their prevalence.

### 3.4 Cache contract

Model verdicts are append-only and lookup-before-call. Existing observations are never silently
overwritten. Axiom columns are also additive; corrections produce new documented artifacts rather
than mutating historical evidence without provenance.

### 3.5 Pilot axiom battery

The pilot similarity-free battery is TFC1, TFC3, M-TDC, LNC1, TF-LNC and PROX1–PROX5. Each axiom
returns +1, 0 or −1 for the same canonical pair.

## 4. Measures

- **Coverage:** proportion of sampled pairs on which the axiom is non-neutral.
- **Agreement:** sign agreement among pairs where the axiom is non-neutral and the canonical model
  verdict is decisive. Always accompanied by evaluable count.
- **Position consistency:** proportion of pairs with identical document preference after swapping.
- **Conditional cycle rate:** directed three-cycles divided by triangles decisive on all edges;
  the all-triangle survival rate is reported alongside it.
- **Joint prediction:** query-grouped out-of-fold accuracy and log loss against a majority model.

Order consistency is not reliability. The two prompts differ systematically, and no claim about
aleatoric noise follows from their agreement.

## 5. Exit criteria

Phase 0 is complete when retrieval, pair generation, both ranker backends, append-only storage,
axiom evaluation and query-level analysis run end to end; obvious relevant/irrelevant sanity pairs
are handled consistently under both orders; and pilot outputs can be regenerated from config.

## 6. Completed results

### 6.1 Ranker sanity checks

FLAN-T5-small showed complete position preference and FLAN-T5-base was insufficiently decisive on
the small sanity set. `google/flan-t5-large` and the Qwen system passed and became the two Phase 1
ranker systems. This is a wiring/content sanity check, not an effectiveness evaluation.

### 6.2 Order sensitivity and conditional cycles

On DL19 top-10, position consistency was 0.714 for Qwen and 0.671 for FLAN-T5-large. Their bias
directions differed, making order swap mandatory. Of 5,050 possible triangles, 2,096 (Qwen) and
1,699 (FLAN) survived as decisive on all three edges; 2 and 6 were cyclic. Cycles are therefore
rare conditional on survival, while order sensitivity is substantial.

### 6.3 Valid pilot conclusions

- Strict-precondition axioms had very low coverage on natural passage pairs.
- The initial ten-feature joint model added only about 1–3 accuracy points over its majority
  baseline, motivating corrected and extended Phase 1 measurement.
- Semantic embeddings were deferred from the pilot because the default fastText artifact was
  large; this was a cost choice, not a semantic conclusion.

Corrections to superseded pilot outputs are documented only in `research-logbook.md`; no invalid
pilot cell is used in the active results above.

## 7. Handoff

Phase 1 repeats the corrected measurement over a second query set, adds query-level uncertainty,
extends the lexical battery, tests relaxed preconditions and a scoped WordNet semantic tier, and
checks whether aggregated LLM preferences improve BM25 at the same rerank depth.
