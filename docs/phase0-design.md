# Phase 0 Design — Setup and Pilot (scientific design and results)

> Companion document: the engineering record — module architecture, preference-store
> schema, work breakdown, runbook and operational risks — lives in
> `phase0-implementation.md`. This document holds the scientific design and the results.

Phase 0 of the research plan (§5, weeks 1–4) stands up the data, the ranker harness, the
axiom toolkit and the preference-logging pipeline, and pilots the whole chain on TREC
DL19. **Milestone: a cached preference dataset and a working agreement pipeline.**

> **Status: complete (2026-07-11).** All §2 exit criteria hold; outcomes, numbers and
> the decisions feeding Phase 1 are recorded in §7.

## 1. Purpose and milestone

Phase 0 exists to make the rest of the thesis measurable: to fix the collections, the
sampling, the ranker prompt, the axiom battery and — crucially — the agreement,
consistency and transitivity definitions that every later phase reports against, and to
produce the first real preference numbers on DL19 so that the open questions the research
plan defers to a pilot (plan §8) can start being answered. The concrete milestone is a
cached preference dataset and a working agreement pipeline.

## 2. Objectives and exit criteria

Phase 0 is done when all of the following hold:

1. **Data.** DL19 topics and the MS MARCO passage corpus are accessible, with a BM25
   first-stage pool retrievable per query.
2. **Harness.** A pairwise-ranker interface exists with (a) a deterministic mock backend
   for pipeline testing and (b) a backend implementing a PRP-style binary prompt, both
   order-swappable.
3. **Preference logging.** Every model verdict is written once to an append-only store
   and is never recomputed (lookup-before-call).
4. **Axioms.** The lexical battery (and STMC where feasible) runs over sampled pairs and
   yields a per-axiom preference for each pair.
5. **Agreement pipeline.** One command produces, from a config: the per-axiom agreement
   profile, position-consistency statistics, and a non-transitivity estimate.
6. **Pilot numbers exist** for DL19 (even on a subset), giving first estimates of the
   quantities the open questions in plan §8 depend on.

## 3. Scientific design decisions

### 3.1 Collections

Pilot on `msmarco-passage/trec-dl-2019/judged` (43 judged queries). End-to-end smoke
tests use `beir/scifact/test` (5k docs, seconds to index locally) so the pipeline is
testable without MS MARCO-scale downloads. (vaswani would be smaller, but its host
`ir.dcs.gla.ac.uk` is unreachable from this network.)

### 3.2 Pair sampling

Two strategies, both seeded:

- `top_k_all_pairs` (pilot default, k=10): all 45 unordered pairs among the BM25 top-10
  per query. Chosen because complete triangles are required to measure non-transitivity,
  and top-10 is where reranking decisions actually matter. 43 queries × 45 pairs × 2
  presentation orders ≈ 3 870 model calls — feasible on CPU with a small model.
- `uniform` (per_query=n): n random unordered pairs from the depth-100 pool, for broader
  coverage in later phases.

Pairs are stored canonically as (doc_id_1 < doc_id_2); presentation order is a separate,
explicit dimension.

### 3.3 Ranker and prompt design

PRP-style binary choice ("Passage A" vs "Passage B"), scored by comparing the model's
likelihood of the two label continuations rather than free generation — deterministic, no
sampling temperature, and yields `prob_a` as a confidence signal (needed by RQ6). The
backend supports both seq2seq (Flan-T5, as in the PRP paper) and causal (Qwen/Llama-style)
models. Order swap is treated as an explicit dimension of the design, not an
implementation afterthought: every pair is presented in both orders and disagreement
across orders is a first-class signal.

The *pilot* model is `google/flan-t5-small` for wiring and
`flan-t5-base`/`Qwen2.5-1.5B-Instruct` as candidates; **the definitive model choice is an
explicit Phase 0 outcome**, not an input. The prompt template is versioned (`v0` for the
seq2seq PRP prompt, `v1` for the chat-style prompt) and stored with every verdict.

### 3.4 Axiom battery

As available in `ir_axioms` 1.1.2 (note LNC2 is not in this release): lexical `TFC1, TFC3,
M_TDC, LNC1, TF_LNC, PROX1–PROX5` plus `REG, ANTI_REG, AND, DIV, LB1`; semantic `STMC1,
STMC2`. The semantic axioms may require large embedding downloads (fastText vectors) — the
pilot measures that cost and may defer them to Phase 1 (RQ2) if prohibitive; the battery
is toggleable.

### 3.5 Agreement, consistency and transitivity definitions

Fixed now so later phases are comparable:

- *Model verdict per unordered pair*: derived from the two presentations; if they
  disagree, the pair is *position-inconsistent* and its verdict is `tie`.
- *Agreement of axiom X*: over pairs where X is non-neutral **and** the model verdict is
  decisive, the fraction where signs match. Reported alongside X's *coverage* (fraction of
  pairs where X is non-neutral).
- *Position-consistency rate*: fraction of pairs with identical verdicts in both orders.
- *Non-transitivity rate*: fraction of cyclic triangles among complete decisive triangles
  (available under `top_k_all_pairs` sampling).
- Primary metric of the pilot: **fidelity** (we characterise the model, not the qrels).

## 4. Pilot protocol

Four conceptual steps:

1. **Smoke test** (automated): SciFact + mock ranker + lexical battery, ~5 queries,
   end-to-end. Runs in CI-time (~minutes), no big downloads.
2. **Backend validation**: flan-t5-small on a handful of real pairs — verifies the scoring
   path, latency measurement and store round-trip.
3. **DL19 pilot run**: prebuilt MS MARCO index, top-10 all-pairs, both orders, pilot
   model; collects the first real preference dataset (~4k verdicts).
4. **First analysis**: agreement profile, consistency and transitivity numbers; decide
   model + sampling scale for Phase 1, and whether STMC axioms stay in the battery.

## 5. Open questions Phase 0 must start answering (plan §8)

- How position-consistent is the model? (→ order_swap stays mandatory or can be sampled)
- How often are axioms neutral (coverage)? (→ battery composition for RQ1)
- Are axiom-covered pairs the ones the model is confident on? (`prob_a` vs coverage)
- Which similarity source for semantic axioms is practical? (fastText download cost)
- Is a small open model's verdict quality sufficient, or does the pilot force a larger
  model / GPU machine for the main study?

## 6. Scientific risks

- **Strict axiom preconditions.** In the SciFact smoke run TFC3/M-TDC/TF-LNC had zero
  coverage and LNC1 2% — their equal-length-style preconditions rarely hold on natural
  pairs. Expected from the literature, but if DL19 looks similar, Phase 1 must consider
  relaxed preconditions (margin parameters) or accept low-coverage axioms.
- **Small pairwise models are position-biased.** `flan-t5-small` picked "Passage A" in
  both presentation orders on an obvious sanity pair — consistent with PRP's finding that
  small Flan-T5 variants cannot rank pairwise. The pilot model must pass an order-swapped
  sanity check before the full run.
- **Prebuilt index mismatch.** The tokenisation of the prebuilt index (stemming,
  stopwords) may differ from what the axioms assume, degrading axiom preferences: this is
  acceptable for the pilot but must be revisited if axiom preferences look degenerate.

## 7. Outcomes and decisions (2026-07-11)

Phase 0 is complete: all §2 exit criteria hold. 43 DL19 judged queries, BM25 top-10
all-pairs → 1,900 canonical pairs, both presentation orders → 3,800 verdicts per model,
collected once and cached. Metrics per model are stored per ranker.

### 7.1 Sanity gates (4-way order-swap)

| model | result | note |
|---|---|---|
| flan-t5-small | FAIL 0/4 correct-when-swapped | answered "Passage A" in all four presentations |
| flan-t5-base | FAIL 3/4 | residual A-bias; probabilities ≈ 0.5 throughout |
| flan-t5-large | PASS 4/4 | decisive (prob ≥ 0.998) |
| Qwen3.6-35B-A3B-AWQ | PASS 4/4 | via vLLM at localhost:9086; `enable_thinking: false` verified — bare "A"/"B" answers, no reasoning preamble |

Consistent with PRP (Qin et al.): within Flan-T5 only *large* can rank pairwise. (The gate
script and procedure are documented in `phase0-implementation.md` §5.2.)

### 7.2 Pilot numbers (DL19, top-10 all-pairs, prompt v0/v1)

| | Qwen3.6-35B-A3B-AWQ | flan-t5-large |
|---|---|---|
| position consistency | 0.714 | 0.671 |
| raw verdict split (a/b/tie) | 1448 / 2276 / 76 — **B-biased** | 2484 / 1316 / 0 — **A-biased** |
| complete decisive triangles (of 5,050 sampled) | 2,096 (41.5%) | 1,699 (33.6%) |
| non-transitivity (cyclic/complete) | 2/2096 = 0.001 | 6/1699 = 0.004 |
| mean latency per presentation | 386 ms (API, 1 token) | 1138 ms (CPU, 2 forward passes) |
| conf.–coverage correlation | 0.004 | −0.131 |

Per-axiom agreement over evaluable pairs (axiom non-neutral ∧ model decisive), with
coverage over the 1,900 pairs:

| axiom | coverage | Qwen agr. | flan-large agr. |
|---|---|---|---|
| TFC1 | 0.757 | 0.478 | 0.466 |
| PROX1 | 0.395 | 0.597 | 0.622 |
| PROX2 | 0.498 | 0.354 | 0.328 |
| PROX3 | 0.128 | 0.675 | 0.707 |
| PROX4 | 0.205 | 0.639 | 0.595 |
| PROX5 | 0.242 | 0.560 | 0.536 |
| LNC1 | 0.063 | 0.639 | 0.539 |
| TF-LNC | 0.049 | 0.547 | 0.576 |
| M-TDC | 0.008 | 0.833 | 0.889 |
| TFC3 | 0.001 | 0.500 | 0.500 |

### 7.3 Findings against the §5 open questions

- **Position consistency is the dominant noise source** (67–71%): order-swap collection
  stays mandatory for every model and every phase. Notably the two models are biased in
  *opposite directions* (Qwen toward the second-shown passage, flan-t5-large toward the
  first), so position bias is not a universal artefact of the prompt template.
- **Decisive preferences are almost perfectly transitive** (≤0.4% cyclic triangles) once
  position-inconsistent pairs are treated as ties. Non-transitivity is not a practical
  obstacle for rank aggregation; complete triangles can be retained at low cost. (Both
  models see the same 5,050 sampled triangles; a triangle is evaluable for cyclicity
  only if the model is decisive on all three edges, so the complete-triangle count is
  itself a consistency metric — the more a model flip-flops on order, the fewer of its
  triangles survive.)
- **The agreement *profile* replicates across architectures.** A 0.8B seq2seq and a 35B
  MoE agree on which axioms track them: TFC1 *below* chance (~0.47) and PROX2 anti-
  agreeing (~0.33–0.35) for both, PROX3 highest (~0.68–0.71). Early evidence that the
  axiomatic account characterises LLM pairwise ranking generally, not one checkpoint —
  and that these rankers are *not* term-frequency rankers (RQ1 has signal).
- **Strict-precondition axioms are as dead on DL19 as on SciFact** (TFC3 0.1%, M-TDC
  0.8%, TF-LNC 4.9%, LNC1 6.3% coverage): Phase 1 must add relaxed-precondition
  (margin-parameterised) variants or accept them as low-coverage curiosities. The
  usable lexical core is TFC1 + PROX1–PROX5 (13–76% coverage); 94.6% of pairs have at
  least one non-neutral axiom.
- **Jointly, the lexical battery explains almost none of the top-10 decisions.**
  Predicting each model's decisive verdicts from all ten axiom preferences at once:
  a majority vote of the axioms scores *below* the majority-class baseline (0.488 vs
  0.573 for Qwen; 0.470 vs 0.600 for flan-t5-large), and a cross-validated logistic
  model — free to invert anti-agreeing axioms — reaches only 0.601/0.611, i.e. ~1–3
  points over the base rate. The fitted structure replicates across both models
  (M-TDC strongly positive, PROX2 negative), so the battery fails in a stable,
  model-independent way — weak signal, not noise.
- **Top-10 pairs are the decision-relevant condition, and that is where the axioms
  fail.** Both pool members of a top-10 pair are already lexically strong, so classical
  axioms mostly re-explain what BM25 decided; the marginal value the LLM adds over
  BM25 — the reordering one deploys a reranker *for* — sits almost entirely in the
  residual. Low agreement on easy-to-explain pairs would be a battery bug; low
  agreement on top-10 pairs is the phenomenon itself.
- **Model confidence and axiom coverage are orthogonal** (corr 0.004 / −0.131): axioms
  do not simply fire on the pairs the model is sure about, so coverage-weighted fidelity
  and confidence-stratified analyses (RQ6) measure genuinely different things.
- **Semantic axioms (STMC1/2) deferred to Phase 1**: their similarity backend
  (`facebook/fasttext-en-vectors`) is a 7.24 GB download (measured, not fetched);
  ir_axioms also offers a WordNet-based term similarity worth evaluating first.

### 7.4 Decisions

1. **Primary ranker: Qwen3.6-35B-A3B-AWQ** (OpenAI backend, prompt v1, logprob-scored,
   thinking disabled). **Contrast model: flan-t5-large** (HF backend, prompt v0) — it
   passed the gate, its profile is sane, and its agreement pattern independently
   replicates the primary's. small/base are kept in the store only as position-bias
   exhibits.
2. **Phase 1 sampling scale**: at 386 ms/call a full DL19 top-10 all-pairs double-order
   sweep costs ~25 min on the local endpoint. Budget allows DL20 as second collection
   and/or k=20 pools (190 pairs/query ≈ 16.3k presentations ≈ 1.8 h/model). Order swap
   (×2) stays in all budgets.
3. **STMC axioms do not enter the Phase 0/1 battery yet** — decide in Phase 1 (RQ2)
   after comparing WordNet similarity vs the 7.24 GB fastText download.
4. **Battery for RQ1** starts from TFC1 + PROX1–5 (+LNC1/TF-LNC/M-TDC as low-coverage
   extras); relaxed preconditions are a Phase 1 work item.
5. **Top-10 all-pairs stays the primary condition throughout; `uniform` depth-100
   sampling is a validity control, not a rescue.** The expected gradient — high axiom
   agreement on wide-gap pairs, near-chance on top-10 — would demonstrate the pipeline
   is sound and the top-10 failure is a property of LLM reranking, not an artefact
   (agreement vs. rank/score gap is the candidate signature figure for RQ1). This
   shifts weight to the second half of the thesis question: characterising the
   residual (RQ3) and finding new axioms that discriminate between lexically close
   documents (RQ4) take priority over tuning the classical battery; relaxed
   preconditions matter chiefly because coverage on hard pairs is what limits the
   axioms that do show signal (M-TDC). The joint-fit analysis above graduates into the
   RQ1 lexical-agreement experiment as its first script.
