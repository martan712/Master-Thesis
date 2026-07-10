# Phase 0 Plan — Setup and Pilot

Phase 0 of the research plan (§5, weeks 1–4): stand up the data, the ranker harness, the
axiom toolkit and the preference-logging pipeline, and pilot the whole chain on TREC DL19.
**Milestone: a cached preference dataset and a working agreement pipeline.** This document
is the detailed, implementable version of that phase.

## 1. Objectives and exit criteria

Phase 0 is done when all of the following hold:

1. **Data.** DL19 topics and the MS MARCO passage corpus are accessible through
   `ir_datasets`/PyTerrier, with a BM25 first-stage pool retrievable per query.
2. **Harness.** A pairwise-ranker interface exists with (a) a deterministic mock backend
   for pipeline testing and (b) a Hugging Face backend implementing a PRP-style binary
   prompt, both order-swappable.
3. **Preference logging.** Every model verdict is written once to the append-only Parquet
   store in `data/preferences/` and is never recomputed (lookup-before-call).
4. **Axioms.** The lexical battery (and STMC where feasible) runs over sampled pairs via
   `ir_axioms` and yields a per-axiom preference for each pair.
5. **Agreement pipeline.** One command produces, from a config: the per-axiom agreement
   profile, position-consistency statistics, and a non-transitivity estimate, written
   under `results/p0_pilot/`.
6. **Pilot numbers exist** for DL19 (even on a subset), giving first estimates of the
   quantities the open questions in plan §8 depend on.

## 2. Architecture

Seven components, all in `src/axiomrank/`, orchestrated by `experiments/p0_pilot/run.py`:

```
config.py        YAML → typed config (dataset, sampling, ranker, axioms, output)
datasets.py      topics + BM25 pooling (PyTerrier, prebuilt Terrier indices), doc text
pairs.py         pair sampling from the pool (canonical unordered pairs)
rankers/base.py  PairwiseRanker interface + PairVerdict
rankers/mock.py  deterministic hash-scored mock (transitive by construction)
rankers/hf.py    transformers backend, PRP prompt v0, label-likelihood scoring
preferences.py   append-only Parquet store, keyed, dedup on read, lookup before call
axioms.py        ir_axioms battery wrapper (AxiomaticPreferences transformer)
agreement.py     canonical verdicts, per-axiom agreement, consistency, transitivity
```

Data flow: `pool → pairs → (store lookup → LLM for misses → store append) → axiom
preferences → agreement report`. Every intermediate is cached under `data/processed/` so
re-runs are incremental; only `results/` outputs are overwritten.

## 3. Design decisions

**Collections.** Pilot on `msmarco-passage/trec-dl-2019/judged` (43 judged queries).
End-to-end smoke tests use `beir/scifact/test` (5k docs, seconds to index locally) so the
pipeline is testable without MS MARCO-scale downloads. (vaswani would be smaller, but its
host `ir.dcs.gla.ac.uk` is unreachable from this network.)

**Index.** Use PyTerrier's prebuilt `terrier_stemmed` MS MARCO passage index (~2 GB
download, one-time) rather than indexing 8.8M passages locally (hours). Recorded in the
config; swapping to a locally built index is a config change.

**Pair sampling.** Two strategies, both seeded:
- `top_k_all_pairs` (pilot default, k=10): all 45 unordered pairs among the BM25 top-10
  per query. Chosen because complete triangles are required to measure non-transitivity,
  and top-10 is where reranking decisions actually matter. 43 queries × 45 pairs × 2
  presentation orders ≈ 3 870 model calls — feasible on CPU with a small model.
- `uniform` (per_query=n): n random unordered pairs from the depth-100 pool, for broader
  coverage in later phases.
Pairs are stored canonically as (doc_id_1 < doc_id_2); presentation order is a separate,
explicit dimension.

**Ranker.** PRP-style binary choice ("Passage A" vs "Passage B"), scored by comparing the
model's likelihood of the two label continuations rather than free generation —
deterministic, no sampling temperature, and yields `prob_a` as a confidence signal (needed
by RQ6). Backend supports both seq2seq (Flan-T5, as in the PRP paper) and causal
(Qwen/Llama-style) models. The *pilot* model is `google/flan-t5-small` for wiring and
`flan-t5-base`/`Qwen2.5-1.5B-Instruct` as candidates; **the definitive model choice is an
explicit Phase 0 outcome**, not an input. Prompt template is versioned (`prompt_version:
v0`) and stored with every verdict.

**Preference store schema** (one row per *presentation*, i.e. per ordered pair):

| column | meaning |
|---|---|
| `dataset, query_id` | collection and topic |
| `doc_id_a, doc_id_b` | documents *in presented order* (A shown first) |
| `model, prompt_version` | ranker identity |
| `verdict` | `a`, `b`, or `tie` (unparseable/refusal → `tie`, logged) |
| `prob_a` | normalised probability of "Passage A" under the two-way choice |
| `score_a, score_b` | raw label log-likelihoods |
| `latency_ms, created_at` | cost bookkeeping (feeds RQ5/RQ6 efficiency baselines) |

Key = (dataset, query_id, doc_id_a, doc_id_b, model, prompt_version). Append-only part
files; dedup keeps the first write. The store is the thesis's reusable core artefact.

**Axiom battery** (as available in `ir_axioms` 1.1.2 — note LNC2 is not in this release):
lexical `TFC1, TFC3, M_TDC, LNC1, TF_LNC, PROX1–PROX5` plus `REG, ANTI_REG, AND, DIV,
LB1`; semantic `STMC1, STMC2`. The semantic axioms may require large embedding downloads
(fastText vectors) — the pilot measures that cost and may defer them to Phase 1 (RQ2)
if prohibitive; the config makes them toggleable.

**Agreement definitions** (fixed now so later phases are comparable):
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

1. **Smoke test** (automated): SciFact + mock ranker + lexical battery, ~5 queries,
   end-to-end into `results/p0_smoke/`. Runs in CI-time (~minutes), no big downloads.
2. **Backend validation**: flan-t5-small on a handful of real pairs — verifies the HF
   scoring path, latency measurement and store round-trip.
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

## 6. Risks

- **CPU-only inference** caps model size; mitigated by top-10 sampling, caching, and the
  option to move collection to a GPU machine later (store schema is machine-agnostic).
- **Prebuilt index mismatch** (stemming/stopwords differ from what axioms assume):
  acceptable for the pilot; revisit if axiom preferences look degenerate.
- **Glasgow hosts down.** `data.pyterrier.org` (prebuilt indices) and `ir.dcs.gla.ac.uk`
  are unreachable from this network; the ir-datasets mirror is up. Resolved: the official
  prebuilt MS MARCO Terrier index is also published on HuggingFace
  (`pyterrier/msmarco-passage.terrier`) and loaded via `pt.Artifact.from_hf` (the
  `hf_artifact` config field). Local indexing (`index_path`) remains the last resort.
- **Small pairwise models are position-biased.** `flan-t5-small` picked "Passage A" in
  both presentation orders on an obvious sanity pair — consistent with PRP's finding that
  small Flan-T5 variants cannot rank pairwise. The pilot model must pass an
  order-swapped sanity check before the full run.
- **Strict axiom preconditions.** In the SciFact smoke run TFC3/M-TDC/TF-LNC had zero
  coverage and LNC1 2% — their equal-length-style preconditions rarely hold on natural
  pairs. Expected from the literature, but if DL19 looks similar, Phase 1 must consider
  relaxed preconditions (margin parameters) or accept low-coverage axioms.
- **`ir_axioms` default tooling semantics** (tokenisation, statistics source) are partly
  implicit; validated by synthetic sanity checks in the test suite (e.g. TFC1 must prefer
  the higher-TF document on a constructed pair).
- **Semantic axiom downloads** (multi-GB embeddings) may not fit the pilot budget; they
  are toggleable and can be deferred to RQ2 without blocking the milestone.

## 7. Work breakdown

1. `config.py` + updated `configs/` (pilot, smoke) — the schema in §3.
2. `datasets.py` + `pairs.py` + unit tests (canonicalisation, determinism, counts).
3. `preferences.py` + tests (append/dedup/lookup round-trip).
4. `rankers/` (base, mock, hf) + prompt v0; mock-based tests.
5. `axioms.py` + synthetic sanity tests.
6. `agreement.py` + hand-computable test case (incl. an inconsistent pair and a cycle).
7. `experiments/p0_pilot/run.py` orchestration; smoke run (vaswani+mock); HF validation;
   then the DL19 run.

Everything lands in small, pointed commits following the repo convention.

## 8. Runbook: finishing Phase 0 (next session)

State as of 2026-07-11: items 1–7 above are **done** — pipeline implemented, 13 tests
passing, SciFact smoke run validated end-to-end, verdict caching confirmed. What remains
is executing the DL19 pilot. The big downloads were cancelled mid-way; everything below
resumes where it left off (HF downloads are resumable, the verdict store is incremental).

### 8.1 Get the data

- **MS MARCO Terrier index** (2.94 GB, resumable, one-time):
  `uv run python -c "from axiomrank import paths; paths.configure_caches(); import pyterrier as pt; print(pt.Artifact.from_hf('pyterrier/msmarco-passage.terrier').path)"`
  Run it in the background; observed HF throughput here swings between ~0.3 and ~4 MB/s,
  so expect 15 minutes to a few hours. Lands in `data/cache/pyterrier/artifacts/`.
- **DL19 topics + passage text** need no separate step: the first
  `run.py --config configs/pilot.yaml` invocation fetches them through ir_datasets
  (~1 GB collection from the ir-datasets mirror, which is reachable; the Glasgow hosts
  `data.pyterrier.org` / `ir.dcs.gla.ac.uk` are not — do not point configs at them).
- Stages 1–2 of the run (pool + pairs) cache under `data/processed/p0_pilot/` and are
  model-independent — they can be built as soon as the index is there.

### 8.2 Get the Flan-T5 contrast models (base and large)

- `uv sync --extra llm` is already done (CPU torch). Prefetch the weights:
  `uv run hf download google/flan-t5-base` (~1 GB) and
  `uv run hf download google/flan-t5-large` (~3 GB); they cache into
  `data/cache/huggingface/`. flan-t5-small is already cached but **failed** the gate
  below — keep it only as a position-bias exhibit.
- **Sanity gate (mandatory before any full collection):** every candidate model must
  pass the 4-way order-swap check — two obvious relevant/irrelevant pairs, each in both
  presentation orders. flan-t5-small chose "Passage A" in all cases (pure position
  bias). Expected CPU latency: base ~0.5 s/call (~45 min for the full pilot), large
  ~2 s/call (~2.5 h).

### 8.3 The primary ranker: Qwen3.6-35B-A3B via local endpoint

- The Qwen model is served externally and reachable at **`http://localhost:9086`**
  (OpenAI-compatible API). Two things must be built for it (additive, nothing migrates):
  1. An `openai` ranker backend (`rankers/openai_api.py`): chat-style prompt as
     `prompt_version: v1`, temperature 0, verdict + `prob_a` derived from single-token
     logprobs of the A/B choice; config gets `base_url`.
  2. **Thinking must be disabled** for this model — pass
     `chat_template_kwargs: {"enable_thinking": false}` (vLLM-style) or the `/no_think`
     directive, and verify in the sanity gate that responses contain no reasoning
     preamble; label-logprob scoring silently breaks if thinking tokens come first.
- The same 4-way sanity gate applies before full collection.

### 8.4 Run the pilot (the actual milestone)

1. `uv run python experiments/p0_pilot/run.py --config configs/pilot.yaml` with
   `ranker.model: google/flan-t5-base` (or large) for the local contrast run —
   background, fully resumable, ~3,870 presentations.
2. Same command with the Qwen backend config for the primary run. The store keys on
   (model, prompt_version), so all models coexist; re-runs cost nothing.
3. Outputs land in `results/p0_pilot/` (agreement.csv, consistency.json per run). Check
   the exit criteria (§1) and read the numbers against the open questions (§5):
   position-consistency per model, axiom coverage (watch the strict-precondition axioms),
   non-transitivity rate, and the latency bookkeeping for the RQ5/6 efficiency baseline.
4. Decide and record in this document: pilot model verdict, sampling scale for Phase 1,
   and whether STMC axioms enter the battery (measure their fastText download first).
