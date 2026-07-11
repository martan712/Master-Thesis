# Phase 0 Implementation — Setup and Pilot (engineering record)

> Companion document: the scientific design, definitions and pilot results live in
> `phase0-design.md`. This document holds the engineering — module architecture,
> preference-store schema, work breakdown, runbook and operational risks.

This is the engineering record of Phase 0: the code that stands up the data, the ranker
harness, the axiom toolkit and the preference-logging pipeline, and the runbook that
executed the DL19 pilot.

## 1. Module architecture

### 1.1 Components

Seven components, all in `src/axiomrank/`, orchestrated by `experiments/p0_pilot/run.py`:

```
config.py        YAML → typed config (dataset, sampling, ranker, axioms, output)
datasets.py      topics + BM25 pooling (PyTerrier, prebuilt Terrier indices), doc text
pairs.py         pair sampling from the pool (canonical unordered pairs)
rankers/base.py  PairwiseRanker interface + PairVerdict
rankers/mock.py  deterministic hash-scored mock (transitive by construction)
rankers/hf.py    transformers backend, PRP prompt v0, label-likelihood scoring
rankers/openai_api.py  OpenAI-compatible endpoint backend (vLLM), chat prompt v1,
                 verdicts from single-token A/B logprobs
preferences.py   append-only Parquet store, keyed, dedup on read, lookup before call
axioms.py        ir_axioms battery wrapper (AxiomaticPreferences transformer)
agreement.py     canonical verdicts, per-axiom agreement, consistency, transitivity
```

Model candidates are vetted by `scripts/sanity_gate.py` (the mandatory 4-way order-swap
check from §6.2) before any full collection; metrics land per ranker under
`results/p0_pilot/metrics/<model>/`.

### 1.2 Data flow

`pool → pairs → (store lookup → LLM for misses → store append) → axiom preferences →
agreement report`. Every intermediate is cached under `data/processed/` so re-runs are
incremental; only `results/` outputs are overwritten.

## 2. Index selection

Use PyTerrier's prebuilt `terrier_stemmed` MS MARCO passage index (~2 GB download,
one-time) rather than indexing 8.8M passages locally (hours). Recorded in the config;
swapping to a locally built index is a config change.

## 3. Preference-store schema and keying

One row per *presentation*, i.e. per ordered pair:

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

## 4. Work breakdown

1. `config.py` + updated `configs/` (pilot, smoke) — the schema in `phase0-design.md` §3.
2. `datasets.py` + `pairs.py` + unit tests (canonicalisation, determinism, counts).
3. `preferences.py` + tests (append/dedup/lookup round-trip).
4. `rankers/` (base, mock, hf) + prompt v0; mock-based tests.
5. `axioms.py` + synthetic sanity tests.
6. `agreement.py` + hand-computable test case (incl. an inconsistent pair and a cycle).
7. `experiments/p0_pilot/run.py` orchestration; smoke run (vaswani+mock); HF validation;
   then the DL19 run.

Everything lands in small, pointed commits following the repo convention.

## 5. Runbook: finishing Phase 0

State as of 2026-07-11: items 1–7 above are **done** — pipeline implemented, 13 tests
passing, SciFact smoke run validated end-to-end, verdict caching confirmed. What remains
is executing the DL19 pilot. The big downloads were cancelled mid-way; everything below
resumes where it left off (HF downloads are resumable, the verdict store is incremental).

### 5.1 Get the data

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

### 5.2 Get the Flan-T5 contrast models (base and large) and run the sanity gate

- `uv sync --extra llm` is already done (CPU torch). Prefetch the weights:
  `uv run hf download google/flan-t5-base` (~1 GB) and
  `uv run hf download google/flan-t5-large` (~3 GB); they cache into
  `data/cache/huggingface/`. flan-t5-small is already cached but **failed** the gate
  below — keep it only as a position-bias exhibit.
- **Sanity gate (mandatory before any full collection), `scripts/sanity_gate.py`:** every
  candidate model must pass the 4-way order-swap check — two obvious relevant/irrelevant
  pairs, each in both presentation orders. flan-t5-small chose "Passage A" in all cases
  (pure position bias). Expected CPU latency: base ~0.5 s/call (~45 min for the full
  pilot), large ~2 s/call (~2.5 h). (Gate results in `phase0-design.md` §7.1.)

### 5.3 The primary ranker: Qwen3.6-35B-A3B via local endpoint

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

### 5.4 Run the pilot (the actual milestone)

1. `uv run python experiments/p0_pilot/run.py --config configs/pilot.yaml` with
   `ranker.model: google/flan-t5-base` (or large) for the local contrast run —
   background, fully resumable, ~3,870 presentations.
2. Same command with the Qwen backend config for the primary run. The store keys on
   (model, prompt_version), so all models coexist; re-runs cost nothing.
3. Outputs land in `results/p0_pilot/` (agreement.csv, consistency.json per run). Check
   the exit criteria (`phase0-design.md` §2) and read the numbers against the open
   questions (`phase0-design.md` §5): position-consistency per model, axiom coverage
   (watch the strict-precondition axioms), non-transitivity rate, and the latency
   bookkeeping for the RQ5/6 efficiency baseline.
4. Decide and record in `phase0-design.md` §7: pilot model verdict, sampling scale for
   Phase 1, and whether STMC axioms enter the battery (measure their fastText download
   first).

## 6. Operational risks

- **CPU-only inference** caps model size; mitigated by top-10 sampling, caching, and the
  option to move collection to a GPU machine later (store schema is machine-agnostic).
- **Glasgow hosts down.** `data.pyterrier.org` (prebuilt indices) and `ir.dcs.gla.ac.uk`
  are unreachable from this network; the ir-datasets mirror is up. Resolved: the official
  prebuilt MS MARCO Terrier index is also published on HuggingFace
  (`pyterrier/msmarco-passage.terrier`) and loaded via `pt.Artifact.from_hf` (the
  `hf_artifact` config field). Local indexing (`index_path`) remains the last resort.
- **Download sizes and resumability.** The one-time downloads (2.94 GB Terrier index,
  ~1 GB base / ~3 GB large Flan-T5 weights, ~1 GB DL19 collection, and the deferred
  7.24 GB fastText model) are all resumable; HF downloads pick up where they left off and
  the verdict store is incremental, so any run can be interrupted and resumed.
- **`ir_axioms` default tooling semantics** (tokenisation, statistics source) are partly
  implicit; validated by synthetic sanity checks in the test suite (e.g. TFC1 must prefer
  the higher-TF document on a constructed pair).
- **Semantic axiom downloads** (multi-GB embeddings) may not fit the pilot budget; they
  are toggleable and can be deferred to RQ2 without blocking the milestone.
