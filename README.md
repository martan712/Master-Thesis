# Axiomatic Reduction of Generative-LLM Pairwise Rankers

MSc thesis project: how far can the pairwise preferences of a generative LLM ranker be
reduced to an interpretable set of retrieval axioms, and what is the nature of the part
that cannot? The research plan lives in [`docs/research-plan.md`](docs/research-plan.md)
and the literature framework in [`docs/literature-overview.md`](docs/literature-overview.md).

## Repository layout

| Path | Purpose | In git? |
|---|---|---|
| `docs/` | Research plan, literature overview, other planning documents | yes |
| `src/axiomrank/` | The main Python package (shared library code) | yes |
| `experiments/` | Config-driven experiment scripts, one directory per phase/RQ | yes |
| `configs/` | YAML experiment configurations | yes |
| `notebooks/` | Exploratory analysis (kept light; real logic moves into `src/`) | yes |
| `scripts/` | Utilities (environment check, ranker sanity gate, downloads, exports) | yes |
| `thesis/` | The thesis manuscript (LaTeX), added in the writing phase | yes |
| `data/` | Dataset caches, raw downloads, the preference dataset, processed features | **no** |
| `results/` | Run files, metrics, figures, tables — one subdirectory per experiment | **no** |
| `models/` | Model checkpoints and fitted surrogates | **no** |

`data/`, `results/` and `models/` are ignored by git (only their READMEs are tracked) and
are created on demand by `axiomrank.paths`. The single most valuable artefact is
`data/preferences/`, the cached LLM pairwise-verdict dataset that the whole thesis reuses
— **back it up** (it is expensive to recompute) but do not commit it.

Inside the package, foundations sit at the top level and everything else is grouped by
function:

| Subpackage / module | Purpose |
|---|---|
| `paths.py`, `config.py` | Path/cache resolution; typed YAML experiment configs |
| `data/` | First-stage retrieval (BM25 pools, indices), pair sampling, the append-only verdict store |
| `rankers/` | Pairwise LLM ranker backends (mock, HF transformers, OpenAI-compatible) |
| `axioms/` | Axiom registry and instantiation, per-pair preference computation, relaxed variants |
| `analysis/` | Verdict collapsing, agreement (+ bootstrap CIs), transitivity, joint fits, gap gradient, figures |
| `ranking/` | Copeland aggregation of pair verdicts (= PRP-allpair) and ir_measures evaluation |
| `pipeline/` | Cached experiment stages, verdict collection against the store, shared measurement recipes |

## Environment

The environment is managed **exclusively with [uv](https://docs.astral.sh/uv/)** — no
pip, venv or conda. Python is pinned to 3.12 (`.python-version`); uv fetches it
automatically.

```bash
uv sync                    # core deps + dev tools (analysis, PyTerrier, ir_axioms)
uv sync --extra llm        # add torch/transformers on the machine that runs the ranker

uv run scripts/verify_setup.py   # smoke-test: imports, Java, cache locations
```

Run everything through `uv run` (e.g. `uv run python experiments/...`), which keeps the
lockfile authoritative. Add dependencies with `uv add <pkg>` (or `uv add --optional llm
<pkg>` / `uv add --dev <pkg>`), never by hand-editing the lock.

## External tooling

Everything is consumed as a library dependency from PyPI — no vendored code, no git
submodules:

- [`ir-axioms`](https://github.com/webis-de/ir_axioms) — the axiom battery, PyTerrier integration (Bondarenko et al., SIGIR '22).
- [`python-terrier`](https://github.com/terrier-org/pyterrier) + [`ir-datasets`](https://ir-datasets.com/) — retrieval, indexing, TREC DL / MS MARCO / BEIR access.
- [`ir-explain`](https://github.com/souravsaha/ir_explain) — **not installed, maybe later.** Only becomes relevant if we compare our axiomatic account against attribution-style explainers (LIRME/EXS, listwise methods; literature overview §2.6). Its 0.1 release hard-pins a 2024 stack (`numpy==1.24.4`, `torch==2.2.0`, …) that cannot coexist with current `ir_axioms`/PyTerrier. If needed: make a separate isolated uv project (`uv init tools/ir_explain_env --python 3.11 && cd tools/ir_explain_env && uv add ir-explain`) and exchange run files/scores with the main project through `data/` and `results/`.
- `torch` / `transformers` *(optional extra)* — the local pairwise LLM ranker (Flan-T5).
- `openai` — client for rankers served behind any OpenAI-compatible endpoint (e.g. a
  vLLM-hosted Qwen); verdicts are scored from single-token logprobs, so the endpoint
  must allow `logprobs`.
- Reference only (read, don't depend on): [catherineschen/axiomatic-ir-interventions](https://github.com/catherineschen/axiomatic-ir-interventions), [webis-de MechIR](https://github.com/webis-de), rank-llm (`uv add rank-llm` if RankVicuna/RankZephyr baselines are run locally).

## Conventions

- **Caches stay inside the repo.** `axiomrank.paths.configure_caches()` points
  `IR_DATASETS_HOME`, `PYTERRIER_HOME` and `HF_HOME` into `data/cache/` — call it before
  importing those libraries. Override via `.env` (see `.env.example`) if needed.
- **Experiments are config-driven and reproducible.** Each experiment reads a YAML from
  `configs/`, fixes its random seed, and writes only under `results/<experiment>/`
  (via `axiomrank.paths.results_dir`). Active RQ4 runners also write a checksum-rich
  `run_manifest.json`; dirty-tree runs are development/exploratory only. See
  `experiments/README.md` for the naming and cache-validity rules.
- **Fidelity vs. effectiveness.** Every experiment declares in its config which of the
  two is its primary metric (research plan §4.1).
- **Preference data format.** LLM verdicts are stored once, append-only, as Parquet in
  `data/preferences/` keyed by (dataset, query id, doc id pair, model, prompt version,
  presentation order), so order-swap consistency checks stay possible.
- Reference lists mirror the Zotero "Master Thesis" collection; citation numbers `[n]` in
  `docs/` refer to `docs/literature-overview.md`.
