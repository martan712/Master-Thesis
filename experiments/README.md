# experiments/

One directory per phase / research question, prefixed for ordering (mirrors the phases in
`docs/research-plan.md` §5):

```
p0_pilot/                 Phase 0 — harness, preference logging, piloted on DL19
rq1_lexical_agreement/    Phase 1 — lexical axiom agreement profiles
rq2_semantic_agreement/   Phase 1 — semantic axioms added
ranking_effectiveness/    Phase 1 — depth-matched effectiveness reference vs BM25
rq3_decomposition/        Phase 2 — combined axiom model + residual analysis
rq4_axioms/               Phase 3 — fitted new-axiom capture, reranking, and ablation
rq4_qualitative/          Phase 3 — reproducible selection packets for manual reversal analysis
rq5_surrogate/            Transitional fitted pairwise-surrogate preview (not yet RQ5)
rq6_cascade/              Phase 4 — selective cascade (stretch)
```

Rules:

- Every script takes a config from `configs/` (`uv run python experiments/<dir>/run.py
  --config configs/<name>.yaml`), fixes the seed from that config, and writes only to
  `results/<dir>/` via `axiomrank.paths.results_dir`, including a copy of the config.
- Shared logic lives in `src/axiomrank/`, not in experiment scripts; a script should read
  as the recipe, not the implementation.
- Each config states its primary metric: `fidelity` (reproducing the model) or
  `effectiveness` (matching qrels) — research plan §4.1.

## Development cache and provenance policy

The current stage fingerprints are configuration guards, not complete content provenance. Until
the content-addressed cache design is implemented, follow these rules for every result-affecting
run:

- Refresh the complete affected stage chain after a change to retrieval, pair sampling, axiom
  implementation, tokenizer behavior, dependency versions, dataset/index choice, or source data.
- Never combine a refreshed pool or pair frame with an older child axiom-preference cache.
- Bump `pipeline.stages.CACHE_SCHEMA_VERSION` for a result-affecting change to shared cached-stage
  behavior.
- Preserve an immutable historical baseline by using a new experiment/variant cache namespace;
  do not silently overwrite it. Additive migrations must be exact-key checked and recorded in the
  research log.
- Runs used for candidate selection must write `run_manifest.json`. A dirty-tree run is
  automatically labelled `exploratory`; thesis-table/freeze runs must come from a clean commit.
- Legacy-migrated caches may support development continuity, but are not sufficient provenance for
  the final confirmation bundle.

Use `scripts/snapshot_artifacts.py` before a high-cost collection or major cache migration. Its
default snapshot covers `data/preferences/` and `results/`, verifies every SHA-256 checksum, and
performs a temporary Parquet restoration test.

The current `rq5_surrogate` preview predicts every pair from pairwise axiom votes and then
uses Copeland aggregation. It is a fitted RQ4 axiom reranker, not yet the linear-time pointwise
scorer required by RQ5. RQ5 starts only when document-level scores can be produced without
enumerating all document pairs and the resulting cost is measured.
