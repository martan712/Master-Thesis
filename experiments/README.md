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

The current `rq5_surrogate` preview predicts every pair from pairwise axiom votes and then
uses Copeland aggregation. It is a fitted RQ4 axiom reranker, not yet the linear-time pointwise
scorer required by RQ5. RQ5 starts only when document-level scores can be produced without
enumerating all document pairs and the resulting cost is measured.
