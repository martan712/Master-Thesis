# thesis/

The LaTeX manuscript lives here once the writing phase starts (plan §5, phase 5).
Final tables and figures are copied in from `results/` so the manuscript never depends on
gitignored paths. Bibliography exports come from the Zotero "Master Thesis" collection.

Ahead of the LaTeX phase, draft chapter-style write-ups live here as Markdown (e.g.
`phase0-writeup.md`, the Phase 0 pilot chapter).

Final-manuscript structure:

1. `foundation-writeup.md` — the compact Phases 0–2 methods and baseline-diagnostics chapter.
2. Phase 3/RQ4 — the main empirical contribution: iterative candidate axioms, fitted reranking,
   BM25/LLM/axiom effectiveness, ablation and locked confirmation.
3. RQ5 — a downstream pointwise efficiency study, only after feature decomposability is shown.

The detailed `phase0-writeup.md`, `phase1-writeup.md` and `phase2-writeup.md` remain transparent
analysis records and sources for appendices. Phase 2 is not intended to survive as a large
standalone final chapter.

The corresponding `docs/phase*-design.md` files are scientific protocol/outcome records;
`docs/phase*-implementation.md` files are engineering and reproduction records. The query is
the inferential unit throughout, and conclusions are conditional on the stated pair filter,
corpus, depth, ranker system and semantic operationalisation.

Research chronology, invalidated interpretations and audit corrections are preserved separately in
[`docs/research-logbook.md`](../docs/research-logbook.md); they are not mixed into the active designs.
