"""axiomrank: axiomatic reduction of generative-LLM pairwise rankers.

Layout — foundations at the top level, everything else grouped by function:

- :mod:`axiomrank.paths`, :mod:`axiomrank.config` — path/cache resolution, typed YAML configs
- :mod:`axiomrank.data` — first-stage retrieval, pair sampling, the append-only verdict store
- :mod:`axiomrank.rankers` — pairwise LLM ranker backends (mock, HF, OpenAI-compatible)
- :mod:`axiomrank.axioms` — axiom registry/instantiation, per-pair preferences, relaxed variants
- :mod:`axiomrank.analysis` — verdict collapsing, agreement, transitivity, joint fits, gap gradient
- :mod:`axiomrank.ranking` — Copeland aggregation (= PRP-allpair) and run evaluation
- :mod:`axiomrank.pipeline` — cached experiment stages, verdict collection, measurement recipes

Import :mod:`axiomrank.paths` and call :func:`axiomrank.paths.configure_caches`
before importing ir_datasets / pyterrier / transformers, so that all dataset and
model caches land inside the repository's ``data/cache`` directory.
"""

__version__ = "0.1.0"
