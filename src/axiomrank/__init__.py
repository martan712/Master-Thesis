"""axiomrank: axiomatic reduction of generative-LLM pairwise rankers.

Import :mod:`axiomrank.paths` and call :func:`axiomrank.paths.configure_caches`
before importing ir_datasets / pyterrier / transformers, so that all dataset and
model caches land inside the repository's ``data/cache`` directory.
"""

__version__ = "0.1.0"
