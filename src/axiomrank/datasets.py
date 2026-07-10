"""Topics, BM25 first-stage pooling and document text, via PyTerrier.

PyTerrier (and its Terrier jars, ir_datasets downloads, etc.) is only imported inside
functions, after cache configuration, so importing this module stays cheap and never
touches the network.
"""

import pandas as pd

from axiomrank import paths
from axiomrank.config import DatasetConfig, FirstStageConfig


def _pyterrier():
    paths.configure_caches()
    import pyterrier as pt

    return pt


def index_ref(cfg: DatasetConfig):
    """Terrier index reference (prebuilt, configured path, or locally built)."""
    return _index_ref(_pyterrier(), cfg)


def _index_ref(pt, cfg: DatasetConfig):
    if cfg.index_path:
        return cfg.index_path
    if cfg.terrier_dataset:
        dataset = pt.get_dataset(cfg.terrier_dataset)
        return dataset.get_index(cfg.index_variant) if cfg.index_variant else dataset.get_index()
    return _build_local_index(pt, cfg)


def _build_local_index(pt, cfg: DatasetConfig):
    """Index the ir_datasets corpus locally (only sensible for small collections)."""
    index_dir = paths.CACHE_DIR / "indices" / cfg.irds_id.replace("/", "__")
    if not (index_dir / "data.properties").exists():
        dataset = pt.get_dataset(f"irds:{cfg.irds_id}")
        indexer = pt.IterDictIndexer(str(index_dir), meta={"docno": 40})
        indexer.index(dataset.get_corpus_iter())
    return str(index_dir)


def bm25_pool(cfg: DatasetConfig, first_stage: FirstStageConfig) -> pd.DataFrame:
    """Retrieve the BM25 pool for all topics, with document text attached.

    Returns one row per (query, document): qid, query, docno, rank, score, text.
    """
    if first_stage.retriever != "bm25":
        raise ValueError(f"Unsupported first-stage retriever: {first_stage.retriever}")
    pt = _pyterrier()
    irds = pt.get_dataset(f"irds:{cfg.irds_id}")
    topics = irds.get_topics()
    retriever = pt.terrier.Retriever(_index_ref(pt, cfg), wmodel="BM25")
    pipeline = (
        pt.rewrite.tokenise()  # strip punctuation Terrier's query parser chokes on
        >> (retriever % first_stage.pool_depth)
        >> pt.text.get_text(irds, "text")
    )
    pool = pipeline.transform(topics)
    return pool[["qid", "query", "docno", "rank", "score", "text"]]
