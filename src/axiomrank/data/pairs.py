"""Sampling document pairs from the first-stage pool.

Pairs are canonical and unordered: doc_id_1 < doc_id_2 lexicographically. The order in
which a pair is *presented* to the ranker is a separate dimension handled by the
experiment loop, never encoded here.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from axiomrank.config import PairsConfig

# The frame every downstream stage (verdict collection, axiom computation) consumes.

PAIR_COLUMNS = ["qid", "query", "doc_id_1", "doc_id_2", "text_1", "text_2"]


def sample_pairs(pool: pd.DataFrame, cfg: PairsConfig, seed: int) -> pd.DataFrame:
    """Sample canonical document pairs per query from the BM25 pool."""
    rng = np.random.default_rng(seed)
    qids = sorted(pool["qid"].unique())
    if cfg.max_queries is not None:
        qids = qids[: cfg.max_queries]

    rows = []
    for qid in qids:
        group = pool[pool["qid"] == qid].sort_values("rank")
        query = group["query"].iloc[0]
        docs = list(zip(group["docno"], group["text"]))
        if cfg.strategy == "top_k_all_pairs":
            selected = list(combinations(docs[: cfg.k], 2))
        elif cfg.strategy == "uniform":
            all_pairs = list(combinations(docs, 2))
            if len(all_pairs) > cfg.per_query:
                idx = rng.choice(len(all_pairs), size=cfg.per_query, replace=False)
                selected = [all_pairs[i] for i in sorted(idx)]
            else:
                selected = all_pairs
        else:
            raise ValueError(f"Unknown pair sampling strategy: {cfg.strategy}")

        for (id_a, text_a), (id_b, text_b) in selected:
            if id_b < id_a:
                id_a, id_b, text_a, text_b = id_b, id_a, text_b, text_a
            rows.append((qid, query, id_a, id_b, text_a, text_b))

    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    return pairs.drop_duplicates(subset=["qid", "doc_id_1", "doc_id_2"], ignore_index=True)
