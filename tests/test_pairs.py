import pandas as pd

from axiomrank.config import PairsConfig
from axiomrank.data.pairs import sample_pairs


def make_pool(n_queries=2, n_docs=6) -> pd.DataFrame:
    rows = []
    for q in range(n_queries):
        for r in range(n_docs):
            rows.append((f"q{q}", f"query {q}", f"d{n_docs - r}", r, float(n_docs - r), f"text {r}"))
    return pd.DataFrame(rows, columns=["qid", "query", "docno", "rank", "score", "text"])


def test_top_k_all_pairs_counts_and_canonical_order():
    pairs = sample_pairs(make_pool(), PairsConfig(strategy="top_k_all_pairs", k=4), seed=1)
    assert len(pairs) == 2 * 6  # C(4,2) per query
    assert (pairs["doc_id_1"] < pairs["doc_id_2"]).all()


def test_uniform_is_deterministic_given_seed():
    cfg = PairsConfig(strategy="uniform", per_query=5)
    a = sample_pairs(make_pool(n_docs=10), cfg, seed=7)
    b = sample_pairs(make_pool(n_docs=10), cfg, seed=7)
    c = sample_pairs(make_pool(n_docs=10), cfg, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_max_queries_limits_queries():
    pairs = sample_pairs(make_pool(n_queries=3), PairsConfig(k=3, max_queries=2), seed=1)
    assert pairs["qid"].nunique() == 2


def test_texts_follow_canonicalisation():
    pool = make_pool(n_queries=1, n_docs=3)
    pairs = sample_pairs(pool, PairsConfig(k=3), seed=1)
    lookup = dict(zip(pool["docno"], pool["text"]))
    for row in pairs.itertuples():
        assert row.text_1 == lookup[row.doc_id_1]
        assert row.text_2 == lookup[row.doc_id_2]
