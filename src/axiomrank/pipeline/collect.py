"""Collecting model verdicts for a pair sample, against the preference store.

Model verdicts live in the append-only preference store and are never recomputed
(lookup before call), regardless of any refresh flag.
"""

import time

import pandas as pd

from axiomrank.config import RankerConfig
from axiomrank.data.preferences import PreferenceStore, new_row
from axiomrank.rankers import make_ranker


def get_all_orders(ranker_cfg: RankerConfig, row):
    """Presentations for one canonical pair: as sampled, plus swapped when configured."""
    orders = [(row.doc_id_1, row.doc_id_2, row.text_1, row.text_2)]
    if ranker_cfg.order_swap:
        orders.append((row.doc_id_2, row.doc_id_1, row.text_2, row.text_1))
    return [
        (row.qid, row.query, doc_a, doc_b, text_a, text_b)
        for doc_a, doc_b, text_a, text_b in orders
    ]


def score_presentation(dataset_id: str, ranker_cfg: RankerConfig, ranker, presentation):
    """Ask the ranker for one presentation's verdict; return a preference-store row."""
    qid, query, doc_a, doc_b, text_a, text_b = presentation

    start = time.perf_counter()
    v = ranker.compare(query, text_a, text_b)
    latency_ms = (time.perf_counter() - start) * 1000
    return new_row(
        dataset=dataset_id,
        query_id=qid,
        doc_id_a=doc_a,
        doc_id_b=doc_b,
        model=ranker.name,
        prompt_version=ranker_cfg.prompt_version,
        verdict=v.verdict,
        prob_a=v.prob_a,
        score_a=v.score_a,
        score_b=v.score_b,
        latency_ms=latency_ms,
    )


def collect_verdicts(
    dataset_id: str,
    ranker_cfg: RankerConfig,
    pairs: pd.DataFrame,
    store: PreferenceStore,
) -> pd.DataFrame:
    """Query the ranker for every presentation not already in the store."""
    ranker = None  # built lazily: skip model loading when everything is cached
    existing = store.load(
        dataset=dataset_id, model=None, prompt_version=ranker_cfg.prompt_version
    )
    have = set(
        zip(existing["query_id"], existing["doc_id_a"], existing["doc_id_b"], existing["model"])
    )

    presentations = []
    for row in pairs.itertuples():
        presentations.extend(get_all_orders(ranker_cfg, row))

    buffer: list[dict] = []
    n_new = 0
    for presentation in presentations:
        qid, query, doc_a, doc_b, text_a, text_b = presentation
        model_name = ranker_cfg.model or "mock"
        if (qid, doc_a, doc_b, model_name) in have:
            continue

        if ranker is None:
            ranker = make_ranker(ranker_cfg)
        buffer.append(score_presentation(dataset_id, ranker_cfg, ranker, presentation))
        n_new += 1
        if len(buffer) >= ranker_cfg.batch_flush:
            store.append(pd.DataFrame(buffer))
            buffer.clear()
            print(f"  ... {n_new} new verdicts", flush=True)
    if buffer:
        store.append(pd.DataFrame(buffer))
    print(f"verdicts: {n_new} newly collected, {len(presentations) - n_new} from cache")

    model_name = ranker.name if ranker is not None else (ranker_cfg.model or "mock")
    df = store.load(
        dataset=dataset_id, model=model_name, prompt_version=ranker_cfg.prompt_version
    )
    wanted = set(zip(pairs["qid"], pairs["doc_id_1"], pairs["doc_id_2"]))
    canon = [tuple(sorted((a, b))) for a, b in zip(df["doc_id_a"], df["doc_id_b"])]
    mask = [(q, c[0], c[1]) in wanted for q, c in zip(df["query_id"], canon)]
    return df[mask].reset_index(drop=True)
