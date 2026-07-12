"""Collecting model verdicts for a pair sample, against the preference store.

Model verdicts live in the append-only preference store and are never recomputed
(lookup before call), regardless of any refresh flag.
"""

import dataclasses
import hashlib
import time

import pandas as pd

from axiomrank.config import RankerConfig
from axiomrank.data.preferences import PreferenceStore, new_row
from axiomrank.rankers import make_ranker


def _protocol_signature(ranker_cfg: RankerConfig) -> dict:
    """Result-affecting ranker settings guarded by the preference-store manifest."""
    if ranker_cfg.backend == "hf":
        from axiomrank.rankers.hf import LABELS, PROMPTS

        prompt = PROMPTS.get(ranker_cfg.prompt_version)
        scoring = {"labels": list(LABELS), "mode": "sequence_log_likelihood"}
    elif ranker_cfg.backend == "openai":
        from axiomrank.rankers.openai_api import PROMPTS

        prompt = PROMPTS.get(ranker_cfg.prompt_version)
        scoring = {"mode": "first_token_top_logprobs"}
    else:
        prompt = "deterministic_hash_score"
        scoring = {"mode": "mock"}
    if prompt is None:
        raise ValueError(f"unknown prompt version {ranker_cfg.prompt_version!r}")
    cfg = dataclasses.asdict(ranker_cfg)
    result_affecting = {
        key: cfg[key]
        for key in (
            "backend", "model", "prompt_version", "max_chars", "device", "dtype",
            "base_url", "extra_body",
        )
    }
    return {
        "implementation_version": 1,
        "settings": result_affecting,
        "prompt_sha256": hashlib.sha256(str(prompt).encode()).hexdigest(),
        "scoring": scoring,
    }


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
    allow_new: bool = True,
) -> pd.DataFrame:
    """Return requested presentations, collecting cache misses only when allowed.

    Analysis-only phases pass ``allow_new=False`` so an incomplete preference store
    fails explicitly instead of unexpectedly loading a model or contacting an endpoint.
    """
    model_name = ranker_cfg.model or "mock"
    store.register_protocol(
        model_name, ranker_cfg.prompt_version, _protocol_signature(ranker_cfg)
    )
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

    missing = [
        presentation
        for presentation in presentations
        if (presentation[0], presentation[2], presentation[3], model_name) not in have
    ]
    if missing and not allow_new:
        raise RuntimeError(
            f"preference cache incomplete for {model_name}: "
            f"{len(missing)}/{len(presentations)} presentations missing"
        )

    buffer: list[dict] = []
    n_new = 0
    for presentation in presentations:
        qid, query, doc_a, doc_b, text_a, text_b = presentation
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

    model_name = ranker.name if ranker is not None else model_name
    df = store.load(
        dataset=dataset_id, model=model_name, prompt_version=ranker_cfg.prompt_version
    )
    wanted = set(zip(pairs["qid"], pairs["doc_id_1"], pairs["doc_id_2"]))
    canon = [tuple(sorted((a, b))) for a, b in zip(df["doc_id_a"], df["doc_id_b"])]
    mask = [(q, c[0], c[1]) in wanted for q, c in zip(df["query_id"], canon)]
    return df[mask].reset_index(drop=True)
