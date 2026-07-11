"""Phase 0 pilot: BM25 pool -> pair sample -> cached LLM verdicts -> axiom agreement.

Usage:
    uv run python experiments/p0_pilot/run.py --config configs/smoke.yaml
    uv run python experiments/p0_pilot/run.py --config configs/pilot.yaml

Stages are cached under data/processed/<experiment>/; delete that directory (or pass
--refresh) to recompute. Model verdicts are cached in the preference store and are never
recomputed regardless of --refresh.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from axiomrank import agreement, paths
from axiomrank.axioms import axiom_preferences
from axiomrank.config import dump_config, load_config
from axiomrank.datasets import bm25_pool, index_ref
from axiomrank.pairs import sample_pairs
from axiomrank.preferences import PreferenceStore, new_row
from axiomrank.rankers import make_ranker


def _cached(path: Path, refresh: bool, compute) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def collect_verdicts(cfg, pairs: pd.DataFrame, store: PreferenceStore) -> pd.DataFrame:
    """Query the ranker for every presentation not already in the store."""
    ranker = None  # built lazily: skip model loading when everything is cached
    existing = store.load(
        dataset=cfg.dataset.irds_id, model=None, prompt_version=cfg.ranker.prompt_version
    )
    have = set(zip(existing["query_id"], existing["doc_id_a"], existing["doc_id_b"], existing["model"]))

    presentations = []
    for row in pairs.itertuples():
        orders = [(row.doc_id_1, row.doc_id_2, row.text_1, row.text_2)]
        if cfg.ranker.order_swap:
            orders.append((row.doc_id_2, row.doc_id_1, row.text_2, row.text_1))
        for doc_a, doc_b, text_a, text_b in orders:
            presentations.append((row.qid, row.query, doc_a, doc_b, text_a, text_b))

    buffer: list[dict] = []
    n_new = 0
    for qid, query, doc_a, doc_b, text_a, text_b in presentations:
        model_name = cfg.ranker.model or "mock"
        if (qid, doc_a, doc_b, model_name) in have:
            continue
        if ranker is None:
            ranker = make_ranker(cfg.ranker)
        start = time.perf_counter()
        v = ranker.compare(query, text_a, text_b)
        latency_ms = (time.perf_counter() - start) * 1000
        buffer.append(
            new_row(
                dataset=cfg.dataset.irds_id,
                query_id=qid,
                doc_id_a=doc_a,
                doc_id_b=doc_b,
                model=ranker.name,
                prompt_version=cfg.ranker.prompt_version,
                verdict=v.verdict,
                prob_a=v.prob_a,
                score_a=v.score_a,
                score_b=v.score_b,
                latency_ms=latency_ms,
            )
        )
        n_new += 1
        if len(buffer) >= cfg.ranker.batch_flush:
            store.append(pd.DataFrame(buffer))
            buffer.clear()
            print(f"  ... {n_new} new verdicts", flush=True)
    if buffer:
        store.append(pd.DataFrame(buffer))
    print(f"verdicts: {n_new} newly collected, {len(presentations) - n_new} from cache")

    model_name = ranker.name if ranker is not None else (cfg.ranker.model or "mock")
    df = store.load(
        dataset=cfg.dataset.irds_id, model=model_name, prompt_version=cfg.ranker.prompt_version
    )
    wanted = set(zip(pairs["qid"], pairs["doc_id_1"], pairs["doc_id_2"]))
    canon = [
        tuple(sorted((a, b)))
        for a, b in zip(df["doc_id_a"], df["doc_id_b"])
    ]
    mask = [
        (q, c[0], c[1]) in wanted for q, c in zip(df["query_id"], canon)
    ]
    return df[mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--refresh", action="store_true", help="recompute cached stages")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed = paths.PROCESSED_DIR / cfg.experiment
    out = paths.results_dir(cfg.experiment)
    dump_config(cfg, out / "config.yaml")

    print(f"[1/4] BM25 pool ({cfg.dataset.irds_id}, depth {cfg.first_stage.pool_depth})")
    pool = _cached(
        processed / "pool.parquet", args.refresh, lambda: bm25_pool(cfg.dataset, cfg.first_stage)
    )
    print(f"      {pool.qid.nunique()} queries, {len(pool)} pooled documents")

    print("[2/4] pair sampling")
    pairs = _cached(
        processed / "pairs.parquet", args.refresh, lambda: sample_pairs(pool, cfg.pairs, cfg.seed)
    )
    print(f"      {len(pairs)} canonical pairs over {pairs.qid.nunique()} queries")

    print("[3/4] model verdicts (cached in preference store)")
    store_df = collect_verdicts(cfg, pairs, PreferenceStore())

    print("[4/4] axiom preferences + agreement")
    names = [n.replace("-", "_") for n in cfg.axioms.names]
    axiom_df = _cached(
        processed / "axiom_prefs.parquet",
        args.refresh,
        lambda: axiom_preferences(
            pool, pairs, cfg.axioms.names, index_location=index_ref(cfg.dataset)
        ),
    )

    verdicts = agreement.model_pair_verdicts(store_df)
    table = agreement.agreement_table(axiom_df, verdicts, names)
    stats = {
        **agreement.consistency_stats(verdicts),
        **agreement.nontransitivity_rate(verdicts),
        "mean_latency_ms": float(store_df["latency_ms"].mean()),
        "model": store_df["model"].iloc[0] if len(store_df) else None,
    }

    # one metrics directory per ranker, so contrast and primary runs coexist
    metrics = out / "metrics" / (cfg.ranker.model or "mock").replace("/", "__")
    metrics.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics / "agreement.csv", index=False)
    with open(metrics / "consistency.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nresults -> {out}")
    print(table.to_string(index=False))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
