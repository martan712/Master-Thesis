"""Collect HF-backend PRP verdicts off-box for the Phase 2 system diagnostic.

Scientific scope and interpretation are in ``docs/phase2-design.md`` §5.4; operational
provenance is recorded in ``docs/research-logbook.md``.

Runs the exact `HFPairwiseRanker` scoring (prompt v0, "Passage A"/"Passage B" label
log-likelihood, order-swap) against a shipped `pairs.parquet`, on whatever `--device`
this machine offers — on bronze that is the Radeon 8060S iGPU under ROCm (`--device cuda
--dtype bfloat16`). Verdicts append to the preference store as fresh, uniquely named
`part-*.parquet` files, which are then rsynced back into the workstation's append-only
`data/preferences/` — no existing part is ever touched.

This deliberately imports only `axiomrank.rankers.hf` + `axiomrank.data.preferences` (the
former lazily pulls torch/transformers), NOT `axiomrank.pipeline`, whose package import
drags in ir_datasets / PyTerrier. It mirrors `axiomrank.pipeline.collect.collect_verdicts`,
which stays the canonical in-repo path; keep the two in step if the store schema changes.

Usage (inside the bronze ROCm container, with `PYTHONPATH=<repo>/src`):

    python scripts/collect_bronze.py \
        --pairs pairs_dl19_top10.parquet \
        --dataset msmarco-passage/trec-dl-2019/judged \
        --model google/flan-t5-xl --device cuda --dtype bfloat16 \
        --out data/preferences
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from axiomrank.data.preferences import PreferenceStore, new_row
from axiomrank.rankers.hf import HFPairwiseRanker


def _presentations(pairs: pd.DataFrame, order_swap: bool):
    """One canonical pair -> its presentations (as sampled, plus swapped when configured)."""
    for row in pairs.itertuples():
        orders = [(row.doc_id_1, row.doc_id_2, row.text_1, row.text_2)]
        if order_swap:
            orders.append((row.doc_id_2, row.doc_id_1, row.text_2, row.text_1))
        for doc_a, doc_b, text_a, text_b in orders:
            yield row.qid, row.query, doc_a, doc_b, text_a, text_b


def collect(args: argparse.Namespace) -> None:
    pairs = pd.read_parquet(args.pairs)
    if args.limit:
        pairs = pairs.head(args.limit)
    store = PreferenceStore(root=Path(args.out))
    ranker = HFPairwiseRanker(
        model_name=args.model,
        prompt_version=args.prompt_version,
        max_chars=args.max_chars,
        device=args.device,
        dtype=args.dtype,
    )

    # Lookup-before-call: skip presentations already in the store (append-only, never recomputed).
    existing = store.load(dataset=args.dataset, model=args.model, prompt_version=args.prompt_version)
    have = set(zip(existing["query_id"], existing["doc_id_a"], existing["doc_id_b"]))

    presentations = list(_presentations(pairs, args.order_swap))
    todo = [p for p in presentations if (p[0], p[2], p[3]) not in have]
    print(
        f"{args.model} on {args.dataset}: {len(pairs)} pairs -> {len(presentations)} presentations, "
        f"{len(presentations) - len(todo)} cached, {len(todo)} to score (device={args.device}, "
        f"dtype={args.dtype})",
        flush=True,
    )
    if not todo:
        return

    print(f"loading {args.model} ...", flush=True)
    ranker._load()  # surface any load/dtype error before the long loop

    buffer: list[dict] = []
    for i, (qid, query, doc_a, doc_b, text_a, text_b) in enumerate(todo, 1):
        start = time.perf_counter()
        v = ranker.compare(query, text_a, text_b)
        buffer.append(
            new_row(
                dataset=args.dataset,
                query_id=qid,
                doc_id_a=doc_a,
                doc_id_b=doc_b,
                model=args.model,
                prompt_version=args.prompt_version,
                verdict=v.verdict,
                prob_a=v.prob_a,
                score_a=v.score_a,
                score_b=v.score_b,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )
        if len(buffer) >= args.batch_flush:
            store.append(pd.DataFrame(buffer))
            buffer.clear()
            print(f"  ... {i}/{len(todo)} scored", flush=True)
    if buffer:
        store.append(pd.DataFrame(buffer))
    print(f"done: {len(todo)} new verdicts written under {args.out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", required=True, help="pairs.parquet with qid/query/doc_id_*/text_*")
    p.add_argument("--dataset", required=True, help="ir_datasets id, the store 'dataset' key")
    p.add_argument("--model", required=True, help="HF model id, e.g. google/flan-t5-xl")
    p.add_argument("--out", default="data/preferences", help="preference store root")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--prompt-version", dest="prompt_version", default="v0")
    p.add_argument("--max-chars", dest="max_chars", type=int, default=2000)
    p.add_argument("--batch-flush", dest="batch_flush", type=int, default=50)
    p.add_argument("--limit", type=int, default=0, help="score only the first N pairs (smoke test)")
    p.add_argument("--no-order-swap", dest="order_swap", action="store_false")
    collect(p.parse_args())


if __name__ == "__main__":
    main()
