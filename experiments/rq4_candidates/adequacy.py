"""Per-document answer-adequacy oracle for the Phase 3 D0 candidate family.

The four D0 candidates (DEFANS, NUMANS, COMPARE, CBP) are cheap deterministic proxies for
one latent property: *answer-adequacy* — does the passage actually deliver the answer to the
query, as opposed to merely being on the same topic (which QCOV/BM25 already capture)? This
script asks the LLM that property directly, one absolute rating per (query, document), and
caches the scalar. The downstream evaluation checks whether the adequacy gap between two
documents predicts the cached pairwise preference, which separates two failure modes we
otherwise cannot distinguish: the property being irrelevant vs. the cheap rules being unable
to detect it.

The rating is deliberately per-document and absolute, never "which of A/B is better": a
pairwise prompt would simply re-run the reranking task and prove nothing. Scoring mirrors the
project's PRP logprob convention (``rankers/openai_api.py``): temperature 0, ``max_tokens 1``,
and the first-token logprobs over the digit labels 0-3 are softmaxed into a distribution whose
expected value is the adequacy scalar. Reasoning tokens are disabled server-side via
``extra_body`` or the first token would be a thinking token.

Verdicts are cached under ``data/adequacy/`` — a new store, entirely separate from the
append-only pairwise preference store, which is never touched. The cache is keyed by
(model, prompt, collection, qid, docno) and is resumable: an interrupted run skips documents
that already have a score.
"""

from __future__ import annotations

import argparse
import math
import os
import uuid
from datetime import datetime, timezone

import pandas as pd

from axiomrank import paths
from axiomrank.config import load_config
from axiomrank.pipeline import stages

PROMPT_VERSION = "adeq_v1"
LABELS = ("0", "1", "2", "3")
PROMPT = (
    "You are rating a single passage against a question.\n"
    "Rate how completely the passage ANSWERS the question — not merely whether it is on "
    "the same topic.\n\n"
    "Scale:\n"
    "0 = does not answer at all (off-topic, or only boilerplate / navigation / metadata)\n"
    "1 = on the topic but does not contain the answer\n"
    "2 = partially answers the question\n"
    "3 = directly and completely answers the question\n\n"
    'Question: "{query}"\n'
    "Passage: {passage}\n\n"
    "Reply with a single digit 0, 1, 2, or 3:"
)
CACHE_COLUMNS = [
    "model", "prompt_version", "collection", "qid", "docno",
    "adequacy", "p0", "p1", "p2", "p3", "argmax", "n_label_logprobs", "created_at",
]


def _store_dir(model_name: str) -> "paths.Path":
    root = paths.DATA_DIR / "adequacy" / model_name.replace("/", "__")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_cache(root) -> pd.DataFrame:
    parts = sorted(root.glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=CACHE_COLUMNS)
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _append(root, rows: list[dict]) -> None:
    if not rows:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = root / f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
    pd.DataFrame(rows)[CACHE_COLUMNS].to_parquet(path, index=False)


def _adequacy_from_logprobs(entries: list[tuple[str, float]]) -> dict:
    """Softmax the digit-label logprobs into a distribution; expected value is the scalar.

    Only the four labels 0-3 are considered; other tokens (whitespace variants aside) are
    ignored. The scalar is E[k] = sum_k p(k)*k over the present labels, so a confident "3"
    and a hesitant "2 vs 3" separate smoothly rather than collapsing to an argmax.
    """
    best: dict[str, float] = {}
    for token, logprob in entries:
        label = token.strip()
        if label in LABELS:
            best[label] = max(best.get(label, float("-inf")), logprob)
    if not best:
        return {"adequacy": float("nan"), "p0": float("nan"), "p1": float("nan"),
                "p2": float("nan"), "p3": float("nan"), "argmax": -1, "n_label_logprobs": 0}
    m = max(best.values())
    weights = {label: math.exp(lp - m) for label, lp in best.items()}
    total = sum(weights.values())
    probs = {label: weights.get(label, 0.0) / total for label in LABELS}
    adequacy = sum(int(label) * probs[label] for label in LABELS)
    argmax = int(max(best, key=best.get))
    return {
        "adequacy": adequacy,
        "p0": probs["0"], "p1": probs["1"], "p2": probs["2"], "p3": probs["3"],
        "argmax": argmax, "n_label_logprobs": len(best),
    }


def _pool_documents(source_cfgs, depth: int | None = None) -> pd.DataFrame:
    """Documents in the rerank scope to score for adequacy.

    Default (``depth is None``): the union of documents over the built pairs — the top-10
    all-pairs set, where cached pairwise labels exist. With ``depth=N`` the scope widens to
    each query's top-N of the BM25 pool by first-stage rank, so a deeper reranking can rescue
    relevant documents BM25 buried below rank 10 (``phase3-adequacy-oracle.md`` §5 follow-up).
    Scoring is resumable, so widening the depth only scores the newly-added documents.
    """
    frames = []
    for cfg in source_cfgs:
        collection = cfg.variant or cfg.dataset.irds_id.replace("/", "_")
        pool = stages.build_pool(cfg, refresh=False)
        if depth is not None:
            ids = (
                pool.sort_values("rank").groupby("qid", sort=False).head(depth)[["qid", "docno"]]
            ).drop_duplicates(["qid", "docno"])
        else:
            pairs = stages.build_pairs(cfg, pool, refresh=False)
            ids = pd.concat(
                [
                    pairs[["qid", "doc_id_1"]].rename(columns={"doc_id_1": "docno"}),
                    pairs[["qid", "doc_id_2"]].rename(columns={"doc_id_2": "docno"}),
                ]
            ).drop_duplicates(["qid", "docno"])
        docs = ids.merge(
            pool[["qid", "query", "docno", "text"]].drop_duplicates(["qid", "docno"]),
            on=["qid", "docno"],
            how="left",
            validate="one_to_one",
        )
        if docs["text"].isna().any():
            raise ValueError(f"missing pool text for some paired documents in {collection}")
        frames.append(docs.assign(collection=collection))
    return pd.concat(frames, ignore_index=True)


def _make_client(ranker_cfg):
    from openai import OpenAI

    return OpenAI(
        base_url=ranker_cfg.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "unused"),
        max_retries=3,
    )


def _score_document(client, model, extra_body, query, text, max_chars) -> dict:
    prompt = PROMPT.format(query=query, passage=text[:max_chars])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=20,
        extra_body=extra_body or None,
    )
    content = response.choices[0].logprobs.content
    if not content:
        return _adequacy_from_logprobs([])
    entries = [(t.token, t.logprob) for t in content[0].top_logprobs]
    return _adequacy_from_logprobs(entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config listing DL19/DL20 sources")
    parser.add_argument("--model-substr", default="qwen", help="ranker model filter")
    parser.add_argument("--limit", type=int, default=None, help="score at most N new documents")
    parser.add_argument(
        "--depth", type=int, default=None,
        help="widen scope to each query's top-N BM25 pool (default: top-10 all-pairs set)",
    )
    parser.add_argument("--flush-every", type=int, default=25)
    args = parser.parse_args()

    paths.configure_caches()
    cfg = load_config(args.config)
    source_cfgs = [load_config(paths.PROJECT_ROOT / source) for source in cfg.sources]
    ranker_cfg = next(
        r for r in source_cfgs[0].all_rankers if args.model_substr in (r.model or "")
    )
    model = ranker_cfg.model

    documents = _pool_documents(source_cfgs, depth=args.depth)
    root = _store_dir(model)
    done = _load_cache(root)
    done_keys = set(zip(done["collection"], done["qid"], done["docno"])) if len(done) else set()
    todo = documents[
        ~documents.apply(lambda r: (r.collection, r.qid, r.docno) in done_keys, axis=1)
    ]
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(
        f"[adequacy] model={model} prompt={PROMPT_VERSION} total docs={len(documents)} "
        f"cached={len(done)} to score={len(todo)}"
    )

    client = _make_client(ranker_cfg)
    buffer: list[dict] = []
    scored = 0
    for row in todo.itertuples(index=False):
        result = _score_document(
            client, model, ranker_cfg.extra_body, row.query, row.text, ranker_cfg.max_chars
        )
        buffer.append(
            {
                "model": model, "prompt_version": PROMPT_VERSION, "collection": row.collection,
                "qid": row.qid, "docno": row.docno,
                "created_at": datetime.now(timezone.utc).isoformat(), **result,
            }
        )
        scored += 1
        if len(buffer) >= args.flush_every:
            _append(root, buffer)
            print(f"[adequacy] {scored}/{len(todo)} scored (flushed)")
            buffer = []
    _append(root, buffer)
    print(f"[adequacy] done: {scored} newly scored -> {root}")


if __name__ == "__main__":
    main()
