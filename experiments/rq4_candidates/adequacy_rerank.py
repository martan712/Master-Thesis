"""Turn the validated Qwen answer-adequacy scalar into a reranking and score its nDCG@10.

The adequacy diagnostic (``docs/phase3-adequacy-oracle.md``) established that the per-document
adequacy scalar ``a(q, d)`` predicts the LLM pairwise preference across model families and tracks
the human qrel grade (Spearman ~0.65-0.72). That was measured on *pairs*. This script asks the
downstream question directly: if we simply sort each query's top-10 by ``a(q, d)`` and score the
resulting run against the collection qrels, does it beat BM25 on nDCG@10 — and how close does it
get to the full PRP-allpair tournament (Copeland over every cached pairwise verdict)?

The comparison is depth-matched by construction. The adequacy cache can score any requested BM25
top-N block (``adequacy.py._pool_documents``); the adequacy and BM25 runs therefore share that block
and differ only in how it is ordered. PRP-allpair remains a depth-10 reference because extending its
tournament requires fresh pairwise verdicts:

- **BM25** — the first-stage order (the gated baseline, phase1-design.md §4).
- **adequacy** — the block sorted by ``a(q, d)`` descending, first-stage rank breaking ties, the tail
  (documents with no adequacy score) held in first-stage order strictly below.
- **PRP-allpair** — Copeland over the cached Qwen pairwise verdicts, the reranking ceiling this cheap
  per-document scalar is trying to approximate at ~4-5x fewer LLM calls per query.

Zero ranker calls, zero downloads, no confirmation-set access: adequacy scalars come from
``data/adequacy/``, the BM25 pool and pairs from the cached stages, and the pairwise verdicts
read-only from the protected preference store via ``merged_cell_frame(allow_new=False)``.
"""

from __future__ import annotations

import argparse
import glob
import json

import pandas as pd

from axiomrank import paths, pipeline, ranking
from axiomrank.config import load_config
from axiomrank.provenance import write_run_manifest

ADEQUACY_MODEL = "models/qwen3.6-35B-A3B-AWQ"


def _load_adequacy() -> dict:
    root = paths.DATA_DIR / "adequacy" / ADEQUACY_MODEL.replace("/", "__")
    parts = sorted(glob.glob(str(root / "part-*.parquet")))
    if not parts:
        raise SystemExit(f"no adequacy scores under {root}; run adequacy.py first")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.dropna(subset=["adequacy"]).drop_duplicates(["collection", "qid", "docno"], keep="last")
    return {(r.collection, str(r.qid), str(r.docno)): r.adequacy for r in df.itertuples(index=False)}


def adequacy_rerank(
    pool: pd.DataFrame, adequacy: dict, collection: str, depth: int | None = None
) -> pd.DataFrame:
    """Rerank each query's top-``depth`` by the adequacy scalar, ``copeland_ranking``'s block layout.

    Documents within the top-``depth`` (by first-stage rank) that carry an adequacy score form the
    reranked block, ordered by ``a(q, d)`` descending with the first-stage rank breaking ties;
    everything else (rank >= depth, or no score) keeps first-stage order strictly below. ``depth=None``
    reranks every scored document. A larger depth lets adequacy rescue relevant documents BM25 buried
    below rank 10. Returns a run frame (``qid``, ``docno``, ``rank``, ``score``) with ``score = -rank``.
    """
    rows = []
    for qid, group in pool.groupby("qid", sort=False):
        scored = {
            d.docno: a
            for d in group.itertuples()
            if (depth is None or d.rank < depth)
            and (a := adequacy.get((collection, str(qid), str(d.docno)))) is not None
        }
        ordered = sorted(
            group.itertuples(),
            key=lambda d: (0 if d.docno in scored else 1, -scored.get(d.docno, 0.0), d.rank),
        )
        for new_rank, doc in enumerate(ordered):
            rows.append((qid, doc.docno, new_rank, float(-new_rank)))
    return pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])


def _assert_coverage(pool: pd.DataFrame, adequacy: dict, collection: str, depth: int) -> None:
    """Fail rather than silently evaluate a partially scored depth."""
    scoped = pool[pool["rank"] < depth]
    missing = [
        (str(row.qid), str(row.docno))
        for row in scoped.itertuples()
        if (collection, str(row.qid), str(row.docno)) not in adequacy
    ]
    if missing:
        preview = ", ".join(f"{qid}/{docno}" for qid, docno in missing[:3])
        raise SystemExit(
            f"adequacy cache is incomplete for {collection} depth {depth}: "
            f"{len(missing)} documents missing (e.g. {preview}); "
            f"run adequacy.py --depth {depth} first"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config listing DL19/DL20 sources")
    parser.add_argument("--refresh", action="store_true", help="recompute cached pool/pairs")
    parser.add_argument(
        "--depths", default="10,20,50,100",
        help="comma-separated rerank depths to sweep (default 10,20,50,100)",
    )
    args = parser.parse_args()
    depths = [int(x) for x in args.depths.split(",")]

    paths.configure_caches()
    cfg = load_config(args.config)
    source_cfgs = [load_config(paths.PROJECT_ROOT / source) for source in cfg.sources]
    adequacy = _load_adequacy()
    metrics = ranking.DEFAULT_METRICS
    names = ranking.metric_names(metrics)

    # The Qwen ranker whose cached pairwise verdicts define the PRP-allpair ceiling.
    ranker_cfg = next(r for r in source_cfgs[0].all_rankers if "qwen" in (r.model or ""))

    reports = []
    for source_cfg in source_cfgs:
        collection = source_cfg.variant or source_cfg.dataset.irds_id.replace("/", "_")
        pool = pipeline.build_pool(source_cfg, args.refresh)

        baseline = ranking.evaluate_run(pool, source_cfg.dataset.irds_id, metrics)

        # PRP-allpair (Copeland over the cached top-10 verdicts) is the depth-10 tournament
        # reference; new pairwise verdicts would be needed to extend it past depth 10.
        merged, _ = pipeline.merged_cell_frame(source_cfg, ranker_cfg, refresh=False)
        prp = ranking.evaluate_run(
            ranking.copeland_ranking(merged, pool), source_cfg.dataset.irds_id, metrics
        )
        _, prp_vs_bm25 = ranking.compare_runs(baseline, prp, metrics, seed=cfg.seed)

        by_depth = {}
        for depth in depths:
            _assert_coverage(pool, adequacy, collection, depth)
            adeq_run = adequacy_rerank(pool, adequacy, collection, depth)
            adeq = ranking.evaluate_run(adeq_run, source_cfg.dataset.irds_id, metrics)
            _, vs_bm25 = ranking.compare_runs(baseline, adeq, metrics, seed=cfg.seed)
            by_depth[depth] = vs_bm25

        reports.append(
            {
                "collection": collection,
                "n_queries": int(len(baseline)),
                "prp_vs_bm25": prp_vs_bm25,
                "adequacy_by_depth": by_depth,
            }
        )

    out_dir = paths.results_dir("rq4_candidates") / "d0v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "adequacy_rerank.json"
    with open(out, "w") as f:
        json.dump(reports, f, indent=2)
    write_run_manifest(
        out.with_name("adequacy_rerank.run_manifest.json"),
        cfg,
        config_source=args.config,
        source_paths=[__file__, paths.PROJECT_ROOT / "src" / "axiomrank"],
        input_paths=[paths.PROJECT_ROOT / source for source in cfg.sources]
        + [pipeline.processed_dir(source_cfg) for source_cfg in source_cfgs]
        + [
            paths.PREFERENCES_DIR,
            paths.DATA_DIR / "adequacy" / ADEQUACY_MODEL.replace("/", "__"),
        ],
        output_paths=[out],
        extra={
            "runner": "experiments/rq4_candidates/adequacy_rerank.py",
            "adequacy_model": ADEQUACY_MODEL,
            "depths": depths,
            "refresh": args.refresh,
        },
    )

    for rep in reports:
        print(f"\n=== {rep['collection']}  ({rep['n_queries']} queries) ===")
        for name in names:
            b = rep["prp_vs_bm25"][name]["mean_baseline"]
            p = rep["prp_vs_bm25"][name]["mean_reranked"]
            print(f"{name}:  BM25={b:.4f}  PRP-allpair@10={p:.4f}")
            for depth in depths:
                s = rep["adequacy_by_depth"][depth][name]
                print(
                    f"    adequacy@{depth:<3} {s['mean_reranked']:.4f}  "
                    f"vs BM25 {s['mean_delta']:+.4f} [{s['delta_ci_lo']:+.4f}, {s['delta_ci_hi']:+.4f}]  "
                    f"W/T/L {s['wins']}/{s['ties']}/{s['losses']}"
                )
    print(f"\n[adequacy-rerank] oracle={ADEQUACY_MODEL} -> {out}")


if __name__ == "__main__":
    main()
