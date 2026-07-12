"""Prepare relevance-improving LLM pair reversals for manual Phase 3 analysis.

This is a selection aid, not a causal attribution method. Copeland ranks are determined by
the complete pair graph, so a single edge cannot be credited with the final nDCG change.
The exported cases satisfy a stricter descriptive condition: within a query whose LLM run
improves nDCG@10, the LLM prefers a more-relevant document that BM25 placed below a
less-relevant document, and the final LLM ranking reverses those two documents.

The script makes no model calls. It joins cached RQ4 predictions, runs, pools, qrels and
axiom preferences and writes a complete candidate table plus compact inspection packets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from axiomrank import paths


COLLECTIONS = {
    "dl19_top10": {
        "dataset": "msmarco-passage/trec-dl-2019/judged",
        "processed": "dl19_top10",
    },
    "dl20_top10": {
        "dataset": "msmarco-passage/trec-dl-2020/judged",
        "processed": "dl20_top10",
    },
}

MODEL_DIRS = {
    "qwen": "models__qwen3.6-35B-A3B-AWQ",
    "flan_large": "google__flan-t5-large",
    "flan_xl": "google__flan-t5-xl",
}

AXIOM_COLUMNS = [
    "TFC1",
    "M_TDC",
    "LNC1",
    "TF_LNC",
    "PROX1",
    "PROX2",
    "PROX3",
    "PROX4",
    "PROX5",
    "AND",
    "DIV",
    "LB1",
    "VERB",
    "QCOV",
]


def _qrels(dataset_id: str) -> dict[tuple[str, str], int]:
    paths.configure_caches()
    import ir_datasets

    return {
        (str(row.query_id), str(row.doc_id)): int(row.relevance)
        for row in ir_datasets.load(dataset_id).qrels_iter()
    }


def _query_deltas(per_query_path: Path, collection: str) -> pd.Series:
    frame = pd.read_csv(per_query_path, dtype={"query_id": str})
    frame = frame[frame["collection"] == collection]
    pivot = frame.pivot(index="query_id", columns="run", values="nDCG@10")
    return pivot["llm"] - pivot["bm25"]


def _preference_maps(metrics_root: Path) -> dict[str, dict[tuple[str, ...], int]]:
    out = {}
    for label, directory in MODEL_DIRS.items():
        frame = pd.read_parquet(metrics_root / directory / "pair_predictions.parquet")
        keys = zip(
            frame["collection"].astype(str),
            frame["query_id"].astype(str),
            frame["doc_id_1"].astype(str),
            frame["doc_id_2"].astype(str),
            strict=True,
        )
        out[label] = dict(zip(keys, frame["target_pref"].astype(int), strict=True))
    return out


def _winner_relative(value: int, winner_is_doc1: bool) -> int:
    """Map a canonical-pair preference to +1 winner, -1 loser, 0 neutral."""
    return int(value if winner_is_doc1 else -value)


def _is_contributory_reversal(
    *,
    query_delta: float,
    bm25_winner: int,
    bm25_loser: int,
    llm_winner: int,
    llm_loser: int,
    rel_winner: int,
    rel_loser: int,
) -> bool:
    """Whether a pair satisfies every descriptive selection condition."""
    return bool(
        query_delta > 0
        and bm25_winner > bm25_loser
        and llm_winner < llm_loser
        and rel_winner > rel_loser
    )


def candidate_reversals(results_root: Path) -> pd.DataFrame:
    metrics_root = results_root / "metrics"
    primary_dir = metrics_root / MODEL_DIRS["qwen"]
    predictions = pd.read_parquet(primary_dir / "pair_predictions.parquet")
    runs = pd.read_parquet(primary_dir / "runs.parquet")
    model_preferences = _preference_maps(metrics_root)
    rows = []

    for collection, spec in COLLECTIONS.items():
        processed = (
            paths.PROCESSED_DIR
            / "rq2_semantic_agreement"
            / spec["processed"]
        )
        pool = pd.read_parquet(processed / "pool.parquet")
        pool["qid"] = pool["qid"].astype(str)
        pool["docno"] = pool["docno"].astype(str)
        axioms = pd.read_parquet(processed / "axiom_prefs.parquet")
        for column in ("qid", "doc_id_1", "doc_id_2"):
            axioms[column] = axioms[column].astype(str)

        qrels = _qrels(spec["dataset"])
        deltas = _query_deltas(primary_dir / "effectiveness_per_query.csv", collection)
        ranks = runs[runs["collection"] == collection].copy()
        ranks["qid"] = ranks["qid"].astype(str)
        ranks["docno"] = ranks["docno"].astype(str)
        rank_map = {
            run: dict(zip(zip(group["qid"], group["docno"], strict=True), group["rank"], strict=True))
            for run, group in ranks.groupby("run")
        }
        pool_map = pool.set_index(["qid", "docno"])[["query", "text", "rank", "score"]]
        axiom_map = axioms.set_index(["qid", "doc_id_1", "doc_id_2"])

        subset = predictions[
            (predictions["collection"] == collection)
            & (predictions["target_pref"] != 0)
        ]
        for pair in subset.itertuples(index=False):
            qid = str(pair.query_id)
            query_delta = float(deltas.get(qid, float("nan")))
            if not query_delta > 0:
                continue

            winner_is_doc1 = pair.target_pref > 0
            winner = str(pair.doc_id_1 if winner_is_doc1 else pair.doc_id_2)
            loser = str(pair.doc_id_2 if winner_is_doc1 else pair.doc_id_1)
            key_winner = (qid, winner)
            key_loser = (qid, loser)
            bm25_winner = int(rank_map["bm25"][key_winner])
            bm25_loser = int(rank_map["bm25"][key_loser])
            llm_winner = int(rank_map["llm"][key_winner])
            llm_loser = int(rank_map["llm"][key_loser])
            rel_winner = qrels.get((qid, winner), 0)
            rel_loser = qrels.get((qid, loser), 0)
            if not _is_contributory_reversal(
                query_delta=query_delta,
                bm25_winner=bm25_winner,
                bm25_loser=bm25_loser,
                llm_winner=llm_winner,
                llm_loser=llm_loser,
                rel_winner=rel_winner,
                rel_loser=rel_loser,
            ):
                continue

            canonical_key = (collection, qid, str(pair.doc_id_1), str(pair.doc_id_2))
            axiom_row = axiom_map.loc[(qid, str(pair.doc_id_1), str(pair.doc_id_2))]
            winner_pool = pool_map.loc[(qid, winner)]
            loser_pool = pool_map.loc[(qid, loser)]
            row = {
                "collection": collection,
                "query_id": qid,
                "query": winner_pool["query"],
                "query_ndcg_delta": query_delta,
                "winner_doc": winner,
                "loser_doc": loser,
                "winner_qrel": rel_winner,
                "loser_qrel": rel_loser,
                "qrel_gap": rel_winner - rel_loser,
                "winner_bm25_rank": bm25_winner,
                "loser_bm25_rank": bm25_loser,
                "winner_llm_rank": llm_winner,
                "loser_llm_rank": llm_loser,
                "bm25_reversal_distance": bm25_winner - bm25_loser,
                "llm_separation": llm_loser - llm_winner,
                "winner_bm25_score": float(winner_pool["score"]),
                "loser_bm25_score": float(loser_pool["score"]),
                "winner_text": winner_pool["text"],
                "loser_text": loser_pool["text"],
                "classical_supports_winner": _winner_relative(
                    int(pair.classical_pref), winner_is_doc1
                ),
                "extended_supports_winner": _winner_relative(
                    int(pair.plus_both_pref), winner_is_doc1
                ),
            }
            for model, preferences in model_preferences.items():
                row[f"{model}_supports_winner"] = _winner_relative(
                    preferences[canonical_key], winner_is_doc1
                )
            for axiom in AXIOM_COLUMNS:
                row[f"axiom_{axiom}"] = _winner_relative(
                    int(axiom_row[axiom]), winner_is_doc1
                )
            rows.append(row)

    candidates = pd.DataFrame(rows)
    return candidates.sort_values(
        [
            "query_ndcg_delta",
            "qrel_gap",
            "bm25_reversal_distance",
            "llm_separation",
        ],
        ascending=False,
    ).reset_index(drop=True)


def _inspection_markdown(candidates: pd.DataFrame, per_collection: int) -> str:
    selected = candidates.groupby("collection", sort=False).head(per_collection)
    lines = [
        "# Candidate relevance-improving pair reversals",
        "",
        "> Automatically selected inspection packets. These are contributory reversals, not",
        "> causal attributions of a query-level nDCG improvement to one pairwise edge.",
        "",
    ]
    for number, row in enumerate(selected.itertuples(index=False), start=1):
        model_support = ", ".join(
            f"{name}={getattr(row, f'{name}_supports_winner'):+d}"
            for name in MODEL_DIRS
        )
        lines.extend(
            [
                f"## {number}. {row.collection} / {row.query_id}: {row.query}",
                "",
                f"- Query nDCG@10 delta: {row.query_ndcg_delta:+.3f}",
                f"- Qrels: preferred {row.winner_qrel}, demoted {row.loser_qrel}",
                f"- BM25 ranks: preferred {row.winner_bm25_rank}, demoted {row.loser_bm25_rank}",
                f"- Qwen ranks: preferred {row.winner_llm_rank}, demoted {row.loser_llm_rank}",
                f"- Cross-model edge support (+1 preferred): {model_support}",
                f"- Fitted classical/extended: {row.classical_supports_winner:+d} / "
                f"{row.extended_supports_winner:+d}",
                "",
                f"**Preferred document `{row.winner_doc}`**",
                "",
                str(row.winner_text),
                "",
                f"**Demoted document `{row.loser_doc}`**",
                "",
                str(row.loser_text),
                "",
                "**Manual notes**",
                "",
                "- Observed distinction:",
                "- Candidate axiom or feature:",
                "- Alternative explanation / counterexample risk:",
                "- Existing-axiom relation:",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=paths.RESULTS_DIR / "rq4_axioms" / "pooled_top10",
    )
    parser.add_argument("--per-collection", type=int, default=20)
    args = parser.parse_args()

    output = args.results_root / "qualitative"
    output.mkdir(parents=True, exist_ok=True)
    candidates = candidate_reversals(args.results_root)
    candidates.to_parquet(output / "candidate_reversals.parquet", index=False)
    candidates.drop(columns=["winner_text", "loser_text"]).to_csv(
        output / "candidate_reversals.csv", index=False
    )
    (output / "inspection_packets.md").write_text(
        _inspection_markdown(candidates, args.per_collection)
    )
    annotation_path = (
        paths.PROJECT_ROOT
        / "docs"
        / "resources"
        / "phase3-qualitative-case-annotations.csv"
    )
    if annotation_path.exists():
        annotations = pd.read_csv(annotation_path, dtype=str)
        annotations.to_csv(output / "manual_annotations.csv", index=False)
        join_keys = ["collection", "query_id", "winner_doc", "loser_doc"]
        typed = candidates.copy()
        for key in join_keys:
            typed[key] = typed[key].astype(str)
        annotated = annotations.merge(
            typed,
            on=join_keys,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if not annotated["_merge"].eq("both").all():
            missing = annotated.loc[annotated["_merge"] != "both", "case_id"].tolist()
            raise ValueError(f"manual annotations do not match selected pairs: {missing}")
        annotated.drop(columns="_merge").to_parquet(
            output / "annotated_cases.parquet", index=False
        )
    print(f"{len(candidates)} candidate reversals -> {output}")


if __name__ == "__main__":
    main()
