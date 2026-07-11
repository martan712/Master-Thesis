"""Phase 1 effectiveness gate: do the pairwise verdicts make a ranker? (phase1-design.md §4)

Per grid cell (one config = collection × top-10 all-pairs, a list of rankers), this
aggregates each ranker's cached pairwise verdicts into a Copeland ranking (= PRP-allpair)
and compares it to the BM25 first-stage baseline on nDCG@10 and MAP, using the collection
qrels. It collects no new verdicts when rq1 has already run — the store is keyed by
(dataset, model, prompt_version) and shared with the rq1 runs — so this is a zero-cost
validation of whether the top-10 residual Phase 1 studies is skill or noise.

Decision rule (phase1-design.md §4): Qwen must clearly beat BM25 on nDCG@10 on both DL19
and DL20, else stop-and-fix (prompt/scoring/model) before drawing axiom conclusions;
flan-t5-large is the contrast model, reported but not gated.

Usage:
    uv run python experiments/ranking_effectiveness/run.py --config configs/eff_dl19_top10.yaml
    uv run python experiments/ranking_effectiveness/run.py --config configs/eff_dl20_top10.yaml --only-model qwen

Verdicts are cached in the preference store (never recomputed); --refresh recomputes the
derived stages (pool, pairs). --only-model substring-filters the configured rankers.
"""

import argparse
import json

from axiomrank import analysis, pipeline, ranking
from axiomrank.config import dump_config, load_config
from axiomrank.data.preferences import PreferenceStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--refresh", action="store_true", help="recompute cached stages")
    parser.add_argument("--only-model", help="substring filter on the configured rankers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.pairs.strategy != "top_k_all_pairs":
        raise SystemExit(
            "ranking_effectiveness needs the complete top-k tournament: set "
            f"pairs.strategy: top_k_all_pairs (got {cfg.pairs.strategy!r})"
        )

    out = pipeline.output_dir(cfg)
    dump_config(cfg, out / "config.yaml")
    metrics = ranking.DEFAULT_METRICS
    names = ranking.metric_names(metrics)

    print(f"[1/3] BM25 pool ({cfg.dataset.irds_id}, depth {cfg.first_stage.pool_depth})")
    pool = pipeline.build_pool(cfg, args.refresh)
    print(f"      {pool.qid.nunique()} queries, {len(pool)} pooled documents")

    print(f"[2/3] pair sampling ({cfg.pairs.strategy}, k={cfg.pairs.k})")
    pairs = pipeline.build_pairs(cfg, pool, args.refresh)
    print(f"      {len(pairs)} canonical pairs over {pairs.qid.nunique()} queries")

    # The BM25 baseline run is the pool as-is; scored once and reused for every ranker.
    baseline_perq = ranking.evaluate_run(pool, cfg.dataset.irds_id, metrics)

    rankers = cfg.all_rankers
    if args.only_model:
        rankers = [r for r in rankers if args.only_model in (r.model or "mock")]
        if not rankers:
            raise SystemExit(f"--only-model {args.only_model!r} matches no configured ranker")

    for ranker_cfg in rankers:
        model_name = ranker_cfg.model or "mock"
        print(f"[3/3] verdicts -> Copeland ranking -> evaluation: {model_name}")
        store_df = pipeline.collect_verdicts(
            cfg.dataset.irds_id, ranker_cfg, pairs, PreferenceStore()
        )
        verdicts = analysis.model_pair_verdicts(store_df)
        reranked_run = ranking.copeland_ranking(verdicts, pool)
        reranked_perq = ranking.evaluate_run(reranked_run, cfg.dataset.irds_id, metrics)

        per_query, summary = ranking.compare_runs(baseline_perq, reranked_perq, metrics)
        consistency = analysis.consistency_stats(verdicts)

        metrics_dir = out / "metrics" / model_name.replace("/", "__")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        per_query.to_csv(metrics_dir / "effectiveness.csv", index=False)
        report = {
            "model": model_name,
            "dataset": cfg.dataset.irds_id,
            "n_queries": int(len(per_query)),
            "metrics": summary,
            "consistency": consistency,
        }
        with open(metrics_dir / "effectiveness.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"      -> {metrics_dir}")
        print(f"      {'metric':>10}  {'BM25':>7}  {'reranked':>9}  {'delta':>7}  W/T/L")
        for name in names:
            s = summary[name]
            print(
                f"      {name:>10}  {s['mean_baseline']:7.4f}  {s['mean_reranked']:9.4f}  "
                f"{s['mean_delta']:+7.4f}  {s['wins']}/{s['ties']}/{s['losses']}"
            )


if __name__ == "__main__":
    main()
