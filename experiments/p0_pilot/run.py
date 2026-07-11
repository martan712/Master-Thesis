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

from axiomrank import analysis, pipeline
from axiomrank.config import dump_config, load_config
from axiomrank.data.preferences import PreferenceStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--refresh", action="store_true", help="recompute cached stages")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = pipeline.output_dir(cfg)
    dump_config(cfg, out / "config.yaml")

    print(f"[1/4] BM25 pool ({cfg.dataset.irds_id}, depth {cfg.first_stage.pool_depth})")
    pool = pipeline.build_pool(cfg, args.refresh)
    print(f"      {pool.qid.nunique()} queries, {len(pool)} pooled documents")

    print("[2/4] pair sampling")
    pairs = pipeline.build_pairs(cfg, pool, args.refresh)
    print(f"      {len(pairs)} canonical pairs over {pairs.qid.nunique()} queries")

    print("[3/4] model verdicts (cached in preference store)")
    store_df = pipeline.collect_verdicts(
        cfg.dataset.irds_id, cfg.ranker, pairs, PreferenceStore()
    )

    print("[4/4] axiom preferences + agreement")
    names = [s.column for s in cfg.axioms.specs]
    axiom_df = pipeline.build_axiom_prefs(cfg, pool, pairs, args.refresh)

    verdicts = analysis.model_pair_verdicts(store_df)
    table = analysis.agreement_table(axiom_df, verdicts, names)
    stats = {
        **analysis.consistency_stats(verdicts),
        **analysis.nontransitivity_rate(verdicts),
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
