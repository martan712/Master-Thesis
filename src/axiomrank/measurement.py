"""The Phase 1 measurement recipe, shared by the rq1 and rq2 runners.

One *grid cell* (a config: collection × sampling condition, with a list of rankers)
produces, per ranker, the full set of Phase 1 outputs under
results/<experiment>/<variant>/metrics/<model>/ (phase1-plan.md §5):

    agreement.csv       per-axiom coverage + agreement with query-bootstrap CIs
    consistency.json    position consistency, decisiveness, transitivity, latency
    joint_fit.json      base rate, majority vote, grouped-CV logistic per feature set
    gap_agreement.csv   agreement and joint accuracy binned by BM25 rank gap
    gap_agreement.png   draft of the RQ1 signature figure
"""

import json

import pandas as pd

from axiomrank import agreement, analysis, pipeline
from axiomrank.axioms import axiom_preferences
from axiomrank.config import ExperimentConfig, dump_config
from axiomrank.datasets import index_ref
from axiomrank.preferences import PreferenceStore


def build_axiom_prefs(
    cfg: ExperimentConfig, pool: pd.DataFrame, pairs: pd.DataFrame, refresh: bool
) -> pd.DataFrame:
    """Cached axiom preferences, recomputed when the configured battery grew."""
    path = pipeline.processed_dir(cfg) / "axiom_prefs.parquet"
    names = [s.column for s in cfg.axioms.specs]
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if set(names) <= set(cached.columns):
            return cached
        print("      battery changed; recomputing axiom preferences")
    return pipeline.cached_frame(
        path,
        True,
        lambda: axiom_preferences(
            pool, pairs, cfg.axioms.specs, index_location=index_ref(cfg.dataset)
        ),
    )


def measure_cell(
    cfg: ExperimentConfig,
    feature_sets: dict[str, list[str]],
    gap_feature_set: str,
    refresh: bool = False,
    only_model: str | None = None,
) -> None:
    processed = pipeline.processed_dir(cfg)
    out = pipeline.output_dir(cfg)
    dump_config(cfg, out / "config.yaml")
    names = [s.column for s in cfg.axioms.specs]

    print(f"[1/4] BM25 pool ({cfg.dataset.irds_id}, depth {cfg.first_stage.pool_depth})")
    pool = pipeline.build_pool(cfg, refresh)
    print(f"      {pool.qid.nunique()} queries, {len(pool)} pooled documents")

    print(f"[2/4] pair sampling ({cfg.pairs.strategy})")
    pairs = pipeline.build_pairs(cfg, pool, refresh)
    print(f"      {len(pairs)} canonical pairs over {pairs.qid.nunique()} queries")

    print(f"[3/4] axiom preferences ({len(names)} columns) -> {processed}")
    axiom_df = build_axiom_prefs(cfg, pool, pairs, refresh)

    rankers = cfg.all_rankers
    if only_model:
        rankers = [r for r in rankers if only_model in (r.model or "mock")]
        if not rankers:
            raise SystemExit(f"--only-model {only_model!r} matches no configured ranker")

    for ranker_cfg in rankers:
        model_name = ranker_cfg.model or "mock"
        print(f"[4/4] verdicts + analysis: {model_name}")
        store_df = pipeline.collect_verdicts(
            cfg.dataset.irds_id, ranker_cfg, pairs, PreferenceStore()
        )
        verdicts = agreement.model_pair_verdicts(store_df)
        merged = analysis.merge_pairs(axiom_df, verdicts)
        merged = analysis.attach_rank_gap(merged, pool)

        table = analysis.agreement_with_ci(merged, names, seed=cfg.seed)
        consistency = {
            **agreement.consistency_stats(verdicts),
            **agreement.nontransitivity_rate(verdicts),
            "mean_latency_ms": float(store_df["latency_ms"].mean()),
            "model": model_name,
        }

        joint = {}
        gap_oof = None
        for set_name, columns in feature_sets.items():
            stats, oof = analysis.joint_fit(merged, columns, seed=cfg.seed)
            joint[set_name] = stats
            if set_name == gap_feature_set:
                gap_oof = oof

        gradient = analysis.gap_gradient(merged, names, oof=gap_oof)

        metrics = out / "metrics" / model_name.replace("/", "__")
        metrics.mkdir(parents=True, exist_ok=True)
        table.to_csv(metrics / "agreement.csv", index=False)
        with open(metrics / "consistency.json", "w") as f:
            json.dump(consistency, f, indent=2)
        with open(metrics / "joint_fit.json", "w") as f:
            json.dump(joint, f, indent=2)
        gradient.to_csv(metrics / "gap_agreement.csv", index=False)
        analysis.gap_figure(
            gradient,
            metrics / "gap_agreement.png",
            title=f"{cfg.experiment}/{cfg.variant or ''} — {model_name}",
        )

        print(f"      -> {metrics}")
        print(table.to_string(index=False))
        summary = {
            name: {
                "base_rate": round(stats["base_rate"], 3),
                "majority_vote": round(stats["majority_vote_accuracy"], 3),
                "cv_accuracy": round(stats["cv_accuracy"], 3),
                "cv_auc": round(stats["cv_auc"], 3),
            }
            for name, stats in joint.items()
        }
        print(json.dumps({"joint_fit": summary, **consistency}, indent=2))
