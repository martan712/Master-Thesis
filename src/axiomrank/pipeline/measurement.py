"""The Phase 1 measurement recipe, shared by the rq1 and rq2 runners.

One *grid cell* (a config: collection × sampling condition, with a list of rankers)
produces, per ranker, the full set of Phase 1 outputs under
results/<experiment>/<variant>/metrics/<model>/ (phase1-implementation.md §3):

    agreement.csv       per-axiom coverage + agreement with query-bootstrap CIs
    consistency.json    position consistency, decisiveness, transitivity, latency
    joint_fit.json      base rate, majority vote, grouped-CV logistic per feature set
    gap_agreement.csv   agreement and joint accuracy binned by BM25 rank gap
    gap_agreement.png   draft of the RQ1 signature figure
"""

import json

from axiomrank import analysis
from axiomrank.config import ExperimentConfig, dump_config
from axiomrank.data.preferences import PreferenceStore
from axiomrank.pipeline import collect, stages


def measure_cell(
    cfg: ExperimentConfig,
    feature_sets: dict[str, list[str]],
    gap_feature_set: str,
    refresh: bool = False,
    only_model: str | None = None,
    analysis_columns: list[str] | None = None,
) -> None:
    processed = stages.processed_dir(cfg)
    out = stages.output_dir(cfg)
    dump_config(cfg, out / "config.yaml")
    configured_names = [s.column for s in cfg.axioms.specs]
    names = configured_names if analysis_columns is None else analysis_columns
    unknown = set(names) - set(configured_names)
    if unknown:
        raise ValueError(f"analysis columns are not configured axioms: {sorted(unknown)}")

    print(f"[1/4] BM25 pool ({cfg.dataset.irds_id}, depth {cfg.first_stage.pool_depth})")
    pool = stages.build_pool(cfg, refresh)
    print(f"      {pool.qid.nunique()} queries, {len(pool)} pooled documents")

    print(f"[2/4] pair sampling ({cfg.pairs.strategy})")
    pairs = stages.build_pairs(cfg, pool, refresh)
    print(f"      {len(pairs)} canonical pairs over {pairs.qid.nunique()} queries")

    print(f"[3/4] axiom preferences ({len(names)} columns) -> {processed}")
    axiom_df = stages.build_axiom_prefs(cfg, pool, pairs, refresh)

    rankers = cfg.all_rankers
    if only_model:
        rankers = [r for r in rankers if only_model in (r.model or "mock")]
        if not rankers:
            raise SystemExit(f"--only-model {only_model!r} matches no configured ranker")

    for ranker_cfg in rankers:
        model_name = ranker_cfg.model or "mock"
        print(f"[4/4] verdicts + analysis: {model_name}")
        store_df = collect.collect_verdicts(
            cfg.dataset.irds_id, ranker_cfg, pairs, PreferenceStore()
        )
        verdicts = analysis.model_pair_verdicts(store_df)
        merged = analysis.merge_pairs(axiom_df, verdicts)
        merged = analysis.attach_rank_gap(merged, pool)

        table = analysis.agreement_with_ci(merged, names, seed=cfg.seed)
        consistency = {
            **analysis.consistency_stats(verdicts),
            **analysis.nontransitivity_rate(verdicts),
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
