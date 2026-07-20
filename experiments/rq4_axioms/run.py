"""RQ4: fitted residual-seed axioms, fidelity, ablation, and reranking effectiveness.

For each LLM target, this zero-call experiment fits query-disjoint OOF axiom surrogates
for four locked variants: classical, +VERB, +QCOV, and +both. It predicts every top-10
pair, Copeland-aggregates those predictions, and compares BM25, the reconstructed LLM,
the fitted variants, and an untuned vote diagnostic on nDCG@10/AP. Qrels are used only
after every pair prediction has been fitted.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from axiomrank import analysis, paths, ranking
from axiomrank.analysis import PAIR_KEY
from axiomrank.config import dump_config, load_config
from axiomrank.pipeline import merged_cell_frame, stages
from axiomrank.provenance import write_run_manifest

DEGENERATE = {"TFC1@len0.2", "TFC1@len0.5", "TFC3", "TFC3@len0.2", "TFC3@len0.5"}
RQ4_COLUMNS = {"VERB", "QCOV", "VERB@m0.2"}
NEW_AXIOMS = ["VERB", "QCOV"]

CAPTURE_BOOTSTRAP = 2_000
EFFECTIVENESS_BOOTSTRAP = 10_000


def _classical_columns(source_cfg) -> list[str]:
    return [
        spec.column
        for spec in source_cfg.axioms.lexical_specs
        if spec.column not in DEGENERATE and spec.column not in RQ4_COLUMNS
    ]


def _feature_sets(classical: list[str]) -> dict[str, list[str]]:
    """Locked nested variants for attribution of each proposed axiom."""
    return {
        "classical": classical,
        "plus_verb": [*classical, "VERB"],
        "plus_qcov": [*classical, "QCOV"],
        "plus_both": [*classical, "VERB", "QCOV"],
    }


def _load_cells(source_cfgs, ranker_cfg) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load model-specific pair frames while preserving collection-local query ids."""
    cells = {}
    parts = []
    offset = 0
    for src in source_cfgs:
        collection = src.variant or src.dataset.irds_id.replace("/", "_")
        merged, _ = merged_cell_frame(src, ranker_cfg, refresh=False)
        merged = merged.assign(collection=collection)
        merged["_group_id"] = collection + ":" + merged["query_id"].astype(str)
        merged.index = pd.RangeIndex(offset, offset + len(merged))
        offset += len(merged)
        cells[collection] = merged
        parts.append(merged)
    return pd.concat(parts), cells


def _query_bootstrap_ci(
    values: np.ndarray, groups: np.ndarray, seed: int, n_bootstrap: int
) -> tuple[float, float]:
    """Cluster bootstrap of a pair-level mean, resampling whole queries."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    by_group = {group: np.where(groups == group)[0] for group in unique}
    boot = np.empty(n_bootstrap)
    for draw in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([by_group[group] for group in sampled])
        boot[draw] = values[rows].mean()
    return tuple(float(x) for x in np.quantile(boot, [0.025, 0.975]))


def _capture(pooled: pd.DataFrame, classical: list[str], seed: int) -> dict:
    """Backward-compatible classical-vs-both fidelity decomposition."""
    frame = pooled.copy()
    frame["query_id"] = frame["_group_id"]
    results = {}
    oof = {}
    for name, features in {"classical": classical, "plus_both": [*classical, *NEW_AXIOMS]}.items():
        results[name], oof[name] = analysis.decompose(frame, features, seed=seed)

    left = oof["classical"][[*PAIR_KEY, "oof_correct", "oof_prob", "y_true"]]
    right = oof["plus_both"][[*PAIR_KEY, "oof_correct", "oof_prob"]]
    paired = left.merge(
        right,
        on=PAIR_KEY,
        suffixes=("_classical", "_plus_both"),
        validate="one_to_one",
    )
    accuracy_lift = (
        paired["oof_correct_plus_both"].astype(float).to_numpy()
        - paired["oof_correct_classical"].astype(float).to_numpy()
    )
    y = paired["y_true"].to_numpy(dtype=float)

    def per_pair_loss(probability):
        p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    logloss_lift = per_pair_loss(paired["oof_prob_classical"]) - per_pair_loss(
        paired["oof_prob_plus_both"]
    )
    groups = paired["query_id"].to_numpy()
    acc_ci = _query_bootstrap_ci(accuracy_lift, groups, seed, CAPTURE_BOOTSTRAP)
    loss_ci = _query_bootstrap_ci(logloss_lift, groups, seed, CAPTURE_BOOTSTRAP)

    def summary(result):
        return {
            "features": result["features"],
            "n_decisive_pairs": result["n_decisive_pairs"],
            "cv_null_accuracy": result["cv_null_accuracy"],
            "cv_accuracy": result["cv_accuracy"],
            "cv_auc": result["cv_auc"],
            "pseudo_r2": result["information"]["pseudo_r2"],
        }

    return {
        "classical": summary(results["classical"]),
        "plus_both": summary(results["plus_both"]),
        "oof_accuracy_lift": float(accuracy_lift.mean()),
        "oof_accuracy_lift_ci": list(acc_ci),
        "oof_logloss_lift": float(logloss_lift.mean()),
        "oof_logloss_lift_ci": list(loss_ci),
    }


def _run_from_preferences(frame, preferences, pool) -> pd.DataFrame:
    verdicts = frame[["query_id", "doc_id_1", "doc_id_2"]].copy()
    verdicts["model_pref"] = np.asarray(preferences, dtype=int)
    return ranking.copeland_ranking(verdicts, pool)


def _comparison(baseline, candidate, metrics, seed) -> dict:
    """Strict paired candidate-minus-baseline effectiveness comparison."""
    _, summary = ranking.compare_runs(
        baseline,
        candidate,
        metrics,
        n_bootstrap=EFFECTIVENESS_BOOTSTRAP,
        seed=seed,
    )
    return summary


def _variant_residual_profiles(
    pooled: pd.DataFrame,
    pair_predictions: pd.DataFrame,
    variants,
) -> pd.DataFrame:
    """Binned remaining OOF error by locked covariate, variant, and decisive target.

    Target ties are excluded because correctness relative to a binary surrogate is not
    defined for them. All predictions are out of fold; this is a diagnostic view of the
    held-out residual, not a refit on the error subset.
    """
    covariates = [
        name for name in ("d_len", "d_qcov", "d_rank", "rank_gap")
        if name in pooled.columns
    ]
    if not covariates:
        return pd.DataFrame()
    decisive = pair_predictions["target_pref"] != 0
    rows = []
    for variant in variants:
        frame = pd.DataFrame(
            {
                "query_id": pair_predictions.loc[decisive, "group_id"],
                "y_true": pair_predictions.loc[decisive, "target_pref"] > 0,
                "oof_correct": pair_predictions.loc[decisive, f"{variant}_correct"].astype(bool),
            }
        )
        for covariate in covariates:
            frame[covariate] = pooled.loc[decisive, covariate]
        profile = analysis.residual_profiles(frame, covariates)
        profile.insert(0, "variant", variant)
        rows.append(profile)
    return pd.concat(rows, ignore_index=True)


def _effectiveness_for_collection(
    *,
    collection: str,
    src,
    pool: pd.DataFrame,
    frame: pd.DataFrame,
    prediction_columns: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate all runs only after model-independent OOF predictions exist."""
    metrics = ranking.DEFAULT_METRICS
    metric_names = ranking.metric_names(metrics)

    run_frames = {"bm25": pool[["qid", "docno", "rank", "score"]].copy()}
    run_frames["llm"] = _run_from_preferences(frame, frame["model_pref"], pool)
    vote = np.sign(frame[feature_sets["plus_both"]].sum(axis=1)).astype(int)
    run_frames["untuned_vote"] = _run_from_preferences(frame, vote, pool)
    for variant in feature_sets:
        run_frames[variant] = _run_from_preferences(
            frame, prediction_columns.loc[frame.index, f"{variant}_pref"], pool
        )

    per_query = {
        name: ranking.evaluate_run(run, src.dataset.irds_id, metrics)
        for name, run in run_frames.items()
    }
    means = {
        name: {metric: float(scores[metric].mean()) for metric in metric_names}
        for name, scores in per_query.items()
    }

    comparisons = {"llm_vs_bm25": _comparison(per_query["bm25"], per_query["llm"], metrics, seed)}
    for variant in feature_sets:
        comparisons[f"{variant}_vs_bm25"] = _comparison(
            per_query["bm25"], per_query[variant], metrics, seed
        )
        comparisons[f"{variant}_vs_llm"] = _comparison(
            per_query["llm"], per_query[variant], metrics, seed
        )
    comparisons["untuned_vote_vs_bm25"] = _comparison(
        per_query["bm25"], per_query["untuned_vote"], metrics, seed
    )
    for candidate, baseline, label in (
        ("plus_verb", "classical", "plus_verb_vs_classical"),
        ("plus_qcov", "classical", "plus_qcov_vs_classical"),
        ("plus_both", "plus_verb", "qcov_given_verb"),
        ("plus_both", "plus_qcov", "verb_given_qcov"),
    ):
        comparisons[label] = _comparison(per_query[baseline], per_query[candidate], metrics, seed)

    fidelity = {
        variant: ranking.surrogate_fidelity(
            prediction_columns.loc[frame.index].rename(
                columns={
                    f"{variant}_prob": "surrogate_prob",
                    f"{variant}_pref": "surrogate_pref",
                    f"{variant}_correct": "correct_on_decisive",
                }
            )[[
                "group_id", "fold", "target_pref", "surrogate_prob",
                "surrogate_pref", "correct_on_decisive",
            ]]
        )
        for variant in feature_sets
    }
    diagnostics = {
        "target_ties": int((frame["model_pref"] == 0).sum()),
        "target_decisive": int((frame["model_pref"] != 0).sum()),
        "untuned_vote_ties": int((vote == 0).sum()),
        "untuned_vote_decisions": int((vote != 0).sum()),
    }
    if "collapse_reason" in frame:
        diagnostics["target_collapse_reasons"] = {
            str(reason): int(count)
            for reason, count in frame["collapse_reason"].value_counts().items()
        }

    per_query_rows = []
    for run_name, scores in per_query.items():
        tagged = scores.copy()
        tagged.insert(0, "run", run_name)
        tagged.insert(0, "collection", collection)
        per_query_rows.append(tagged)
    persisted_runs = []
    for run_name, run in run_frames.items():
        tagged = run.copy()
        tagged.insert(0, "run", run_name)
        tagged.insert(0, "collection", collection)
        persisted_runs.append(tagged)

    report = {
        "dataset": src.dataset.irds_id,
        "means": means,
        "comparisons": comparisons,
        "fidelity": fidelity,
        "tie_diagnostics": diagnostics,
    }
    return report, pd.concat(per_query_rows, ignore_index=True), pd.concat(persisted_runs, ignore_index=True)


def _run_model(source_cfgs, ranker_cfg, out, seed):
    model_name = ranker_cfg.model or "mock"
    metrics_dir = out / "metrics" / model_name.replace("/", "__")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pooled, cells = _load_cells(source_cfgs, ranker_cfg)
    classical = _classical_columns(source_cfgs[0])
    feature_sets = _feature_sets(classical)
    folds = ranking.assign_query_folds(pooled["_group_id"], n_folds=5)

    base_columns = ["collection", "query_id", "doc_id_1", "doc_id_2", "_group_id", "model_pref"]
    if "collapse_reason" in pooled:
        base_columns.append("collapse_reason")
    pair_predictions = pooled[base_columns].rename(
        columns={"_group_id": "group_id", "model_pref": "target_pref"}
    ).copy()
    pair_predictions["fold"] = folds
    model_metadata = {}
    for variant, features in feature_sets.items():
        prediction, metadata = ranking.fit_oof_surrogate(
            pooled,
            features,
            group_col="_group_id",
            folds=folds,
            seed=seed,
        )
        pair_predictions[f"{variant}_prob"] = prediction["surrogate_prob"]
        pair_predictions[f"{variant}_pref"] = prediction["surrogate_pref"]
        pair_predictions[f"{variant}_correct"] = prediction["correct_on_decisive"]
        model_metadata[variant] = metadata

    # Fit and prediction are complete before qrels-backed evaluation begins below.
    effectiveness = {}
    per_query_outputs = []
    run_outputs = []
    agreement_outputs = []
    for src in source_cfgs:
        collection = src.variant or src.dataset.irds_id.replace("/", "_")
        frame = cells[collection]
        pool = stages.build_pool(src, refresh=False)
        report, per_query, runs = _effectiveness_for_collection(
            collection=collection,
            src=src,
            pool=pool,
            frame=frame,
            prediction_columns=pair_predictions,
            feature_sets=feature_sets,
            seed=seed,
        )
        effectiveness[collection] = report
        per_query_outputs.append(per_query)
        run_outputs.append(runs)
        agreement = analysis.agreement_with_ci(frame, NEW_AXIOMS, seed=seed)
        agreement.insert(0, "collection", collection)
        agreement_outputs.append(agreement)

    pair_predictions.to_parquet(metrics_dir / "pair_predictions.parquet", index=False)
    pair_predictions[["collection", "query_id", "group_id", "fold"]].drop_duplicates().to_csv(
        metrics_dir / "folds.csv", index=False
    )
    pd.concat(per_query_outputs, ignore_index=True).to_csv(
        metrics_dir / "effectiveness_per_query.csv", index=False
    )
    pd.concat(run_outputs, ignore_index=True).to_parquet(metrics_dir / "runs.parquet", index=False)
    pd.concat(agreement_outputs, ignore_index=True).to_csv(
        metrics_dir / "new_axiom_agreement.csv", index=False
    )
    _variant_residual_profiles(pooled, pair_predictions, feature_sets).to_csv(
        metrics_dir / "residual_profiles.csv", index=False
    )
    with open(metrics_dir / "surrogate_models.json", "w") as handle:
        json.dump(model_metadata, handle, indent=2)
    with open(metrics_dir / "effectiveness.json", "w") as handle:
        json.dump(effectiveness, handle, indent=2)
    with open(metrics_dir / "capture.json", "w") as handle:
        json.dump(_capture(pooled, classical, seed), handle, indent=2)

    primary = ranking.metric_names()[0]
    print(f"[rq4] {model_name} -> {metrics_dir}")
    for collection, report in effectiveness.items():
        means = report["means"]
        print(
            f"      {collection}: BM25 {means['bm25'][primary]:.4f}, "
            f"LLM {means['llm'][primary]:.4f}, classical {means['classical'][primary]:.4f}, "
            f"+both {means['plus_both'][primary]:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only-model", help="substring filter on configured rankers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.sources:
        raise SystemExit("rq4 config needs a `sources:` list of grid-cell configs")
    source_cfgs = [load_config(paths.PROJECT_ROOT / path) for path in cfg.sources]
    rankers = source_cfgs[0].all_rankers
    if args.only_model:
        rankers = [ranker for ranker in rankers if args.only_model in (ranker.model or "mock")]
        if not rankers:
            raise SystemExit(f"--only-model {args.only_model!r} matches no configured ranker")

    out = stages.output_dir(cfg)
    dump_config(cfg, out / "config.yaml")
    for ranker_cfg in rankers:
        _run_model(source_cfgs, ranker_cfg, out, cfg.seed)
    write_run_manifest(
        out / "run_manifest.json",
        cfg,
        config_source=args.config,
        source_paths=[
            Path(__file__),
            paths.PROJECT_ROOT / "src" / "axiomrank" / "axioms",
            paths.PROJECT_ROOT / "src" / "axiomrank" / "analysis",
            paths.PROJECT_ROOT / "src" / "axiomrank" / "ranking",
            paths.PROJECT_ROOT / "src" / "axiomrank" / "pipeline",
        ],
        input_paths=[
            paths.PROJECT_ROOT / source for source in cfg.sources
        ]
        + [stages.processed_dir(source_cfg) for source_cfg in source_cfgs]
        + [paths.PREFERENCES_DIR],
        output_paths=[out],
        extra={
            "runner": "experiments/rq4_axioms/run.py",
            "selected_models": [ranker.model or "mock" for ranker in rankers],
            "source_configs": list(cfg.sources),
            "cache_policy": "development-config-guard-with-explicit-refresh",
        },
    )


if __name__ == "__main__":
    main()
