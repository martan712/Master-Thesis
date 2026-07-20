"""Evaluate deterministic Phase 3 candidate axioms on development data.

The runner computes candidate preferences from cached DL19/DL20 pools, joins them to
cached LLM labels, and compares query-disjoint OOF surrogates. It makes no ranker calls.
The evaluation is exploratory because the qualitative casebook that motivated these
features was drawn from the same development collections; it never accesses the locked
external confirmation collection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from axiomrank import analysis, paths, ranking
from axiomrank.axioms import axiom_preferences
from axiomrank.axioms.answering import candidate_query_precondition
from axiomrank.config import dump_config, load_config
from axiomrank.pipeline import merged_cell_frame, stages
from axiomrank.provenance import write_run_manifest

DEGENERATE = {"TFC1@len0.2", "TFC1@len0.5", "TFC3", "TFC3@len0.2", "TFC3@len0.5"}
NON_CLASSICAL = {"VERB", "QCOV", "VERB@m0.2"}
PAIR_BOOTSTRAP = 2_000
EFFECTIVENESS_BOOTSTRAP = 10_000
# Minimum distinct evaluable queries for a query bootstrap to express honest 95%
# uncertainty; below this the agreement interval is reported as NA rather than a
# degenerate one- or two-cluster range.
MIN_CI_CLUSTERS = 5


def _classical_columns(source_cfg) -> list[str]:
    return [
        spec.column
        for spec in source_cfg.axioms.lexical_specs
        if spec.column not in DEGENERATE and spec.column not in NON_CLASSICAL
    ]


def _feature_sets(classical: list[str], candidates: list[str]) -> dict[str, list[str]]:
    variants = {"classical": list(classical)}
    variants.update({f"plus_{name.lower()}": [*classical, name] for name in candidates})
    variants["plus_all_d0"] = [*classical, *candidates]
    return variants


def _candidate_preferences(source_cfg, candidate_specs) -> pd.DataFrame:
    pool = stages.build_pool(source_cfg, refresh=False)
    pairs = stages.build_pairs(source_cfg, pool, refresh=False)
    preferences = axiom_preferences(pool, pairs, candidate_specs)
    queries = pairs[["qid", "query"]].drop_duplicates("qid")
    return preferences.merge(queries, on="qid", validate="many_to_one")


def _join_candidate(base: pd.DataFrame, candidate: pd.DataFrame, collection: str) -> pd.DataFrame:
    """Exact one-to-one join of base cells and candidate preferences on the pair key.

    Uses an outer merge with an indicator so extra *and* missing candidate keys both fail,
    matching the repository's exact-pair merge policy (analysis.merge_pairs).
    """
    merged = base.merge(
        candidate.rename(columns={"qid": "query_id"}),
        on=["query_id", "doc_id_1", "doc_id_2"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(
            f"candidate/base pair keys do not match exactly for {collection}: {counts}"
        )
    return merged.drop(columns="_merge")


def _load_cells(source_cfgs, ranker_cfg, candidate_frames):
    cells = {}
    parts = []
    offset = 0
    for source_cfg in source_cfgs:
        collection = source_cfg.variant or source_cfg.dataset.irds_id.replace("/", "_")
        base, _ = merged_cell_frame(source_cfg, ranker_cfg, refresh=False)
        merged = _join_candidate(base, candidate_frames[collection], collection)
        merged = merged.assign(collection=collection)
        merged["_group_id"] = collection + ":" + merged["query_id"].astype(str)
        merged.index = pd.RangeIndex(offset, offset + len(merged))
        offset += len(merged)
        cells[collection] = merged
        parts.append(merged)
    return pd.concat(parts), cells


def _query_bootstrap(values: np.ndarray, groups: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    draws = np.empty(PAIR_BOOTSTRAP)
    for draw in range(PAIR_BOOTSTRAP):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([rows_by_group[group] for group in sampled])
        draws[draw] = values[rows].mean()
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _incremental_fidelity(
    predictions: pd.DataFrame, baseline: str, candidate: str, seed: int
) -> dict:
    decisive = predictions["target_pref"] != 0
    frame = predictions.loc[decisive]
    y = (frame["target_pref"] > 0).to_numpy(dtype=float)
    baseline_correct = frame[f"{baseline}_correct"].to_numpy(dtype=float)
    candidate_correct = frame[f"{candidate}_correct"].to_numpy(dtype=float)
    accuracy_lift = candidate_correct - baseline_correct

    def loss(probability):
        probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
        return -(y * np.log(probability) + (1 - y) * np.log(1 - probability))

    logloss_lift = loss(frame[f"{baseline}_prob"]) - loss(frame[f"{candidate}_prob"])
    groups = frame["group_id"].to_numpy()
    return {
        "accuracy_lift": float(accuracy_lift.mean()),
        "accuracy_lift_ci": _query_bootstrap(accuracy_lift, groups, seed),
        "logloss_lift": float(logloss_lift.mean()),
        "logloss_lift_ci": _query_bootstrap(logloss_lift, groups, seed),
    }


def _run_from_preferences(frame, preferences, pool):
    verdicts = frame[["query_id", "doc_id_1", "doc_id_2"]].copy()
    verdicts["model_pref"] = np.asarray(preferences, dtype=int)
    return ranking.copeland_ranking(verdicts, pool)


def _evaluate_collection(source_cfg, pool, frame, predictions, variants, seed):
    metrics = ranking.DEFAULT_METRICS
    names = ranking.metric_names(metrics)
    runs = {
        "bm25": pool[["qid", "docno", "rank", "score"]].copy(),
        "llm": _run_from_preferences(frame, frame["model_pref"], pool),
    }
    for variant in variants:
        runs[variant] = _run_from_preferences(
            frame, predictions.loc[frame.index, f"{variant}_pref"], pool
        )

    per_query = {
        name: ranking.evaluate_run(run, source_cfg.dataset.irds_id, metrics)
        for name, run in runs.items()
    }
    means = {
        run: {metric: float(scores[metric].mean()) for metric in names}
        for run, scores in per_query.items()
    }
    comparisons = {}
    for candidate in ["llm", *variants]:
        _, comparisons[f"{candidate}_vs_bm25"] = ranking.compare_runs(
            per_query["bm25"],
            per_query[candidate],
            metrics,
            n_bootstrap=EFFECTIVENESS_BOOTSTRAP,
            seed=seed,
        )
    for candidate in variants:
        if candidate == "classical":
            continue
        _, comparisons[f"{candidate}_vs_classical"] = ranking.compare_runs(
            per_query["classical"],
            per_query[candidate],
            metrics,
            n_bootstrap=EFFECTIVENESS_BOOTSTRAP,
            seed=seed,
        )
        _, comparisons[f"{candidate}_vs_llm"] = ranking.compare_runs(
            per_query["llm"],
            per_query[candidate],
            metrics,
            n_bootstrap=EFFECTIVENESS_BOOTSTRAP,
            seed=seed,
        )
    return {"means": means, "comparisons": comparisons}, per_query, runs


def _candidate_agreement(frame: pd.DataFrame, candidates: list[str], seed: int) -> pd.DataFrame:
    table = analysis.agreement_with_ci(
        frame, candidates, seed=seed, min_ci_clusters=MIN_CI_CLUSTERS
    )
    active_queries = {
        candidate: int(frame.loc[frame[candidate] != 0, "query_id"].nunique())
        for candidate in candidates
    }
    table["n_active_queries"] = table["axiom"].map(active_queries)
    table["n_queries"] = int(frame["query_id"].nunique())
    query_rows = frame[["query_id", "query"]].drop_duplicates("query_id")
    eligible_queries = {
        candidate: set(
            query_rows.loc[
                query_rows["query"].map(
                    lambda query: candidate_query_precondition(candidate, query)
                ),
                "query_id",
            ]
        )
        for candidate in candidates
    }
    table["n_eligible_queries"] = table["axiom"].map(
        {candidate: len(qids) for candidate, qids in eligible_queries.items()}
    )
    conditional_coverage = {}
    positive = {}
    negative = {}
    for candidate in candidates:
        eligible = frame["query_id"].isin(eligible_queries[candidate])
        conditional_coverage[candidate] = (
            float((frame.loc[eligible, candidate] != 0).mean()) if eligible.any() else float("nan")
        )
        positive[candidate] = int((frame[candidate] > 0).sum())
        negative[candidate] = int((frame[candidate] < 0).sum())
    table["eligible_pair_coverage"] = table["axiom"].map(conditional_coverage)
    table["n_positive"] = table["axiom"].map(positive)
    table["n_negative"] = table["axiom"].map(negative)
    return table


def _frozen_fold_prob(
    pooled: pd.DataFrame, features: list[str], fold_models: list[dict], folds, zero: str | None
) -> np.ndarray:
    """Reapply the fitted per-fold coefficients out of fold, optionally zeroing one feature.

    With ``zero=None`` this reproduces the surrogate's OOF probability. With ``zero`` set to a
    candidate feature, all coefficients stay frozen and only that feature is set to 0, which
    isolates the candidate's *direct* pair effect from the full add-one refit (which also
    moves the classical coefficients).
    """
    matrix = pooled[features].to_numpy(dtype=float)
    if zero is not None:
        matrix = matrix.copy()
        matrix[:, features.index(zero)] = 0.0
    folds = np.asarray(folds)
    probability = np.full(len(pooled), np.nan)
    for fold_model in fold_models:
        test = folds == fold_model["fold"]
        coef = np.array([fold_model["coefficients"][name] for name in features])
        logit = fold_model["intercept"] + matrix[test] @ coef
        probability[test] = 1.0 / (1.0 + np.exp(-logit))
    return probability


def _feature_zero_effects(pooled, predictions, feature_sets, models, candidate_names, folds):
    """Direct feature-zero counterfactuals, held-out and with frozen coefficients.

    For each candidate, in both its own add-one model and the all-D0 union model, count the
    held-out pairs and queries whose hard preference changes when only that candidate feature
    is zeroed. This is the query-level effect attributable to the feature itself, as opposed
    to the full add-one ablation, which conflates it with refitting the classical weights.
    """
    from axiomrank.ranking.surrogate import _hard_preference

    group = predictions["group_id"].to_numpy()
    effects = {}
    for candidate in candidate_names:
        effects[candidate] = {}
        for host in (f"plus_{candidate.lower()}", "plus_all_d0"):
            features = feature_sets[host]
            base_prob = _frozen_fold_prob(pooled, features, models[host]["fold_models"], folds, None)
            # Frozen-coefficient reconstruction must match the surrogate's own OOF output.
            assert np.allclose(base_prob, predictions[f"{host}_prob"].to_numpy(), atol=1e-9)
            zero_prob = _frozen_fold_prob(
                pooled, features, models[host]["fold_models"], folds, candidate
            )
            changed = _hard_preference(base_prob) != _hard_preference(zero_prob)
            effects[candidate][host] = {
                "n_pairs_changed": int(changed.sum()),
                "n_queries_changed": int(pd.unique(group[changed]).size),
            }
    return effects


def _run_model(source_cfgs, ranker_cfg, candidate_frames, candidate_names, out, seed):
    model_name = ranker_cfg.model or "mock"
    model_dir = out / "metrics" / model_name.replace("/", "__")
    model_dir.mkdir(parents=True, exist_ok=True)
    pooled, cells = _load_cells(source_cfgs, ranker_cfg, candidate_frames)
    feature_sets = _feature_sets(_classical_columns(source_cfgs[0]), candidate_names)
    folds = ranking.assign_query_folds(pooled["_group_id"], n_folds=5)

    predictions = pooled[
        ["collection", "query_id", "doc_id_1", "doc_id_2", "_group_id", "model_pref"]
    ].rename(columns={"_group_id": "group_id", "model_pref": "target_pref"})
    predictions["fold"] = folds
    models = {}
    for variant, features in feature_sets.items():
        prediction, metadata = ranking.fit_oof_surrogate(
            pooled, features, group_col="_group_id", folds=folds, seed=seed
        )
        predictions[f"{variant}_prob"] = prediction["surrogate_prob"]
        predictions[f"{variant}_pref"] = prediction["surrogate_pref"]
        predictions[f"{variant}_correct"] = prediction["correct_on_decisive"]
        models[variant] = metadata

    fidelity = {
        variant: {
            "metrics": models[variant]["fidelity"],
            **(
                {}
                if variant == "classical"
                else _incremental_fidelity(predictions, "classical", variant, seed)
            ),
        }
        for variant in feature_sets
    }
    feature_zero = _feature_zero_effects(
        pooled, predictions, feature_sets, models, candidate_names, folds
    )

    effectiveness = {}
    agreement_parts = []
    per_query_parts = []
    run_parts = []
    for source_cfg in source_cfgs:
        collection = source_cfg.variant or source_cfg.dataset.irds_id.replace("/", "_")
        frame = cells[collection]
        pool = stages.build_pool(source_cfg, refresh=False)
        report, per_query, runs = _evaluate_collection(
            source_cfg, pool, frame, predictions, feature_sets, seed
        )
        effectiveness[collection] = report
        agreement = _candidate_agreement(frame, candidate_names, seed)
        agreement.insert(0, "collection", collection)
        agreement_parts.append(agreement)
        for run_name, scores in per_query.items():
            tagged = scores.copy()
            tagged.insert(0, "run", run_name)
            tagged.insert(0, "collection", collection)
            per_query_parts.append(tagged)
        for run_name, run in runs.items():
            tagged = run.copy()
            tagged.insert(0, "run", run_name)
            tagged.insert(0, "collection", collection)
            run_parts.append(tagged)

    predictions.to_parquet(model_dir / "pair_predictions.parquet", index=False)
    pd.concat(agreement_parts, ignore_index=True).to_csv(
        model_dir / "candidate_agreement.csv", index=False
    )
    pd.concat(per_query_parts, ignore_index=True).to_csv(
        model_dir / "effectiveness_per_query.csv", index=False
    )
    pd.concat(run_parts, ignore_index=True).to_parquet(model_dir / "runs.parquet", index=False)
    for filename, payload in (
        ("surrogate_models.json", models),
        ("fidelity.json", fidelity),
        ("feature_zero.json", feature_zero),
        ("effectiveness.json", effectiveness),
    ):
        with open(model_dir / filename, "w") as handle:
            json.dump(payload, handle, indent=2)

    primary = ranking.metric_names()[0]
    print(f"[rq4-candidates] {model_name} -> {model_dir}")
    for collection, report in effectiveness.items():
        means = report["means"]
        print(
            f"  {collection}: BM25 {means['bm25'][primary]:.4f}, "
            f"LLM {means['llm'][primary]:.4f}, classical {means['classical'][primary]:.4f}, "
            f"+all D0 {means['plus_all_d0'][primary]:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only-model", help="substring filter on configured rankers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.sources or not cfg.axioms.specs:
        raise SystemExit("candidate evaluation needs `sources` and `axioms`")
    source_cfgs = [load_config(paths.PROJECT_ROOT / source) for source in cfg.sources]
    candidates = [spec.column for spec in cfg.axioms.specs]
    candidate_frames = {}
    for source_cfg in source_cfgs:
        collection = source_cfg.variant or source_cfg.dataset.irds_id.replace("/", "_")
        candidate_frames[collection] = _candidate_preferences(source_cfg, cfg.axioms.specs)

    out = stages.output_dir(cfg)
    dump_config(cfg, out / "config.yaml")
    for collection, frame in candidate_frames.items():
        frame.to_parquet(out / f"candidate_preferences_{collection}.parquet", index=False)

    rankers = source_cfgs[0].all_rankers
    if args.only_model:
        rankers = [ranker for ranker in rankers if args.only_model in (ranker.model or "mock")]
    if not rankers:
        raise SystemExit(f"--only-model {args.only_model!r} matches no configured ranker")
    for ranker_cfg in rankers:
        _run_model(source_cfgs, ranker_cfg, candidate_frames, candidates, out, cfg.seed)
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
            "runner": "experiments/rq4_candidates/run.py",
            "selected_models": [ranker.model or "mock" for ranker in rankers],
            "source_configs": list(cfg.sources),
            "candidate_columns": candidates,
            "cache_policy": "development-config-guard-with-explicit-refresh",
            "legacy_provenance_note": (
                "provenance.json may describe an older retained run; run_manifest.json is "
                "authoritative for runs made by this version"
            ),
        },
    )


if __name__ == "__main__":
    main()
