"""RQ3: the explained/residual decomposition (phase2-design.md, phase2-implementation.md §3).

Pools the source grid cells named in the config (the two DL19/DL20 top-10 cells) from
cache — zero model calls — and, per ranker, fits the combined axiom model, decomposes its
verdicts into explained/residual with the information and noise-floor views, characterises
the residual (profiles, a CV residual model with lift over axiom-only, clusters), and emits
the joint-level gap gradient. Qwen is primary; flan-t5-large replicates.

Usage:
    uv run python experiments/rq3_decomposition/run.py --config configs/rq3_pooled_top10.yaml
    uv run python experiments/rq3_decomposition/run.py --config configs/rq3_pooled_top10.yaml --only-model qwen
"""

import argparse
import json

import pandas as pd

from axiomrank import analysis, paths
from axiomrank.analysis import CONTENT_COVARIATES, COVARIATE_COLUMNS, SIGNED_COVARIATES
from axiomrank.config import load_config
from axiomrank.pipeline import merged_cell_frame, stages

# Degenerate columns dropped from the RQ3 feature set (phase1-design.md §9.3 decision 1):
# TFC1@len{0.2,0.5} are bit-identical to strict TFC1, and TFC3 + its variants have <=4
# evaluable pairs.
DEGENERATE = {"TFC1@len0.2", "TFC1@len0.5", "TFC3", "TFC3@len0.2", "TFC3@len0.5"}


def _feature_sets(source_cfg) -> dict[str, list[str]]:
    lexical = [s.column for s in source_cfg.axioms.lexical_specs if s.column not in DEGENERATE]
    semantic = [s.column for s in source_cfg.axioms.semantic_specs]
    sets = {"lexical": lexical}
    if semantic:
        sets["lexical_semantic"] = lexical + semantic
    return sets


def _position_consistency(merged: pd.DataFrame) -> float | None:
    both = merged[merged["n_presentations"] >= 2]
    return float(both["position_consistent"].mean()) if len(both) else None


def _pool(source_cfgs, ranker_cfg, refresh) -> tuple[pd.DataFrame, dict]:
    """Pool the source cells into one frame, namespacing query ids by collection so the
    grouped-CV folds stay query-clean across collections."""
    per_collection: dict[str, pd.DataFrame] = {}
    for src in source_cfgs:
        collection = src.variant or src.dataset.irds_id.replace("/", "_")
        merged, _ = merged_cell_frame(src, ranker_cfg, refresh)
        merged = merged.assign(collection=collection)
        merged["query_id"] = collection + ":" + merged["query_id"].astype(str)
        per_collection[collection] = merged
        print(f"      {collection}: {len(merged)} pairs")
    return pd.concat(per_collection.values(), ignore_index=True), per_collection


def _decompose_all(merged, feature_sets, seed) -> dict:
    pos_cons = _position_consistency(merged)
    out = {}
    for name, cols in feature_sets.items():
        result, _ = analysis.decompose(
            merged, cols, position_consistency=pos_cons, seed=seed,
            nonlinear=(name == "lexical"),
        )
        out[name] = result
    return out


def _run_model(pooled, per_collection, feature_sets, seed, metrics_dir):
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Headline decomposition on the pooled cell, both feature sets.
    decomposition = _decompose_all(pooled, feature_sets, seed)
    with open(metrics_dir / "decomposition.json", "w") as f:
        json.dump(decomposition, f, indent=2)

    # Interpretable coefficients of the headline (lexical) combined model.
    coefs = decomposition["lexical"]["coefficients"]
    pd.DataFrame(
        {"axiom": list(coefs), "coefficient": list(coefs.values())}
    ).sort_values("coefficient", key=abs, ascending=False).to_csv(
        metrics_dir / "coefficients.csv", index=False
    )

    # Residual analysis off the headline (lexical) decomposition's OOF.
    _, oof = analysis.decompose(
        pooled, feature_sets["lexical"], position_consistency=_position_consistency(pooled),
        seed=seed,
    )
    oof_cov = oof.merge(pooled[[*analysis.PAIR_KEY, *COVARIATE_COLUMNS]], on=analysis.PAIR_KEY)

    analysis.residual_profiles(oof_cov, COVARIATE_COLUMNS).to_csv(
        metrics_dir / "residual_profiles.csv", index=False
    )
    residual = {
        "all_covariates": analysis.residual_model(oof_cov, SIGNED_COVARIATES, seed=seed),
        "content_only": analysis.residual_model(oof_cov, CONTENT_COVARIATES, seed=seed),
    }
    with open(metrics_dir / "residual_model.json", "w") as f:
        json.dump(residual, f, indent=2)
    analysis.residual_clusters(oof_cov, SIGNED_COVARIATES, seed=seed).to_csv(
        metrics_dir / "residual_clusters.csv", index=False
    )

    # Joint-level gap gradient (design §4 open item) off the same OOF.
    gradient = analysis.gap_gradient(
        pooled, feature_sets["lexical"], oof=oof.rename(columns={"is_residual": "_r"}),
        bm25=True,
    )
    gradient.to_csv(metrics_dir / "gap_gradient.csv", index=False)

    # Per-collection robustness: the headline decomposition on each source cell alone.
    by_col = metrics_dir / "by_collection"
    for collection, frame in per_collection.items():
        col_dir = by_col / collection
        col_dir.mkdir(parents=True, exist_ok=True)
        with open(col_dir / "decomposition.json", "w") as f:
            json.dump(_decompose_all(frame, feature_sets, seed), f, indent=2)

    return decomposition, residual


def _run_uniform_gradient(pooled_uniform, feature_sets, seed, metrics_dir):
    """The wide-gap arm of the gap-gradient open item (design §4): the combined model's
    OOF accuracy across the full BM25 rank-gap range, from the uniform50 cells."""
    _, oof = analysis.decompose(
        pooled_uniform, feature_sets["lexical"],
        position_consistency=_position_consistency(pooled_uniform), seed=seed,
    )
    gradient = analysis.gap_gradient(
        pooled_uniform, feature_sets["lexical"], oof=oof, bm25=True
    )
    gradient.to_csv(metrics_dir / "gap_gradient_uniform.csv", index=False)
    joint = gradient.drop_duplicates("gap_bin")[["gap_bin", "joint_cv_accuracy", "n_pairs"]]
    print(f"      wide-gap gradient -> {metrics_dir / 'gap_gradient_uniform.csv'}")
    print(joint.to_string(index=False))


def _report(decomposition, residual, metrics_dir):
    lex = decomposition["lexical"]
    print(f"      -> {metrics_dir}")
    print(json.dumps({
        "cv_accuracy": round(lex["cv_accuracy"], 3),
        "base_rate": round(lex["base_rate"], 3),
        "pseudo_r2": round(lex["information"]["pseudo_r2"], 4),
        "reliability_ceiling": lex["reliability_ceiling"],
        "nonlinear_headroom": round(lex.get("nonlinear_headroom", float("nan")), 3),
        "residual_lift_content": round(residual["content_only"]["lift"], 4),
        "residual_lift_content_ci": [
            round(residual["content_only"]["lift_ci_lo"], 4),
            round(residual["content_only"]["lift_ci_hi"], 4),
        ],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only-model", help="substring filter on the configured rankers")
    parser.add_argument("--refresh", action="store_true", help="recompute cached stages")
    parser.add_argument(
        "--uniform", action="store_true",
        help="also emit the wide-gap gradient from the config's uniform_sources (design §4)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.sources:
        raise SystemExit("rq3 config needs a `sources:` list of grid-cell configs to pool")
    source_cfgs = [load_config(paths.PROJECT_ROOT / s) for s in cfg.sources]
    feature_sets = _feature_sets(source_cfgs[0])
    uniform_cfgs = [load_config(paths.PROJECT_ROOT / s) for s in cfg.uniform_sources]
    if args.uniform and not uniform_cfgs:
        raise SystemExit("--uniform needs a `uniform_sources:` list in the rq3 config")

    rankers = source_cfgs[0].all_rankers
    if args.only_model:
        rankers = [r for r in rankers if args.only_model in (r.model or "mock")]
        if not rankers:
            raise SystemExit(f"--only-model {args.only_model!r} matches no configured ranker")

    out = stages.output_dir(cfg)
    from axiomrank.config import dump_config
    dump_config(cfg, out / "config.yaml")

    for ranker_cfg in rankers:
        model_name = ranker_cfg.model or "mock"
        metrics_dir = out / "metrics" / model_name.replace("/", "__")

        print(f"[rq3] {model_name}: pooling {len(source_cfgs)} top-10 cell(s)")
        pooled, per_collection = _pool(source_cfgs, ranker_cfg, args.refresh)
        decomposition, residual = _run_model(
            pooled, per_collection, feature_sets, cfg.seed, metrics_dir
        )
        _report(decomposition, residual, metrics_dir)

        if args.uniform:
            print(f"[rq3] {model_name}: pooling {len(uniform_cfgs)} uniform cell(s)")
            pooled_uniform, _ = _pool(uniform_cfgs, ranker_cfg, args.refresh)
            uniform_features = _feature_sets(uniform_cfgs[0])
            _run_uniform_gradient(pooled_uniform, uniform_features, cfg.seed, metrics_dir)


if __name__ == "__main__":
    main()
