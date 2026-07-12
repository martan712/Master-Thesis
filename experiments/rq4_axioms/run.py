"""RQ4 (Phase 3): capture test + reranking validation of the VERB/QCOV residual axioms.

Two arms, both zero model calls (everything from the cached rq2 stages + preference store):

1. **Capture (decomposition).** Pool the two DL19/DL20 top-10 cells and, per ranker, fit the
   combined axiom model on (a) the classical lexical battery and (b) classical + {VERB, QCOV}
   on the same query-grouped folds. Report CV accuracy, pseudo-R² and the reducible-residual
   upper bound for each, plus the out-of-fold accuracy lift with a paired query-bootstrap CI
   (phase3-design.md §2 step 3). Qwen is primary; flan-t5-large and flan-t5-xl replicate.

2. **Reranking.** Per collection, aggregate the axiom battery into a per-pair preference by
   majority vote (the canonical untuned ir_axioms aggregate = sign of the column sum), Copeland
   it into a run, and evaluate nDCG@10 / MAP against the qrels — classical-only vs
   classical + {VERB, QCOV}. Report the lift over the classical battery with a paired
   query-bootstrap CI (phase3-design.md §2 step 4; §3.8 depth-ceiling: lift, not absolute nDCG).

Usage:
    uv run --no-sync python experiments/rq4_axioms/run.py --config configs/rq4_axioms.yaml
    uv run --no-sync python experiments/rq4_axioms/run.py --config configs/rq4_axioms.yaml --only-model qwen
"""

import argparse
import json

import numpy as np
import pandas as pd

from axiomrank import analysis, paths, ranking
from axiomrank.analysis import PAIR_KEY
from axiomrank.config import dump_config, load_config
from axiomrank.pipeline import merged_cell_frame, stages

# Degenerate columns dropped from the classical feature set (matches rq3 runner).
DEGENERATE = {"TFC1@len0.2", "TFC1@len0.5", "TFC3", "TFC3@len0.2", "TFC3@len0.5"}
# The RQ4 axiom columns (excluded from "classical").
RQ4_COLUMNS = {"VERB", "QCOV", "VERB@m0.2"}
# The headline additions tested by the capture/reranking arms (VERB_R is auxiliary).
NEW_AXIOMS = ["VERB", "QCOV"]

N_BOOT = 2000


def _classical_columns(source_cfg) -> list[str]:
    return [
        s.column
        for s in source_cfg.axioms.lexical_specs
        if s.column not in DEGENERATE and s.column not in RQ4_COLUMNS
    ]


def _pool(source_cfgs, ranker_cfg):
    """Pool the source cells, namespacing query ids by collection (as the rq3 runner does)."""
    per_collection = {}
    for src in source_cfgs:
        collection = src.variant or src.dataset.irds_id.replace("/", "_")
        merged, _ = merged_cell_frame(src, ranker_cfg, refresh=False)
        merged = merged.assign(collection=collection)
        merged["query_id"] = collection + ":" + merged["query_id"].astype(str)
        per_collection[collection] = merged
    return pd.concat(per_collection.values(), ignore_index=True), per_collection


def _bootstrap_ci(per_pair: np.ndarray, groups: np.ndarray, seed: int) -> tuple[float, float]:
    """Query-grouped bootstrap 95% CI of a per-pair mean (resample query groups)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in unique}
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in picked])
        boot[b] = per_pair[rows].mean()
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _capture(pooled, classical_cols, pos_cons, seed) -> dict:
    """Classical vs classical+{VERB,QCOV} decomposition, with paired OOF lift + CI."""
    augmented_cols = classical_cols + NEW_AXIOMS
    res_c, oof_c = analysis.decompose(pooled, classical_cols, pos_cons, seed=seed)
    res_a, oof_a = analysis.decompose(pooled, augmented_cols, pos_cons, seed=seed)

    # Align the two OOF frames on the pair key (same decisive subset, same folds).
    m = oof_c[[*PAIR_KEY, "oof_correct", "oof_prob", "y_true"]].merge(
        oof_a[[*PAIR_KEY, "oof_correct", "oof_prob"]], on=PAIR_KEY, suffixes=("_c", "_a")
    )
    lift = m["oof_correct_a"].astype(float).to_numpy() - m["oof_correct_c"].astype(float).to_numpy()
    groups = m["query_id"].to_numpy()
    ci_lo, ci_hi = _bootstrap_ci(lift, groups, seed)

    # Per-pair log-loss (cross-entropy) lift: positive = augmented sharpens the probability.
    # This is the information view Phase 2 treated as the honest figure (§3.1), and is more
    # sensitive than the 0.5-thresholded accuracy lift.
    eps = 1e-12
    y = m["y_true"].astype(float).to_numpy()

    def _ce(p):
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    ll_lift = _ce(m["oof_prob_c"].to_numpy()) - _ce(m["oof_prob_a"].to_numpy())
    ll_lo, ll_hi = _bootstrap_ci(ll_lift, groups, seed)

    def summary(res):
        return {
            "n_decisive_pairs": res["n_decisive_pairs"],
            "base_rate": res["base_rate"],
            "cv_accuracy": res["cv_accuracy"],
            "cv_auc": res["cv_auc"],
            "pseudo_r2": res["information"]["pseudo_r2"],
            "reliability_ceiling": res["reliability_ceiling"],
            "reducible_residual_upper": res["reducible_residual_upper"],
        }

    return {
        "classical": summary(res_c),
        "augmented": summary(res_a),
        "cv_accuracy_lift": res_a["cv_accuracy"] - res_c["cv_accuracy"],
        "pseudo_r2_lift": res_a["information"]["pseudo_r2"] - res_c["information"]["pseudo_r2"],
        "oof_lift": float(lift.mean()),
        "oof_lift_ci": [ci_lo, ci_hi],
        "logloss_lift": float(ll_lift.mean()),
        "logloss_lift_ci": [ll_lo, ll_hi],
        "new_axiom_coefficients": {
            k: res_a["coefficients"][k] for k in NEW_AXIOMS if k in res_a["coefficients"]
        },
    }


def _vote_verdicts(axiom_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Majority-vote aggregate of the axiom columns into a collapsed pair-verdict frame."""
    votes = axiom_df[cols].sum(axis=1)
    out = axiom_df[["qid", "doc_id_1", "doc_id_2"]].rename(columns={"qid": "query_id"}).copy()
    out["model_pref"] = np.sign(votes).astype(int)
    return out


def _rerank_metrics(axiom_df, pool, dataset_id, cols, metrics):
    verdicts = _vote_verdicts(axiom_df, cols)
    run = ranking.copeland_ranking(verdicts, pool)
    return ranking.evaluate_run(run, dataset_id, metrics)


def _reranking(source_cfgs, seed) -> dict:
    """Per-collection nDCG@10/MAP: classical-only vs classical+{VERB,QCOV} axiom aggregate."""
    metrics = ranking.DEFAULT_METRICS
    names = ranking.metric_names(metrics)
    out = {}
    for src in source_cfgs:
        collection = src.variant or src.dataset.irds_id.replace("/", "_")
        pool = stages.build_pool(src, refresh=False)
        pairs = stages.build_pairs(src, pool, refresh=False)
        axiom_df = stages.build_axiom_prefs(src, pool, pairs, refresh=False)
        classical_cols = _classical_columns(src)

        base = _rerank_metrics(axiom_df, pool, src.dataset.irds_id, classical_cols, metrics)
        aug = _rerank_metrics(
            axiom_df, pool, src.dataset.irds_id, classical_cols + NEW_AXIOMS, metrics
        )
        per_query, summary = ranking.compare_runs(base, aug, metrics)

        # Paired query-bootstrap CI on the per-query delta (aug - classical), per metric.
        col_out = {}
        for name in names:
            delta = per_query[f"{name}_delta"].to_numpy(dtype=float)
            groups = np.arange(len(delta))  # one query per row -> resample queries directly
            ci_lo, ci_hi = _bootstrap_ci(delta, groups, seed)
            col_out[name] = {
                "classical_mean": summary[name]["mean_baseline"],
                "augmented_mean": summary[name]["mean_reranked"],
                "lift": summary[name]["mean_delta"],
                "lift_ci": [ci_lo, ci_hi],
                "wins": summary[name]["wins"],
                "ties": summary[name]["ties"],
                "losses": summary[name]["losses"],
                "n_queries": summary[name]["n_queries"],
            }
        out[collection] = col_out
    return out


def _position_consistency(merged: pd.DataFrame) -> float | None:
    both = merged[merged["n_presentations"] >= 2]
    return float(both["position_consistent"].mean()) if len(both) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only-model", help="substring filter on the configured rankers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.sources:
        raise SystemExit("rq4 config needs a `sources:` list of grid-cell configs")
    source_cfgs = [load_config(paths.PROJECT_ROOT / s) for s in cfg.sources]
    classical_cols = _classical_columns(source_cfgs[0])

    rankers = source_cfgs[0].all_rankers
    if args.only_model:
        rankers = [r for r in rankers if args.only_model in (r.model or "mock")]
        if not rankers:
            raise SystemExit(f"--only-model {args.only_model!r} matches no configured ranker")

    out = stages.output_dir(cfg)
    dump_config(cfg, out / "config.yaml")

    # Reranking is model-independent (axiom aggregate), computed once.
    print("[rq4] reranking arm (axiom aggregate, per collection)")
    rerank = _reranking(source_cfgs, cfg.seed)
    with open(out / "reranking.json", "w") as f:
        json.dump(rerank, f, indent=2)
    for collection, cols in rerank.items():
        for name, s in cols.items():
            print(
                f"      {collection} {name:>7}: classical {s['classical_mean']:.4f} -> "
                f"+{'{:+.4f}'.format(s['lift'])} [{s['lift_ci'][0]:+.4f}, {s['lift_ci'][1]:+.4f}] "
                f"W/T/L {s['wins']}/{s['ties']}/{s['losses']}"
            )

    for ranker_cfg in rankers:
        model_name = ranker_cfg.model or "mock"
        print(f"[rq4] capture arm: {model_name}")
        pooled, _ = _pool(source_cfgs, ranker_cfg)
        capture = _capture(pooled, classical_cols, _position_consistency(pooled), cfg.seed)

        metrics_dir = out / "metrics" / model_name.replace("/", "__")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / "capture.json", "w") as f:
            json.dump(capture, f, indent=2)

        c, a = capture["classical"], capture["augmented"]
        print(
            f"      classical  acc {c['cv_accuracy']:.4f}  R² {c['pseudo_r2']:.4f}  "
            f"reducible↑ {c['reducible_residual_upper']:.4f}"
        )
        print(
            f"      augmented  acc {a['cv_accuracy']:.4f}  R² {a['pseudo_r2']:.4f}  "
            f"reducible↑ {a['reducible_residual_upper']:.4f}"
        )
        print(
            f"      OOF acc lift {capture['oof_lift']:+.4f} "
            f"[{capture['oof_lift_ci'][0]:+.4f}, {capture['oof_lift_ci'][1]:+.4f}]  "
            f"logloss lift {capture['logloss_lift']:+.4f} "
            f"[{capture['logloss_lift_ci'][0]:+.4f}, {capture['logloss_lift_ci'][1]:+.4f}]"
        )
        print(
            f"      ΔR² {capture['pseudo_r2_lift']:+.4f}  coefs {capture['new_axiom_coefficients']}"
        )


if __name__ == "__main__":
    main()
