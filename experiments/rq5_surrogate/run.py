"""RQ5 preview: the fitted axiom-surrogate reranker (Phase 3 bridge, docs/phase3-design.md).

The Phase 1 effectiveness gate showed the *untuned* axiom majority vote reranks BM25's top-10
to roughly BM25 itself (every axiom one equal vote). This asks the fair question instead: how
far can a *weighted* axiom model get toward the LLM? Per LLM target, we fit an L2 logistic on
the axiom battery to predict that model's pairwise verdicts — query-grouped out-of-fold, so a
held-out query's pairs are scored by a model trained only on other queries (no leakage) — then
Copeland-aggregate the surrogate's predicted preferences into a run and compare it, per
collection, to BM25, the untuned axiom vote, the LLM, and the perfect-top-10 oracle.

The surrogate predicts on ALL top-10 pairs (decisive or tied for the LLM), so it never peeks at
which pairs the LLM tied on; its preference is the hard sign of P(doc_1 preferred) − 0.5. Both
the classical battery and classical + {VERB, QCOV} are fitted, to show the two increment-1
axioms' marginal value to the surrogate. Zero model calls — everything is from the cached
rq2 stages + preference store.

Usage:
    uv run --no-sync python experiments/rq5_surrogate/run.py --config configs/rq5_surrogate.yaml
    uv run --no-sync python experiments/rq5_surrogate/run.py --config configs/rq5_surrogate.yaml --only-model qwen
"""

import argparse
import json

import numpy as np
import pandas as pd

from axiomrank import analysis, paths, ranking
from axiomrank.config import dump_config, load_config
from axiomrank.pipeline import merged_cell_frame, stages

# Degenerate columns dropped from the feature set (matches the rq3/rq4 runners).
DEGENERATE = {"TFC1@len0.2", "TFC1@len0.5", "TFC3", "TFC3@len0.2", "TFC3@len0.5"}
RQ4_COLUMNS = {"VERB", "QCOV", "VERB@m0.2"}
NEW_AXIOMS = ["VERB", "QCOV"]
N_BOOT = 2000


def _classical_columns(source_cfg) -> list[str]:
    return [
        s.column
        for s in source_cfg.axioms.lexical_specs
        if s.column not in DEGENERATE and s.column not in RQ4_COLUMNS
    ]


def _boot_ci(delta: np.ndarray, seed: int) -> tuple[float, float]:
    """Query-bootstrap 95% CI of a per-query mean delta (one row per query)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(delta, dtype=float)
    boot = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(N_BOOT)])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _surrogate_oof_pref(pooled: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    """Out-of-fold surrogate preference (+1/-1) for EVERY pair in `pooled`.

    Query-grouped folds; per fold, fit the same L2 logistic as the decomposition on the
    training queries' DECISIVE pairs, then predict P(doc_1 preferred) on ALL held-out pairs.
    Hard sign of (p - 0.5) is the surrogate's Copeland preference.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    X = pooled[features].to_numpy(dtype=float)
    pref = pooled["model_pref"].to_numpy()
    groups = pooled["_nqid"].to_numpy()
    prob = np.full(len(pooled), np.nan)

    n_folds = min(5, len(np.unique(groups)))
    for train, test in GroupKFold(n_splits=n_folds).split(X, pref, groups):
        dec = train[pref[train] != 0]  # train only on the LLM's decisive pairs
        y = pref[dec] > 0
        clf = LogisticRegression(max_iter=1000, random_state=seed).fit(X[dec], y)
        prob[test] = clf.predict_proba(X[test])[:, 1]
    return np.where(prob >= 0.5, 1, -1).astype(int)


def _run_from_pref(frame: pd.DataFrame, pref: np.ndarray, pool, dataset_id, metrics):
    verdicts = frame[["query_id", "doc_id_1", "doc_id_2"]].copy()
    verdicts["model_pref"] = pref
    run = ranking.copeland_ranking(verdicts, pool)
    return ranking.evaluate_run(run, dataset_id, metrics)


def _oracle_run(pool: pd.DataFrame, dataset_id: str, depth: int = 10) -> pd.DataFrame:
    """Perfect reorder of BM25's top-`depth` by qrels grade — the top-10 nDCG ceiling."""
    import ir_datasets

    qrels: dict = {}
    for q in ir_datasets.load(dataset_id).qrels_iter():
        qrels.setdefault(q.query_id, {})[q.doc_id] = q.relevance
    rows = []
    for qid, g in pool.groupby("qid", sort=False):
        g = g.sort_values("rank")
        top = g.head(depth).copy()
        grades = qrels.get(str(qid), {})
        top["_g"] = [grades.get(str(d), 0) for d in top["docno"]]
        top = top.sort_values(["_g", "rank"], ascending=[False, True])
        order = list(top["docno"]) + list(g.iloc[depth:]["docno"])
        rows += [(qid, d, nr, float(-nr)) for nr, d in enumerate(order)]
    return pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only-model", help="substring filter on the configured rankers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.sources:
        raise SystemExit("rq5 config needs a `sources:` list of grid-cell configs")
    source_cfgs = [load_config(paths.PROJECT_ROOT / s) for s in cfg.sources]
    classical = _classical_columns(source_cfgs[0])
    feature_sets = {"classical": classical, "classical+VERB+QCOV": classical + NEW_AXIOMS}

    metrics = ranking.DEFAULT_METRICS
    names = ranking.metric_names(metrics)
    out = stages.output_dir(cfg)
    dump_config(cfg, out / "config.yaml")

    # Per collection: pool, BM25 baseline, oracle ceiling (all model-independent).
    per_collection = {}
    for src in source_cfgs:
        collection = src.variant or src.dataset.irds_id.replace("/", "_")
        pool = stages.build_pool(src, refresh=False)
        base = ranking.evaluate_run(pool, src.dataset.irds_id, metrics).set_index("query_id")
        oracle = ranking.evaluate_run(
            _oracle_run(pool, src.dataset.irds_id), src.dataset.irds_id, metrics
        ).set_index("query_id")
        per_collection[collection] = {"src": src, "pool": pool, "base": base, "oracle": oracle}

    rankers = source_cfgs[0].all_rankers
    if args.only_model:
        rankers = [r for r in rankers if args.only_model in (r.model or "mock")]
        if not rankers:
            raise SystemExit(f"--only-model {args.only_model!r} matches no configured ranker")

    report: dict = {}
    for ranker_cfg in rankers:
        model_name = ranker_cfg.model or "mock"
        # Build each cell's merged frame (top-10 pairs only) and pool for CV + reranking.
        cells = {}
        pooled_parts = []
        for collection, ctx in per_collection.items():
            merged, _ = merged_cell_frame(ctx["src"], ranker_cfg, refresh=False)
            if merged.empty:
                break
            merged = merged.assign(collection=collection)
            merged["_nqid"] = collection + ":" + merged["query_id"].astype(str)
            cells[collection] = merged
            pooled_parts.append(merged)
        if len(cells) != len(per_collection):
            print(f"[rq5] {model_name}: no cached verdicts for a cell — skipping")
            continue
        pooled = pd.concat(pooled_parts, ignore_index=True)

        # Fit the surrogate OOF once per feature set over the pooled cells.
        surrogate_pref = {
            set_name: pd.Series(_surrogate_oof_pref(pooled, cols, cfg.seed), index=pooled.index)
            for set_name, cols in feature_sets.items()
        }

        report[model_name] = {}
        for collection, merged in cells.items():
            ctx = per_collection[collection]
            pool, base, oracle = ctx["pool"], ctx["base"], ctx["oracle"]
            dataset_id = ctx["src"].dataset.irds_id
            sub = pooled["collection"] == collection

            runs = {
                "axiom_vote": _run_from_pref(
                    merged,
                    np.sign(merged[feature_sets["classical+VERB+QCOV"]].sum(axis=1)).astype(int),
                    pool, dataset_id, metrics,
                ),
                "llm": _run_from_pref(
                    merged, merged["model_pref"].to_numpy(), pool, dataset_id, metrics
                ),
            }
            for set_name in feature_sets:
                runs[f"surrogate[{set_name}]"] = _run_from_pref(
                    merged, surrogate_pref[set_name][sub.values].to_numpy(),
                    pool, dataset_id, metrics,
                )

            col_report = {"bm25": {n: float(base[n].mean()) for n in names},
                          "oracle": {n: float(oracle[n].mean()) for n in names}}
            for label, perq in runs.items():
                perq = perq.set_index("query_id")
                entry = {}
                for n in names:
                    delta = (perq[n] - base[n]).dropna()
                    lo, hi = _boot_ci(delta.values, cfg.seed)
                    entry[n] = {
                        "value": float(perq[n].mean()),
                        "delta_vs_bm25": float(delta.mean()),
                        "ci": [lo, hi],
                    }
                col_report[label] = entry
            report[model_name][collection] = col_report

    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "surrogate_reranking.json", "w") as f:
        json.dump(report, f, indent=2)

    # Console table.
    primary = names[0]
    for model_name, cols in report.items():
        print(f"\n=== surrogate of {model_name}  ({primary}, Δ vs BM25 [95% CI]; * excludes 0)")
        for collection, cr in cols.items():
            print(f"  {collection}:  BM25 {cr['bm25'][primary]:.4f}   "
                  f"oracle {cr['oracle'][primary]:.4f}")
            order = ["axiom_vote", "surrogate[classical]", "surrogate[classical+VERB+QCOV]", "llm"]
            for label in order:
                s = cr[label][primary]
                lo, hi = s["ci"]
                star = "*" if (lo > 0 or hi < 0) else " "
                print(f"    {label:34s} {s['value']:.4f}   "
                      f"{s['delta_vs_bm25']:+.4f} [{lo:+.4f},{hi:+.4f}] {star}")
    print(f"\n-> {metrics_dir / 'surrogate_reranking.json'}")


if __name__ == "__main__":
    main()
