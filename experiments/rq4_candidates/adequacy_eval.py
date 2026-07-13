"""Evaluate whether the per-document answer-adequacy gap predicts pairwise preference.

For each target model and collection, the cached pairwise label ``model_pref`` (from the
protected preference store, loaded read-only via ``merged_cell_frame`` with
``allow_new=False``) is compared against ``delta = adequacy(d1) - adequacy(d2)`` from the Qwen
adequacy oracle (``adequacy.py``). Three questions, per the diagnostic design in
``docs/phase3-adequacy-oracle.md`` §3:

1. does ``sign(delta)`` agree with the preference on decisive pairs (is the property
   decision-relevant)?
2. what is the ranking AUC of ``delta`` against the preference (the ceiling any detector of
   this property could reach)?
3. does Qwen-adequacy predict the *FLAN* preferences, not just Qwen's own (circularity
   control — cross-model transfer is the honest test)?

No ranker calls, no downloads, no confirmation-set access.
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from axiomrank import paths
from axiomrank.config import load_config
from axiomrank.pipeline.frames import merged_cell_frame

ADEQUACY_MODEL = "models/qwen3.6-35B-A3B-AWQ"
DEADBAND = 0.10  # |delta| below this is treated as no adequacy discrimination (a tie)


def _load_adequacy() -> dict:
    root = paths.DATA_DIR / "adequacy" / ADEQUACY_MODEL.replace("/", "__")
    parts = sorted(glob.glob(str(root / "part-*.parquet")))
    if not parts:
        raise SystemExit(f"no adequacy scores under {root}; run adequacy.py first")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.dropna(subset=["adequacy"]).drop_duplicates(["collection", "qid", "docno"], keep="last")
    return {(r.collection, str(r.qid), str(r.docno)): r.adequacy for r in df.itertuples(index=False)}


def _evaluate(merged: pd.DataFrame, adequacy: dict, collection: str) -> dict:
    a1 = merged.apply(lambda r: adequacy.get((collection, str(r.query_id), str(r.doc_id_1))), axis=1)
    a2 = merged.apply(lambda r: adequacy.get((collection, str(r.query_id), str(r.doc_id_2))), axis=1)
    frame = merged.assign(delta=a1.to_numpy() - a2.to_numpy()).dropna(subset=["delta"])
    decisive = frame[frame["model_pref"] != 0].copy()
    y = decisive["model_pref"].to_numpy()
    delta = decisive["delta"].to_numpy()

    discriminating = np.abs(delta) > DEADBAND
    n_disc = int(discriminating.sum())
    sign_agree = (
        float((np.sign(delta[discriminating]) == y[discriminating]).mean()) if n_disc else float("nan")
    )
    # Ranking AUC of delta against the positive class (doc_1 preferred).
    pos = y > 0
    auc = float("nan")
    if len(np.unique(pos)) == 2:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(pos, delta))
    return {
        "n_pairs": int(len(frame)),
        "n_decisive": int(len(decisive)),
        "n_discriminating": n_disc,
        "discriminating_coverage": float(n_disc / len(decisive)) if len(decisive) else float("nan"),
        "sign_agreement": sign_agree,
        "auc": auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    paths.configure_caches()
    cfg = load_config(args.config)
    source_cfgs = [load_config(paths.PROJECT_ROOT / source) for source in cfg.sources]
    adequacy = _load_adequacy()

    rows = []
    for ranker_cfg in source_cfgs[0].all_rankers:
        target = ranker_cfg.model or "mock"
        for source_cfg in source_cfgs:
            collection = source_cfg.variant or source_cfg.dataset.irds_id.replace("/", "_")
            merged, _ = merged_cell_frame(source_cfg, ranker_cfg, refresh=False)
            report = _evaluate(merged, adequacy, collection)
            report.update({"target": target, "collection": collection})
            rows.append(report)

    table = pd.DataFrame(rows)[
        ["target", "collection", "n_decisive", "n_discriminating",
         "discriminating_coverage", "sign_agreement", "auc"]
    ]
    out = paths.results_dir("rq4_candidates") / "d0v2" / "adequacy_eval.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    print(f"\n[adequacy-eval] oracle={ADEQUACY_MODEL} deadband={DEADBAND} -> {out}")
    print(
        "Qwen rows are the self-consistency upper bound; the FLAN rows are the honest "
        "cross-model transfer test."
    )


if __name__ == "__main__":
    main()
