"""Non-axiom covariates of a pair, for the RQ3 residual analysis (phase2-design.md §3.4).

Everything here is derivable from the cached frames (pool, pairs, preference store) with no
model call and no download. Covariates come in two shapes:

- **signed** (doc_1 minus doc_2): directional features a residual model uses to *predict*
  which document the LLM prefers — the same shape as the axiom columns (+1 favours doc_1).
- **magnitude** (order-invariant): difficulty/stratification features for the residual
  profiles (rank/score gap, verbosity, query type, the model's own confidence margin).

The content-based signed covariates (`CONTENT_COVARIATES`) are the ones the design's §6.1
RQ4-main-act gate requires signal in; BM25/rank strength and the model's confidence are
deliberately excluded from that set (design §3.4: predictable-from-confidence is a
calibration finding, not an axiom seed).
"""

import re

import numpy as np
import pandas as pd

from axiomrank.analysis.agreement import attach_rank_gap
from axiomrank.analysis.verdicts import PAIR_KEY

_WORD = re.compile(r"[a-z0-9]+")
_WH = {"who", "what", "when", "where", "why", "how", "which", "whom", "whose", "is", "are",
       "do", "does", "can", "could", "should", "would", "will"}

# Signed (doc_1 − doc_2) covariates a residual model predicts the LLM's verdict sign from.
SIGNED_COVARIATES = ["d_rank", "d_score", "d_len", "d_qcov"]
# Order-invariant covariates for the residual profiles / difficulty stratification.
MAGNITUDE_COVARIATES = [
    "rank_gap", "score_gap", "len_max", "len_ratio", "query_len", "query_is_question",
    "conf_margin_prob", "conf_margin_score",
]
COVARIATE_COLUMNS = SIGNED_COVARIATES + MAGNITUDE_COVARIATES
# The content/directional subset the §6.1 gate reads (excludes rank/score strength and
# the model's own confidence).
CONTENT_COVARIATES = ["d_len", "d_qcov"]


def _terms(text: str) -> set[str]:
    return set(_WORD.findall(str(text).lower()))


def _tokens(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def _pair_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Length, query-coverage and query-shape covariates from the pair text (word-level)."""
    rows = []
    for r in pairs.itertuples(index=False):
        q = _terms(r.query)
        t1, t2 = _tokens(r.text_1), _tokens(r.text_2)
        s1, s2 = set(t1), set(t2)
        len_1, len_2 = len(t1), len(t2)
        cov_1 = len(q & s1) / len(q) if q else 0.0
        cov_2 = len(q & s2) / len(q) if q else 0.0
        first = _tokens(r.query)[:1]
        rows.append(
            {
                "query_id": r.qid,
                "doc_id_1": r.doc_id_1,
                "doc_id_2": r.doc_id_2,
                "d_len": float(len_1 - len_2),
                "len_max": float(max(len_1, len_2)),
                "len_ratio": float(max(len_1, len_2) / max(min(len_1, len_2), 1)),
                "d_qcov": float(cov_1 - cov_2),
                "query_len": float(len(q)),
                "query_is_question": float("?" in str(r.query) or (first and first[0] in _WH)),
            }
        )
    return pd.DataFrame(rows)


def _confidence(store_df: pd.DataFrame) -> pd.DataFrame:
    """Per canonical pair, the model's mean decision-confidence margin over its
    presentations: |prob_a − 0.5| and |score_a − score_b| (both order-invariant)."""
    df = store_df.copy()
    lo = df[["doc_id_a", "doc_id_b"]].min(axis=1)
    hi = df[["doc_id_a", "doc_id_b"]].max(axis=1)
    df["doc_id_1"], df["doc_id_2"] = lo, hi
    df["conf_margin_prob"] = (df["prob_a"].astype(float) - 0.5).abs()
    df["conf_margin_score"] = (df["score_a"].astype(float) - df["score_b"].astype(float)).abs()
    return (
        df.groupby(PAIR_KEY, sort=True)[["conf_margin_prob", "conf_margin_score"]]
        .mean()
        .reset_index()
    )


def attach_covariates(
    merged: pd.DataFrame,
    pool: pd.DataFrame,
    pairs: pd.DataFrame,
    store_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add the non-axiom covariate columns (`COVARIATE_COLUMNS`) to a merged pair frame.

    `merged` is one row per canonical pair with the model verdict and axiom columns;
    `pool` supplies BM25 rank/score, `pairs` the text, and `store_df` (optional) the
    per-presentation confidence. Missing confidence (single-order pairs) stays NaN.
    """
    df = merged if "rank_gap" in merged.columns else attach_rank_gap(merged, pool)
    df = df.copy()
    df["d_rank"] = (df["rank_1"] - df["rank_2"]).astype(float)
    df["rank_max"] = df[["rank_1", "rank_2"]].max(axis=1).astype(float)

    scores = pool[["qid", "docno", "score"]].rename(columns={"qid": "query_id"})
    for side in ("1", "2"):
        s = scores.rename(columns={"docno": f"doc_id_{side}", "score": f"score_{side}"})
        df = df.merge(s, on=["query_id", f"doc_id_{side}"], how="left")
    df["d_score"] = (df["score_1"] - df["score_2"]).astype(float)
    df["score_gap"] = df["d_score"].abs()
    df["score_max"] = df[["score_1", "score_2"]].max(axis=1).astype(float)

    df = df.merge(_pair_features(pairs), on=PAIR_KEY, how="left")

    if store_df is not None and len(store_df):
        df = df.merge(_confidence(store_df), on=PAIR_KEY, how="left")
    else:
        df["conf_margin_prob"] = np.nan
        df["conf_margin_score"] = np.nan
    return df
