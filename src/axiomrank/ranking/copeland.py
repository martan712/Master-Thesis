"""Aggregating pairwise verdicts into a ranking via Copeland scoring.

This is the generic pairwise->ranking path: `copeland_ranking` aggregates *any*
collapsed pair-verdict frame (the LLM's cached verdicts today, axiom majority votes
later) into a ranked run compatible with ir_measures/pyterrier; nothing here is
LLM-specific. Copeland scoring over a *complete* top-k tournament (every pair scored)
uses the same win/loss aggregation family as PRP-allpair in the listwise-reranking
literature (Qin et al., "Large Language Models are Effective Text Rankers with Pairwise
Ranking Prompting"). Our protocol first collapses order-inconsistent presentations to
ties, so numerical results are not exact protocol replications of the published runs.
"""

import pandas as pd


def copeland_ranking(verdicts: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Aggregate collapsed pair verdicts into a run via Copeland scoring.

    `verdicts` is any `analysis.model_pair_verdicts`-shaped frame: columns
    `query_id`, `doc_id_1`, `doc_id_2`, `model_pref` in {-1, 0, +1} (+1 prefers
    doc_id_1; position-inconsistent pairs already collapse to 0). `pool` is the
    first-stage frame (`qid`, `docno`, `rank`, `score`; `rank` is 0-based).

    Per query, a document's Copeland score is (wins - losses) over the pairs it appears
    in; ties (model_pref 0) contribute nothing. Documents that appear in at least one
    pair form the reranked block, ordered by Copeland score descending; equal scores are
    broken by first-stage rank (stable, deterministic). Pool documents that appear in no
    pair — the tail below the reranked top-k — keep their first-stage order strictly
    below every reranked document.

    Returns a run frame (`qid`, `docno`, `rank`, `score`) with `rank` the new 0-based
    rank and `score` = -rank, monotone with rank so ir_measures/pyterrier order it
    exactly as ranked.
    """
    rows = []
    for qid, pool_group in pool.groupby("qid", sort=False):
        query_verdicts = verdicts[verdicts["query_id"] == qid]

        copeland: dict[str, int] = {}
        reranked: set[str] = set()
        for v in query_verdicts.itertuples():
            reranked.add(v.doc_id_1)
            reranked.add(v.doc_id_2)
            if v.model_pref > 0:  # prefers doc_id_1
                copeland[v.doc_id_1] = copeland.get(v.doc_id_1, 0) + 1
                copeland[v.doc_id_2] = copeland.get(v.doc_id_2, 0) - 1
            elif v.model_pref < 0:  # prefers doc_id_2
                copeland[v.doc_id_1] = copeland.get(v.doc_id_1, 0) - 1
                copeland[v.doc_id_2] = copeland.get(v.doc_id_2, 0) + 1
            # model_pref == 0: a tie contributes nothing to either document's score.

        # Sort key: reranked block first (0) then the tail (1); within the reranked
        # block by Copeland descending; first-stage rank breaks every remaining tie and
        # is unique per query, so the order is fully deterministic regardless of sort
        # stability.
        ordered = sorted(
            pool_group.itertuples(),
            key=lambda d: (
                0 if d.docno in reranked else 1,
                -copeland.get(d.docno, 0),
                d.rank,
            ),
        )
        for new_rank, doc in enumerate(ordered):
            rows.append((qid, doc.docno, new_rank, float(-new_rank)))

    return pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])
