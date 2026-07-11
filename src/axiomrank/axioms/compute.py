"""Computing per-axiom preferences over sampled pairs."""

import pandas as pd

from axiomrank import paths
from axiomrank.axioms.registry import build_axioms, coerce_spec
from axiomrank.config import AxiomSpec
from axiomrank.data.pairs import PAIR_COLUMNS  # noqa: F401  (documents the expected input)


def axiom_preferences(
    pool: pd.DataFrame,
    pairs: pd.DataFrame,
    specs: list[AxiomSpec | str],
    index_location=None,
) -> pd.DataFrame:
    """Per-axiom preference for every canonical pair.

    Returns one row per pair: qid, doc_id_1, doc_id_2 plus one column per axiom spec
    (named by its alias) with a value in {-1, 0, +1} (+1 = axiom prefers doc_id_1).
    Evaluates each axiom on exactly the sampled pairs. (ir_axioms' AxiomaticPreferences
    transformer instead crosses every document a query's pairs touch — ~40x the work
    on the uniform depth-100 cells — for identical values on the pairs that are kept.)
    """
    paths.configure_caches()
    from ir_axioms.model import Document, Query
    from tqdm.auto import tqdm

    specs = [coerce_spec(spec) for spec in specs]
    axioms = build_axioms(specs, index_location=index_location)
    names = [spec.column for spec in specs]

    # rank/score ride along on the Document objects, as on the old pool-crossing path.
    meta = pool.set_index(["qid", "docno"])[["rank", "score"]]

    def document(qid, docno, text) -> Document:
        row = meta.loc[(qid, docno)]
        return Document(
            id=str(docno), text=str(text), score=float(row["score"]), rank=int(row["rank"])
        )

    records = []
    progress = tqdm(
        pairs.itertuples(index=False),
        total=len(pairs),
        desc=f"axiom preferences ({len(names)} axioms)",
        unit="pair",
    )
    for row in progress:
        query = Query(id=str(row.qid), text=str(row.query))
        doc_1 = document(row.qid, row.doc_id_1, row.text_1)
        doc_2 = document(row.qid, row.doc_id_2, row.text_2)
        record = {"qid": row.qid, "doc_id_1": row.doc_id_1, "doc_id_2": row.doc_id_2}
        for name, axiom in zip(names, axioms):
            # Axioms return floats; clamp to sign so downstream code sees {-1, 0, 1}.
            value = axiom.preference(input=query, output1=doc_1, output2=doc_2)
            record[name] = int(value > 0) - int(value < 0)
        records.append(record)

    return pd.DataFrame(records, columns=["qid", "doc_id_1", "doc_id_2", *names])
