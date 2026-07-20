"""Fresh-process determinism/antisymmetry probe invoked by the test suite."""

from __future__ import annotations

import json

import pandas as pd

from axiomrank.axioms import axiom_preferences
from axiomrank.config import AxiomSpec

SPECS = [
    "TFC1",
    "LNC1",
    "TF-LNC",
    "PROX1",
    "PROX2",
    "PROX3",
    "PROX4",
    "PROX5",
    "AND",
    "DIV",
    "LB1",
    "VERB",
    "QCOV",
    AxiomSpec(name="VERB_R", alias="VERB@m0.2", params={"margin_fraction": 0.2}),
    "DEFANS",
    "NUMANS",
    "COMPARE",
    "CBP",
]

CASES = [
    (
        "q1",
        "ant zebra",
        "ant zebra answer with useful supporting details",
        "zebra filler filler ant",
    ),
    (
        "q2",
        "define photosynthesis",
        "Photosynthesis means converting light into chemical energy in plants.",
        "A page mentioning photosynthesis and related links.",
    ),
    (
        "q3",
        "how many moons does mars have",
        "Mars has two moons named Phobos and Deimos.",
        "Mars formed 4.5 billion years ago and has moons.",
    ),
    (
        "q4",
        "difference between mitosis and meiosis",
        "Mitosis differs from meiosis because it makes two identical cells.",
        "Mitosis and meiosis are listed in this biology navigation page.",
    ),
]


def frames():
    pool_rows = []
    pair_rows = []
    for qid, query, text_1, text_2 in CASES:
        pool_rows.extend(
            [
                (qid, query, f"{qid}-d1", 0, 2.0, text_1),
                (qid, query, f"{qid}-d2", 1, 1.0, text_2),
            ]
        )
        pair_rows.append((qid, query, f"{qid}-d1", f"{qid}-d2", text_1, text_2))
    pool = pd.DataFrame(
        pool_rows, columns=["qid", "query", "docno", "rank", "score", "text"]
    )
    pairs = pd.DataFrame(
        pair_rows,
        columns=["qid", "query", "doc_id_1", "doc_id_2", "text_1", "text_2"],
    )
    return pool, pairs


def main() -> None:
    pool, pairs = frames()
    forward = axiom_preferences(pool, pairs, SPECS)
    reversed_pairs = pairs.rename(
        columns={
            "doc_id_1": "doc_id_2",
            "doc_id_2": "doc_id_1",
            "text_1": "text_2",
            "text_2": "text_1",
        }
    )[pairs.columns]
    backward = axiom_preferences(pool, reversed_pairs, SPECS)
    columns = [column for column in forward if column not in {"qid", "doc_id_1", "doc_id_2"}]
    for column in columns:
        if not forward[column].eq(-backward[column]).all():
            raise AssertionError(f"axiom is not antisymmetric under document swap: {column}")
    payload = forward.sort_values("qid").to_dict(orient="records")
    print("AXIOM_DETERMINISM_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
