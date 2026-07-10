"""Computing per-axiom preferences over sampled pairs via ir_axioms.

Axiom names in configs use the paper spelling (M-TDC, TF-LNC); they are normalised to
ir_axioms' Python identifiers here. ir_axioms is imported lazily, after cache
configuration, because importing it starts the Terrier JVM.
"""

import pandas as pd

from axiomrank import paths
from axiomrank.pairs import PAIR_COLUMNS  # noqa: F401  (documents the expected input)


def _normalise(name: str) -> str:
    return name.replace("-", "_")


def build_axioms(names: list[str], index_location=None) -> list:
    """Instantiate axioms by config name.

    Statistics-dependent axioms (TFC3, M-TDC, ...) need collection statistics, which
    ir_axioms cannot default; when `index_location` (a Terrier index ref or path) is
    given, a TerrierIndexStatistics tool is bound into ir_axioms' injector first.
    """
    paths.configure_caches()
    import ir_axioms.axiom as ax

    if index_location is not None:
        from ir_axioms.dependency_injection import injector
        from ir_axioms.tools import TerrierIndexStatistics
        from ir_axioms.tools.index_statistics.base import IndexStatistics

        injector.binder.bind(
            IndexStatistics, to=TerrierIndexStatistics(index_location=index_location)
        )

    instances = []
    for name in names:
        factory = getattr(ax, _normalise(name), None)
        if factory is None:
            raise ValueError(f"Unknown ir_axioms axiom: {name}")
        instances.append(factory())
    return instances


def axiom_preferences(
    pool: pd.DataFrame,
    pairs: pd.DataFrame,
    axiom_names: list[str],
    index_location=None,
) -> pd.DataFrame:
    """Per-axiom preference for every canonical pair.

    Returns one row per pair: qid, doc_id_1, doc_id_2 plus one column per axiom with a
    value in {-1, 0, +1} (+1 = axiom prefers doc_id_1). Uses ir_axioms'
    AxiomaticPreferences transformer, restricted per query to the documents that actually
    occur in the sampled pairs.
    """
    paths.configure_caches()
    from ir_axioms.integrations.pyterrier import AxiomaticPreferences

    axioms = build_axioms(axiom_names, index_location=index_location)
    names = [_normalise(n) for n in axiom_names]
    transformer = AxiomaticPreferences(axioms=axioms, axiom_names=names, text_field="text")

    used = pd.concat(
        [
            pairs[["qid", "doc_id_1"]].rename(columns={"doc_id_1": "docno"}),
            pairs[["qid", "doc_id_2"]].rename(columns={"doc_id_2": "docno"}),
        ]
    ).drop_duplicates()
    subpool = pool.merge(used, on=["qid", "docno"])

    crossed = transformer.transform(subpool)
    crossed = crossed.rename(columns={"docno_a": "doc_id_1", "docno_b": "doc_id_2"})
    pref_cols = {f"{n}_preference": n for n in names}
    crossed = crossed.rename(columns=pref_cols)

    result = pairs[["qid", "doc_id_1", "doc_id_2"]].merge(
        crossed[["qid", "doc_id_1", "doc_id_2", *names]],
        on=["qid", "doc_id_1", "doc_id_2"],
        how="left",
    )
    for n in names:
        # Axioms return floats; clamp to sign so downstream code sees {-1, 0, 1}.
        result[n] = result[n].fillna(0).apply(lambda v: (v > 0) - (v < 0))
    return result
