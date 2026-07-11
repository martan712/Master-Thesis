"""Computing per-axiom preferences over sampled pairs via ir_axioms.

Axiom names in configs use the paper spelling (M-TDC, TF-LNC); they are normalised to
Python identifiers here. Batteries are lists of AxiomSpec: a factory name plus optional
params — margin_fraction, precondition_margin (rebinds the LEN length precondition of
TFC1/TFC3) and similarity (wordnet | fasttext, for STMC/REG-style axioms) — and an
alias naming the output column, so one axiom can appear at several settings.

ir_axioms is imported lazily, after cache configuration, because importing it starts
the Terrier JVM.
"""

import pandas as pd

from axiomrank import paths
from axiomrank.config import AxiomSpec
from axiomrank.pairs import PAIR_COLUMNS  # noqa: F401  (documents the expected input)


def _normalise(name: str) -> str:
    return name.replace("-", "_")


def _coerce(spec: AxiomSpec | str) -> AxiomSpec:
    """Accept bare axiom names wherever specs are expected (defaults, no alias)."""
    return AxiomSpec(name=spec) if isinstance(spec, str) else spec


def _factory(name: str):
    """Resolve an axiom factory: our relaxed variants first, then ir_axioms."""
    from axiomrank import relaxed

    import ir_axioms.axiom as ax

    normalised = _normalise(name)
    factory = getattr(relaxed, normalised, None) or getattr(ax, normalised, None)
    if factory is None:
        raise ValueError(f"Unknown axiom: {name}")
    return factory


def _term_similarity(kind: str):
    from ir_axioms.tools import FastTextTermSimilarity, WordNetSynonymSetTermSimilarity

    if kind == "wordnet":
        return WordNetSynonymSetTermSimilarity()
    if kind == "fasttext":
        # 7.24 GB model download on first similarity() call — gated in phase1-design §5.1.
        return FastTextTermSimilarity()
    raise ValueError(f"Unknown term similarity backend: {kind}")


def _instantiate(spec: AxiomSpec):
    params = dict(spec.params)
    kwargs = {}
    if "similarity" in params:
        kwargs["term_similarity"] = _term_similarity(params.pop("similarity"))
    if "precondition_margin" in params:
        from ir_axioms.precondition import LEN

        kwargs["precondition"] = LEN(margin_fraction=params.pop("precondition_margin"))
    kwargs.update(params)  # remaining params (margin_fraction, ...) pass through
    return _factory(spec.name)(**kwargs)


def build_axioms(specs: list[AxiomSpec | str], index_location=None) -> list:
    """Instantiate the battery from config specs.

    Statistics-dependent axioms (TFC3, M-TDC, ...) need collection statistics, which
    ir_axioms cannot default; when `index_location` (a Terrier index ref or path) is
    given, a TerrierIndexStatistics tool is bound into ir_axioms' injector first.
    """
    paths.configure_caches()

    if index_location is not None:
        from ir_axioms.dependency_injection import injector
        from ir_axioms.tools import TerrierIndexStatistics
        from ir_axioms.tools.index_statistics.base import IndexStatistics

        injector.binder.bind(
            IndexStatistics, to=TerrierIndexStatistics(index_location=index_location)
        )

    return [_instantiate(_coerce(spec)) for spec in specs]


def axiom_preferences(
    pool: pd.DataFrame,
    pairs: pd.DataFrame,
    specs: list[AxiomSpec | str],
    index_location=None,
) -> pd.DataFrame:
    """Per-axiom preference for every canonical pair.

    Returns one row per pair: qid, doc_id_1, doc_id_2 plus one column per axiom spec
    (named by its alias) with a value in {-1, 0, +1} (+1 = axiom prefers doc_id_1).
    Uses ir_axioms' AxiomaticPreferences transformer, restricted per query to the
    documents that actually occur in the sampled pairs.
    """
    paths.configure_caches()
    from ir_axioms.integrations.pyterrier import AxiomaticPreferences

    specs = [_coerce(spec) for spec in specs]
    axioms = build_axioms(specs, index_location=index_location)
    names = [spec.column for spec in specs]
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
