"""Computing per-axiom preferences over sampled pairs via ir_axioms.

Axiom names in configs use the paper spelling (M-TDC, TF-LNC); they are normalised to
Python identifiers here. Batteries are lists of AxiomSpec: a factory name plus optional
params — margin_fraction, precondition_margin (rebinds the LEN length precondition of
TFC1/TFC3) and similarity (wordnet | fasttext, for STMC/REG-style axioms) — and an
alias naming the output column, so one axiom can appear at several settings.

ir_axioms is imported lazily, after cache configuration, because importing it starts
the Terrier JVM.
"""

from functools import cached_property, lru_cache

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


def _cached_spacy_tokenizer():
    """ir_axioms' default tokenizer with tokenisation memoised per text. Axioms
    otherwise re-run the full spaCy pipeline on the same document for every
    axiom x pair, which dominates the axiom stage (~0.6 s/pair measured). The full
    en_core_web_sm pipeline is kept — excluding parser/NER shifts lemmas on ~0.2%
    of texts, and bit-identical tokenisation is worth more than the extra 2-3x."""
    from ir_axioms.tools.tokenizer.spacy import SpacyTermTokenizer

    class CachedSpacyTermTokenizer(SpacyTermTokenizer):
        @cached_property
        def _terms_cached(self):
            base = super().terms
            return lru_cache(maxsize=1 << 18)(lambda text: tuple(base(text)))

        def terms(self, text: str):
            return self._terms_cached(text)

    return CachedSpacyTermTokenizer()


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

    from ir_axioms.dependency_injection import injector
    from ir_axioms.tools.tokenizer.base import TermTokenizer
    from ir_axioms.utils.injection import reset_binding_scopes

    injector.binder.bind(TermTokenizer, to=_cached_spacy_tokenizer())

    if index_location is not None:
        from ir_axioms.tools import TerrierIndexStatistics
        from ir_axioms.tools.index_statistics.base import IndexStatistics

        injector.binder.bind(
            IndexStatistics, to=TerrierIndexStatistics(index_location=index_location)
        )

    # Drop any singletons already built against the previous bindings.
    reset_binding_scopes(injector)

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
    Evaluates each axiom on exactly the sampled pairs. (ir_axioms' AxiomaticPreferences
    transformer instead crosses every document a query's pairs touch — ~40x the work
    on the uniform depth-100 cells — for identical values on the pairs that are kept.)
    """
    paths.configure_caches()
    from ir_axioms.model import Document, Query
    from tqdm.auto import tqdm

    specs = [_coerce(spec) for spec in specs]
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
