"""Resolving axiom specs into instantiated ir_axioms axioms.

Batteries are lists of AxiomSpec: a factory name plus optional params —
margin_fraction, precondition_margin (rebinds the LEN length precondition of
TFC1/TFC3) and similarity (wordnet | fasttext, for STMC/REG-style axioms) — and an
alias naming the output column, so one axiom can appear at several settings.
"""

from axiomrank import paths
from axiomrank.axioms.tokenizer import cached_spacy_tokenizer
from axiomrank.config import AxiomSpec


def _normalise(name: str) -> str:
    return name.replace("-", "_")


def coerce_spec(spec: AxiomSpec | str) -> AxiomSpec:
    """Accept bare axiom names wherever specs are expected (defaults, no alias)."""
    return AxiomSpec(name=spec) if isinstance(spec, str) else spec


def _factory(name: str):
    """Resolve an axiom factory: our rq4/relaxed variants first, then ir_axioms."""
    from axiomrank.axioms import relaxed, rq4

    import ir_axioms.axiom as ax

    normalised = _normalise(name)
    factory = (
        getattr(rq4, normalised, None)
        or getattr(relaxed, normalised, None)
        or getattr(ax, normalised, None)
    )
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

    from ir_axioms.dependency_injection import injector
    from ir_axioms.tools.tokenizer.base import TermTokenizer
    from ir_axioms.utils.injection import reset_binding_scopes

    injector.binder.bind(TermTokenizer, to=cached_spacy_tokenizer())

    if index_location is not None:
        from ir_axioms.tools import TerrierIndexStatistics
        from ir_axioms.tools.index_statistics.base import IndexStatistics

        injector.binder.bind(
            IndexStatistics, to=TerrierIndexStatistics(index_location=index_location)
        )

    # Drop any singletons already built against the previous bindings.
    reset_binding_scopes(injector)

    return [_instantiate(coerce_spec(spec)) for spec in specs]
