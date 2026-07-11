"""A memoising wrapper around ir_axioms' default spaCy tokenizer."""

from functools import cached_property, lru_cache


def cached_spacy_tokenizer():
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
