"""Axiom batteries: factory resolution, instantiation, and per-pair preferences.

Axiom names in configs use the paper spelling (M-TDC, TF-LNC); the registry normalises
them to Python identifiers and resolves our relaxed variants (:mod:`axiomrank.axioms.relaxed`)
before falling back to ir_axioms. ir_axioms is imported lazily, after cache
configuration, because importing it starts the Terrier JVM.
"""

from axiomrank.axioms.compute import axiom_preferences
from axiomrank.axioms.registry import build_axioms

__all__ = ["axiom_preferences", "build_axioms"]
