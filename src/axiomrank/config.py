"""Typed experiment configuration, loaded from YAML files in configs/."""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    irds_id: str
    hf_artifact: str | None = None  # prebuilt index on HuggingFace (pt.Artifact.from_hf)
    terrier_dataset: str | None = None
    index_variant: str | None = None
    index_path: str | None = None  # local index directory; overrides everything else


@dataclass
class FirstStageConfig:
    retriever: str = "bm25"
    pool_depth: int = 100


@dataclass
class PairsConfig:
    strategy: str = "top_k_all_pairs"  # top_k_all_pairs | uniform
    k: int = 10
    per_query: int = 200
    max_queries: int | None = None


@dataclass
class RankerConfig:
    backend: str = "mock"  # mock | hf | openai
    model: str | None = None
    prompt_version: str = "v0"
    order_swap: bool = True
    max_chars: int = 2000  # passage text truncation before prompting
    batch_flush: int = 50  # verdicts per store flush
    base_url: str | None = None  # openai backend: OpenAI-compatible endpoint
    extra_body: dict | None = None  # openai backend: extra request fields (vLLM options)


@dataclass
class AxiomSpec:
    """One axiom column: an ir_axioms (or axiomrank.axioms.relaxed) factory name plus params.

    `alias` names the output column (defaults to `name`), so the same axiom can appear
    several times at different parameter settings, e.g. TFC1 with a relaxed length
    precondition next to the strict default.
    """

    name: str
    alias: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def column(self) -> str:
        return (self.alias or self.name).replace("-", "_")


def _axiom_spec(entry: str | dict) -> AxiomSpec:
    if isinstance(entry, str):
        return AxiomSpec(name=entry)
    return _build(AxiomSpec, entry)


@dataclass
class AxiomsConfig:
    # Entries are either a bare axiom name or {name, alias, params} (see AxiomSpec).
    lexical: list[str | dict] = field(default_factory=list)
    semantic: list[str | dict] = field(default_factory=list)

    @property
    def lexical_specs(self) -> list["AxiomSpec"]:
        return [_axiom_spec(e) for e in self.lexical]

    @property
    def semantic_specs(self) -> list["AxiomSpec"]:
        return [_axiom_spec(e) for e in self.semantic]

    @property
    def specs(self) -> list["AxiomSpec"]:
        return [*self.lexical_specs, *self.semantic_specs]

    @property
    def names(self) -> list[str]:
        return [s.alias or s.name for s in self.specs]


@dataclass
class ExperimentConfig:
    experiment: str
    seed: int = 42
    primary_metric: str = "fidelity"  # fidelity | effectiveness (plan §4.1)
    variant: str | None = None  # grid cell, e.g. dl19_top10; scopes outputs/caches
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(irds_id="vaswani"))
    first_stage: FirstStageConfig = field(default_factory=FirstStageConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    rankers: list[RankerConfig] = field(default_factory=list)  # multi-model experiments
    axioms: AxiomsConfig = field(default_factory=AxiomsConfig)

    @property
    def all_rankers(self) -> list[RankerConfig]:
        return self.rankers if self.rankers else [self.ranker]


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    nested_types = {
        "dataset": DatasetConfig,
        "first_stage": FirstStageConfig,
        "pairs": PairsConfig,
        "ranker": RankerConfig,
        "axioms": AxiomsConfig,
    }
    kwargs = {}
    for name, value in data.items():
        nested = nested_types.get(name)
        if nested and isinstance(value, dict):
            kwargs[name] = _build(nested, value)
        elif name == "rankers" and isinstance(value, list):
            kwargs[name] = [_build(RankerConfig, v) for v in value]
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _build(ExperimentConfig, raw)


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    """Write the exact config an experiment ran with next to its outputs."""
    import dataclasses

    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
