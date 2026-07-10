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
    backend: str = "mock"  # mock | hf
    model: str | None = None
    prompt_version: str = "v0"
    order_swap: bool = True
    max_chars: int = 2000  # passage text truncation before prompting
    batch_flush: int = 50  # verdicts per store flush


@dataclass
class AxiomsConfig:
    lexical: list[str] = field(default_factory=list)
    semantic: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [*self.lexical, *self.semantic]


@dataclass
class ExperimentConfig:
    experiment: str
    seed: int = 42
    primary_metric: str = "fidelity"  # fidelity | effectiveness (plan §4.1)
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(irds_id="vaswani"))
    first_stage: FirstStageConfig = field(default_factory=FirstStageConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    axioms: AxiomsConfig = field(default_factory=AxiomsConfig)


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
        kwargs[name] = _build(nested, value) if nested and isinstance(value, dict) else value
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
