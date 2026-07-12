"""Cache-provenance checks for model-independent pipeline stages."""

import json

import pandas as pd

from axiomrank.config import ExperimentConfig, PairsConfig
from axiomrank.pipeline import stages


def _pool(n_docs=5):
    return pd.DataFrame(
        {
            "qid": ["q1"] * n_docs,
            "query": ["query"] * n_docs,
            "docno": [f"d{i}" for i in range(n_docs)],
            "rank": list(range(n_docs)),
            "score": list(reversed(range(n_docs))),
            "text": [f"text {i}" for i in range(n_docs)],
        }
    )


def test_cached_frame_recomputes_when_fingerprint_changes(tmp_path):
    path = tmp_path / "stage.parquet"
    calls = []

    def compute(value):
        calls.append(value)
        return pd.DataFrame({"value": [value]})

    assert stages.cached_frame(path, False, lambda: compute(1), "fp-a").iloc[0, 0] == 1
    assert stages.cached_frame(path, False, lambda: compute(2), "fp-a").iloc[0, 0] == 1
    assert stages.cached_frame(path, False, lambda: compute(3), "fp-b").iloc[0, 0] == 3
    assert calls == [1, 3]


def test_legacy_cache_is_registered_only_after_validation(tmp_path):
    path = tmp_path / "stage.parquet"
    pd.DataFrame({"value": [7]}).to_parquet(path, index=False)
    out = stages.cached_frame(
        path,
        False,
        lambda: pd.DataFrame({"value": [99]}),
        "fp",
        legacy_validator=lambda df: df["value"].eq(7).all(),
    )
    assert out["value"].tolist() == [7]
    meta = json.loads(path.with_suffix(".parquet.meta.json").read_text())
    assert meta == {"fingerprint": "fp", "legacy_migrated": True}


def test_pair_cache_is_invalidated_when_sampling_config_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(stages.paths, "PROCESSED_DIR", tmp_path)
    cfg = ExperimentConfig(experiment="cache-test", pairs=PairsConfig(k=3))
    pool = _pool()
    first = stages.build_pairs(cfg, pool)
    assert len(first) == 3

    cfg.pairs.k = 4
    second = stages.build_pairs(cfg, pool)
    assert len(second) == 6

