"""Axiom-spec parsing and multi-ranker config handling (phase1-implementation.md §3)."""

from pathlib import Path

from axiomrank.config import AxiomsConfig, RankerConfig, load_config

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_bare_axiom_names_become_default_specs():
    cfg = AxiomsConfig(lexical=["TFC1", "TF-LNC"])
    specs = cfg.specs
    assert [s.name for s in specs] == ["TFC1", "TF-LNC"]
    assert [s.column for s in specs] == ["TFC1", "TF_LNC"]
    assert all(s.params == {} for s in specs)


def test_dict_axiom_entries_carry_alias_and_params():
    cfg = AxiomsConfig(
        lexical=[
            "TFC1",
            {"name": "TFC1", "alias": "TFC1@len0.2", "params": {"precondition_margin": 0.2}},
        ],
        semantic=[{"name": "STMC1", "alias": "STMC1@wn", "params": {"similarity": "wordnet"}}],
    )
    relaxed = cfg.lexical_specs[1]
    assert relaxed.name == "TFC1"
    assert relaxed.column == "TFC1@len0.2"
    assert relaxed.params == {"precondition_margin": 0.2}
    assert cfg.names == ["TFC1", "TFC1@len0.2", "STMC1@wn"]
    assert [s.column for s in cfg.specs] == cfg.names  # no dashes in these aliases


def test_rankers_list_parses_and_all_rankers_falls_back(tmp_path):
    multi = tmp_path / "multi.yaml"
    multi.write_text(
        """
experiment: t
variant: cell
rankers:
  - backend: openai
    model: m1
  - backend: hf
    model: m2
"""
    )
    cfg = load_config(multi)
    assert cfg.variant == "cell"
    assert [r.model for r in cfg.all_rankers] == ["m1", "m2"]

    single = tmp_path / "single.yaml"
    single.write_text("experiment: t\nranker:\n  backend: mock\n")
    cfg = load_config(single)
    assert cfg.variant is None
    assert cfg.all_rankers == [cfg.ranker]


def test_ranker_device_dtype_default_to_cpu_fp32(tmp_path):
    # Existing configs must stay byte-for-byte reproducible: the new fields default to the
    # historical CPU/fp32 behaviour, and are only overridden by an explicit hf/GPU rung.
    assert RankerConfig().device == "cpu"
    assert RankerConfig().dtype == "float32"

    cfg_path = tmp_path / "gpu.yaml"
    cfg_path.write_text(
        """
experiment: t
rankers:
  - backend: hf
    model: google/flan-t5-xl
    device: cuda
    dtype: bfloat16
"""
    )
    ranker = load_config(cfg_path).all_rankers[0]
    assert (ranker.device, ranker.dtype) == ("cuda", "bfloat16")


def test_grid_configs_parse_with_unique_columns():
    grid = sorted(CONFIGS_DIR.glob("rq[12]_*.yaml")) + [CONFIGS_DIR / "p1_smoke.yaml"]
    assert len(grid) == 7  # 4 rq1 cells + 2 rq2 cells + smoke
    for path in grid:
        cfg = load_config(path)
        assert cfg.variant, path.name
        assert cfg.all_rankers, path.name
        columns = [s.column for s in cfg.axioms.specs]
        assert len(columns) == len(set(columns)), f"duplicate axiom columns in {path.name}"


def test_phase2_feature_sets_exclude_later_rq4_axioms():
    import importlib.util

    def load_runner(relative, name):
        spec = importlib.util.spec_from_file_location(name, CONFIGS_DIR.parent / relative)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    rq2_runner = load_runner("experiments/rq2_semantic_agreement/run.py", "rq2_runner")
    rq3_runner = load_runner("experiments/rq3_decomposition/run.py", "rq3_runner")

    cfg = load_config(CONFIGS_DIR / "rq2_dl19_top10.yaml")
    rq2 = rq2_runner._analysis_feature_sets(cfg)
    rq3 = rq3_runner._feature_sets(cfg)
    later = {"VERB", "QCOV", "VERB@m0.2"}
    assert later.isdisjoint(rq2["lexical"])
    assert later.isdisjoint(rq2["combined"])
    assert later.isdisjoint(rq3["lexical"])
    assert later.isdisjoint(rq3["lexical_semantic"])
