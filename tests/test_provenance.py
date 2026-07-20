"""Common experiment-manifest provenance checks."""

import json

from axiomrank.config import AxiomsConfig, ExperimentConfig
from axiomrank.provenance import sha256_file, write_run_manifest


def test_run_manifest_records_config_environment_sources_and_artifacts(tmp_path):
    config_source = tmp_path / "experiment.yaml"
    config_source.write_text("experiment: test\n")
    source = tmp_path / "axiom.py"
    source.write_text("VERSION = 1\n")
    consumed = tmp_path / "input.parquet"
    consumed.write_bytes(b"input")
    produced = tmp_path / "result.json"
    produced.write_text('{"result": 1}\n')
    destination = tmp_path / "run_manifest.json"
    cfg = ExperimentConfig(experiment="test", axioms=AxiomsConfig(lexical=["TFC1"]))

    manifest = write_run_manifest(
        destination,
        cfg,
        config_source=config_source,
        source_paths=[source],
        input_paths=[consumed],
        output_paths=[produced],
        extra={"runner": "test"},
    )

    saved = json.loads(destination.read_text())
    assert saved == manifest
    assert saved["config"]["experiment"] == "test"
    assert saved["axiom_specs"] == [{"alias": None, "name": "TFC1", "params": {}}]
    assert saved["config_source"]["sha256"] == sha256_file(config_source)
    assert saved["source_files"][0]["sha256"] == sha256_file(source)
    assert saved["inputs"][0]["sha256"] == sha256_file(consumed)
    assert saved["outputs"][0]["sha256"] == sha256_file(produced)
    assert saved["environment"]["uv_lock_sha256"]
    assert saved["git"]["revision"]
    assert saved["research_status"] in {"exploratory", "clean_development"}
