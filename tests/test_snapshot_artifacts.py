"""Integrity and recovery tests for local research-artifact snapshots."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "snapshot_artifacts.py"
SPEC = importlib.util.spec_from_file_location("snapshot_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
snapshot_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot_artifacts)
create_snapshot = snapshot_artifacts.create_snapshot
restore_test = snapshot_artifacts.restore_test
verify_snapshot = snapshot_artifacts.verify_snapshot


def _project(tmp_path):
    project = tmp_path / "project"
    preferences = project / "data" / "preferences"
    preferences.mkdir(parents=True)
    pd.DataFrame({"query_id": ["q1"], "verdict": ["a"]}).to_parquet(
        preferences / "part-test.parquet", index=False
    )
    results = project / "results"
    results.mkdir()
    (results / "report.json").write_text('{"ok": true}\n')
    return project


def test_snapshot_is_checksummed_and_restorable(tmp_path):
    project = _project(tmp_path)
    snapshot = project / "data" / "backups" / "snapshot"
    manifest_path = create_snapshot(project, snapshot)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["sources"] == ["data/preferences", "results"]
    assert verify_snapshot(snapshot)["n_files"] == 2
    restored = restore_test(snapshot)
    assert restored["restored_rows"] == 1
    assert restored["restored_file"].endswith("part-test.parquet")


def test_snapshot_verification_detects_tampering(tmp_path):
    project = _project(tmp_path)
    snapshot = project / "data" / "backups" / "snapshot"
    create_snapshot(project, snapshot)
    (snapshot / "results" / "report.json").write_text("changed\n")

    with pytest.raises(ValueError, match="snapshot size mismatch|snapshot checksum mismatch"):
        verify_snapshot(snapshot)
