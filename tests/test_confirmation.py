"""The external confirmation collection must remain inaccessible during development."""

from pathlib import Path

import pytest
import yaml

from axiomrank.confirmation import (
    LOCK_MANIFEST,
    UNLOCK_MANIFEST,
    ConfirmationLockedError,
    assert_dataset_access_allowed,
    locked_dataset_ids,
)
from axiomrank.paths import CONFIGS_DIR


HOLDOUT = "beir/nfcorpus/test"


def test_confirmation_manifest_is_locked_and_not_unlocked():
    manifest = yaml.safe_load(LOCK_MANIFEST.read_text())
    assert manifest["status"] == "locked"
    assert manifest["dataset"]["irds_id"] == HOLDOUT
    assert manifest["fitting"]["holdout_refitting"] == "forbidden"
    assert HOLDOUT in locked_dataset_ids()
    assert not UNLOCK_MANIFEST.exists()


def test_confirmation_guard_fails_before_any_dataset_import():
    with pytest.raises(ConfirmationLockedError, match="untouched Phase 3 confirmation"):
        assert_dataset_access_allowed(HOLDOUT)
    assert_dataset_access_allowed("msmarco-passage/trec-dl-2019/judged")


def test_no_experiment_config_references_the_holdout():
    offenders = [
        path
        for path in Path(CONFIGS_DIR).rglob("*.yaml")
        if HOLDOUT in path.read_text()
    ]
    assert offenders == []
