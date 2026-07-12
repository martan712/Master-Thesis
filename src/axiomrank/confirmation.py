"""Fail-closed guard for datasets reserved as untouched confirmation sets."""

from __future__ import annotations

from functools import lru_cache

import yaml

from axiomrank import paths


LOCK_MANIFEST = paths.PROJECT_ROOT / "docs" / "phase3-confirmation-lock.yaml"
UNLOCK_MANIFEST = paths.CONFIGS_DIR / "confirmation" / "UNLOCKED.yaml"


class ConfirmationLockedError(RuntimeError):
    """Raised before any reserved confirmation dataset is accessed."""


@lru_cache(maxsize=1)
def _manifest() -> dict:
    try:
        manifest = yaml.safe_load(LOCK_MANIFEST.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ConfirmationLockedError(
            f"cannot verify confirmation lock manifest {LOCK_MANIFEST}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("status") != "locked":
        raise ConfirmationLockedError("confirmation lock manifest is missing or not locked")
    return manifest


def locked_dataset_ids() -> frozenset[str]:
    """Dataset identifiers reserved by the active lock manifest."""
    return frozenset({_manifest()["dataset"]["irds_id"]})


def assert_dataset_access_allowed(dataset_id: str) -> None:
    """Reject holdout access unless a deliberate unlock manifest exists.

    This prevents accidental access through the shared retrieval/evaluation paths. It is an
    engineering guard, not a security boundary; the research protocol also forbids direct
    dataset access that bypasses these functions.
    """
    if dataset_id in locked_dataset_ids() and not UNLOCK_MANIFEST.exists():
        raise ConfirmationLockedError(
            f"{dataset_id!r} is the untouched Phase 3 confirmation set; access is locked. "
            f"Freeze the development manifest and create {UNLOCK_MANIFEST} in an explicit "
            "unlock commit before the one-shot confirmation run."
        )
