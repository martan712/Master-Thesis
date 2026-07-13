"""Schema checks for the Phase 3 development candidate registry."""

import yaml

from axiomrank.paths import PROJECT_ROOT


def test_candidate_registry_has_unique_ids_and_explicit_preconditions():
    path = PROJECT_ROOT / "docs" / "resources" / "phase3-candidate-registry.yaml"
    registry = yaml.safe_load(path.read_text())
    candidates = registry["candidates"]
    ids = [candidate["id"] for candidate in candidates]
    assert len(ids) == len(set(ids))
    assert registry["status"] == "development_v2"
    assert registry["shared_precondition"]["max_logged_revisions"] == 2
    assert registry["shared_precondition"]["completed_revisions"] == 2
    for candidate in candidates:
        assert candidate["status"] in {"specified", "tested_development_d0v2"}
        assert candidate["query_precondition"]
        assert candidate["cost"].startswith("D")
        assert "evidence" in candidate
    implemented = [candidate for candidate in candidates if candidate["status"].startswith("tested")]
    assert {candidate["id"] for candidate in implemented} == {
        "DEFANS", "NUMANS", "CBP", "COMPARE"
    }
    assert all(candidate["implementation_alias"].endswith("_d0v2") for candidate in implemented)


def test_confirmation_dataset_is_not_a_candidate_registry_input():
    path = PROJECT_ROOT / "docs" / "resources" / "phase3-candidate-registry.yaml"
    assert "nfcorpus" not in path.read_text().lower()
