"""Configured statistics-free axioms must be stable across fresh hash-seeded processes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

PROBE = Path(__file__).parent / "fixtures" / "axiom_determinism_probe.py"
MARKER = "AXIOM_DETERMINISM_JSON="


def _run(seed: int):
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(seed)
    result = subprocess.run(
        [sys.executable, str(PROBE)],
        cwd=Path(__file__).parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    line = next(line for line in result.stdout.splitlines() if line.startswith(MARKER))
    return json.loads(line.removeprefix(MARKER))


def test_axiom_battery_is_stable_across_python_hash_seeds():
    assert _run(0) == _run(1) == _run(777)
