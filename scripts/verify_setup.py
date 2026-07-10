"""Smoke-test the environment: run with `uv run scripts/verify_setup.py`.

Checks imports, Java (needed by PyTerrier), and that the cache/data directories resolve
and are writable. Exits non-zero on the first hard failure.
"""

import importlib
import shutil
import subprocess
import sys

from axiomrank import paths

FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global FAIL
    mark = "ok " if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL = 1


def main() -> int:
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    check("running inside the uv project venv", ".venv" in sys.executable, sys.executable)

    java = shutil.which("java")
    check("java on PATH (required by PyTerrier)", java is not None, java or "not found")
    if java:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True)
        print(f"      {out.stderr.splitlines()[0] if out.stderr else out.stdout.splitlines()[0]}")

    paths.configure_caches()
    paths.ensure_data_dirs()
    print(f"      project root: {paths.PROJECT_ROOT}")
    print(f"      caches:       {paths.CACHE_DIR}")
    check("data/results/models directories writable", paths.PREFERENCES_DIR.is_dir())

    for mod in ("numpy", "pandas", "sklearn", "matplotlib", "yaml", "pyarrow",
                "ir_datasets", "pyterrier", "ir_axioms"):
        try:
            m = importlib.import_module(mod)
            check(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001 — report any import-time failure
            check(f"import {mod}", False, repr(e))

    for mod in ("torch", "transformers"):
        try:
            m = importlib.import_module(mod)
            print(f"[ok ] import {mod} (extra: llm) — {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"[ -- ] {mod} not installed (fine unless running the LLM; `uv sync --extra llm`)")

    print("\nAll checks passed." if FAIL == 0 else "\nSome checks FAILED.")
    return FAIL


if __name__ == "__main__":
    raise SystemExit(main())
