"""Canonical locations for data, results and models, resolved from the repo root.

Everything large lives under the repository but outside git (see .gitignore):

- data/cache/        ir_datasets, PyTerrier and Hugging Face caches
- data/raw/          anything downloaded by hand rather than through a library
- data/preferences/  the cached LLM pairwise-verdict dataset (the reusable core artefact)
- data/processed/    derived intermediate datasets (axiom feature matrices, pair samples)
- results/<exp>/     runs, metrics, figures and tables, one directory per experiment
- models/            checkpoints and fitted surrogate models

Directories are created on first use; a fresh clone needs no manual mkdir.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
PREFERENCES_DIR = DATA_DIR / "preferences"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def configure_caches() -> None:
    """Point ir_datasets, PyTerrier and Hugging Face at data/cache.

    Must run before importing ir_datasets, pyterrier or transformers, because
    those libraries read their environment variables at import time. Explicitly
    set variables (e.g. from a .env file) take precedence.
    """
    defaults = {
        "IR_DATASETS_HOME": CACHE_DIR / "ir_datasets",
        "PYTERRIER_HOME": CACHE_DIR / "pyterrier",
        "HF_HOME": CACHE_DIR / "huggingface",
    }
    for var, path in defaults.items():
        os.environ.setdefault(var, str(path))
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)


def results_dir(experiment: str) -> Path:
    """Return (and create) results/<experiment>, the output root for one experiment."""
    path = RESULTS_DIR / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_data_dirs() -> None:
    """Create the standard data/model directories if missing."""
    for path in (CACHE_DIR, RAW_DIR, PREFERENCES_DIR, PROCESSED_DIR, RESULTS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
