"""Common run manifests for auditable development experiments."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from axiomrank import paths
from axiomrank.config import ExperimentConfig

PACKAGE_NAMES = (
    "axiomrank",
    "ir-axioms",
    "python-terrier",
    "ir-datasets",
    "spacy",
    "en-core-web-sm",
    "numpy",
    "pandas",
    "scikit-learn",
    "pyarrow",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_state(project_root: Path) -> dict:
    revision = _command(["git", "rev-parse", "HEAD"], project_root)
    branch = _command(["git", "branch", "--show-current"], project_root)
    status = _command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], project_root
    )
    dirty = bool(status) if status is not None else None
    dirty_digest = None
    if dirty:
        digest = hashlib.sha256()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
        ).stdout
        digest.update(diff)
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=project_root,
            capture_output=True,
            check=True,
        ).stdout.split(b"\0")
        for raw_path in sorted(path for path in untracked if path):
            relative = Path(os.fsdecode(raw_path))
            digest.update(raw_path + b"\0")
            target = project_root / relative
            if target.is_file():
                digest.update(bytes.fromhex(sha256_file(target)))
        dirty_digest = digest.hexdigest()
    return {
        "revision": revision,
        "branch": branch,
        "dirty": dirty,
        "dirty_diff_sha256": dirty_digest,
        "status_porcelain": status.splitlines() if status else [],
    }


def _java_version() -> str | None:
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    lines = (result.stderr or result.stdout).splitlines()
    return lines[0] if lines else None


def _versions() -> dict[str, str | None]:
    versions = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _label(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _file_record(path: Path, project_root: Path) -> dict:
    record = {"path": _label(path, project_root)}
    if not path.exists():
        return {**record, "exists": False}
    if not path.is_file():
        return {**record, "exists": True, "type": "directory"}
    return {
        **record,
        "exists": True,
        "type": "file",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _artifact_records(
    values: Iterable[str | Path],
    project_root: Path,
    exclude: Path | None = None,
    source_files: bool = False,
) -> list[dict]:
    files: dict[str, Path] = {}
    for value in values:
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        if path.is_dir():
            candidates = (
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and not (
                    source_files
                    and ("__pycache__" in candidate.parts or candidate.suffix == ".pyc")
                )
            )
        else:
            candidates = (path,)
        for candidate in candidates:
            if exclude is not None and candidate.resolve() == exclude.resolve():
                continue
            files[_label(candidate, project_root)] = candidate
    return [_file_record(files[name], project_root) for name in sorted(files)]


def write_run_manifest(
    destination: Path,
    cfg: ExperimentConfig,
    *,
    config_source: str | Path | None = None,
    source_paths: Iterable[str | Path] = (),
    input_paths: Iterable[str | Path] = (),
    output_paths: Iterable[str | Path] = (),
    extra: dict | None = None,
    project_root: Path = paths.PROJECT_ROOT,
) -> dict:
    """Write an atomic, checksum-rich manifest for one completed experiment run."""
    project_root = project_root.resolve()
    destination = destination.resolve()
    config_path = Path(config_source) if config_source is not None else None
    if config_path is not None and not config_path.is_absolute():
        config_path = project_root / config_path
    config = dataclasses.asdict(cfg)
    canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    git = _git_state(project_root)
    extra = extra or {}
    research_status = extra.get("research_status_override")
    if research_status is None:
        research_status = "exploratory" if git["dirty"] is not False else "clean_development"
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_status": research_status,
        "config": config,
        "config_sha256": sha256_bytes(canonical_config),
        "config_source": (
            _file_record(config_path, project_root) if config_path is not None else None
        ),
        "git": git,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "java": _java_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": _versions(),
            "uv_lock_sha256": sha256_file(project_root / "uv.lock"),
        },
        "axiom_specs": [dataclasses.asdict(spec) for spec in cfg.axioms.specs],
        "source_files": _artifact_records(source_paths, project_root, source_files=True),
        "inputs": _artifact_records(input_paths, project_root),
        "outputs": _artifact_records(output_paths, project_root, exclude=destination),
        "extra": extra,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, destination)
    return manifest
