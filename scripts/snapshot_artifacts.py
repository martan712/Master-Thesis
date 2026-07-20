"""Create and verify recoverable snapshots of expensive local research artifacts.

The default snapshot contains the append-only preference store and all current results.
Snapshots live below ``data/backups/`` (already excluded from Git through ``data/**``),
carry a SHA-256 inventory, and can be checked without touching the live artifacts.

Usage:
    uv run --no-sync python scripts/snapshot_artifacts.py create
    uv run --no-sync python scripts/snapshot_artifacts.py verify DATA/BACKUP/PATH
    uv run --no-sync python scripts/snapshot_artifacts.py restore-test DATA/BACKUP/PATH
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from axiomrank import paths

MANIFEST_NAME = "snapshot-manifest.json"
DEFAULT_SOURCES = (Path("data/preferences"), Path("results"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _inventory(snapshot: Path) -> list[dict]:
    records = []
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        if path.name == MANIFEST_NAME:
            continue
        records.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def create_snapshot(
    project_root: Path,
    destination: Path,
    sources: tuple[Path, ...] = DEFAULT_SOURCES,
) -> Path:
    """Copy sources into a new snapshot and write its checksum manifest."""
    project_root = project_root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"snapshot destination already exists: {destination}")
    if not _inside(destination, project_root):
        raise ValueError("snapshot destination must be inside the project root")

    destination.mkdir(parents=True)
    copied_sources = []
    for relative in sources:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"snapshot source must be project-relative: {relative}")
        source = project_root / relative
        if not source.exists():
            raise FileNotFoundError(f"snapshot source does not exist: {source}")
        if _inside(destination, source):
            raise ValueError(f"snapshot destination cannot be inside a source: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, copy_function=shutil.copy2)
        else:
            shutil.copy2(source, target)
        copied_sources.append(relative.as_posix())

    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "project_root": str(project_root),
        "sources": copied_sources,
        "files": _inventory(destination),
    }
    manifest_path = destination / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def verify_snapshot(snapshot: Path) -> dict:
    """Verify every recorded artifact and reject missing or unrecorded files."""
    snapshot = snapshot.resolve()
    manifest_path = snapshot / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError(f"unsupported or malformed snapshot manifest: {manifest_path}")

    expected = {record["path"]: record for record in manifest["files"]}
    actual = {
        path.relative_to(snapshot).as_posix(): path
        for path in snapshot.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"snapshot inventory mismatch; missing={missing}, extra={extra}")
    for relative, record in expected.items():
        path = actual[relative]
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"snapshot size mismatch: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"snapshot checksum mismatch: {relative}")
    return {
        "snapshot": str(snapshot),
        "n_files": len(expected),
        "size_bytes": sum(record["size_bytes"] for record in expected.values()),
        "manifest_sha256": sha256_file(manifest_path),
    }


def restore_test(snapshot: Path) -> dict:
    """Restore one preference Parquet to temporary storage and read it successfully."""
    verification = verify_snapshot(snapshot)
    manifest = json.loads((snapshot / MANIFEST_NAME).read_text())
    candidates = [
        snapshot / record["path"]
        for record in manifest["files"]
        if record["path"].startswith("data/preferences/part-")
        and record["path"].endswith(".parquet")
    ]
    if not candidates:
        raise ValueError("snapshot contains no preference-store Parquet part to restore")
    source = sorted(candidates)[0]
    with tempfile.TemporaryDirectory(prefix="axiomrank-restore-") as directory:
        restored = Path(directory) / source.name
        shutil.copy2(source, restored)
        frame = pd.read_parquet(restored)
    return {
        **verification,
        "restored_file": source.relative_to(snapshot).as_posix(),
        "restored_rows": len(frame),
        "restored_columns": list(frame.columns),
    }


def _default_destination() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return paths.DATA_DIR / "backups" / f"development-artifacts-{stamp}"


def _record_latest(snapshot: Path, status: dict) -> None:
    latest = snapshot.parent / "LATEST.json"
    payload = {**status, "verified_at_utc": _utc_now()}
    latest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--destination", type=Path, default=None)
    for command in ("verify", "restore-test"):
        subparsers.add_parser(command).add_argument("snapshot", type=Path)
    args = parser.parse_args()

    if args.command == "create":
        destination = args.destination or _default_destination()
        create_snapshot(paths.PROJECT_ROOT, destination)
        status = restore_test(destination)
        _record_latest(destination.resolve(), status)
    elif args.command == "verify":
        status = verify_snapshot(args.snapshot)
    else:
        status = restore_test(args.snapshot)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
