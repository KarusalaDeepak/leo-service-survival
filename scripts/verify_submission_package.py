#!/usr/bin/env python3
"""Verify the SHA-256 manifest in an extracted Paper 4 submission package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Extracted package root (defaults to the script's project root).",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = (args.manifest or root / "PACKAGE_MANIFEST.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for record in manifest.get("files", []):
        relative = record["path"]
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            errors.append(f"size mismatch: {relative}")
            continue
        if sha256(path) != record["sha256"]:
            errors.append(f"hash mismatch: {relative}")

    expected = {record["path"] for record in manifest.get("files", [])}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "paper/build" not in path.relative_to(root).as_posix()
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
    }
    for relative in sorted(actual - expected):
        errors.append(f"unindexed: {relative}")
    for relative in sorted(expected - actual):
        errors.append(f"indexed but absent: {relative}")

    report = {
        "manifest": str(manifest_path),
        "indexed_files": len(expected),
        "passed": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
