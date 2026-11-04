#!/usr/bin/env python3
"""Build a relative, checksum-verifiable manifest for a submission package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


EXCLUDED_PARTS = {".pytest_cache", "__pycache__", ".git", "build"}
EXCLUDED_SUFFIXES = {".pyc", ".DS_Store"}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def included_files(root: Path, manifest: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path == manifest:
            continue
        relative = path.relative_to(root)
        if set(relative.parts).intersection(EXCLUDED_PARTS):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise ValueError(f"submission package contains a symbolic link: {relative}")
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output = (args.output or root / "PACKAGE_MANIFEST.json").resolve()
    if output.parent != root:
        raise ValueError("manifest output must be directly inside the package root")

    records = []
    for path in included_files(root, output):
        records.append(
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(root).as_posix(),
                "sha256": digest(path),
            }
        )

    manifest = {
        "manifest_version": "2.1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "sha256",
        "field_trial_completed": False,
        "scheduler_release_decision": "RETAIN_MATCHED_SIMPLE_BASELINE",
        "claim_boundary": (
            "This manifest verifies the redistribution-safe Paper 4 package. "
            "It does not assert live field acquisition, candidate authorization, "
            "RACS superiority, canary execution, or deployment benefit."
        ),
        "files": records,
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(output),
                "indexed_files": len(records),
                "manifest_sha256": digest(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
