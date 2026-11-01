from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/result_to_script_manifest.csv"
FORBIDDEN_MODEL_EXTENSIONS = {
    ".pkl",
    ".pickle",
    ".joblib",
    ".onnx",
    ".ubj",
    ".model",
}


def _released_files() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for folder in (ROOT / "results", ROOT / "figures")
        for path in folder.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and path.name != ".DS_Store"
        and "__pycache__" not in path.parts
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_result_manifest_has_exactly_one_portable_mapping_per_artifact() -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = [row["artifact_path"] for row in rows]

    assert len(paths) == len(set(paths))
    assert set(paths) == _released_files()
    assert all(not path.startswith("/") for path in paths)
    assert all("/Users/" not in str(row) for row in rows)
    assert all(row["producer_entrypoint"] for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)
    for row in rows:
        assert _sha256(ROOT / row["artifact_path"]) == row["sha256"]


def test_public_tree_contains_no_executable_model_objects() -> None:
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_MODEL_EXTENSIONS
    ]
    assert offenders == []

