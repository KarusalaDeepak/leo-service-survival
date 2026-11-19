#!/usr/bin/env python3
"""Allowlisted code/aggregate reviewer archive, no manuscripts or input rows."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER3 = ROOT.parents[1] / "03_Paper_RACS_EO_Scheduling/Code_and_Manuscript"


def build(destination):
    files = []
    for name in ("run_r30_wetlinks_eo_trace.py", "run_r31_long_contact_probe.py",
                 "run_r32_checkpoint_capacity.py", "run_r34_causal_chunks.py",
                 "summarize_r34_causal_chunks.py", "package_r34_review.py"):
        files.append((ROOT / "scripts" / name, "paper4/scripts/" + name))
    files.append((ROOT / "src/leo_service_survival/metrics.py", "paper4/src/leo_service_survival/metrics.py"))
    for name in ("matched_scheduler.py", "parameter_registry.py"):
        files.append((PAPER3 / "scripts" / name, "paper3/scripts/" + name))
    for name in ("test_r34_causal_chunks.py",):
        files.append((ROOT / "tests" / name, "paper4/tests/" + name))
    for name in ("r34_causal_chunk_protocol_2026-11-19.md", "r34_duration_matching_addendum_2026-11-19.md",
                 "r34_correction_and_results.md", "r34_reviewer_instructions.md"):
        files.append((ROOT / "docs" / name, "paper4/docs/" + name))
    for panel in ("r34_causal_native_verified_2026-11-19", "r34_causal_duration_verified_2026-11-19"):
        for name in ("runs.csv", "summary.csv", "paired_inference.csv", "inference_scope.json"):
            files.append((ROOT / "results" / panel / name, "aggregates/" + panel + "/" + name))
    manifest = []
    for source, target in files:
        manifest.append(dict(path=target, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("refusing to replace a reviewer archive")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, target in files:
            archive.write(source, target)
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        archive.writestr("README.txt", "Start with paper4/docs/r34_reviewer_instructions.md. Author-written code and aggregate outputs only. No raw data, row-level predictions, manuscript, credentials, or Git history. Repository visibility is controlled separately.\n")
    print(f"Archived {len(files)} allowlisted files: {destination}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("destination", type=Path)
    build(p.parse_args().destination)
