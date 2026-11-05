"""Pin the separate RACS code/evidence release without copying its input data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


PINNED_FILES = (
    "scripts/run_camera_ready_diagnostics.py",
    "scripts/matched_scheduler.py",
    "results/camera_ready_diagnostics/camera_ready_diagnostics.json",
    "results/camera_ready_diagnostics/common_ledger_by_instance.csv",
    "results/camera_ready_diagnostics/full_margin_mechanisms_by_instance.csv",
    "results/camera_ready_diagnostics/publication_manifest.json",
    "results/reviewer_revision/restricted_plan_optimum.csv",
    "results/reviewer_revision/instance_results.csv",
    "results/parameter_robustness_complete_2026-11-02/score_outer_results.csv",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_summary(path: Path, policy: str) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stress"] == "combined" and row["policy"] == policy]
    if len(rows) != 9 or len({row["instance"] for row in rows}) != 9:
        raise ValueError(f"expected nine distinct combined-stress scenarios for {policy}")
    completed = sum(int(row["completed"]) for row in rows)
    tasks = sum(int(row["tasks"]) for row in rows)
    return {
        "scenario_count": len(rows),
        "equal_scenario_mean": sum(int(row["completed"]) / int(row["tasks"]) for row in rows) / len(rows),
        "pooled_completed_requests": completed,
        "pooled_requested_tasks": tasks,
        "pooled_request_completion": completed / tasks,
        "interpretation": "The primary comparison uses equal-scenario means; pooled counts use different weights.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper3-root", type=Path, required=True)
    parser.add_argument("--release-bundle", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.paper3_root.resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != args.expected_commit:
        raise ValueError(f"Paper 3 HEAD differs from expected commit: {actual}")
    records = []
    for relative in PINNED_FILES:
        path = root / relative
        records.append({"path": relative, "sha256": digest(path), "bytes": path.stat().st_size})
    result = {
        "schema_version": 1,
        "source_repository": "https://github.com/KarusalaDeepak/leo-eo-scheduling",
        "source_commit": actual,
        "release_bundle": {"filename": args.release_bundle.name, "sha256": digest(args.release_bundle)},
        "release_manifest": {"filename": args.release_manifest.name, "sha256": digest(args.release_manifest)},
        "source_binding": "The separate code/evidence package and exact file hashes identify the scheduling diagnostics used by Paper 4 R3.",
        "file_fingerprints": records,
        "completion_aggregation": {
            "racs": completion_summary(root / "results/parameter_robustness_complete_2026-11-02/score_outer_results.csv", "full_racs"),
            "full_margin": completion_summary(root / "results/reviewer_revision/instance_results.csv", "full_margin"),
        },
        "data_boundary": "Only hashes and aggregate totals are included; raw contacts, observation windows, task records, schedules, and manuscript files are not copied.",
        "evidence_tier": "controlled_synthetic_benchmark",
        "operational_validation": "NOT_RUN",
        "shared_scheduler_path_outcome_experiment": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output), "source_commit": actual}, indent=2))


if __name__ == "__main__":
    main()
