#!/usr/bin/env python3
"""Report generated-development C-grid sensitivity without changing a frozen run."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from leo_service_survival.eos_full_chain_predictor import audit_full_chain_regularization_grid


def refresh_robustness_hashes(directory: Path) -> None:
    """Refresh generated hash metadata after an explicitly recorded metadata correction."""

    robustness_path = directory / "robustness_manifest.json"
    if robustness_path.exists():
        robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
        robustness["artifact_sha256"] = {
            str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != robustness_path
        }
        robustness_path.write_text(json.dumps(robustness, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_directory(directory: Path) -> dict:
    manifest_path = directory / "predictor_manifest.json"
    reference_path = directory / "predictor_reference.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = pd.read_csv(reference_path)
    frozen_c = float(manifest["model"]["regularization_c"])
    table = audit_full_chain_regularization_grid(reference, frozen_regularization_c=frozen_c)
    csv_path = directory / "predictor_regularization_grid_sensitivity.csv"
    output_manifest = directory / "predictor_regularization_grid_sensitivity_manifest.json"
    if csv_path.exists() or output_manifest.exists():
        raise FileExistsError("regularization-range audit already exists; use a fresh run")
    table.to_csv(csv_path, index=False)
    best_c = float(table.iloc[0]["regularization_c"])
    payload = {
        "frozen_model_sha256": manifest["model_sha256"],
        "frozen_regularization_c": frozen_c,
        "best_extended_grid_c": best_c,
        "frozen_choice_retained_as_extended_best": frozen_c == best_c,
        "extended_best_on_grid_boundary": best_c in (
            float(table["regularization_c"].min()), float(table["regularization_c"].max()),
        ),
        "extended_c_grid": sorted(table["regularization_c"].tolist()),
        "model_or_calibration_changed": False,
        "validation_or_evaluation_outcomes_used": False,
        "partition_roles_read_for_fitting": ["training", "tuning"],
        "source_training_partition": manifest["partitions"]["training"],
        "source_tuning_partition": manifest["partitions"]["tuning"],
        "table_sha256": sha256(csv_path.read_bytes()).hexdigest(),
        "evidence_tier": "synthetic_simulation",
        "claim_boundary": "Search-range sensitivity on generated development data, not physical coefficient validation.",
    }
    output_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refresh_robustness_hashes(directory)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--refresh-hashes-only", action="store_true")
    args = parser.parse_args()
    if args.refresh_hashes_only:
        refresh_robustness_hashes(args.run_directory)
        print("Robustness artifact hashes refreshed; no model, data or outcomes changed.")
        return
    print(json.dumps(audit_directory(args.run_directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
