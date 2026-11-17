"""Export an explicit aggregate-only R5 parameter-evidence allowlist.

Never copies measured episode rows, EOS source/task/action records, generated
training examples, fitted model objects, or third-party scenario files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/parameter_revision_r5_v1"
SOURCES = {
    "full_chain": ("results/eos_full_chain_parameter_revision_v1", (
        "policy_summary.csv", "paired_inference.json", "paired_seed_results.csv",
        "full_chain_audit.csv", "robustness_summary.csv", "robustness_manifest.json",
        "predictor_coefficients.csv", "predictor_regularization_selection.csv",
        "predictor_probability_diagnostics.csv", "predictor_calibration_bins.csv",
        "predictor_manifest.json", "predictor_regularization_grid_sensitivity.csv",
        "predictor_regularization_grid_sensitivity_manifest.json")),
    "endpoint": ("results/endpoint_parameter_sensitivity_v1", (
        "endpoint_parameter_sensitivity.csv", "sensitivity_coverage.csv",
        "all_profile_support.csv", "source_cadence.csv", "summary.json", "manifest.json")),
    "development": ("results/parameter_defensibility_v1", (
        "declared_protocol.json", "development_selection_freeze.json", "temporal_fold_audit.json",
        "search.csv", "scores.csv", "summary.csv", "paired.csv", "calibration_sensitivity.csv",
        "feature_guard_sensitivity.csv", "feature_guard_site_calendar_batches.csv",
        "feature_coverage_budget_sensitivity.csv", "governance_logic_sensitivity.csv",
        "monitor_logic_sensitivity.csv", "declared_hazard_cap_protocol.json",
        "hazard_cap_sensitivity.csv", "hazard_cap_hits.csv", "parameter_coverage_addendum.csv",
        "software_correctness_note.json", "run_manifest.json")),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_text(value: str) -> str:
    value = value.replace(str(ROOT), "Implementation")
    value = value.replace("/Users/dkarusal/anaconda3/bin/python3", "python3")
    return re.sub(r"/Users/[^\s\"']+", "<local-path>", value)


def portable(value):
    if isinstance(value, dict):
        return {portable_text(str(k)): portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        return portable_text(value)
    return value


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite the aggregate export: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for group, (folder, names) in SOURCES.items():
        for name in names:
            source = ROOT / folder / name
            if not source.is_file():
                raise FileNotFoundError(source)
            target = OUTPUT / group / name
            target.parent.mkdir(exist_ok=True)
            if source.suffix == ".json":
                target.write_text(json.dumps(portable(json.loads(source.read_text())), indent=2, sort_keys=True) + "\n")
            elif name == "source_cadence.csv":
                with source.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    rows = list(reader)
                    fields = reader.fieldnames
                for row in rows:
                    row["source_file"] = Path(row["source_file"]).name
                with target.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                shutil.copy2(source, target)
            if "/Users/" in target.read_text():
                raise ValueError(f"Private absolute path remains in export: {target}")
            entries.append({"source": target.relative_to(ROOT).as_posix(),
                "package_path": f"{group}/{name}", "sha256": sha(target),
                "original_source": source.relative_to(ROOT).as_posix(),
                "original_sha256": sha(source), "export_transformation": "aggregate copy; JSON paths made portable; cadence path reduced to basename"})
    registry = OUTPUT / "parameter_registry.csv"
    shutil.copy2(ROOT / "configs/parameter_registry.csv", registry)
    entries.append({"source": registry.relative_to(ROOT).as_posix(), "package_path": registry.name,
                    "sha256": sha(registry), "original_source": "configs/parameter_registry.csv",
                    "original_sha256": sha(registry), "export_transformation": "exact copy"})
    index = {"revision": "R5", "aggregate_only": True, "field_trial_status": "NOT_RUN",
        "published_code_release": "v1.3.1", "new_code_pushed": False,
        "primary_empirical_fit_replaced": False,
        "interpretation": "New retrospective numerical-choice audits and generated-predictor validation; no physical coefficient identification, untouched confirmation, field benefit or universal superiority.",
        "excluded": ["raw measurements", "measured episode predictions", "EOS source scenarios",
            "task/action/outcome rows", "generated predictor references and validation rows", "fitted executable model objects"],
        "files": entries}
    (OUTPUT / "evidence_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "aggregate_files": len(entries), "index_sha256": sha(OUTPUT / "evidence_index.json")}, indent=2))


if __name__ == "__main__":
    main()
