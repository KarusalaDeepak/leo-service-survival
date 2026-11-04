#!/usr/bin/env python3
"""Validate the Objective 4 prospective-trial package without running a trial."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_trial import (  # noqa: E402
    load_ledger_schema,
    validate_campaign_register,
    validate_ledger_record,
    validate_trial_configuration,
)
from build_prospective_trial_manifest import MANIFEST_PATHS  # noqa: E402


EXPECTED_SOFTWARE_PIPELINE_STATUS = "SEVEN_STEP_PIPELINE_IMPLEMENTED_AND_TESTED"
EXPECTED_CONTROLLED_REHEARSAL_STATUS = "FAKE_DRIVER_TEST_COMPLETE"
EXPECTED_CONTROLLED_STATUS_PATH = (
    "prospective_trial/controlled_dry_run/STATUS.json"
)
EXPECTED_REHEARSAL_FILES = {
    "STATUS.json",
    "outputs/01_qualification_report.json",
    "outputs/02_campaign_registry_lock.json",
    "outputs/03_synchronized_telemetry.jsonl",
    "outputs/03_telemetry_audit.json",
    "outputs/rejected_branch/04_gate_report.json",
    "outputs/rejected_branch/05_baseline_and_rollback_report.json",
    "outputs/authorized_branch/04_gate_report.json",
    "outputs/authorized_branch/06_canary_plan.json",
    "outputs/authorized_branch/06_canary_checkpoint.json",
    "outputs/authorized_branch/06_final_test_unlock.json",
    "outputs/authorized_branch/06_final_test_plan.json",
    "outputs/authorized_branch/07_action_outcome_ledger.jsonl",
    "outputs/authorized_branch/07_end_to_end_audit.json",
    "outputs/authorized_branch/07_prespecified_analysis.json",
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_required_package_paths(root: Path) -> dict[str, Any]:
    """Require every source, contract, schema, and test in the release manifest."""

    duplicates = sorted(
        {item for item in MANIFEST_PATHS if MANIFEST_PATHS.count(item) > 1}
    )
    missing = sorted(item for item in MANIFEST_PATHS if not (root / item).is_file())
    structured_errors: list[str] = []
    for relative in MANIFEST_PATHS:
        path = root / relative
        if not path.is_file():
            continue
        if relative.startswith(("prospective_trial/config/", "prospective_trial/schema/")):
            if path.suffix == ".json":
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    structured_errors.append(f"{relative}: invalid JSON: {exc}")
                    continue
                if not isinstance(value, dict) or not value:
                    structured_errors.append(
                        f"{relative}: top-level JSON must be a non-empty object"
                    )
            elif path.suffix == ".csv":
                try:
                    frame = pd.read_csv(path, keep_default_na=False)
                except Exception as exc:  # pragma: no cover - pandas error text varies
                    structured_errors.append(f"{relative}: invalid CSV: {exc}")
                    continue
                if not len(frame.columns) or frame.columns.duplicated().any():
                    structured_errors.append(
                        f"{relative}: CSV must have non-duplicate columns"
                    )
    return {
        "passed": not duplicates and not missing and not structured_errors,
        "declared_path_count": len(MANIFEST_PATHS),
        "duplicates": duplicates,
        "missing": missing,
        "structured_file_errors": structured_errors,
    }


def validate_status_boundaries(status: Mapping[str, Any]) -> dict[str, Any]:
    """Keep implemented software, fake rehearsal, and unrun field work separate."""

    errors: list[str] = []
    if status.get("field_trial_status") != "NOT_RUN":
        errors.append("field_trial_status must be NOT_RUN")
    if status.get("field_trial_completed") is not False:
        errors.append("field_trial_completed must be false")
    if status.get("benefit_claim_supported") is not False:
        errors.append("benefit_claim_supported must be false")
    if not status.get("external_dependencies"):
        errors.append("external_dependencies must remain explicit")
    if status.get("software_pipeline_status") != EXPECTED_SOFTWARE_PIPELINE_STATUS:
        errors.append(
            f"software_pipeline_status must be {EXPECTED_SOFTWARE_PIPELINE_STATUS}"
        )
    if status.get("controlled_rehearsal_status") != EXPECTED_CONTROLLED_REHEARSAL_STATUS:
        errors.append(
            f"controlled_rehearsal_status must be {EXPECTED_CONTROLLED_REHEARSAL_STATUS}"
        )
    if status.get("controlled_rehearsal_status_path") != EXPECTED_CONTROLLED_STATUS_PATH:
        errors.append(
            f"controlled_rehearsal_status_path must be {EXPECTED_CONTROLLED_STATUS_PATH}"
        )

    software_steps = status.get("software_steps")
    if not isinstance(software_steps, list):
        errors.append("software_steps must be an array")
    else:
        numbers = [item.get("step") for item in software_steps if isinstance(item, dict)]
        if numbers != list(range(1, 8)):
            errors.append("software_steps must contain ordered steps 1 through 7")
        for item in software_steps:
            if not isinstance(item, dict):
                errors.append("every software_steps item must be an object")
                continue
            step = item.get("step")
            if item.get("status") != "IMPLEMENTED_AND_TESTED":
                errors.append(f"software step {step} status must be IMPLEMENTED_AND_TESTED")
            if item.get("live_execution_status") != "NOT_RUN":
                errors.append(f"software step {step} live_execution_status must be NOT_RUN")
            if not str(item.get("mechanism", "")).strip():
                errors.append(f"software step {step} must name its mechanism")

    external_steps = status.get("external_execution_steps")
    if not isinstance(external_steps, list):
        errors.append("external_execution_steps must be an array")
    else:
        numbers = [item.get("step") for item in external_steps if isinstance(item, dict)]
        if numbers != list(range(1, 8)):
            errors.append("external_execution_steps must contain ordered steps 1 through 7")
        for item in external_steps:
            if not isinstance(item, dict):
                errors.append("every external_execution_steps item must be an object")
                continue
            if item.get("status") != "NOT_RUN":
                errors.append(
                    f"external execution step {item.get('step')} status must be NOT_RUN"
                )

    return {
        "passed": not errors,
        "software_pipeline_status": status.get("software_pipeline_status"),
        "controlled_rehearsal_status": status.get("controlled_rehearsal_status"),
        "field_trial_status": status.get("field_trial_status"),
        "software_step_count": len(software_steps) if isinstance(software_steps, list) else 0,
        "external_step_count": len(external_steps) if isinstance(external_steps, list) else 0,
        "errors": errors,
    }


def validate_controlled_rehearsal(package: Path) -> dict[str, Any]:
    """Verify the fake rehearsal's claim boundary and its transitive file hashes."""

    root = package / "controlled_dry_run"
    status_path = root / "STATUS.json"
    manifest_path = root / "manifest.json"
    errors: list[str] = []
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"passed": False, "errors": [str(exc)]}

    if status.get("evidence_status") != "FAKE_DRIVER_TEST":
        errors.append("controlled status evidence_status must be FAKE_DRIVER_TEST")
    if status.get("field_trial_status") != "NOT_RUN":
        errors.append("controlled status field_trial_status must be NOT_RUN")
    if status.get("software_rehearsal_complete") is not True:
        errors.append("controlled status software_rehearsal_complete must be true")
    if status.get("analysis_benefit_claim_supported") is not False:
        errors.append("controlled status must not support a benefit claim")
    if status.get("prospective_live_actions_installed") is not False:
        errors.append("controlled status must not claim live installed actions")
    if manifest.get("evidence_status") != "FAKE_DRIVER_TEST":
        errors.append("controlled manifest evidence_status must be FAKE_DRIVER_TEST")
    if manifest.get("field_trial_status") != "NOT_RUN":
        errors.append("controlled manifest field_trial_status must be NOT_RUN")

    entries = manifest.get("files")
    indexed: set[str] = set()
    if not isinstance(entries, list):
        errors.append("controlled manifest files must be an array")
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("controlled manifest file entry must be an object")
            continue
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str):
            errors.append("controlled manifest file path must be a string")
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"unsafe controlled manifest path: {relative}")
            continue
        if relative in indexed:
            errors.append(f"duplicate controlled manifest path: {relative}")
            continue
        indexed.add(relative)
        path = root / relative_path
        if not path.is_file():
            errors.append(f"controlled manifest file missing: {relative}")
            continue
        if _sha256_file(path) != expected_hash:
            errors.append(f"controlled manifest hash mismatch: {relative}")
        if entry.get("bytes") != path.stat().st_size:
            errors.append(f"controlled manifest byte count mismatch: {relative}")

    missing_expected = sorted(EXPECTED_REHEARSAL_FILES - indexed)
    if missing_expected:
        errors.append(
            "controlled manifest omits required seven-step artifacts: "
            + ", ".join(missing_expected)
        )
    return {
        "passed": not errors,
        "evidence_status": status.get("evidence_status"),
        "field_trial_status": status.get("field_trial_status"),
        "indexed_file_count": len(indexed),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-locked-campaigns",
        action="store_true",
        help="Require every included campaign to be locked; use only at trial freeze.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = ROOT / "prospective_trial"
    config_path = package / "config" / "gate_canary_rollback.json"
    register_path = package / "config" / "acquisition_family_register.csv"
    schema_path = package / "schema" / "decision_ledger.schema.json"
    action_template_path = package / "schema" / "action_record.template.json"
    outcome_template_path = package / "schema" / "outcome_record.template.json"
    dictionary_path = package / "schema" / "decision_ledger_data_dictionary.csv"
    status_path = package / "STATUS.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    register = pd.read_csv(register_path, keep_default_na=False)
    schema = load_ledger_schema(schema_path)
    dictionary = pd.read_csv(dictionary_path, keep_default_na=False)
    action_template = json.loads(action_template_path.read_text(encoding="utf-8"))
    outcome_template = json.loads(outcome_template_path.read_text(encoding="utf-8"))

    gate = config["gate"]
    reports = {
        "configuration": validate_trial_configuration(config),
        "campaign_register": validate_campaign_register(
            register,
            require_locked=args.require_locked_campaigns,
            minimum_gate_campaigns=gate["minimum_independent_gate_campaigns"],
            minimum_gate_families=gate["minimum_independent_gate_families"],
        ),
        "action_template": validate_ledger_record(action_template, schema),
        "outcome_template": validate_ledger_record(outcome_template, schema),
    }

    schema_fields = set(schema["fields"])
    dictionary_fields = set(dictionary["field"])
    dictionary_ok = schema_fields == dictionary_fields and not dictionary["field"].duplicated().any()
    status_report = validate_status_boundaries(status)
    rehearsal_report = validate_controlled_rehearsal(package)
    required_paths_report = validate_required_package_paths(ROOT)

    output = {
        "package": str(package),
        "protocol_status": config["protocol_status"],
        "field_trial_status": status.get("field_trial_status"),
        "reports": {
            name: report.checks.to_dict(orient="records")
            for name, report in reports.items()
        },
        "schema_dictionary_match": dictionary_ok,
        "status_boundary": status_report,
        "controlled_rehearsal": rehearsal_report,
        "required_package_paths": required_paths_report,
        "machine_readable_not_run_status": status_report["passed"],
        "passed": bool(
            all(report.passed for report in reports.values())
            and dictionary_ok
            and status_report["passed"]
            and rehearsal_report["passed"]
            and required_paths_report["passed"]
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
