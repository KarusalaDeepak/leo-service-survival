#!/usr/bin/env python3
"""Create or verify hashes for the local prospective-trial execution package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_trial import (  # noqa: E402
    build_reproducibility_manifest,
    verify_reproducibility_manifest,
)


MANIFEST_PATHS = (
    "prospective_trial/STATUS.json",
    "prospective_trial/README.md",
    "prospective_trial/config/acquisition_family_register.csv",
    "prospective_trial/config/campaign_registry_lock.template.csv",
    "prospective_trial/config/testbed_qualification.template.json",
    "prospective_trial/config/gate_canary_rollback.json",
    "prospective_trial/config/gate_opportunity_derivation.template.json",
    "prospective_trial/config/final_test_analysis.template.json",
    "prospective_trial/config/route_driver.template.json",
    "prospective_trial/preparation/README.md",
    "prospective_trial/protocol/preregistration.md",
    "prospective_trial/protocol/statistical_analysis_plan.md",
    "prospective_trial/protocol/execution_runbook.md",
    "prospective_trial/schema/action_record.template.json",
    "prospective_trial/schema/application_outcome_acknowledgement.schema.json",
    "prospective_trial/schema/application_outcome_acknowledgement.template.json",
    "prospective_trial/schema/decision_ledger.schema.json",
    "prospective_trial/schema/decision_ledger_data_dictionary.csv",
    "prospective_trial/schema/gate_evidence.schema.json",
    "prospective_trial/schema/gate_evidence.template.json",
    "prospective_trial/schema/gate_opportunity_manifest.schema.json",
    "prospective_trial/schema/gate_opportunity_table.schema.json",
    "prospective_trial/schema/locked_gate_evidence_v2.schema.json",
    "prospective_trial/schema/outcome_record.template.json",
    "prospective_trial/schema/release_artifact_authentication.schema.json",
    "prospective_trial/schema/expected_probe_schedule.template.csv",
    "prospective_trial/schema/synchronized_telemetry.schema.json",
    "prospective_trial/schema/trial_freeze_manifest.schema.json",
    # The compact controlled rehearsal is transitively complete: its own manifest
    # authenticates every fake input and generated output without presenting any
    # of them as field evidence.
    "prospective_trial/controlled_dry_run/README.md",
    "prospective_trial/controlled_dry_run/STATUS.json",
    "prospective_trial/controlled_dry_run/manifest.json",
    "scripts/append_prospective_outcome.py",
    "scripts/emit_application_outcome_acknowledgement.py",
    "scripts/plan_blinded_pilot.py",
    "scripts/build_prospective_trial_manifest.py",
    "scripts/build_closed_loop_audit_report.py",
    "scripts/build_fixed_canary_plan.py",
    "scripts/build_fixed_final_test_plan.py",
    "scripts/build_gate_opportunity_manifest.py",
    "scripts/build_locked_gate_evidence.py",
    "scripts/build_trial_freeze_manifest.py",
    "scripts/evaluate_canary_checkpoint.py",
    "scripts/execute_canary_rollback.py",
    "scripts/ingest_synchronized_telemetry.py",
    "scripts/lock_campaign_registry.py",
    "scripts/qualify_field_testbed.py",
    "scripts/run_controlled_seven_step_dry_run.py",
    "scripts/run_final_test_itt_analysis.py",
    "scripts/run_frozen_evidence_gate.py",
    "scripts/run_prospective_action.py",
    "scripts/unlock_untouched_final_test.py",
    "scripts/validate_prospective_trial_package.py",
    "scripts/verify_rejection_fallback.py",
    "src/leo_service_survival/__init__.py",
    "src/leo_service_survival/application_acknowledgement.py",
    "src/leo_service_survival/controlled_dry_run.py",
    "src/leo_service_survival/field_preparation.py",
    "src/leo_service_survival/gate_freeze.py",
    "src/leo_service_survival/live_actuation.py",
    "src/leo_service_survival/outcome_evidence.py",
    "src/leo_service_survival/pilot_power.py",
    "src/leo_service_survival/prospective_release.py",
    "src/leo_service_survival/prospective_trial.py",
    "tests/test_controlled_seven_step_dry_run.py",
    "tests/test_application_acknowledgement_emitter.py",
    "tests/test_field_preparation.py",
    "tests/test_gate_freeze_integration.py",
    "tests/test_live_actuation.py",
    "tests/test_outcome_evidence.py",
    "tests/test_pilot_power.py",
    "tests/test_prospective_release.py",
    "tests/test_prospective_trial.py",
    "tests/test_prospective_trial_package_validation.py",
    "docs/application_outcome_acknowledgement_emitter.md",
    "docs/blinded_pilot_power_planning.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify:
        manifest = json.loads(args.verify.read_text(encoding="utf-8"))
        report = verify_reproducibility_manifest(ROOT, manifest)
        print(report.checks.to_json(orient="records", indent=2))
        return 0 if report.passed else 1

    manifest = build_reproducibility_manifest(
        ROOT,
        MANIFEST_PATHS,
        protocol_status="DRAFT_NOT_REGISTERED",
    )
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite {args.output}; use --force")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
