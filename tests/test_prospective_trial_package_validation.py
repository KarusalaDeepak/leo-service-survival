from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_prospective_trial_package import (  # noqa: E402
    validate_controlled_rehearsal,
    validate_required_package_paths,
    validate_status_boundaries,
)


def _root_status() -> dict:
    return {
        "field_trial_status": "NOT_RUN",
        "field_trial_completed": False,
        "benefit_claim_supported": False,
        "external_dependencies": ["live hardware and approvals"],
        "software_pipeline_status": "SEVEN_STEP_PIPELINE_IMPLEMENTED_AND_TESTED",
        "controlled_rehearsal_status": "FAKE_DRIVER_TEST_COMPLETE",
        "controlled_rehearsal_status_path": (
            "prospective_trial/controlled_dry_run/STATUS.json"
        ),
        "software_steps": [
            {
                "step": step,
                "status": "IMPLEMENTED_AND_TESTED",
                "live_execution_status": "NOT_RUN",
                "mechanism": f"software mechanism {step}",
            }
            for step in range(1, 8)
        ],
        "external_execution_steps": [
            {"step": step, "status": "NOT_RUN"} for step in range(1, 8)
        ],
    }


def test_status_boundary_separates_implemented_software_from_unrun_field_work() -> None:
    report = validate_status_boundaries(_root_status())
    assert report["passed"] is True
    assert report["software_step_count"] == 7
    assert report["external_step_count"] == 7

    invalid = deepcopy(_root_status())
    invalid["software_steps"][3]["live_execution_status"] = "COMPLETE"
    invalid["external_execution_steps"][3]["status"] = "COMPLETE"
    report = validate_status_boundaries(invalid)
    assert report["passed"] is False
    assert any("step 4" in error for error in report["errors"])


def test_release_manifest_declares_every_required_package_file() -> None:
    report = validate_required_package_paths(ROOT)
    assert report["passed"] is True, report
    assert report["declared_path_count"] >= 60


def test_checked_in_controlled_rehearsal_is_fake_and_hash_complete() -> None:
    report = validate_controlled_rehearsal(ROOT / "prospective_trial")
    assert report["passed"] is True, report
    assert report["evidence_status"] == "FAKE_DRIVER_TEST"
    assert report["field_trial_status"] == "NOT_RUN"
    assert report["indexed_file_count"] >= 30
