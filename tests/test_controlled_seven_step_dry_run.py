from __future__ import annotations

import json
from pathlib import Path

import pytest

from leo_service_survival.controlled_dry_run import (
    run_controlled_seven_step_dry_run,
)


ROOT = Path(__file__).resolve().parents[1]


def test_seven_step_dry_run_exercises_both_release_branches(tmp_path: Path) -> None:
    output = tmp_path / "run"
    status = run_controlled_seven_step_dry_run(output, package_root=ROOT)
    assert status["evidence_status"] == "FAKE_DRIVER_TEST"
    assert status["field_trial_status"] == "NOT_RUN"
    assert status["software_rehearsal_complete"] is True
    assert status["gate_evidence_status"] == "FAKE_RAW_OPPORTUNITY_DERIVED_AND_HASH_LOCKED"
    assert status["steps"]["4_frozen_gate_rejected_fixture"] == "REJECTED"
    assert status["steps"]["4_frozen_gate_authorized_fixture"] == "AUTHORIZED"
    assert status["steps"]["5_baseline_retention_and_rollback"] == "VERIFIED"
    assert status["steps"]["6_canary_checkpoint"] == "CANARY_CHECKPOINT_PASSED"
    assert status["steps"]["6_untouched_final_test_unlock"] == "UNTOUCHED_FINAL_TEST_UNLOCKED"
    assert status["analysis_benefit_claim_supported"] is False
    assert status["raw_evidence_derived_gate_run"] is True
    assert status["prospective_live_actions_installed"] is False

    rejection = json.loads(
        (output / "outputs" / "rejected_branch" / "05_baseline_and_rollback_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert rejection["candidate_installed_in_production"] is False
    assert rejection["production_baseline_verification"]["passed"] is True
    analysis = json.loads(
        (output / "outputs" / "authorized_branch" / "07_prespecified_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    assert analysis["benefit_claim_supported"] is False
    assert analysis["risk_difference_candidate_minus_baseline"] == 0.0


def test_dry_run_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_status = run_controlled_seven_step_dry_run(left, package_root=ROOT)
    right_status = run_controlled_seven_step_dry_run(right, package_root=ROOT)
    assert left_status == right_status
    for branch in ("rejected_fixture", "authorized_fixture"):
        left_evidence = json.loads(
            (left / "inputs" / branch / "gate_evidence.raw_derived.json").read_text(
                encoding="utf-8"
            )
        )
        right_evidence = json.loads(
            (right / "inputs" / branch / "gate_evidence.raw_derived.json").read_text(
                encoding="utf-8"
            )
        )
        assert left_evidence["pooled_metrics"] == right_evidence["pooled_metrics"]
        assert left_evidence["point_estimates"] == right_evidence["point_estimates"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_controlled_seven_step_dry_run(left, package_root=ROOT)
