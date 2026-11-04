from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from leo_service_survival.outcome_evidence import (
    FAKE_DRIVER_TEST,
    LIVE_APPLICATION_ACKNOWLEDGEMENT,
    OUTCOME_ACK_PURPOSE,
    load_verified_outcome_evidence,
    verify_outcome_evidence_binding,
)
from leo_service_survival.prospective_release import sign_artifact
from leo_service_survival.prospective_trial import (
    append_hash_chained_record,
    load_ledger_schema,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "prospective_trial" / "schema" / "decision_ledger.schema.json"
ACTION_TEMPLATE = ROOT / "prospective_trial" / "schema" / "action_record.template.json"
KEY = b"outcome-evidence-test-key-at-least-32-bytes"


def _action(*, fake: bool = False) -> dict:
    action = json.loads(ACTION_TEMPLATE.read_text(encoding="utf-8"))
    action.update(
        {
            "recorded_wall_ns": 100,
            "recorded_monotonic_ns": 100,
            "study_id": "study-1",
            "trial_id": "trial-1",
            "campaign_id": "campaign-1",
            "acquisition_family_id": "family-1",
            "session_id": "session-1",
            "task_id": "task-1",
            "decision_id": "decision-1",
            "evidence_tier": (
                "controlled_fake_driver_test" if fake else "field_live_randomized"
            ),
            "gate_state": "authorized",
            "gate_reason": "all_frozen_conditions_passed",
            "guard_state": "pass",
            "monitor_state": "pass",
            "canary_state": "candidate",
            "policy_arm": "gated_candidate",
            "available_path_ids": ["baseline", "candidate"],
            "proposed_path_id": "candidate",
            "baseline_path_id": "baseline",
            "intended_path_id": "candidate",
            "pre_fib_path_id": "baseline",
            "post_fib_path_id": "candidate",
            "installed_path_id": "candidate",
            "final_fib_path_id": "candidate",
            "install_requested": True,
            "install_acknowledged": True,
            "install_verified": True,
            "first_packet_observed": True,
            "first_packet_path_id": "candidate",
            "fallback_triggered": False,
            "fallback_reason": "",
        }
    )
    return action


def _body(action: dict, *, fake: bool = False) -> dict:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "application_outcome_acknowledgement",
        "evidence_status": (
            FAKE_DRIVER_TEST if fake else LIVE_APPLICATION_ACKNOWLEDGEMENT
        ),
        "evidence_origin": "fake_driver_test" if fake else "executed_live",
        "acknowledgement_id": "ack-1",
        "acknowledgement_source": "destination-application-observer",
        "acknowledged_wall_ns": 450,
        "claim_boundary": "One task acknowledgement; no broader performance claim.",
        **{
            field: action[field]
            for field in (
                "study_id",
                "trial_id",
                "protocol_version",
                "campaign_id",
                "acquisition_family_id",
                "session_id",
                "task_id",
                "decision_id",
                "model_sha256",
                "config_sha256",
            )
        },
        "required_bytes": 100,
        "delivered_bytes": 100,
        "deadline_wall_ns": 500,
        "completion_wall_ns": 400,
        "observation_valid": True,
        "application_complete": True,
        "no_action": False,
        "controller_failure": False,
        "late_install": False,
        "missingness_reason": "",
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_live_acknowledgement_is_authenticated_and_deadline_is_derived(
    tmp_path: Path,
) -> None:
    action = _action()
    evidence_path = tmp_path / "outcome.json"
    signed = sign_artifact(_body(action), KEY, purpose=OUTCOME_ACK_PURPOSE)
    _write(evidence_path, signed)

    evidence = load_verified_outcome_evidence(
        evidence_path, action=action, signing_key=KEY
    )
    assert evidence.authenticated
    assert evidence.met_deadline
    record = {
        **{
            field: signed[field]
            for field in (
                "study_id",
                "trial_id",
                "protocol_version",
                "campaign_id",
                "acquisition_family_id",
                "session_id",
                "task_id",
                "decision_id",
                "model_sha256",
                "config_sha256",
                "required_bytes",
                "delivered_bytes",
                "deadline_wall_ns",
                "completion_wall_ns",
                "observation_valid",
                "application_complete",
                "no_action",
                "controller_failure",
                "late_install",
                "missingness_reason",
            )
        },
        "met_deadline": True,
        "outcome_evidence_sha256": evidence.evidence_file_sha256,
        "outcome_evidence_status": LIVE_APPLICATION_ACKNOWLEDGEMENT,
        "outcome_evidence_origin": "executed_live",
        "application_acknowledgement_sha256": evidence.acknowledgement_sha256,
        "application_acknowledgement_id": "ack-1",
        "application_acknowledgement_source": signed["acknowledgement_source"],
        "application_acknowledgement_wall_ns": signed["acknowledged_wall_ns"],
        "application_acknowledgement_authenticated": True,
        "application_acknowledgement_key_id": evidence.authentication_key_id,
    }
    assert verify_outcome_evidence_binding(
        record, evidence_path, action=action, signing_key=KEY
    )
    evidence_path.write_bytes(evidence_path.read_bytes() + b"\n")
    assert not verify_outcome_evidence_binding(
        record, evidence_path, action=action, signing_key=KEY
    )


def test_observer_cannot_self_declare_a_contradictory_deadline_flag(
    tmp_path: Path,
) -> None:
    action = _action()
    body = _body(action)
    body["completion_wall_ns"] = 600
    body["acknowledged_wall_ns"] = 650
    body["met_deadline"] = True
    path = tmp_path / "contradictory.json"
    _write(path, sign_artifact(body, KEY, purpose=OUTCOME_ACK_PURPOSE))
    with pytest.raises(ValueError, match="derived outcome fields"):
        load_verified_outcome_evidence(path, action=action, signing_key=KEY)


def test_tampered_signed_acknowledgement_is_rejected(tmp_path: Path) -> None:
    action = _action()
    path = tmp_path / "tampered.json"
    signed = sign_artifact(_body(action), KEY, purpose=OUTCOME_ACK_PURPOSE)
    signed["acknowledgement_source"] = "tampered-source"
    _write(path, signed)
    with pytest.raises(ValueError, match="prospective-trial validation failed"):
        load_verified_outcome_evidence(path, action=action, signing_key=KEY)


def test_wrong_decision_or_task_identity_is_rejected(tmp_path: Path) -> None:
    action = _action()
    body = _body(action)
    body["decision_id"] = "different-decision"
    body["task_id"] = "different-task"
    path = tmp_path / "wrong-identity.json"
    _write(path, sign_artifact(body, KEY, purpose=OUTCOME_ACK_PURPOSE))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_verified_outcome_evidence(path, action=action, signing_key=KEY)


def test_missing_acknowledgement_and_missing_live_key_fail_closed(
    tmp_path: Path,
) -> None:
    action = _action()
    with pytest.raises(FileNotFoundError):
        load_verified_outcome_evidence(
            tmp_path / "missing.json", action=action, signing_key=KEY
        )
    path = tmp_path / "live.json"
    _write(path, sign_artifact(_body(action), KEY, purpose=OUTCOME_ACK_PURPOSE))
    with pytest.raises(ValueError, match="signing key is required"):
        load_verified_outcome_evidence(path, action=action)


def test_fake_driver_acknowledgement_stays_explicitly_non_field(
    tmp_path: Path,
) -> None:
    action = _action(fake=True)
    path = tmp_path / "fake.json"
    _write(path, _body(action, fake=True))
    evidence = load_verified_outcome_evidence(path, action=action)
    assert evidence.met_deadline
    assert not evidence.authenticated
    assert evidence.authentication_key_id == ""


def test_append_cli_uses_acknowledgement_and_derives_late_failure(
    tmp_path: Path,
) -> None:
    action = _action()
    schema = load_ledger_schema(SCHEMA_PATH)
    ledger = tmp_path / "ledger.jsonl"
    append_hash_chained_record(ledger, action, schema)
    body = _body(action)
    body["completion_wall_ns"] = 600
    body["acknowledged_wall_ns"] = 650
    evidence_path = tmp_path / "outcome.json"
    _write(evidence_path, sign_artifact(body, KEY, purpose=OUTCOME_ACK_PURPOSE))
    key_path = tmp_path / "operator.key"
    key_path.write_bytes(KEY)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "append_prospective_outcome.py"),
            "--ledger",
            str(ledger),
            "--schema",
            str(SCHEMA_PATH),
            "--decision-id",
            action["decision_id"],
            "--outcome-evidence",
            str(evidence_path),
            "--signing-key",
            str(key_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert outcome["met_deadline"] is False
    assert outcome["conservative_on_time_completion"] is False
    assert outcome["failure_stage"] == "deadline"
    assert outcome["application_acknowledgement_authenticated"] is True

