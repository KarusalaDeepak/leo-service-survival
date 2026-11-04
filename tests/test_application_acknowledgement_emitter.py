from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from leo_service_survival.application_acknowledgement import (
    ApplicationOutcomeObservation,
    build_live_application_acknowledgement,
    load_application_acknowledgement_contract,
)
from leo_service_survival.outcome_evidence import load_verified_outcome_evidence
from leo_service_survival.prospective_trial import (
    append_hash_chained_record,
    load_ledger_schema,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER_SCHEMA = ROOT / "prospective_trial" / "schema" / "decision_ledger.schema.json"
ACK_SCHEMA = (
    ROOT
    / "prospective_trial"
    / "schema"
    / "application_outcome_acknowledgement.schema.json"
)
ACTION_TEMPLATE = ROOT / "prospective_trial" / "schema" / "action_record.template.json"
EMITTER = ROOT / "scripts" / "emit_application_outcome_acknowledgement.py"
APPENDER = ROOT / "scripts" / "append_prospective_outcome.py"
KEY = b"endpoint-acknowledgement-test-key-at-least-32-bytes"


def _action() -> dict:
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
            "model_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "evidence_tier": "field_live_randomized",
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
            "rollback_triggered": False,
            "rollback_reason": "",
        }
    )
    return action


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ledger(tmp_path: Path) -> tuple[Path, dict]:
    action = _action()
    ledger = tmp_path / "ledger.jsonl"
    append_hash_chained_record(ledger, action, load_ledger_schema(LEDGER_SCHEMA))
    return ledger, action


def _command(tmp_path: Path, output: Path) -> tuple[list[str], Path, dict]:
    ledger, action = _ledger(tmp_path)
    key_path = tmp_path / "operator.key"
    key_path.write_bytes(KEY)
    command = [
        sys.executable,
        str(EMITTER),
        "--ledger",
        str(ledger),
        "--ledger-schema",
        str(LEDGER_SCHEMA),
        "--acknowledgement-schema",
        str(ACK_SCHEMA),
        "--decision-id",
        action["decision_id"],
        "--signing-key",
        str(key_path),
        "--output",
        str(output),
        "--acknowledgement-id",
        "ack-endpoint-1",
        "--acknowledgement-source",
        "remote-anchor/application-observer/v1",
        "--acknowledged-wall-ns",
        "450",
        "--required-bytes",
        "100",
        "--delivered-bytes",
        "100",
        "--deadline-wall-ns",
        "500",
        "--completion-wall-ns",
        "400",
        "--observation-valid",
        "true",
        "--application-complete",
        "true",
        "--no-action",
        "false",
        "--controller-failure",
        "false",
        "--late-install",
        "false",
    ]
    return command, ledger, action


def test_cli_emits_immutable_authenticated_evidence_consumable_by_appender(
    tmp_path: Path,
) -> None:
    output = tmp_path / "application-ack.json"
    command, ledger, action = _command(tmp_path, output)
    emitted = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    summary = json.loads(emitted.stdout)
    assert summary["outcome_appended"] is False
    assert summary["authentication_purpose"] == "O4_APPLICATION_OUTCOME_ACKNOWLEDGEMENT_V1"
    assert stat.S_IMODE(output.stat().st_mode) == 0o400

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision_id"] == action["decision_id"]
    assert payload["required_bytes"] == 100
    assert payload["delivered_bytes"] == 100
    assert "met_deadline" not in payload
    verified = load_verified_outcome_evidence(output, action=action, signing_key=KEY)
    assert verified.authenticated
    assert verified.met_deadline

    appended = subprocess.run(
        [
            sys.executable,
            str(APPENDER),
            "--ledger",
            str(ledger),
            "--schema",
            str(LEDGER_SCHEMA),
            "--decision-id",
            action["decision_id"],
            "--outcome-evidence",
            str(output),
            "--signing-key",
            str(tmp_path / "operator.key"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert appended.returncode == 0, appended.stderr
    outcome = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert outcome["record_type"] == "outcome"
    assert outcome["conservative_on_time_completion"] is True
    assert outcome["application_acknowledgement_authenticated"] is True
    assert outcome["application_acknowledgement_sha256"] == payload["artifact_sha256"]


def test_cli_refuses_to_replace_existing_evidence(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("operator-owned existing evidence\n", encoding="utf-8")
    original = output.read_bytes()
    command, _, _ = _command(tmp_path, output)
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing to replace existing acknowledgement" in result.stderr
    assert output.read_bytes() == original


def test_invalid_observation_is_rejected_before_signing() -> None:
    contract = load_application_acknowledgement_contract(ACK_SCHEMA)
    observation = ApplicationOutcomeObservation(
        required_bytes=100,
        delivered_bytes=100,
        deadline_wall_ns=500,
        completion_wall_ns=400,
        observation_valid=False,
        application_complete=True,
        no_action=False,
        controller_failure=False,
        late_install=False,
        missingness_reason="",
    )
    with pytest.raises(ValueError, match="invalid observations require"):
        build_live_application_acknowledgement(
            action=_action(),
            observation=observation,
            acknowledgement_id="ack-1",
            acknowledgement_source="endpoint-observer",
            acknowledged_wall_ns=450,
            signing_key=KEY,
            contract=contract,
        )


def test_schema_purpose_change_fails_closed(tmp_path: Path) -> None:
    schema = json.loads(ACK_SCHEMA.read_text(encoding="utf-8"))
    changed = deepcopy(schema)
    changed["properties"]["authentication"]["properties"]["purpose"]["const"] = (
        "WRONG_PURPOSE"
    )
    changed_path = tmp_path / "changed-schema.json"
    _write_json(changed_path, changed)
    with pytest.raises(ValueError, match="purpose differs"):
        load_application_acknowledgement_contract(changed_path)


def test_cli_requires_explicit_validity_booleans(tmp_path: Path) -> None:
    output = tmp_path / "application-ack.json"
    command, _, _ = _command(tmp_path, output)
    index = command.index("--observation-valid")
    command[index + 1] = "unknown"
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "expected true or false" in result.stderr
    assert not output.exists()
