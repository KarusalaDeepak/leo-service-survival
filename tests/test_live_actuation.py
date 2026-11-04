from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from leo_service_survival.field_preparation import (
    LOCK_REQUIRED_COLUMNS,
    ProbeResult,
    append_telemetry_record,
    build_campaign_registry_lock,
    probe_command_sha256,
    probe_observation_sha256,
    validate_route_driver_configuration,
    validate_testbed_qualification,
)
from leo_service_survival.live_actuation import (
    verify_live_action_bindings,
    verify_live_qualification,
)
from leo_service_survival.gate_freeze import (
    CONFIRMATORY_LIVE,
    build_trial_freeze_manifest,
)
from leo_service_survival.prospective_release import sign_artifact
from leo_service_survival.prospective_trial import sha256_file


KEY = b"live-actuation-test-key-at-least-32-bytes"
MODEL_HASH = "a" * 64
PROBES = (
    "cellular_link",
    "leo_link",
    "dual_interface",
    "remote_anchor",
    "clock_error",
    "session_continuity",
    "permission_bundle",
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _live_configs(tmp_path: Path) -> tuple[Path, Path, Path, Path, tuple[ProbeResult, ...]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    commands = {name: ["/usr/bin/id", "-u"] for name in PROBES}
    components = {
        name: {
            "asset_or_service_id": f"asset-{name}",
            "status": "QUALIFIED",
            "evidence_sha256": sha256(name.encode()).hexdigest(),
        }
        for name in (
            "cellular_5g_terminal",
            "leo_terminal",
            "dual_access_gateway",
            "remote_anchor",
            "clock_sync",
            "session_preserving_path",
        )
    }
    components["cellular_5g_terminal"].update(
        {"interface": "wwan0", "service_active": True, "telemetry_export_available": True}
    )
    components["leo_terminal"].update(
        {"interface": "sat0", "service_active": True, "telemetry_export_available": True}
    )
    components["dual_access_gateway"].update(
        {
            "cellular_interface": "wwan0",
            "leo_interface": "sat0",
            "independent_fib_observer": True,
            "emergency_baseline_restore": True,
        }
    )
    components["remote_anchor"].update(
        {"controlled_endpoint": True, "application_acknowledgement": True}
    )
    components["clock_sync"].update({"method": "PTP", "maximum_error_ms": 0.5})
    components["session_preserving_path"].update(
        {"mechanism": "anchored_tunnel", "preserves_session_identity": True}
    )
    testbed = {
        "qualification_status": "LIVE_REQUIRED",
        "evidence_source": "live_hardware",
        "field_trial_completed": False,
        "maximum_clock_error_ms": 1.0,
        "components": components,
        "permissions": {
            name: {
                "status": "APPROVED",
                "approval_id": f"approval-{name}",
                "evidence_sha256": sha256(f"permission-{name}".encode()).hexdigest(),
            }
            for name in ("institutional", "operator", "network_change", "ethics_or_exemption")
        },
        "read_only_probes": commands,
    }
    paths = ("reactive_baseline", "cellular", "leo")
    route = {
        "status": "FROZEN_DEPLOYMENT_CONFIG",
        "evidence_source": "live_hardware",
        "destination": "198.51.100.10",
        "baseline_path_id": "reactive_baseline",
        "candidate_path_ids": ["cellular", "leo"],
        "maximum_total_loop_ms": 100.0,
        "first_packet_timeout_ms": 500.0,
        "snapshot_command": ["/usr/bin/id", "-u"],
        "install_commands": {path: ["/usr/bin/id", "-u"] for path in paths},
        "first_packet_commands": {path: ["/usr/bin/id", "-u"] for path in paths},
        "isolated_rollback_drill": {
            "isolation_attested": True,
            "namespace_id": "o4-test-namespace",
            "inject_fault_command": ["/usr/bin/id", "-u"],
            "restore_baseline_command": ["/usr/bin/id", "-u"],
            "verify_baseline_command": ["/usr/bin/id", "-u"],
            "observe_first_packet_command": ["/usr/bin/id", "-u"],
        },
        "contract": {
            "snapshot_stdout_first_line": "resolved path",
            "install_exit_zero": "acknowledged",
            "first_packet_exit_zero": "observed",
            "first_packet_stdout_first_line": "independently observed path",
            "commands_use_shell": False,
            "live_mutation_default": False,
        },
    }
    testbed_path = tmp_path / "testbed.json"
    route_path = tmp_path / "route.json"
    schema_path = tmp_path / "schema.json"
    trial_path = tmp_path / "trial.json"
    _write_json(testbed_path, testbed)
    _write_json(route_path, route)
    _write_json(schema_path, {"schema": "frozen-ledger"})
    _write_json(
        trial_path,
        {
            "protocol_id": "O4-V1",
            "protocol_status": "FROZEN_REGISTERED",
            "field_trial_completed": False,
        },
    )
    results = tuple(
        ProbeResult(
            probe_id=name,
            passed=True,
            observed=f"live result {name}",
            evidence_sha256=probe_observation_sha256(f"live result {name}"),
            command_sha256=probe_command_sha256(commands[name]),
            evidence_origin="executed_live",
        )
        for name in PROBES
    )
    return testbed_path, route_path, schema_path, trial_path, results


def _qualified_fixture(tmp_path: Path) -> dict:
    testbed_path, route_path, schema_path, trial_path, probes = _live_configs(tmp_path)
    testbed = json.loads(testbed_path.read_text())
    route = json.loads(route_path.read_text())
    testbed_report = validate_testbed_qualification(testbed, probe_results=probes)
    route_report = validate_route_driver_configuration(route)
    assert testbed_report.live_eligible and route_report.live_eligible
    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "live_testbed_qualification",
        "field_trial_status": "NOT_RUN",
        "qualification_status": "LIVE_QUALIFIED",
        "evidence_source": "live_hardware",
        "probe_evidence_origin": "AUTHENTICATED_LIVE_EXECUTION",
        "testbed_config_sha256": sha256_file(testbed_path),
        "route_config_sha256": sha256_file(route_path),
        "ledger_schema_sha256": sha256_file(schema_path),
        "testbed_evidence_status": "LIVE_QUALIFIED",
        "testbed_live_eligible": True,
        "route_evidence_status": "LIVE_QUALIFIED",
        "route_live_eligible": True,
        "probe_results": [item.__dict__ for item in probes],
        "testbed_checks": testbed_report.checks.to_dict(orient="records"),
        "route_checks": route_report.checks.to_dict(orient="records"),
        "passed": True,
    }
    qualification = sign_artifact(payload, KEY, purpose="O4_LIVE_TESTBED_QUALIFICATION_V1")
    qualification_path = tmp_path / "qualification.json"
    _write_json(qualification_path, qualification)
    return {
        "testbed": testbed_path,
        "route": route_path,
        "schema": schema_path,
        "trial": trial_path,
        "qualification": qualification_path,
        "qualification_value": qualification,
    }


def _telemetry(path: Path, campaign: str, family: str, session: str) -> None:
    append_telemetry_record(
        path,
        {
            "schema_version": "1.0.0",
            "record_type": "network_probe",
            "evidence_status": "live_hardware",
            "wall_time_ns": 10,
            "monotonic_time_ns": 10,
            "clock_epoch_id": "clock-1",
            "maximum_clock_error_ms": 0.5,
            "clock_evidence_sha256": "b" * 64,
            "study_id": "O4",
            "campaign_id": campaign,
            "acquisition_family_id": family,
            "session_id": session,
            "source_id": "probe-source",
            "sequence_number": 0,
            "probe_id": "probe-1",
            "path_id": "cellular",
            "disposition": "reply",
            "sent_wall_time_ns": 1,
            "received_wall_time_ns": 2,
            "rtt_ms": 1.0,
            "failure_reason": "",
        },
    )


def _bindings_fixture(tmp_path: Path) -> dict:
    values = _qualified_fixture(tmp_path)
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"frozen live model")
    model_hash = sha256_file(model_path)
    telemetry = tmp_path / "telemetry.jsonl"
    _telemetry(telemetry, "GATE-01", "GATE-F01", "session-1")
    rows = []
    roles = [
        ("DEV-01", "DEV-F01", "development"),
        ("GATE-01", "GATE-F01", "gate_evidence"),
        ("GATE-02", "GATE-F02", "gate_evidence"),
        ("GATE-03", "GATE-F03", "gate_evidence"),
        ("GATE-04", "GATE-F01", "gate_evidence"),
        ("GATE-05", "GATE-F02", "gate_evidence"),
        ("TEST-01", "TEST-F01", "final_test"),
    ]
    for index, (campaign, family, role) in enumerate(roles):
        unique = sha256(f"{campaign}-raw".encode()).hexdigest()
        if campaign == "GATE-01":
            unique = sha256_file(telemetry)
        row = {
            "study_id": "O4",
            "campaign_id": campaign,
            "acquisition_family_id": family,
            "role": role,
            "acquisition_pipeline_id": f"pipeline-{family}",
            "site_id": f"site-{family}",
            "operator_id": f"operator-{family}",
            "terminal_id": f"terminal-{family}",
            "route_id": f"route-{family}",
            "server_id": f"server-{family}",
            "collection_start_utc": f"2026-01-{index + 1:02d}T00:00:00Z",
            "collection_end_utc": f"2026-01-{index + 1:02d}T01:00:00Z",
            "protocol_version": "O4-V1",
            "model_sha256": model_hash,
            "config_sha256": sha256_file(values["trial"]),
            "testbed_qualification_sha256": sha256_file(values["qualification"]),
            "raw_capture_sha256": unique,
            "gate_opportunity_sha256": (
                sha256(f"{campaign}-gate-opportunity".encode()).hexdigest()
                if role == "gate_evidence"
                else ""
            ),
            "telemetry_manifest_sha256": sha256(f"{campaign}-telemetry".encode()).hexdigest(),
            "outcome_manifest_sha256": sha256(f"{campaign}-outcome".encode()).hexdigest(),
            "clock_sync_method": "PTP",
            "maximum_clock_error_ms": 0.5,
            "status": "complete",
            "evidence_tier": "field_shadow_concurrent",
            "role_access_group": f"access-{role}",
        }
        rows.append(row)
    frame = pd.DataFrame(rows, columns=sorted(LOCK_REQUIRED_COLUMNS))
    lock = build_campaign_registry_lock(
        frame,
        protocol_sha256=sha256_file(values["trial"]),
        qualification_sha256=sha256_file(values["qualification"]),
        lock_timestamp_utc="2026-02-01T00:00:00Z",
    )
    lock_path = tmp_path / "campaign-lock.json"
    _write_json(lock_path, lock)
    analysis_path = tmp_path / "analysis-plan.txt"
    derivation_path = tmp_path / "opportunity-derivation.py"
    analysis_path.write_text("frozen analysis", encoding="utf-8")
    derivation_path.write_text("# frozen derivation\n", encoding="utf-8")
    manifest = build_trial_freeze_manifest(
        execution_mode=CONFIRMATORY_LIVE,
        protocol_id="O4-V1",
        freeze_timestamp_utc="2026-02-02T00:00:00Z",
        artifact_paths={
            "testbed_qualification": values["qualification"],
            "route_config": values["route"],
            "campaign_registry_lock": lock_path,
            "ledger_schema": values["schema"],
            "model": model_path,
            "trial_config": values["trial"],
            "analysis_plan": analysis_path,
            "opportunity_derivation": derivation_path,
        },
        signing_key=KEY,
    )
    manifest_path = tmp_path / "trial-freeze.json"
    _write_json(manifest_path, manifest.to_dict())
    return {
        **values,
        "telemetry": telemetry,
        "lock": lock_path,
        "freeze_manifest": manifest_path,
        "model_hash": model_hash,
    }


def _verify(values: dict, **changes: object):
    args = {
        "qualification_path": values["qualification"],
        "testbed_config_path": values["testbed"],
        "route_config_path": values["route"],
        "trial_config_path": values["trial"],
        "ledger_schema_path": values["schema"],
        "campaign_lock_path": values["lock"],
        "trial_freeze_manifest_path": values["freeze_manifest"],
        "raw_telemetry_path": values["telemetry"],
        "signing_key": KEY,
        "campaign_id": "GATE-01",
        "acquisition_family_id": "GATE-F01",
        "session_id": "session-1",
        "release_phase": "canary",
        "model_sha256": values["model_hash"],
    }
    args.update(changes)
    return verify_live_action_bindings(**args)


def test_live_bindings_accept_only_exact_authenticated_chain(tmp_path: Path) -> None:
    values = _bindings_fixture(tmp_path)
    result = _verify(values)
    assert result.campaign_role == "gate_evidence"
    assert result.raw_telemetry_sha256 == sha256_file(values["telemetry"])


def test_live_qualification_rejects_imported_or_tampered_probe(tmp_path: Path) -> None:
    values = _qualified_fixture(tmp_path)
    qualification = deepcopy(values["qualification_value"])
    body = {key: value for key, value in qualification.items() if key not in {"artifact_sha256", "authentication"}}
    body["probe_results"][0]["evidence_origin"] = "imported_evidence"
    forged = sign_artifact(body, KEY, purpose="O4_LIVE_TESTBED_QUALIFICATION_V1")
    _write_json(values["qualification"], forged)
    with pytest.raises(ValueError, match="probe_evidence"):
        verify_live_qualification(
            qualification_path=values["qualification"],
            testbed_config_path=values["testbed"],
            route_config_path=values["route"],
            ledger_schema_path=values["schema"],
            signing_key=KEY,
        )


def test_live_bindings_reject_changed_route_wrong_role_and_telemetry(tmp_path: Path) -> None:
    values = _bindings_fixture(tmp_path)
    route = json.loads(values["route"].read_text())
    route["destination"] = "198.51.100.11"
    _write_json(values["route"], route)
    with pytest.raises(RuntimeError, match="qualified frozen file changed"):
        _verify(values)

    values = _bindings_fixture(tmp_path / "role")
    with pytest.raises(RuntimeError, match="requires a locked final_test"):
        _verify(values, release_phase="final_test", final_test_unlock={})

    values = _bindings_fixture(tmp_path / "telemetry")
    values["telemetry"].write_text(values["telemetry"].read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match the locked campaign hash"):
        _verify(values)
