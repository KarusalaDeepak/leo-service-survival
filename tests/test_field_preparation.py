from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from leo_service_survival.field_preparation import (
    FAKE_DRIVER_TEST,
    LIVE_REQUIRED,
    ProbeResult,
    append_telemetry_record,
    audit_telemetry_stream,
    build_campaign_registry_lock,
    ingest_telemetry_bundle,
    probe_command_sha256,
    probe_observation_sha256,
    run_read_only_qualification_probes,
    validate_lockable_campaign_registry,
    validate_route_driver_configuration,
    validate_testbed_qualification,
    verify_campaign_registry_lock,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "prospective_trial"
HEX = "a" * 64


class DeterministicFakeProbeRunner:
    def run(self, probe_id: str, command: tuple[str, ...]) -> ProbeResult:
        observed = json.dumps(
            {"probe_id": probe_id, "command": list(command), "result": "pass"},
            sort_keys=True,
        )
        return ProbeResult(
            probe_id=probe_id,
            passed=True,
            observed=observed,
            evidence_sha256=probe_observation_sha256(observed),
            command_sha256=probe_command_sha256(command),
            evidence_origin="fake_execution",
        )


def _fake_testbed() -> dict:
    component_names = (
        "cellular_5g_terminal",
        "leo_terminal",
        "dual_access_gateway",
        "remote_anchor",
        "clock_sync",
        "session_preserving_path",
    )
    components = {
        name: {
            "asset_or_service_id": f"FAKE-{name.upper()}",
            "status": FAKE_DRIVER_TEST,
            "evidence_sha256": HEX,
        }
        for name in component_names
    }
    components["clock_sync"].update(
        {"method": "FAKE_DETERMINISTIC", "maximum_error_ms": 0.0}
    )
    components["session_preserving_path"].update(
        {
            "mechanism": "FAKE_DETERMINISTIC",
            "preserves_session_identity": True,
        }
    )
    components["cellular_5g_terminal"].update(
        {
            "interface": "fake-cellular0",
            "service_active": True,
            "telemetry_export_available": True,
        }
    )
    components["leo_terminal"].update(
        {
            "interface": "fake-leo0",
            "service_active": True,
            "telemetry_export_available": True,
        }
    )
    components["dual_access_gateway"].update(
        {
            "cellular_interface": "fake-cellular0",
            "leo_interface": "fake-leo0",
            "independent_fib_observer": True,
            "emergency_baseline_restore": True,
        }
    )
    components["remote_anchor"].update(
        {"controlled_endpoint": True, "application_acknowledgement": True}
    )
    probes = {
        name: ["/usr/bin/true"]
        for name in (
            "cellular_link",
            "leo_link",
            "dual_interface",
            "remote_anchor",
            "clock_error",
            "session_continuity",
            "permission_bundle",
        )
    }
    return {
        "schema_version": "1.0.0",
        "qualification_status": FAKE_DRIVER_TEST,
        "evidence_source": FAKE_DRIVER_TEST,
        "field_trial_completed": False,
        "maximum_clock_error_ms": 1.0,
        "components": components,
        "permissions": {
            name: {"status": FAKE_DRIVER_TEST}
            for name in (
                "institutional",
                "operator",
                "network_change",
                "ethics_or_exemption",
            )
        },
        "read_only_probes": probes,
    }


def _fake_route() -> dict:
    paths = ("reactive_baseline", "cellular", "leo")
    return {
        "status": FAKE_DRIVER_TEST,
        "evidence_source": FAKE_DRIVER_TEST,
        "destination": "198.51.100.10",
        "baseline_path_id": "reactive_baseline",
        "candidate_path_ids": ["cellular", "leo"],
        "maximum_total_loop_ms": 100.0,
        "first_packet_timeout_ms": 500.0,
        "snapshot_command": ["/usr/bin/true"],
        "install_commands": {path: ["/usr/bin/true"] for path in paths},
        "first_packet_commands": {path: ["/usr/bin/true"] for path in paths},
        "isolated_rollback_drill": {
            "isolation_attested": True,
            "namespace_id": "FAKE-ISOLATED-NAMESPACE",
            "inject_fault_command": ["/usr/bin/true"],
            "restore_baseline_command": ["/usr/bin/true"],
            "verify_baseline_command": ["/usr/bin/true"],
            "observe_first_packet_command": ["/usr/bin/true"],
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


def _fake_registry() -> pd.DataFrame:
    def digest(label: str) -> str:
        return sha256(label.encode("utf-8")).hexdigest()

    rows = []
    definitions = (
        ("C-DEV", "F-DEV", "development", "2026-01-01", "2026-01-02"),
        ("C-GATE-1", "F-GATE-1", "gate_evidence", "2026-02-01", "2026-02-02"),
        ("C-GATE-2", "F-GATE-2", "gate_evidence", "2026-02-03", "2026-02-04"),
        ("C-GATE-3", "F-GATE-3", "gate_evidence", "2026-02-05", "2026-02-06"),
        ("C-GATE-4", "F-GATE-1", "gate_evidence", "2026-02-07", "2026-02-08"),
        ("C-GATE-5", "F-GATE-2", "gate_evidence", "2026-02-09", "2026-02-10"),
        ("C-TEST", "F-TEST", "final_test", "2026-03-01", "2026-03-02"),
    )
    for index, (campaign, family, role, start, end) in enumerate(definitions):
        rows.append(
            {
                "study_id": "O4-FAKE",
                "campaign_id": campaign,
                "acquisition_family_id": family,
                "role": role,
                "acquisition_pipeline_id": f"PIPE-{index}",
                "site_id": f"SITE-{index}",
                "operator_id": f"OP-{index}",
                "terminal_id": f"TERM-{index}",
                "route_id": f"ROUTE-{index}",
                "server_id": f"SERVER-{index}",
                "collection_start_utc": f"{start}T00:00:00Z",
                "collection_end_utc": f"{end}T00:00:00Z",
                "protocol_version": "O4-V1",
                "model_sha256": digest("model"),
                "config_sha256": digest("config"),
                "testbed_qualification_sha256": digest("qualification"),
                "raw_capture_sha256": digest(f"raw-{index}"),
                "gate_opportunity_sha256": (
                    digest(f"gate-opportunity-{index}")
                    if role == "gate_evidence"
                    else ""
                ),
                "telemetry_manifest_sha256": digest(f"telemetry-{index}"),
                "outcome_manifest_sha256": digest(f"outcome-{index}"),
                "clock_sync_method": "FAKE_DETERMINISTIC",
                "maximum_clock_error_ms": 0.0,
                "status": "complete",
                "evidence_tier": "hardware_in_loop",
                "role_access_group": f"ACCESS-{role}",
            }
        )
    return pd.DataFrame(rows)


def _lock() -> dict:
    return build_campaign_registry_lock(
        _fake_registry(),
        protocol_sha256=sha256(b"protocol").hexdigest(),
        qualification_sha256=sha256(b"qualification").hexdigest(),
        lock_timestamp_utc="2026-04-01T00:00:00Z",
        allow_fake=True,
    )


def _common(record_type: str, sequence: int, *, source: str = "FAKE-SOURCE") -> dict:
    return {
        "schema_version": "1.0.0",
        "record_type": record_type,
        "evidence_status": FAKE_DRIVER_TEST,
        "wall_time_ns": 1_800_000_000_000_000_000 + sequence * 1_000_000,
        "monotonic_time_ns": sequence * 1_000_000,
        "clock_epoch_id": "FAKE-CLOCK-EPOCH",
        "maximum_clock_error_ms": 0.0,
        "clock_evidence_sha256": HEX,
        "study_id": "O4-FAKE",
        "campaign_id": "C-GATE-1",
        "acquisition_family_id": "F-GATE-1",
        "session_id": "FAKE-SESSION",
        "source_id": source,
        "sequence_number": sequence,
    }


def _telemetry_records() -> list[dict]:
    records: list[dict] = []
    reply = _common("network_probe", 1)
    reply.update(
        {
            "probe_id": "PROBE-REPLY",
            "path_id": "cellular",
            "disposition": "reply",
            "sent_wall_time_ns": reply["wall_time_ns"],
            "received_wall_time_ns": reply["wall_time_ns"] + 20_000_000,
            "rtt_ms": 20.0,
            "failure_reason": "",
        }
    )
    records.append(reply)
    for sequence, disposition in ((2, "loss"), (3, "timeout")):
        record = _common("network_probe", sequence)
        record.update(
            {
                "probe_id": f"PROBE-{disposition.upper()}",
                "path_id": "leo",
                "disposition": disposition,
                "sent_wall_time_ns": record["wall_time_ns"],
                "received_wall_time_ns": None,
                "rtt_ms": None,
                "failure_reason": f"scheduled_{disposition}",
            }
        )
        records.append(record)

    rf = _common("rf_beam_handover", 4)
    rf.update(
        {
            "path_id": "leo",
            "access_type": "leo",
            "rf_metrics": {"snr_db": 8.5},
            "beam_id": "FAKE-BEAM",
            "satellite_id": "FAKE-SATELLITE",
            "gateway_id": "FAKE-GATEWAY",
            "handover_event": "none",
        }
    )
    records.append(rf)

    stages = ("proposed", "install_ack", "post_fib", "first_packet", "final_fib")
    for offset, stage in enumerate(stages, start=5):
        route = _common("route_observation", offset)
        route.update(
            {
                "decision_id": "FAKE-DECISION",
                "task_id": "FAKE-TASK",
                "stage": stage,
                "path_id": "cellular",
                "destination": "198.51.100.10",
                "acknowledged": True,
                "evidence_sha256": sha256(stage.encode("utf-8")).hexdigest(),
            }
        )
        records.append(route)

    outcome = _common("application_eo_outcome", 10)
    outcome.update(
        {
            "decision_id": "FAKE-DECISION",
            "task_id": "FAKE-TASK",
            "workload_type": "earth_observation",
            "required_bytes": 1000,
            "delivered_bytes": 1000,
            "deadline_wall_time_ns": outcome["wall_time_ns"] + 2_000_000,
            "completion_wall_time_ns": outcome["wall_time_ns"] + 1_000_000,
            "application_complete": True,
            "met_deadline": True,
            "no_action": False,
            "failure_reason": "",
        }
    )
    records.append(outcome)
    return records


def test_unexecuted_testbed_template_stays_live_required() -> None:
    config = json.loads(
        (PACKAGE / "config" / "testbed_qualification.template.json").read_text(
            encoding="utf-8"
        )
    )
    report = validate_testbed_qualification(config)
    assert report.passed, report.checks.to_dict(orient="records")
    assert report.evidence_status == LIVE_REQUIRED
    assert not report.live_eligible
    assert config["field_trial_completed"] is False


def test_deterministic_fake_probe_integration_is_never_live_evidence() -> None:
    config = _fake_testbed()
    results = run_read_only_qualification_probes(
        config, DeterministicFakeProbeRunner()
    )
    report = validate_testbed_qualification(
        config,
        probe_results=results,
        allow_fake_driver=True,
    )
    assert report.passed, report.checks.to_dict(orient="records")
    assert report.evidence_status == FAKE_DRIVER_TEST
    assert not report.live_eligible
    refused = validate_testbed_qualification(config, probe_results=results)
    assert not refused.passed


def test_fake_route_driver_requires_opt_in_and_cannot_be_live_qualified() -> None:
    refused = validate_route_driver_configuration(_fake_route())
    assert not refused.passed
    accepted = validate_route_driver_configuration(
        _fake_route(), allow_fake_driver=True
    )
    assert accepted.passed, accepted.checks.to_dict(orient="records")
    assert accepted.evidence_status == FAKE_DRIVER_TEST
    assert not accepted.live_eligible


def test_route_template_contains_live_rejection_branch_contract() -> None:
    route = json.loads(
        (PACKAGE / "config" / "route_driver.template.json").read_text(
            encoding="utf-8"
        )
    )
    assert route["evidence_source"] == LIVE_REQUIRED
    assert route["maximum_total_loop_ms"] > 0
    assert route["first_packet_timeout_ms"] > 0
    drill = route["isolated_rollback_drill"]
    assert drill["isolation_attested"] is False
    assert {
        "inject_fault_command",
        "restore_baseline_command",
        "verify_baseline_command",
        "observe_first_packet_command",
    }.issubset(drill)


def test_campaign_lock_is_deterministic_and_tamper_evident() -> None:
    frame = _fake_registry()
    report = validate_lockable_campaign_registry(frame, allow_fake=True)
    assert report.passed, report.checks.to_dict(orient="records")
    lock = _lock()
    assert lock["evidence_status"] == FAKE_DRIVER_TEST
    assert lock["live_field_evidence"] is False
    assert verify_campaign_registry_lock(lock).passed

    tampered = json.loads(json.dumps(lock))
    tampered["campaigns"][0]["role"] = "final_test"
    assert not verify_campaign_registry_lock(tampered).passed


def test_campaign_lock_rejects_role_leakage_and_raw_capture_reuse() -> None:
    frame = _fake_registry()
    frame.loc[1, "acquisition_family_id"] = frame.loc[0, "acquisition_family_id"]
    frame.loc[1, "raw_capture_sha256"] = frame.loc[0, "raw_capture_sha256"]
    frame.loc[2, "gate_opportunity_sha256"] = frame.loc[
        1, "gate_opportunity_sha256"
    ]
    report = validate_lockable_campaign_registry(frame, allow_fake=True)
    checks = report.checks.set_index("check")
    assert not bool(checks.loc["registry.family_role_separation", "passed"])
    assert not bool(checks.loc["registry.no_raw_capture_reuse", "passed"])
    assert not bool(checks.loc["registry.no_gate_opportunity_reuse", "passed"])


def test_synchronized_ingestion_preserves_reply_loss_timeout_and_lifecycle(
    tmp_path: Path,
) -> None:
    target = tmp_path / "telemetry.jsonl"
    report = ingest_telemetry_bundle(
        _telemetry_records(),
        target,
        expected_probe_ids=("PROBE-REPLY", "PROBE-LOSS", "PROBE-TIMEOUT"),
        campaign_lock=_lock(),
        require_complete_action_outcome=True,
    )
    assert report.passed, report.checks.to_dict(orient="records")
    assert report.evidence_status == FAKE_DRIVER_TEST
    assert not report.live_eligible
    dispositions = {
        json.loads(line).get("disposition")
        for line in target.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["record_type"] == "network_probe"
    }
    assert dispositions == {"reply", "loss", "timeout"}


def test_missing_scheduled_timeout_and_hash_tampering_are_detected(
    tmp_path: Path,
) -> None:
    target = tmp_path / "telemetry.jsonl"
    records = [
        record
        for record in _telemetry_records()
        if record.get("probe_id") != "PROBE-TIMEOUT"
    ]
    for record in records:
        append_telemetry_record(target, record)
    incomplete = audit_telemetry_stream(
        target,
        expected_probe_ids=("PROBE-REPLY", "PROBE-LOSS", "PROBE-TIMEOUT"),
    )
    assert not incomplete.passed
    assert not bool(
        incomplete.checks.set_index("check").loc[
            "telemetry.loss_preserving_probe_schedule", "passed"
        ]
    )

    lines = target.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["disposition"] = "timeout"
    lines[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not audit_telemetry_stream(target).passed
