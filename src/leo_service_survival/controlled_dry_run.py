"""Deterministic seven-step Objective 4 software rehearsal.

Every artifact created here is ``FAKE_DRIVER_TEST``.  The rehearsal exercises
the production validation, release, fallback, canary, ledger, and audit code,
but it never converts fake devices or supplied metric fixtures into field
evidence.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .field_preparation import (
    FAKE_DRIVER_TEST,
    ProbeResult,
    build_campaign_registry_lock,
    ingest_telemetry_bundle,
    probe_command_sha256,
    probe_observation_sha256,
    run_read_only_qualification_probes,
    validate_route_driver_configuration,
    validate_testbed_qualification,
    verify_campaign_registry_lock,
)
from .gate_freeze import (
    CONTROLLED_FIXTURE,
    GATE_OPPORTUNITY_COLUMNS,
    assert_gate_evidence_recomputed,
    build_locked_gate_evidence,
    build_trial_freeze_manifest,
)
from .prospective_release import (
    RollbackDrillResult,
    build_end_to_end_audit_report,
    build_fixed_canary_plan,
    build_fixed_final_test_plan,
    evaluate_canary_checkpoint,
    run_frozen_evidence_gate,
    run_prespecified_final_test_analysis,
    run_rejection_branch,
    unlock_untouched_final_test,
    verify_gate_report,
    verify_signed_artifact,
)
from .prospective_trial import (
    ClockSample,
    ClosedLoopController,
    DecisionContext,
    FirstPacketObservation,
    InstallAcknowledgement,
    RouteSnapshot,
    append_hash_chained_record,
    audit_decision_lifecycles,
    load_ledger_schema,
    sha256_file,
)
from .outcome_evidence import load_verified_outcome_evidence


FAKE_SIGNING_KEY = b"objective-4-controlled-fake-driver-key-v1"
MODEL_SHA256 = sha256(b"O4-FAKE-FROZEN-MODEL-V1").hexdigest()
FIXED_WALL_NS = 1_800_000_000_000_000_000


def _claim_boundary() -> str:
    return (
        "FAKE_DRIVER_TEST exercises software paths only. Gate metrics are derived "
        "from hash-locked fake opportunity tables, not field captures. This is not "
        "live hardware, an independent field campaign, a prospective installed "
        "action, an EO outcome, or evidence of candidate benefit. Field execution "
        "remains NOT_RUN."
    )


def _write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class _FakeProbeRunner:
    def run(self, probe_id: str, command: Sequence[str]) -> ProbeResult:
        observed = json.dumps(
            {"command": list(command), "probe_id": probe_id, "result": "pass"},
            sort_keys=True,
        )
        return ProbeResult(
            probe_id=probe_id,
            passed=True,
            observed=observed,
            evidence_sha256=probe_observation_sha256(observed),
            command_sha256=probe_command_sha256(command),
            evidence_origin="fake_driver_test",
        )


class _FakeClock:
    def __init__(self, start_ns: int = FIXED_WALL_NS) -> None:
        self.value = start_ns

    def sample(self) -> ClockSample:
        result = ClockSample(self.value, self.value)
        self.value += 1_000_000
        return result


class _FakeRouteDriver:
    def __init__(self, state: str = "reactive_baseline") -> None:
        self.state = state

    def snapshot(self, destination: str) -> RouteSnapshot:
        return RouteSnapshot(self.state, f"FAKE snapshot {destination}={self.state}")

    def install(self, path_id: str, destination: str) -> InstallAcknowledgement:
        self.state = path_id
        return InstallAcknowledgement(True, f"FAKE installed {path_id}")

    def observe_first_packet(
        self, path_id: str, destination: str, timeout_ms: float
    ) -> FirstPacketObservation:
        return FirstPacketObservation(True, path_id, "FAKE first packet observed")


class _FakeRollbackDrill:
    def run_isolated_rollback_drill(
        self,
        *,
        isolated_namespace_id: str,
        destination: str,
        baseline_path_id: str,
    ) -> RollbackDrillResult:
        return RollbackDrillResult(
            isolated_namespace_id=isolated_namespace_id,
            fault_injected=True,
            rollback_acknowledged=True,
            baseline_verified=True,
            first_packet_observed=True,
            details=(
                f"FAKE isolated rollback {destination} to {baseline_path_id}; "
                "no production route was changed"
            ),
        )


def _fake_testbed() -> dict[str, Any]:
    names = (
        "cellular_5g_terminal",
        "leo_terminal",
        "dual_access_gateway",
        "remote_anchor",
        "clock_sync",
        "session_preserving_path",
    )
    components: dict[str, dict[str, Any]] = {
        name: {
            "asset_or_service_id": f"FAKE-{name.upper()}",
            "status": FAKE_DRIVER_TEST,
            "evidence_sha256": sha256(name.encode()).hexdigest(),
        }
        for name in names
    }
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
    components["clock_sync"].update(
        {"method": "FAKE_DETERMINISTIC", "maximum_error_ms": 0.0}
    )
    components["session_preserving_path"].update(
        {"mechanism": "FAKE_DETERMINISTIC", "preserves_session_identity": True}
    )
    probe_ids = (
        "cellular_link",
        "leo_link",
        "dual_interface",
        "remote_anchor",
        "clock_error",
        "session_continuity",
        "permission_bundle",
    )
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
        "read_only_probes": {name: ["/usr/bin/true"] for name in probe_ids},
        "claim_boundary": _claim_boundary(),
    }


def _fake_route_config() -> dict[str, Any]:
    paths = ("reactive_baseline", "cellular", "leo")
    return {
        "schema_version": "1.0.0",
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
            "namespace_id": "FAKE-ISOLATED-ROLLBACK-NAMESPACE",
            "inject_fault_command": ["/usr/bin/true"],
            "restore_baseline_command": ["/usr/bin/true"],
            "verify_baseline_command": ["/usr/bin/true"],
            "observe_first_packet_command": ["/usr/bin/true"],
        },
        "contract": {
            "snapshot_stdout_first_line": "resolved path",
            "install_exit_zero": "acknowledged",
            "first_packet_exit_zero": "observed",
            "first_packet_stdout_first_line": "resolved packet path",
            "commands_use_shell": False,
            "live_mutation_default": False,
        },
        "claim_boundary": _claim_boundary(),
    }


def _campaign_register(
    config_hash: str,
    qualification_hash: str,
    *,
    gate_opportunity_hashes: Mapping[str, str],
) -> pd.DataFrame:
    definitions = (
        ("FAKE-DEV-01", "FAKE-DEV-F01", "development", "2026-01-01"),
        ("FAKE-GATE-01", "FAKE-GATE-F01", "gate_evidence", "2026-02-01"),
        ("FAKE-GATE-02", "FAKE-GATE-F02", "gate_evidence", "2026-02-03"),
        ("FAKE-GATE-03", "FAKE-GATE-F03", "gate_evidence", "2026-02-05"),
        ("FAKE-GATE-04", "FAKE-GATE-F01", "gate_evidence", "2026-02-07"),
        ("FAKE-GATE-05", "FAKE-GATE-F02", "gate_evidence", "2026-02-09"),
        ("FAKE-TEST-01", "FAKE-TEST-F01", "final_test", "2026-03-01"),
    )
    rows: list[dict[str, Any]] = []
    for index, (campaign, family, role, start) in enumerate(definitions):
        rows.append(
            {
                "study_id": "O4-CONTROLLED-DRY-RUN",
                "campaign_id": campaign,
                "acquisition_family_id": family,
                "role": role,
                "acquisition_pipeline_id": f"FAKE-PIPE-{index:02d}",
                "site_id": f"FAKE-SITE-{index:02d}",
                "operator_id": f"FAKE-OPERATOR-{index:02d}",
                "terminal_id": f"FAKE-TERMINAL-{index:02d}",
                "route_id": f"FAKE-ROUTE-{index:02d}",
                "server_id": f"FAKE-SERVER-{index:02d}",
                "collection_start_utc": f"{start}T00:00:00Z",
                "collection_end_utc": f"{start}T01:00:00Z",
                "protocol_version": "O4-PROSPECTIVE-CLOSED-LOOP-V1",
                "model_sha256": MODEL_SHA256,
                "config_sha256": config_hash,
                "testbed_qualification_sha256": qualification_hash,
                "raw_capture_sha256": sha256(
                    f"raw-telemetry-{campaign}".encode()
                ).hexdigest(),
                "gate_opportunity_sha256": gate_opportunity_hashes.get(
                    campaign, ""
                ),
                "telemetry_manifest_sha256": sha256(f"telemetry-{campaign}".encode()).hexdigest(),
                "outcome_manifest_sha256": sha256(f"outcome-{campaign}".encode()).hexdigest(),
                "clock_sync_method": "FAKE_DETERMINISTIC",
                "maximum_clock_error_ms": 0.0,
                "status": "locked",
                "evidence_tier": "controlled_fake_driver_test",
                "role_access_group": f"FAKE-ACCESS-{role}",
                "notes": "FAKE_DRIVER_TEST only; not a field campaign",
            }
        )
    return pd.DataFrame(rows)


def _telemetry_records() -> list[dict[str, Any]]:
    common = {
        "schema_version": "1.0.0",
        "evidence_status": FAKE_DRIVER_TEST,
        "clock_epoch_id": "FAKE-CLOCK-EPOCH",
        "maximum_clock_error_ms": 0.0,
        "clock_evidence_sha256": sha256(b"fake-clock").hexdigest(),
        "study_id": "O4-CONTROLLED-DRY-RUN",
        "campaign_id": "FAKE-GATE-01",
        "acquisition_family_id": "FAKE-GATE-F01",
        "session_id": "FAKE-SESSION-TELEMETRY",
        "source_id": "FAKE-SYNCHRONIZED-SOURCE",
    }

    def base(record_type: str, sequence: int) -> dict[str, Any]:
        return {
            **common,
            "record_type": record_type,
            "wall_time_ns": FIXED_WALL_NS + sequence * 1_000_000,
            "monotonic_time_ns": sequence * 1_000_000,
            "sequence_number": sequence,
        }

    records: list[dict[str, Any]] = []
    for sequence, disposition, path_id in (
        (1, "reply", "cellular"),
        (2, "loss", "leo"),
        (3, "timeout", "leo"),
    ):
        item = base("network_probe", sequence)
        item.update(
            {
                "probe_id": f"FAKE-PROBE-{sequence}",
                "path_id": path_id,
                "disposition": disposition,
                "sent_wall_time_ns": item["wall_time_ns"],
                "received_wall_time_ns": item["wall_time_ns"] + 20_000_000 if disposition == "reply" else None,
                "rtt_ms": 20.0 if disposition == "reply" else None,
                "failure_reason": "" if disposition == "reply" else f"scheduled_{disposition}",
            }
        )
        records.append(item)
    rf = base("rf_beam_handover", 4)
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
    for sequence, stage in enumerate(
        ("proposed", "install_ack", "post_fib", "first_packet", "final_fib"),
        start=5,
    ):
        route = base("route_observation", sequence)
        route.update(
            {
                "decision_id": "FAKE-TELEMETRY-DECISION",
                "task_id": "FAKE-TELEMETRY-TASK",
                "stage": stage,
                "path_id": "cellular",
                "destination": "198.51.100.10",
                "acknowledged": True,
                "evidence_sha256": sha256(stage.encode()).hexdigest(),
            }
        )
        records.append(route)
    outcome = base("application_eo_outcome", 10)
    outcome.update(
        {
            "decision_id": "FAKE-TELEMETRY-DECISION",
            "task_id": "FAKE-TELEMETRY-TASK",
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


def _write_gate_opportunity_tables(
    directory: Path, *, reject: bool
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Write raw per-campaign fixtures before the campaign register is locked."""

    directory.mkdir(parents=True, exist_ok=True)
    families = (
        "FAKE-GATE-F01",
        "FAKE-GATE-F02",
        "FAKE-GATE-F03",
        "FAKE-GATE-F01",
        "FAKE-GATE-F02",
    )
    entries: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    for campaign_index, family in enumerate(families, start=1):
        campaign = f"FAKE-GATE-{campaign_index:02d}"
        rows = []
        for opportunity_index in range(20):
            rows.append(
                {
                    "campaign_id": campaign,
                    "acquisition_family_id": family,
                    "opportunity_group_id": f"{campaign}-OPP-{opportunity_index:03d}",
                    "cluster_id": f"{campaign}-CLUSTER-{opportunity_index // 2:02d}",
                    "feature_present_count": 10,
                    "feature_required_count": 10,
                    "ood_flag": int(reject),
                    "predicted_success_probability": 1.0,
                    "candidate_success": 1,
                    "baseline_success": 0,
                    "control_loop_ms": 40.0,
                }
            )
        table_path = directory / f"{campaign}.csv"
        pd.DataFrame(rows, columns=GATE_OPPORTUNITY_COLUMNS).to_csv(
            table_path, index=False
        )
        digest = sha256_file(table_path)
        hashes[campaign] = digest
        entries.append(
            {
                "campaign_id": campaign,
                "acquisition_family_id": family,
                "path": str(table_path.resolve()),
                "sha256": digest,
            }
        )
    return entries, hashes


def _freeze_and_derive_gate_evidence(
    *,
    branch_dir: Path,
    package: Path,
    qualification_path: Path,
    route_path: Path,
    config_path: Path,
    analysis_plan_path: Path,
    derivation_path: Path,
    model_path: Path,
    register: pd.DataFrame,
    campaign_lock: Mapping[str, Any],
    opportunity_entries: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Freeze all artifacts and derive gate metrics from locked raw CSVs."""

    register_path = branch_dir / "campaign_register.csv"
    register.to_csv(register_path, index=False)
    lock_path = branch_dir / "campaign_registry_lock.json"
    _write_json(lock_path, campaign_lock)
    opportunity_manifest = {
        "schema_version": "1.0.0",
        "evidence_status": FAKE_DRIVER_TEST,
        "campaign_registry_sha256": campaign_lock["registry_sha256"],
        "tables": list(opportunity_entries),
        "claim_boundary": _claim_boundary(),
    }
    opportunity_manifest_path = branch_dir / "opportunity_manifest.json"
    _write_json(opportunity_manifest_path, opportunity_manifest)
    manifest = build_trial_freeze_manifest(
        execution_mode=CONTROLLED_FIXTURE,
        protocol_id="O4-PROSPECTIVE-CLOSED-LOOP-V1",
        freeze_timestamp_utc="2026-04-01T00:00:00Z",
        artifact_paths={
            "testbed_qualification": qualification_path,
            "route_config": route_path,
            "campaign_registry_lock": lock_path,
            "ledger_schema": package / "prospective_trial" / "schema" / "decision_ledger.schema.json",
            "model": model_path,
            "trial_config": config_path,
            "analysis_plan": analysis_plan_path,
            "opportunity_derivation": derivation_path,
        },
        signing_key=FAKE_SIGNING_KEY,
    )
    freeze_path = branch_dir / "trial_freeze_manifest.json"
    _write_json(freeze_path, manifest.to_dict())
    evidence = build_locked_gate_evidence(
        freeze_manifest=manifest,
        opportunity_manifest_path=opportunity_manifest_path,
        signing_key=FAKE_SIGNING_KEY,
        execution_mode=CONTROLLED_FIXTURE,
    )
    assert_gate_evidence_recomputed(
        evidence,
        freeze_manifest=manifest,
        opportunity_manifest_path=opportunity_manifest_path,
        signing_key=FAKE_SIGNING_KEY,
        execution_mode=CONTROLLED_FIXTURE,
    )
    evidence_path = branch_dir / "gate_evidence.raw_derived.json"
    _write_json(evidence_path, evidence)
    return {
        "evidence": evidence,
        "evidence_path": evidence_path,
        "freeze": manifest.to_dict(),
        "freeze_path": freeze_path,
        "lock_path": lock_path,
        "opportunity_manifest_path": opportunity_manifest_path,
        "register_csv_path": register_path,
    }


def _decision_context(arm: str) -> DecisionContext:
    return DecisionContext(
        destination="198.51.100.10",
        baseline_path_id="reactive_baseline",
        proposed_path_id="cellular",
        gate_state="authorized",
        gate_reason="FAKE controlled-metric-fixture dry-run branch",
        guard_state="pass",
        monitor_state="pass",
        canary_state="candidate" if arm == "gated_candidate" else "baseline",
        collect_ms=1.0,
        feature_ms=1.0,
        decision_ms=1.0,
        disseminate_ms=1.0,
    )


def _append_decision(
    *,
    ledger: Path,
    schema: Mapping[str, Any],
    action_template: Mapping[str, Any],
    outcome_template: Mapping[str, Any],
    config_hash: str,
    testbed_config_hash: str,
    route_config_hash: str,
    ledger_schema_hash: str,
    qualification_hash: str,
    campaign_lock_hash: str,
    gate_hash: str,
    assignment_hash: str,
    block_id: str,
    index: int,
    arm_index: int,
    arm: str,
    success: bool,
) -> None:
    result = ClosedLoopController(
        _FakeRouteDriver(),
        clock=_FakeClock(FIXED_WALL_NS + index * 100_000_000),
    ).execute(_decision_context(arm))
    decision_id = f"FAKE-FINAL-{index:02d}"
    session_id = f"FAKE-{arm.upper()}-SESSION-{arm_index % 2}"
    installed = result.installed_path_id or result.final_fib_path_id
    action = dict(action_template)
    action.update(
        {
            "recorded_wall_ns": result.action_end.wall_ns,
            "recorded_monotonic_ns": result.action_end.monotonic_ns,
            "study_id": "O4-CONTROLLED-DRY-RUN",
            "trial_id": "FAKE-TRIAL-V1",
            "campaign_id": "FAKE-TEST-01",
            "acquisition_family_id": "FAKE-TEST-F01",
            "session_id": session_id,
            "task_id": f"FAKE-TASK-{index:02d}",
            "decision_id": decision_id,
            "model_sha256": MODEL_SHA256,
            "config_sha256": config_hash,
            "testbed_config_sha256": testbed_config_hash,
            "route_config_sha256": route_config_hash,
            "ledger_schema_sha256": ledger_schema_hash,
            "qualification_artifact_sha256": qualification_hash,
            "campaign_lock_sha256": campaign_lock_hash,
            "campaign_role": "final_test",
            "evidence_tier": "controlled_fake_driver_test",
            "block_id": block_id,
            "release_phase": "final_test",
            "gate_report_sha256": gate_hash,
            "assignment_plan_sha256": assignment_hash,
            "policy_arm": arm,
            "gate_state": "authorized",
            "gate_reason": "FAKE_DRIVER_TEST controlled metric fixture",
            "guard_state": "pass",
            "monitor_state": "pass",
            "canary_state": "candidate" if arm == "gated_candidate" else "baseline",
            "available_path_ids": ["reactive_baseline", "cellular", "leo"],
            "proposed_path_id": "cellular",
            "baseline_path_id": "reactive_baseline",
            "intended_path_id": result.intended_path_id,
            "pre_fib_path_id": result.pre_fib_path_id or "reactive_baseline",
            "install_requested": result.pre_fib_path_id != result.intended_path_id,
            "install_acknowledged": result.install_acknowledged,
            "install_ack_evidence_sha256": result.install_ack_evidence_sha256,
            "post_fib_path_id": result.post_fib_path_id or result.intended_path_id,
            "installed_path_id": installed,
            "install_verified": result.install_verified,
            "first_packet_observed": result.first_packet_observed,
            "first_packet_path_id": result.intended_path_id,
            "first_packet_evidence_sha256": result.first_packet_evidence_sha256,
            "final_fib_path_id": result.final_fib_path_id or result.intended_path_id,
            "fallback_triggered": result.fallback_triggered,
            "fallback_reason": result.fallback_reason,
            "rollback_triggered": result.rollback_triggered,
            "rollback_reason": result.rollback_reason,
            "rollback_acknowledged": result.rollback_acknowledged,
            "rollback_install_verified": result.rollback_install_verified,
            "rollback_first_packet_observed": result.rollback_first_packet_observed,
            "rollback_first_packet_path_id": result.rollback_first_packet_path_id or "NONE",
            "rollback_ack_evidence_sha256": result.rollback_ack_evidence_sha256,
            "rollback_first_packet_evidence_sha256": result.rollback_first_packet_evidence_sha256,
            "rollback_verified": result.rollback_verified,
            "telemetry_age_ms": 5.0,
            "collect_ms": 1.0,
            "feature_ms": 1.0,
            "decision_ms": 1.0,
            "disseminate_ms": 1.0,
            "install_ms": result.install_ms,
            "switch_ms": result.switch_ms,
            "total_loop_ms": result.total_loop_ms,
            "rollback_install_ms": result.rollback_install_ms,
            "rollback_switch_ms": result.rollback_switch_ms,
            "rollback_total_ms": result.rollback_total_ms,
            "action_total_ms": result.action_total_ms,
            "raw_telemetry_sha256": sha256(decision_id.encode()).hexdigest(),
        }
    )
    append_hash_chained_record(ledger, action, schema)
    required_bytes = 1000
    delivered_bytes = 1000 if success else 0
    deadline_wall_ns = result.action_end.wall_ns + 50_000_000
    completion_wall_ns = result.action_end.wall_ns + (
        20_000_000 if success else 60_000_000
    )
    fake_acknowledgement = {
        "schema_version": "1.0.0",
        "artifact_type": "application_outcome_acknowledgement",
        "evidence_status": FAKE_DRIVER_TEST,
        "evidence_origin": "fake_driver_test",
        "acknowledgement_id": f"FAKE-ACK-{decision_id}",
        "acknowledgement_source": "controlled_fake_application_observer",
        "acknowledged_wall_ns": completion_wall_ns + 1_000_000,
        "claim_boundary": _claim_boundary(),
        "study_id": "O4-CONTROLLED-DRY-RUN",
        "trial_id": "FAKE-TRIAL-V1",
        "protocol_version": action["protocol_version"],
        "campaign_id": "FAKE-TEST-01",
        "acquisition_family_id": "FAKE-TEST-F01",
        "session_id": session_id,
        "task_id": f"FAKE-TASK-{index:02d}",
        "decision_id": decision_id,
        "model_sha256": MODEL_SHA256,
        "config_sha256": config_hash,
        "required_bytes": required_bytes,
        "delivered_bytes": delivered_bytes,
        "deadline_wall_ns": deadline_wall_ns,
        "completion_wall_ns": completion_wall_ns,
        "observation_valid": True,
        "application_complete": success,
        "no_action": False,
        "controller_failure": False,
        "late_install": False,
        "missingness_reason": "",
    }
    acknowledgement_path = (
        ledger.parent / "outcome_evidence" / f"{decision_id}.json"
    )
    _write_json(acknowledgement_path, fake_acknowledgement)
    verified_outcome = load_verified_outcome_evidence(
        acknowledgement_path,
        action=action,
    )
    outcome = dict(outcome_template)
    outcome.update(
        {
            "recorded_wall_ns": result.action_end.wall_ns + 10_000_000,
            "recorded_monotonic_ns": result.action_end.monotonic_ns + 10_000_000,
            "study_id": "O4-CONTROLLED-DRY-RUN",
            "trial_id": "FAKE-TRIAL-V1",
            "campaign_id": "FAKE-TEST-01",
            "acquisition_family_id": "FAKE-TEST-F01",
            "session_id": session_id,
            "task_id": f"FAKE-TASK-{index:02d}",
            "decision_id": decision_id,
            "model_sha256": MODEL_SHA256,
            "config_sha256": config_hash,
            "evidence_tier": "controlled_fake_driver_test",
            "required_bytes": required_bytes,
            "delivered_bytes": delivered_bytes,
            "deadline_wall_ns": deadline_wall_ns,
            "completion_wall_ns": completion_wall_ns,
            "observation_valid": True,
            "application_complete": success,
            "met_deadline": verified_outcome.met_deadline,
            "no_action": False,
            "controller_failure": False,
            "late_install": False,
            "conservative_on_time_completion": success,
            "failure_stage": "none" if success else "delivery",
            "missingness_reason": "",
            "outcome_evidence_sha256": verified_outcome.evidence_file_sha256,
            "outcome_evidence_status": FAKE_DRIVER_TEST,
            "outcome_evidence_origin": "fake_driver_test",
            "application_acknowledgement_id": fake_acknowledgement[
                "acknowledgement_id"
            ],
            "application_acknowledgement_source": fake_acknowledgement[
                "acknowledgement_source"
            ],
            "application_acknowledgement_wall_ns": fake_acknowledgement[
                "acknowledged_wall_ns"
            ],
            "application_acknowledgement_sha256": verified_outcome.acknowledgement_sha256,
            "application_acknowledgement_authenticated": False,
            "application_acknowledgement_key_id": "",
        }
    )
    append_hash_chained_record(ledger, outcome, schema)


def _file_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "1.0.0",
        "evidence_status": FAKE_DRIVER_TEST,
        "field_trial_status": "NOT_RUN",
        "files": files,
        "claim_boundary": _claim_boundary(),
    }


def run_controlled_seven_step_dry_run(
    output_root: str | Path,
    *,
    package_root: str | Path,
) -> dict[str, Any]:
    """Run both release branches with deterministic fake inputs.

    The target must be absent or empty.  This avoids silently overwriting a
    previous rehearsal and makes byte-for-byte comparison possible.
    """

    root = Path(output_root)
    package = Path(package_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty dry-run directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    outputs = root / "outputs"
    rejected_dir = outputs / "rejected_branch"
    authorized_dir = outputs / "authorized_branch"
    rejected_inputs = inputs / "rejected_fixture"
    authorized_inputs = inputs / "authorized_fixture"
    for directory in (
        inputs,
        rejected_inputs,
        authorized_inputs,
        rejected_dir,
        authorized_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Qualify the driver contract without representing fake devices as live.
    testbed = _fake_testbed()
    route = _fake_route_config()
    testbed_path = inputs / "testbed_config.json"
    route_path = inputs / "route_config.json"
    _write_json(testbed_path, testbed)
    _write_json(route_path, route)
    probes = run_read_only_qualification_probes(testbed, _FakeProbeRunner())
    testbed_report = validate_testbed_qualification(
        testbed, probe_results=probes, allow_fake_driver=True
    )
    route_report = validate_route_driver_configuration(route, allow_fake_driver=True)
    if not testbed_report.passed or not route_report.passed:
        raise RuntimeError("controlled fake qualification failed")
    qualification = {
        "schema_version": "1.0.0",
        "step": 1,
        "evidence_status": FAKE_DRIVER_TEST,
        "field_trial_status": "NOT_RUN",
        "live_eligible": False,
        "passed": True,
        "testbed_evidence_status": FAKE_DRIVER_TEST,
        "route_evidence_status": FAKE_DRIVER_TEST,
        "testbed_live_eligible": False,
        "route_live_eligible": False,
        "testbed_checks": testbed_report.checks.to_dict(orient="records"),
        "route_checks": route_report.checks.to_dict(orient="records"),
        "probe_results": [asdict(item) for item in probes],
        "claim_boundary": _claim_boundary(),
    }
    qualification_path = outputs / "01_qualification_report.json"
    _write_json(qualification_path, qualification)

    # Freeze a rehearsal copy.  The repository's live protocol stays draft.
    config = json.loads(
        (package / "prospective_trial" / "config" / "gate_canary_rollback.json").read_text(
            encoding="utf-8"
        )
    )
    config["protocol_status"] = "FROZEN_REGISTERED"
    config["claim_boundary"] = _claim_boundary()
    config_path = inputs / "trial_config.fake_frozen.json"
    _write_json(config_path, config)
    config_hash = sha256_file(config_path)

    model_path = inputs / "fake_model_artifact.bin"
    model_path.write_bytes(b"O4-FAKE-FROZEN-MODEL-V1")
    if sha256_file(model_path) != MODEL_SHA256:
        raise RuntimeError("fake model fixture hash changed")
    derivation_path = inputs / "opportunity_derivation.fake_frozen.json"
    _write_json(
        derivation_path,
        {
            "schema_version": "1.0.0",
            "bootstrap_seed": 20261103,
            "bootstrap_replicates": 1000,
            "confidence_level": 0.95,
            "cluster_column": "cluster_id",
            "calibration_metric": "absolute_mean_probability_error",
            "evidence_status": FAKE_DRIVER_TEST,
            "claim_boundary": _claim_boundary(),
        },
    )
    analysis_spec = {
        "schema_version": "1.0.0",
        "status": "FROZEN_ANALYSIS_SPEC",
        "estimand": "intention_to_treat_risk_difference_candidate_minus_baseline",
        "endpoint": "conservative_on_time_task_completion",
        "candidate_arm": "gated_candidate",
        "baseline_arm": "reactive_baseline",
        "cluster_field": "block_id",
        "minimum_independent_clusters_per_arm": 10,
        "bootstrap_replicates": 1000,
        "confidence_level": 0.95,
        "bootstrap_seed": 20261103,
        "minimum_practical_gain": 0.02,
        "missing_outcome_rule": "count_as_failure",
        "config_sha256": config_hash,
        "model_sha256": MODEL_SHA256,
        "evidence_status": FAKE_DRIVER_TEST,
        "claim_boundary": _claim_boundary(),
    }
    analysis_spec_path = inputs / "final_test_analysis.fake_frozen.json"
    _write_json(analysis_spec_path, analysis_spec)

    # 2. Each branch freezes its own raw tables before its register is locked.
    branch_packages: dict[str, dict[str, Any]] = {}
    for name, branch_input, reject in (
        ("rejected", rejected_inputs, True),
        ("authorized", authorized_inputs, False),
    ):
        entries, gate_opportunity_hashes = _write_gate_opportunity_tables(
            branch_input / "opportunities", reject=reject
        )
        branch_register = _campaign_register(
            config_hash,
            sha256_file(qualification_path),
            gate_opportunity_hashes=gate_opportunity_hashes,
        )
        branch_lock = build_campaign_registry_lock(
            branch_register,
            protocol_sha256=config_hash,
            qualification_sha256=sha256_file(qualification_path),
            lock_timestamp_utc="2026-04-01T00:00:00Z",
            allow_fake=True,
        )
        verify_campaign_registry_lock(branch_lock).raise_for_errors()
        frozen = _freeze_and_derive_gate_evidence(
            branch_dir=branch_input,
            package=package,
            qualification_path=qualification_path,
            route_path=route_path,
            config_path=config_path,
            analysis_plan_path=analysis_spec_path,
            derivation_path=derivation_path,
            model_path=model_path,
            register=branch_register,
            campaign_lock=branch_lock,
            opportunity_entries=entries,
        )
        branch_packages[name] = {
            "register": branch_register,
            "register_path": frozen["register_csv_path"],
            "lock": branch_lock,
            **frozen,
        }
    register = branch_packages["authorized"]["register"]
    register_path = branch_packages["authorized"]["register_path"]
    campaign_lock = branch_packages["authorized"]["lock"]
    _write_json(outputs / "02_campaign_registry_lock.json", campaign_lock)

    # 3. Preserve reply/loss/timeout, RF/beam, route, and application records.
    telemetry = _telemetry_records()
    (inputs / "telemetry_input.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in telemetry),
        encoding="utf-8",
    )
    telemetry_path = outputs / "03_synchronized_telemetry.jsonl"
    telemetry_report = ingest_telemetry_bundle(
        telemetry,
        telemetry_path,
        expected_probe_ids=["FAKE-PROBE-1", "FAKE-PROBE-2", "FAKE-PROBE-3"],
        campaign_lock=campaign_lock,
        require_complete_action_outcome=True,
    )
    _write_json(
        outputs / "03_telemetry_audit.json",
        {
            "evidence_status": telemetry_report.evidence_status,
            "live_eligible": telemetry_report.live_eligible,
            "checks": telemetry_report.checks.to_dict(orient="records"),
            "claim_boundary": _claim_boundary(),
        },
    )

    # 4/5. Controlled rejected fixture: baseline retention plus rollback drill.
    rejected_evidence = branch_packages["rejected"]["evidence"]
    rejected_evidence_path = branch_packages["rejected"]["evidence_path"]
    rejected_register_path = branch_packages["rejected"]["lock_path"]
    rejected_gate = run_frozen_evidence_gate(
        config_path=config_path,
        register_path=rejected_register_path,
        evidence_path=rejected_evidence_path,
        signing_key=FAKE_SIGNING_KEY,
        created_wall_ns=FIXED_WALL_NS + 100,
        execution_mode=CONTROLLED_FIXTURE,
        freeze_manifest_path=branch_packages["rejected"]["freeze_path"],
        opportunity_manifest_path=branch_packages["rejected"]["opportunity_manifest_path"],
    )
    if rejected_gate["decision"] != "REJECTED":
        raise RuntimeError("rejected dry-run branch did not reject")
    _write_json(rejected_dir / "04_gate_report.json", rejected_gate)
    rejection_result = run_rejection_branch(
        gate_report=rejected_gate,
        signing_key=FAKE_SIGNING_KEY,
        controller=ClosedLoopController(_FakeRouteDriver("cellular"), clock=_FakeClock()),
        context=_decision_context("gated_candidate"),
        drill_driver=_FakeRollbackDrill(),
        isolated_namespace_id="FAKE-ISOLATED-ROLLBACK-NAMESPACE",
        created_wall_ns=FIXED_WALL_NS + 200,
    )
    if rejection_result["status"] != "VERIFIED":
        raise RuntimeError("rejection branch did not verify baseline and rollback")
    _write_json(rejected_dir / "05_baseline_and_rollback_report.json", rejection_result)

    # 4/6. Controlled passing fixture: fixed canary and untouched-test unlock.
    authorized_evidence = branch_packages["authorized"]["evidence"]
    authorized_evidence_path = branch_packages["authorized"]["evidence_path"]
    authorized_gate = run_frozen_evidence_gate(
        config_path=config_path,
        register_path=branch_packages["authorized"]["lock_path"],
        evidence_path=authorized_evidence_path,
        signing_key=FAKE_SIGNING_KEY,
        created_wall_ns=FIXED_WALL_NS + 300,
        execution_mode=CONTROLLED_FIXTURE,
        freeze_manifest_path=branch_packages["authorized"]["freeze_path"],
        opportunity_manifest_path=branch_packages["authorized"]["opportunity_manifest_path"],
    )
    if authorized_gate["decision"] != "AUTHORIZED":
        raise RuntimeError("authorized dry-run branch did not authorize")
    _write_json(authorized_dir / "04_gate_report.json", authorized_gate)
    canary_plan = build_fixed_canary_plan(
        gate_report=authorized_gate,
        signing_key=FAKE_SIGNING_KEY,
        block_ids=[f"FAKE-CANARY-BLOCK-{index:02d}" for index in range(20)],
        allocation_seed="FAKE-FIXED-ALLOCATION-SEED-2026",
        allocation_fraction=0.10,
        created_wall_ns=FIXED_WALL_NS + 400,
    )
    events = [
        {
            "block_id": item["block_id"],
            "executed_arm": item["assigned_arm"],
            "install_acknowledged": True,
            "install_verified": True,
            "installed_path_mismatch": False,
            "baseline_path_id": "reactive_baseline",
            "intended_path_id": (
                "cellular"
                if item["assigned_arm"] == "gated_candidate"
                else "reactive_baseline"
            ),
            "first_packet_observed": True,
            "first_packet_path_id": (
                "cellular"
                if item["assigned_arm"] == "gated_candidate"
                else "reactive_baseline"
            ),
            "first_packet_evidence_sha256": sha256(
                f"first-packet:{item['block_id']}".encode("utf-8")
            ).hexdigest(),
            "rollback_triggered": False,
            "rollback_acknowledged": False,
            "rollback_install_verified": False,
            "rollback_first_packet_observed": False,
            "rollback_first_packet_path_id": "NONE",
            "rollback_ack_evidence_sha256": sha256(b"").hexdigest(),
            "rollback_first_packet_evidence_sha256": sha256(b"").hexdigest(),
            "rollback_verified": False,
            "emergency_stop": False,
            "unauthorized_candidate_action": False,
            "active_shift_alarm": False,
            "endpoint_safety_boundary_crossed": False,
            "clock_or_logging_failure": False,
            "total_loop_ms": 40.0,
            "rollback_total_ms": 0.0,
            "action_total_ms": 40.0,
            "evidence_status": FAKE_DRIVER_TEST,
        }
        for item in canary_plan["assignments"]
    ]
    checkpoint = evaluate_canary_checkpoint(
        gate_report=authorized_gate,
        canary_plan=canary_plan,
        event_records=events,
        config=config,
        signing_key=FAKE_SIGNING_KEY,
        created_wall_ns=FIXED_WALL_NS + 500,
    )
    unlock = unlock_untouched_final_test(
        gate_report=authorized_gate,
        canary_plan=canary_plan,
        checkpoint_report=checkpoint,
        register_path=register_path,
        signing_key=FAKE_SIGNING_KEY,
        checkpoint_approval_id="FAKE-APPROVAL-NOT-VALID-FOR-LIVE-USE",
        no_retuning_attestation=True,
        created_wall_ns=FIXED_WALL_NS + 600,
    )
    _write_json(authorized_dir / "06_canary_plan.json", canary_plan)
    _write_json(authorized_dir / "06_canary_events.json", events)
    _write_json(authorized_dir / "06_canary_checkpoint.json", checkpoint)
    _write_json(authorized_dir / "06_final_test_unlock.json", unlock)
    final_plan = build_fixed_final_test_plan(
        gate_report=authorized_gate,
        final_test_unlock=unlock,
        signing_key=FAKE_SIGNING_KEY,
        analysis_spec_path=analysis_spec_path,
        block_ids=[f"FAKE-FINAL-BLOCK-{index:02d}" for index in range(20)],
        allocation_seed="FAKE-FINAL-TEST-ALLOCATION-SEED-2026",
        candidate_allocation_fraction=0.50,
        created_wall_ns=FIXED_WALL_NS + 650,
    )
    _write_json(authorized_dir / "06_final_test_plan.json", final_plan)

    # 7. Controller outputs feed the signed, tamper-evident outcome chain.
    schema_path = package / "prospective_trial" / "schema" / "decision_ledger.schema.json"
    schema = load_ledger_schema(schema_path)
    action_template = json.loads(
        (package / "prospective_trial" / "schema" / "action_record.template.json").read_text(
            encoding="utf-8"
        )
    )
    outcome_template = json.loads(
        (package / "prospective_trial" / "schema" / "outcome_record.template.json").read_text(
            encoding="utf-8"
        )
    )
    ledger = authorized_dir / "07_action_outcome_ledger.jsonl"
    for index, assignment in enumerate(final_plan["assignments"]):
        arm = assignment["assigned_arm"]
        arm_index = sum(
            1
            for prior in final_plan["assignments"][:index]
            if prior["assigned_arm"] == arm
        )
        _append_decision(
            ledger=ledger,
            schema=schema,
            action_template=action_template,
            outcome_template=outcome_template,
            config_hash=config_hash,
            testbed_config_hash=sha256_file(testbed_path),
            route_config_hash=sha256_file(route_path),
            ledger_schema_hash=sha256_file(schema_path),
            qualification_hash=sha256_file(qualification_path),
            campaign_lock_hash=sha256_file(
                branch_packages["authorized"]["lock_path"]
            ),
            gate_hash=authorized_gate["artifact_sha256"],
            assignment_hash=final_plan["artifact_sha256"],
            block_id=assignment["block_id"],
            index=index,
            arm_index=arm_index,
            arm=arm,
            success=arm_index % 2 == 0,
        )
    audit_decision_lifecycles(ledger, schema, require_mature_outcomes=True).raise_for_errors()
    audit = build_end_to_end_audit_report(
        ledger_path=ledger,
        schema_path=schema_path,
        register_path=register_path,
        gate_report=authorized_gate,
        checkpoint_report=checkpoint,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        signing_key=FAKE_SIGNING_KEY,
        created_wall_ns=FIXED_WALL_NS + 700,
    )
    _write_json(authorized_dir / "07_end_to_end_audit.json", audit)

    analysis = run_prespecified_final_test_analysis(
        ledger_path=ledger,
        schema_path=schema_path,
        register_path=register_path,
        analysis_spec_path=analysis_spec_path,
        gate_report=authorized_gate,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        audit_report=audit,
        signing_key=FAKE_SIGNING_KEY,
        created_wall_ns=FIXED_WALL_NS + 800,
    )
    _write_json(authorized_dir / "07_prespecified_analysis.json", analysis)

    verify_gate_report(rejected_gate, FAKE_SIGNING_KEY).raise_for_errors()
    verify_gate_report(authorized_gate, FAKE_SIGNING_KEY).raise_for_errors()
    for artifact, purpose in (
        (rejection_result, "O4_REJECTION_BRANCH_V1"),
        (canary_plan, "O4_FIXED_CANARY_PLAN_V1"),
        (checkpoint, "O4_CANARY_CHECKPOINT_V1"),
        (unlock, "O4_FINAL_TEST_UNLOCK_V1"),
        (final_plan, "O4_FIXED_FINAL_TEST_PLAN_V1"),
        (audit, "O4_END_TO_END_AUDIT_V1"),
        (analysis, "O4_FINAL_TEST_ITT_ANALYSIS_V1"),
    ):
        verify_signed_artifact(artifact, FAKE_SIGNING_KEY, purpose=purpose).raise_for_errors()
    if analysis.get("benefit_claim_supported") is not False:
        raise RuntimeError("controlled dry run must not support a field benefit claim")

    status = {
        "schema_version": "1.0.0",
        "evidence_status": FAKE_DRIVER_TEST,
        "field_trial_status": "NOT_RUN",
        "software_rehearsal_complete": True,
        "gate_evidence_status": "FAKE_RAW_OPPORTUNITY_DERIVED_AND_HASH_LOCKED",
        "steps": {
            "1_testbed_qualification": "FAKE_DRIVER_TEST_PASS",
            "2_role_separated_campaign_lock": "FAKE_DRIVER_TEST_PASS",
            "3_synchronized_loss_preserving_telemetry": "FAKE_DRIVER_TEST_PASS",
            "4_frozen_gate_rejected_fixture": rejected_gate["decision"],
            "4_frozen_gate_authorized_fixture": authorized_gate["decision"],
            "5_baseline_retention_and_rollback": rejection_result["status"],
            "6_canary_checkpoint": checkpoint["status"],
            "6_untouched_final_test_unlock": unlock["status"],
            "7_trace_to_outcome_audit": f"FAKE_DRIVER_TEST_ONLY:{audit['claim_level']}",
            "7_prespecified_analysis": f"FAKE_DRIVER_TEST_ONLY:{analysis['status']}",
        },
        "analysis_benefit_claim_supported": False,
        "live_hardware_obtained": False,
        "live_campaigns_collected": False,
        "raw_evidence_derived_gate_run": True,
        "raw_evidence_source": "FAKE_DRIVER_TEST_OPPORTUNITY_TABLES",
        "prospective_live_actions_installed": False,
        "claim_boundary": _claim_boundary(),
    }
    _write_json(root / "STATUS.json", status)
    (root / "README.md").write_text(
        "# Objective 4 controlled seven-step dry run\n\n"
        "This directory is generated by `scripts/run_controlled_seven_step_dry_run.py`. "
        "Every record is labelled `FAKE_DRIVER_TEST`. Gate metrics are recomputed "
        "from hash-locked fake opportunity tables; those tables are not field captures. The run "
        "exercises both rejected/fallback and authorized/canary/final-test software "
        "paths. It is not field evidence and does not complete the live experiment.\n\n"
        "See `STATUS.json` for branch outcomes and `manifest.json` for hashes.\n",
        encoding="utf-8",
    )
    _write_json(root / "manifest.json", _file_manifest(root))
    return status
