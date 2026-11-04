from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from leo_service_survival.prospective_release import (
    RollbackDrillResult,
    build_end_to_end_audit_report,
    build_fixed_canary_plan,
    build_fixed_final_test_plan,
    evaluate_canary_checkpoint,
    run_frozen_evidence_gate,
    run_prespecified_final_test_analysis,
    run_rejection_branch,
    resolve_itt_fallback_status,
    unlock_untouched_final_test,
    verify_gate_report,
    verify_signed_artifact,
)
from leo_service_survival.gate_freeze import CONTROLLED_FIXTURE
from leo_service_survival.prospective_trial import (
    ClockSample,
    ClosedLoopController,
    DecisionContext,
    FirstPacketObservation,
    InstallAcknowledgement,
    RouteSnapshot,
    append_hash_chained_record,
    load_ledger_schema,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "prospective_trial"
SCHEMA_PATH = PACKAGE / "schema" / "decision_ledger.schema.json"
KEY = b"objective-4-test-release-key-32-bytes-minimum"
MODEL_HASH = "a" * 64


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _frozen_package(
    tmp_path: Path,
    *,
    gate_campaigns: int = 5,
    metric_changes: dict[str, float] | None = None,
) -> tuple[Path, Path, Path]:
    config = json.loads(
        (PACKAGE / "config" / "gate_canary_rollback.json").read_text(encoding="utf-8")
    )
    config["protocol_status"] = "FROZEN_REGISTERED"
    config_path = tmp_path / "trial.json"
    _write_json(config_path, config)
    config_hash = sha256_file(config_path)

    columns = pd.read_csv(
        PACKAGE / "config" / "acquisition_family_register.csv",
        nrows=0,
    ).columns.tolist()
    rows: list[dict[str, object]] = []

    def row(campaign: str, family: str, role: str, pipeline: str) -> dict[str, object]:
        fingerprint = family.lower()
        value = {
            "study_id": "O4-PROSPECTIVE",
            "campaign_id": campaign,
            "acquisition_family_id": family,
            "role": role,
            "acquisition_pipeline_id": pipeline,
            "site_id": f"site-{fingerprint}",
            "operator_id": f"operator-{fingerprint}",
            "terminal_id": f"terminal-{fingerprint}",
            "route_id": f"route-{fingerprint}",
            "server_id": f"server-{fingerprint}",
            "collection_start_utc": "2026-01-01T00:00:00Z",
            "collection_end_utc": "2026-01-01T01:00:00Z",
            "protocol_version": config["protocol_id"],
            "model_sha256": MODEL_HASH,
            "config_sha256": config_hash,
            "clock_sync_method": "PTP",
            "maximum_clock_error_ms": 1.0,
            "status": "locked",
            "evidence_tier": "field_live_randomized" if role == "final_test" else "field_shadow_concurrent",
            "notes": "test fixture",
        }
        return {column: value[column] for column in columns}

    rows.append(row("DEV-01", "DEV-F01", "development", "PIPE-DEV"))
    family_sequence = ("GATE-F01", "GATE-F02", "GATE-F03")
    for index in range(gate_campaigns):
        family = family_sequence[index % len(family_sequence)]
        rows.append(row(f"GATE-{index + 1:02d}", family, "gate_evidence", f"PIPE-{family}"))
    rows.append(row("TEST-01", "TEST-F01", "final_test", "PIPE-TEST"))
    register = pd.DataFrame(rows, columns=columns)
    register_path = tmp_path / "register.csv"
    register.to_csv(register_path, index=False)

    gate_rows = register.loc[register["role"].eq("gate_evidence")]
    campaign_records = [
        {
            "campaign_id": item.campaign_id,
            "acquisition_family_id": item.acquisition_family_id,
            "paired_opportunity_groups": 2,
            "raw_evidence_sha256": sha256(item.campaign_id.encode("utf-8")).hexdigest(),
        }
        for item in gate_rows.itertuples(index=False)
    ]
    metrics = {
        "paired_opportunity_groups": 2 * gate_campaigns,
        "feature_coverage": 0.99,
        "ood_fraction": 0.05,
        "calibration_gap_upper": 0.03,
        "candidate_minus_baseline_effect_lower": 0.01,
        "effect_ci_half_width": 0.03,
        "control_loop_p95_ms": 50.0,
    }
    metrics.update(metric_changes or {})
    evidence = {
        "schema_version": "1.0.0",
        "evidence_status": "FAKE_DRIVER_TEST",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_hash,
        "model_sha256": MODEL_HASH,
        "campaigns": campaign_records,
        "pooled_metrics": metrics,
    }
    evidence_path = tmp_path / "gate_evidence.json"
    _write_json(evidence_path, evidence)
    return config_path, register_path, evidence_path


def _gate(tmp_path: Path, **package_args: object) -> tuple[dict, Path, Path, Path]:
    config, register, evidence = _frozen_package(tmp_path, **package_args)
    report = run_frozen_evidence_gate(
        config_path=config,
        register_path=register,
        evidence_path=evidence,
        signing_key=KEY,
        created_wall_ns=100,
        execution_mode=CONTROLLED_FIXTURE,
    )
    return report, config, register, evidence


def test_frozen_gate_authorizes_only_hash_bound_complete_evidence(tmp_path: Path) -> None:
    report, _, _, _ = _gate(tmp_path)
    assert report["decision"] == "AUTHORIZED"
    assert report["candidate_authorized"] is True
    assert verify_gate_report(report, KEY).passed
    tampered = json.loads(json.dumps(report))
    tampered["decision"] = "REJECTED"
    assert not verify_gate_report(tampered, KEY).passed


def test_frozen_gate_distinguishes_rejection_and_insufficiency(tmp_path: Path) -> None:
    rejected, _, _, _ = _gate(
        tmp_path / "reject",
        metric_changes={"ood_fraction": 0.20},
    )
    assert rejected["decision"] == "REJECTED"
    assert rejected["failed_conditions"] == ["ood_fraction"]

    insufficient, _, _, _ = _gate(tmp_path / "insufficient", gate_campaigns=4)
    assert insufficient["decision"] == "INSUFFICIENT_EVIDENCE"
    assert "independent_gate_campaigns" in insufficient["failed_conditions"]


def test_gate_refuses_draft_protocol(tmp_path: Path) -> None:
    config, register, evidence = _frozen_package(tmp_path)
    value = json.loads(config.read_text(encoding="utf-8"))
    value["protocol_status"] = "DRAFT_NOT_REGISTERED"
    _write_json(config, value)
    with pytest.raises(ValueError, match="FROZEN_REGISTERED"):
        run_frozen_evidence_gate(
            config_path=config,
            register_path=register,
            evidence_path=evidence,
            signing_key=KEY,
            execution_mode=CONTROLLED_FIXTURE,
        )


class FakeClock:
    def __init__(self) -> None:
        self.value = 1_000_000_000

    def sample(self) -> ClockSample:
        sample = ClockSample(self.value, self.value)
        self.value += 1_000_000
        return sample


class FakeRouteDriver:
    def __init__(self, state: str = "candidate") -> None:
        self.state = state
        self.install_calls: list[str] = []

    def snapshot(self, destination: str) -> RouteSnapshot:
        return RouteSnapshot(self.state, destination)

    def install(self, path_id: str, destination: str) -> InstallAcknowledgement:
        self.install_calls.append(path_id)
        self.state = path_id
        return InstallAcknowledgement(True, f"installed {path_id}")

    def observe_first_packet(
        self, path_id: str, destination: str, timeout_ms: float
    ) -> FirstPacketObservation:
        return FirstPacketObservation(True, path_id, "observed")


class FakeDrill:
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
            details=f"fake drill for {destination} to {baseline_path_id}",
        )


def _context() -> DecisionContext:
    return DecisionContext(
        destination="198.51.100.1",
        baseline_path_id="baseline",
        proposed_path_id="candidate",
        gate_state="not_evaluated",
        gate_reason="fixture",
        guard_state="unavailable",
        monitor_state="unavailable",
        canary_state="shadow",
        collect_ms=1.0,
        feature_ms=1.0,
        decision_ms=1.0,
        disseminate_ms=1.0,
    )


def test_rejected_gate_verifies_production_baseline_and_isolated_rollback(tmp_path: Path) -> None:
    report, _, _, _ = _gate(tmp_path, metric_changes={"ood_fraction": 0.20})
    driver = FakeRouteDriver()
    result = run_rejection_branch(
        gate_report=report,
        signing_key=KEY,
        controller=ClosedLoopController(driver, clock=FakeClock()),
        context=_context(),
        drill_driver=FakeDrill(),
        isolated_namespace_id="ns-o4-drill",
        created_wall_ns=200,
    )
    assert result["status"] == "VERIFIED"
    assert result["candidate_installed_in_production"] is False
    assert driver.install_calls == ["baseline"]
    assert verify_signed_artifact(result, KEY, purpose="O4_REJECTION_BRANCH_V1").passed


def _canary_events(plan: dict, *, fail_candidate: bool = False) -> list[dict]:
    events: list[dict] = []
    failed = False
    for assignment in plan["assignments"]:
        candidate = assignment["assigned_arm"] == "gated_candidate"
        inject = candidate and fail_candidate and not failed
        failed = failed or inject
        intended = "candidate" if candidate else "baseline"
        rollback_total = 5.0 if inject else 0.0
        events.append(
            {
                "block_id": assignment["block_id"],
                "executed_arm": assignment["assigned_arm"],
                "install_acknowledged": not inject,
                "install_verified": not inject,
                "installed_path_mismatch": False,
                "baseline_path_id": "baseline",
                "intended_path_id": intended,
                "first_packet_observed": not inject,
                "first_packet_path_id": intended if not inject else "NONE",
                "first_packet_evidence_sha256": "1" * 64,
                "rollback_triggered": inject,
                "rollback_acknowledged": inject,
                "rollback_install_verified": inject,
                "rollback_first_packet_observed": inject,
                "rollback_first_packet_path_id": "baseline" if inject else "NONE",
                "rollback_ack_evidence_sha256": "2" * 64,
                "rollback_first_packet_evidence_sha256": "3" * 64,
                "rollback_verified": inject,
                "emergency_stop": False,
                "unauthorized_candidate_action": False,
                "active_shift_alarm": False,
                "endpoint_safety_boundary_crossed": False,
                "clock_or_logging_failure": False,
                "total_loop_ms": 40.0,
                "rollback_total_ms": rollback_total,
                "action_total_ms": 40.0 + rollback_total,
            }
        )
    return events


def _release_chain(
    tmp_path: Path,
    *,
    final_block_count: int = 20,
) -> tuple[dict, dict, dict, dict, dict, Path, Path]:
    gate, config_path, register_path, _ = _gate(tmp_path)
    blocks = [f"block-{index:03d}" for index in range(20)]
    plan = build_fixed_canary_plan(
        gate_report=gate,
        signing_key=KEY,
        block_ids=blocks,
        allocation_seed="fixed-allocation-seed-2026",
        allocation_fraction=0.10,
        created_wall_ns=300,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = evaluate_canary_checkpoint(
        gate_report=gate,
        canary_plan=plan,
        event_records=_canary_events(plan),
        config=config,
        signing_key=KEY,
        created_wall_ns=400,
    )
    unlock = unlock_untouched_final_test(
        gate_report=gate,
        canary_plan=plan,
        checkpoint_report=checkpoint,
        register_path=register_path,
        signing_key=KEY,
        checkpoint_approval_id="registered-checkpoint-001",
        no_retuning_attestation=True,
        created_wall_ns=500,
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
        "bootstrap_seed": 12345,
        "minimum_practical_gain": 0.02,
        "missing_outcome_rule": "count_as_failure",
        "config_sha256": sha256_file(config_path),
        "model_sha256": MODEL_HASH,
    }
    _write_json(tmp_path / "analysis.json", analysis_spec)
    final_plan = build_fixed_final_test_plan(
        gate_report=gate,
        final_test_unlock=unlock,
        signing_key=KEY,
        analysis_spec_path=tmp_path / "analysis.json",
        block_ids=[f"final-block-{index:02d}" for index in range(final_block_count)],
        allocation_seed="fixed-final-test-seed-2026",
        candidate_allocation_fraction=0.50,
        created_wall_ns=550,
    )
    return gate, plan, checkpoint, unlock, final_plan, config_path, register_path


def test_canary_is_fixed_stops_on_fault_and_cannot_unlock(tmp_path: Path) -> None:
    gate, config_path, register_path, _ = _gate(tmp_path)
    plan = build_fixed_canary_plan(
        gate_report=gate,
        signing_key=KEY,
        block_ids=[f"b-{index}" for index in range(20)],
        allocation_seed="frozen-seed-123456789",
        allocation_fraction=0.10,
    )
    assert plan["candidate_blocks"] == 2
    stopped = evaluate_canary_checkpoint(
        gate_report=gate,
        canary_plan=plan,
        event_records=_canary_events(plan, fail_candidate=True),
        config=json.loads(config_path.read_text(encoding="utf-8")),
        signing_key=KEY,
    )
    assert stopped["status"] == "STOPPED_ROLLBACK_VERIFIED"
    assert stopped["automatic_rollback_required"] is True
    assert stopped["automatic_rollback_completed"] is True
    with pytest.raises(ValueError, match="remains locked"):
        unlock_untouched_final_test(
            gate_report=gate,
            canary_plan=plan,
            checkpoint_report=stopped,
            register_path=register_path,
            signing_key=KEY,
            checkpoint_approval_id="approval",
            no_retuning_attestation=True,
        )


def test_canary_rejects_omitted_safety_fields_and_marks_unverified_rollback(
    tmp_path: Path,
) -> None:
    gate, config_path, _, _ = _gate(tmp_path)
    plan = build_fixed_canary_plan(
        gate_report=gate,
        signing_key=KEY,
        block_ids=[f"strict-{index}" for index in range(20)],
        allocation_seed="strict-event-contract-seed",
        allocation_fraction=0.10,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    omitted = _canary_events(plan)
    omitted[0].pop("emergency_stop")
    with pytest.raises(ValueError, match="missing=.*emergency_stop"):
        evaluate_canary_checkpoint(
            gate_report=gate,
            canary_plan=plan,
            event_records=omitted,
            config=config,
            signing_key=KEY,
        )

    unverified = _canary_events(plan, fail_candidate=True)
    failed = next(item for item in unverified if not item["install_acknowledged"])
    failed["rollback_triggered"] = False
    failed["rollback_verified"] = False
    failed["rollback_acknowledged"] = False
    failed["rollback_install_verified"] = False
    failed["rollback_first_packet_observed"] = False
    failed["rollback_first_packet_path_id"] = "NONE"
    failed["rollback_total_ms"] = 0.0
    failed["action_total_ms"] = failed["total_loop_ms"]
    checkpoint = evaluate_canary_checkpoint(
        gate_report=gate,
        canary_plan=plan,
        event_records=unverified,
        config=config,
        signing_key=KEY,
    )
    assert checkpoint["status"] == "STOPPED_ROLLBACK_UNVERIFIED"
    assert checkpoint["automatic_rollback_completed"] is False


def test_canary_rejects_claimed_rollback_without_independent_baseline_packet(
    tmp_path: Path,
) -> None:
    gate, config_path, _, _ = _gate(tmp_path)
    plan = build_fixed_canary_plan(
        gate_report=gate,
        signing_key=KEY,
        block_ids=[f"proof-{index}" for index in range(20)],
        allocation_seed="proof-event-contract-seed",
        allocation_fraction=0.10,
    )
    events = _canary_events(plan, fail_candidate=True)
    failed = next(item for item in events if item["rollback_triggered"])
    failed["rollback_first_packet_path_id"] = "candidate"
    with pytest.raises(ValueError, match="first-packet proof"):
        evaluate_canary_checkpoint(
            gate_report=gate,
            canary_plan=plan,
            event_records=events,
            config=json.loads(config_path.read_text(encoding="utf-8")),
            signing_key=KEY,
        )


def test_clean_canary_unlocks_final_test_without_retuning(tmp_path: Path) -> None:
    gate, plan, checkpoint, unlock, final_plan, _, _ = _release_chain(tmp_path)
    assert checkpoint["status"] == "CANARY_CHECKPOINT_PASSED"
    assert unlock["status"] == "UNTOUCHED_FINAL_TEST_UNLOCKED"
    assert unlock["frozen_artifacts"] == gate["frozen_artifacts"]
    assert unlock["no_retuning_attestation"] is True
    assert final_plan["final_test_unlock_sha256"] == unlock["artifact_sha256"]
    assert final_plan["candidate_blocks"] == 10
    assert final_plan["baseline_blocks"] == 10
    assert verify_signed_artifact(unlock, KEY, purpose="O4_FINAL_TEST_UNLOCK_V1").passed


def test_authorized_baseline_assignment_is_not_mislabelled_as_fallback() -> None:
    assert resolve_itt_fallback_status(
        gate_decision="AUTHORIZED",
        assigned_arm="reactive_baseline",
        controller_fallback_triggered=True,
        controller_fallback_reason="candidate_not_authorized:baseline",
    ) == (False, "")
    assert resolve_itt_fallback_status(
        gate_decision="AUTHORIZED",
        assigned_arm="gated_candidate",
        controller_fallback_triggered=True,
        controller_fallback_reason="runtime_safety_failure",
    ) == (True, "runtime_safety_failure")


def _template(name: str) -> dict:
    return json.loads((PACKAGE / "schema" / name).read_text(encoding="utf-8"))


def _append_decision(
    ledger: Path,
    schema: dict,
    *,
    decision_id: str,
    block_id: str,
    arm: str,
    session: str,
    success: bool | None,
    config_hash: str,
    gate_hash: str,
    plan_hash: str,
    model_hash: str = MODEL_HASH,
    release_phase: str = "final_test",
    no_action: bool = False,
    controller_failure: bool = False,
    late_install: bool = False,
) -> None:
    installed = "candidate" if arm == "gated_candidate" else "baseline"
    action = _template("action_record.template.json")
    action.update(
        {
            "recorded_wall_ns": 1000 + len(decision_id),
            "recorded_monotonic_ns": 1000 + len(decision_id),
            "study_id": "O4-PROSPECTIVE",
            "trial_id": "TRIAL-1",
            "campaign_id": "TEST-01",
            "acquisition_family_id": "TEST-F01",
            "session_id": session,
            "task_id": f"task-{decision_id}",
            "decision_id": decision_id,
            "model_sha256": model_hash,
            "config_sha256": config_hash,
            "evidence_tier": "field_live_randomized",
            "block_id": block_id,
            "release_phase": release_phase,
            "gate_report_sha256": gate_hash,
            "assignment_plan_sha256": plan_hash,
            "policy_arm": arm,
            "gate_state": "authorized",
            "gate_reason": "all_frozen_conditions_passed",
            "guard_state": "pass",
            "monitor_state": "pass",
            "canary_state": "candidate" if arm == "gated_candidate" else "baseline",
            "available_path_ids": ["baseline", "candidate"],
            "proposed_path_id": "candidate",
            "baseline_path_id": "baseline",
            "intended_path_id": installed,
            "pre_fib_path_id": installed,
            "install_requested": False,
            "install_acknowledged": True,
            "post_fib_path_id": installed,
            "installed_path_id": installed,
            "install_verified": True,
            "first_packet_observed": True,
            "first_packet_path_id": installed,
            "final_fib_path_id": installed,
            "fallback_triggered": False,
            "fallback_reason": "",
            "rollback_triggered": False,
            "rollback_reason": "",
            "rollback_verified": True,
            "telemetry_age_ms": 5.0,
            "collect_ms": 1.0,
            "feature_ms": 1.0,
            "decision_ms": 1.0,
            "disseminate_ms": 1.0,
            "install_ms": 0.0,
            "switch_ms": 1.0,
            "total_loop_ms": 5.0,
            "action_total_ms": 5.0,
            "raw_telemetry_sha256": sha256(decision_id.encode("utf-8")).hexdigest(),
        }
    )
    append_hash_chained_record(ledger, action, schema)
    if success is None:
        return
    outcome = _template("outcome_record.template.json")
    outcome.update(
        {
            "recorded_wall_ns": 2000 + len(decision_id),
            "recorded_monotonic_ns": 2000 + len(decision_id),
            "study_id": "O4-PROSPECTIVE",
            "trial_id": "TRIAL-1",
            "campaign_id": "TEST-01",
            "acquisition_family_id": "TEST-F01",
            "session_id": session,
            "task_id": f"task-{decision_id}",
            "decision_id": decision_id,
            "model_sha256": model_hash,
            "config_sha256": config_hash,
            "evidence_tier": "field_live_randomized",
            "required_bytes": 100,
            "delivered_bytes": 100 if success else 0,
            "deadline_wall_ns": 5000,
            "completion_wall_ns": 4000 if success else 6000,
            "observation_valid": True,
            "application_complete": success,
            "met_deadline": success,
            "no_action": no_action,
            "controller_failure": controller_failure,
            "late_install": late_install,
            "conservative_on_time_completion": bool(
                success and not no_action and not controller_failure and not late_install
            ),
            "failure_stage": "none" if success else "delivery",
            "missingness_reason": "",
            "outcome_evidence_sha256": "4" * 64,
            "outcome_evidence_status": "LIVE_APPLICATION_ACKNOWLEDGEMENT",
            "outcome_evidence_origin": "executed_live",
            "application_acknowledgement_id": f"ack-{decision_id}",
            "application_acknowledgement_source": "test-application",
            "application_acknowledgement_wall_ns": 7000,
            "application_acknowledgement_sha256": "5" * 64,
            "application_acknowledgement_authenticated": True,
            "application_acknowledgement_key_id": "0123456789abcdef",
        }
    )
    append_hash_chained_record(ledger, outcome, schema)


def _final_ledger(
    tmp_path: Path,
    config_hash: str,
    gate: dict,
    final_plan: dict,
    *,
    include_missing: bool = False,
    tasks_per_block: int = 1,
    flip_first_assignment: bool = False,
    inject_no_action: bool = False,
    model_hash: str = MODEL_HASH,
    release_phase: str = "final_test",
) -> Path:
    ledger = tmp_path / "final.jsonl"
    schema = load_ledger_schema(SCHEMA_PATH)
    arm_index = {"gated_candidate": 0, "reactive_baseline": 0}
    missing_injected = False
    no_action_injected = False
    assignment_flipped = False
    for assignment in final_plan["assignments"]:
        arm = assignment["assigned_arm"]
        index = arm_index[arm]
        arm_index[arm] += 1
        for task_index in range(tasks_per_block):
            success: bool | None = arm == "gated_candidate"
            if include_missing and arm == "reactive_baseline" and not missing_injected:
                success = None
                missing_injected = True
            observed_arm = arm
            if flip_first_assignment and not assignment_flipped:
                observed_arm = (
                    "reactive_baseline"
                    if arm == "gated_candidate"
                    else "gated_candidate"
                )
                assignment_flipped = True
            no_action = bool(
                inject_no_action
                and success is not None
                and not no_action_injected
                and arm == "gated_candidate"
            )
            if no_action:
                no_action_injected = True
                success = False
            _append_decision(
                ledger,
                schema,
                decision_id=f"{assignment['block_id']}-task-{task_index}",
                block_id=assignment["block_id"],
                arm=observed_arm,
                session=f"{arm}-session-{index % 2}",
                success=success,
                config_hash=config_hash,
                gate_hash=gate["artifact_sha256"],
                plan_hash=final_plan["artifact_sha256"],
                model_hash=model_hash,
                release_phase=release_phase,
                no_action=no_action,
            )
    return ledger


def test_end_to_end_audit_reconciles_routes_timing_outcomes_and_chain(tmp_path: Path) -> None:
    gate, _, checkpoint, unlock, final_plan, config_path, register_path = _release_chain(tmp_path)
    ledger = _final_ledger(tmp_path, sha256_file(config_path), gate, final_plan)
    audit = build_end_to_end_audit_report(
        ledger_path=ledger,
        schema_path=SCHEMA_PATH,
        register_path=register_path,
        gate_report=gate,
        checkpoint_report=checkpoint,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        signing_key=KEY,
        created_wall_ns=600,
    )
    assert audit["integrity_passed"] is True
    assert audit["claim_level"] == "FINAL_TEST_DATA_READY_FOR_PRESPECIFIED_ANALYSIS"
    assert audit["counts"]["actions"] == 20
    assert audit["action_reconciliation_discrepancies"] == {
        key: 0 for key in audit["action_reconciliation_discrepancies"]
    }
    assert audit["timing"]["total_loop_ms"]["p95_ms"] == 5.0
    assert audit["benefit_claim_supported"] is False


def test_final_test_itt_analysis_counts_missing_as_failure_and_applies_gain(tmp_path: Path) -> None:
    gate, _, checkpoint, unlock, final_plan, config_path, register_path = _release_chain(tmp_path)
    ledger = _final_ledger(
        tmp_path,
        sha256_file(config_path),
        gate,
        final_plan,
        include_missing=True,
        tasks_per_block=2,
        inject_no_action=True,
    )
    spec_path = tmp_path / "analysis.json"
    audit = build_end_to_end_audit_report(
        ledger_path=ledger,
        schema_path=SCHEMA_PATH,
        register_path=register_path,
        gate_report=gate,
        checkpoint_report=checkpoint,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        signing_key=KEY,
    )
    report = run_prespecified_final_test_analysis(
        ledger_path=ledger,
        schema_path=SCHEMA_PATH,
        register_path=register_path,
        analysis_spec_path=spec_path,
        gate_report=gate,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        audit_report=audit,
        signing_key=KEY,
        created_wall_ns=700,
    )
    assert report["status"] == "ANALYSIS_COMPLETE"
    assert report["missing_outcomes_counted_as_failure"] == 1
    assert report["arm_task_counts"] == {
        "gated_candidate": 20,
        "reactive_baseline": 20,
    }
    assert report["no_action_counted_as_failure"] == 1
    assert report["candidate_completion_risk"] == 0.95
    assert report["baseline_completion_risk"] == 0.0
    assert report["risk_difference_candidate_minus_baseline"] == 0.95
    assert report["risk_difference_interval"]["lower"] > 0.02
    assert report["prespecified_gain_supported"] is True
    assert report["benefit_claim_supported"] is False
    assert report["confirmatory_live_evidence"] is False
    assert verify_signed_artifact(
        report, KEY, purpose="O4_FINAL_TEST_ITT_ANALYSIS_V1"
    ).passed


def test_final_analysis_with_too_few_independent_blocks_cannot_claim_benefit(
    tmp_path: Path,
) -> None:
    gate, _, checkpoint, unlock, final_plan, config_path, register_path = _release_chain(
        tmp_path, final_block_count=18
    )
    ledger = _final_ledger(tmp_path, sha256_file(config_path), gate, final_plan)
    audit = build_end_to_end_audit_report(
        ledger_path=ledger,
        schema_path=SCHEMA_PATH,
        register_path=register_path,
        gate_report=gate,
        checkpoint_report=checkpoint,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        signing_key=KEY,
    )
    report = run_prespecified_final_test_analysis(
        ledger_path=ledger,
        schema_path=SCHEMA_PATH,
        register_path=register_path,
        analysis_spec_path=tmp_path / "analysis.json",
        gate_report=gate,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        audit_report=audit,
        signing_key=KEY,
    )
    assert report["arm_cluster_counts"] == {
        "gated_candidate": 9,
        "reactive_baseline": 9,
    }
    assert report["status"] == "ANALYSIS_INSUFFICIENT"
    assert report["risk_difference_interval"] == {"lower": None, "upper": None}
    assert report["prespecified_gain_supported"] is False
    assert report["benefit_claim_supported"] is False


def test_final_analysis_rejects_wrong_assignment_even_with_multiple_tasks_per_block(
    tmp_path: Path,
) -> None:
    gate, _, checkpoint, unlock, final_plan, config_path, register_path = _release_chain(tmp_path)
    ledger = _final_ledger(
        tmp_path,
        sha256_file(config_path),
        gate,
        final_plan,
        tasks_per_block=2,
        flip_first_assignment=True,
    )
    audit = build_end_to_end_audit_report(
        ledger_path=ledger,
        schema_path=SCHEMA_PATH,
        register_path=register_path,
        gate_report=gate,
        checkpoint_report=checkpoint,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        signing_key=KEY,
    )
    with pytest.raises(ValueError, match="action arm differs from signed ITT assignment"):
        run_prespecified_final_test_analysis(
            ledger_path=ledger,
            schema_path=SCHEMA_PATH,
            register_path=register_path,
            analysis_spec_path=tmp_path / "analysis.json",
            gate_report=gate,
            final_test_unlock=unlock,
            final_test_plan=final_plan,
            audit_report=audit,
            signing_key=KEY,
        )


@pytest.mark.parametrize(
    ("ledger_kwargs", "message"),
    [
        ({"model_hash": "b" * 64}, "model or configuration outside the frozen gate"),
        ({"release_phase": "canary"}, "non-final-test release phases"),
    ],
)
def test_final_analysis_rejects_action_artifact_or_phase_drift(
    tmp_path: Path,
    ledger_kwargs: dict,
    message: str,
) -> None:
    gate, _, checkpoint, unlock, final_plan, config_path, register_path = _release_chain(tmp_path)
    ledger = _final_ledger(
        tmp_path,
        sha256_file(config_path),
        gate,
        final_plan,
        **ledger_kwargs,
    )
    audit = build_end_to_end_audit_report(
        ledger_path=ledger,
        schema_path=SCHEMA_PATH,
        register_path=register_path,
        gate_report=gate,
        checkpoint_report=checkpoint,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        signing_key=KEY,
    )
    with pytest.raises(ValueError, match=message):
        run_prespecified_final_test_analysis(
            ledger_path=ledger,
            schema_path=SCHEMA_PATH,
            register_path=register_path,
            analysis_spec_path=tmp_path / "analysis.json",
            gate_report=gate,
            final_test_unlock=unlock,
            final_test_plan=final_plan,
            audit_report=audit,
            signing_key=KEY,
        )


def test_final_analysis_rejects_tampered_audit_and_post_plan_spec_change(
    tmp_path: Path,
) -> None:
    gate, _, checkpoint, unlock, final_plan, config_path, register_path = _release_chain(tmp_path)
    ledger = _final_ledger(tmp_path, sha256_file(config_path), gate, final_plan)
    audit = build_end_to_end_audit_report(
        ledger_path=ledger,
        schema_path=SCHEMA_PATH,
        register_path=register_path,
        gate_report=gate,
        checkpoint_report=checkpoint,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        signing_key=KEY,
    )
    tampered_audit = json.loads(json.dumps(audit))
    tampered_audit["counts"]["actions"] += 1
    with pytest.raises(ValueError, match="prospective-trial validation failed"):
        run_prespecified_final_test_analysis(
            ledger_path=ledger,
            schema_path=SCHEMA_PATH,
            register_path=register_path,
            analysis_spec_path=tmp_path / "analysis.json",
            gate_report=gate,
            final_test_unlock=unlock,
            final_test_plan=final_plan,
            audit_report=tampered_audit,
            signing_key=KEY,
        )

    spec_path = tmp_path / "analysis.json"
    retuned = json.loads(spec_path.read_text(encoding="utf-8"))
    retuned["minimum_practical_gain"] = -0.50
    _write_json(spec_path, retuned)
    with pytest.raises(ValueError, match="differs from the signed final-test plan"):
        run_prespecified_final_test_analysis(
            ledger_path=ledger,
            schema_path=SCHEMA_PATH,
            register_path=register_path,
            analysis_spec_path=spec_path,
            gate_report=gate,
            final_test_unlock=unlock,
            final_test_plan=final_plan,
            audit_report=audit,
            signing_key=KEY,
        )


def test_final_analysis_refuses_retuned_model_hash(tmp_path: Path) -> None:
    gate, _, checkpoint, unlock, final_plan, config_path, register_path = _release_chain(tmp_path)
    ledger = _final_ledger(tmp_path, sha256_file(config_path), gate, final_plan)
    audit = build_end_to_end_audit_report(
        ledger_path=ledger,
        schema_path=SCHEMA_PATH,
        register_path=register_path,
        gate_report=gate,
        checkpoint_report=checkpoint,
        final_test_unlock=unlock,
        final_test_plan=final_plan,
        signing_key=KEY,
    )
    spec = {
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
        "bootstrap_seed": 1,
        "minimum_practical_gain": 0.02,
        "missing_outcome_rule": "count_as_failure",
        "config_sha256": sha256_file(config_path),
        "model_sha256": "b" * 64,
    }
    spec_path = tmp_path / "retuned.json"
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="different frozen artifacts"):
        run_prespecified_final_test_analysis(
            ledger_path=ledger,
            schema_path=SCHEMA_PATH,
            register_path=register_path,
            analysis_spec_path=spec_path,
            gate_report=gate,
            final_test_unlock=unlock,
            final_test_plan=final_plan,
            audit_report=audit,
            signing_key=KEY,
        )
