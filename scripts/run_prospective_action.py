#!/usr/bin/env python3
"""Execute and record one prospective route action on configured hardware.

The script refuses template configurations and disables route mutation unless
``--execute-live-route-change`` is supplied.  It must be run only after the
protocol, acquisition register, route commands, and approvals are frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_trial import (  # noqa: E402
    ClosedLoopController,
    CommandRouteDriver,
    DecisionContext,
    append_hash_chained_record,
    load_ledger_schema,
    sha256_file,
)
from leo_service_survival.live_actuation import (  # noqa: E402
    verify_live_action_bindings,
)
from leo_service_survival.prospective_release import (  # noqa: E402
    load_signing_key,
    resolve_itt_fallback_status,
    verify_gate_report,
    verify_signed_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--testbed-config", type=Path, required=True)
    parser.add_argument("--qualification-artifact", type=Path, required=True)
    parser.add_argument("--campaign-lock", type=Path, required=True)
    parser.add_argument("--trial-freeze-manifest", type=Path, required=True)
    parser.add_argument("--trial-config", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--acquisition-family-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--raw-telemetry-sha256", required=True)
    parser.add_argument("--raw-telemetry-file", type=Path, required=True)
    parser.add_argument("--evidence-tier", choices=("hardware_in_loop", "field_shadow_concurrent", "field_live_randomized"), required=True)
    parser.add_argument("--proposed-path-id", required=True)
    parser.add_argument("--available-path-id", action="append", required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--assignment-plan", type=Path)
    parser.add_argument("--final-test-unlock", type=Path)
    parser.add_argument("--release-phase", choices=("rejection_verification", "canary", "final_test"), required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--guard-state", choices=("pass", "fallback", "unavailable"), required=True)
    parser.add_argument("--monitor-state", choices=("pass", "alarm", "not_mature", "unavailable"), required=True)
    parser.add_argument("--telemetry-age-ms", type=float, required=True)
    parser.add_argument("--collect-ms", type=float, required=True)
    parser.add_argument("--feature-ms", type=float, required=True)
    parser.add_argument("--decision-ms", type=float, required=True)
    parser.add_argument("--disseminate-ms", type=float, required=True)
    parser.add_argument("--execute-live-route-change", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route = json.loads(args.route_config.read_text(encoding="utf-8"))
    trial = json.loads(args.trial_config.read_text(encoding="utf-8"))
    if route.get("status") != "FROZEN_DEPLOYMENT_CONFIG":
        raise RuntimeError("route configuration is not frozen for deployment")
    if trial.get("protocol_status") != "FROZEN_REGISTERED":
        raise RuntimeError("trial protocol is not frozen and registered")
    if not args.execute_live_route_change:
        raise RuntimeError(
            "live mutation is disabled; use the validated tests for dry-run failure injection"
        )

    signing_key = load_signing_key(args.signing_key)
    gate_report = json.loads(args.gate_report.read_text(encoding="utf-8"))
    verify_gate_report(gate_report, signing_key).raise_for_errors()
    if gate_report["frozen_artifacts"]["config_sha256"] != sha256_file(args.trial_config):
        raise RuntimeError("signed gate report is bound to a different trial configuration")
    if gate_report["frozen_artifacts"]["model_sha256"] != args.model_sha256:
        raise RuntimeError("signed gate report is bound to a different model")
    freeze_manifest = json.loads(
        args.trial_freeze_manifest.read_text(encoding="utf-8")
    )
    if gate_report.get("trial_freeze_manifest_sha256") != freeze_manifest.get(
        "manifest_sha256"
    ):
        raise RuntimeError(
            "signed gate report is not bound to the supplied trial freeze manifest"
        )

    decision_to_state = {
        "AUTHORIZED": "authorized",
        "REJECTED": "rejected",
        "INSUFFICIENT_EVIDENCE": "insufficient_evidence",
    }
    gate_state = decision_to_state[gate_report["decision"]]
    gate_reason = (
        "all_frozen_conditions_passed"
        if gate_state == "authorized"
        else ",".join(gate_report.get("failed_conditions", ())) or gate_state
    )
    canary_state = "baseline"
    assigned_arm = "reactive_baseline"
    assignment_plan_sha256 = "0" * 64
    unlock = None
    if gate_state == "authorized":
        if args.assignment_plan is None:
            raise RuntimeError("authorized action requires a signed phase-specific assignment plan")
        assignment_plan = json.loads(args.assignment_plan.read_text(encoding="utf-8"))
        purpose = (
            "O4_FIXED_CANARY_PLAN_V1"
            if args.release_phase == "canary"
            else "O4_FIXED_FINAL_TEST_PLAN_V1"
            if args.release_phase == "final_test"
            else None
        )
        if purpose is None:
            raise RuntimeError("authorized gate cannot execute the rejection-verification phase")
        verify_signed_artifact(
            assignment_plan, signing_key, purpose=purpose
        ).raise_for_errors()
        if assignment_plan.get("gate_report_sha256") != gate_report.get("artifact_sha256"):
            raise RuntimeError("assignment plan is not bound to the supplied gate report")
        if args.release_phase == "final_test":
            if args.final_test_unlock is None:
                raise RuntimeError("final-test action requires a signed unlock token")
            unlock = json.loads(args.final_test_unlock.read_text(encoding="utf-8"))
            verify_signed_artifact(
                unlock, signing_key, purpose="O4_FINAL_TEST_UNLOCK_V1"
            ).raise_for_errors()
            if assignment_plan.get("final_test_unlock_sha256") != unlock.get("artifact_sha256"):
                raise RuntimeError("final-test assignment plan is not bound to the unlock token")
        elif args.final_test_unlock is not None:
            raise RuntimeError("final-test unlock must not be supplied for a canary action")
        matches = [
            item
            for item in assignment_plan.get("assignments", ())
            if item.get("block_id") == args.block_id
        ]
        if len(matches) != 1:
            raise RuntimeError("block id must have exactly one signed phase assignment")
        assigned_arm = str(matches[0].get("assigned_arm"))
        assignment_plan_sha256 = str(assignment_plan["artifact_sha256"])
        canary_state = (
            "candidate"
            if assigned_arm == "gated_candidate"
            else "baseline"
        )
    else:
        if args.release_phase != "rejection_verification":
            raise RuntimeError("rejected or insufficient gate permits only rejection verification")
        if args.assignment_plan is not None or args.final_test_unlock is not None:
            raise RuntimeError("signed assignment cannot override a rejected or insufficient gate")

    bindings = verify_live_action_bindings(
        qualification_path=args.qualification_artifact,
        testbed_config_path=args.testbed_config,
        route_config_path=args.route_config,
        trial_config_path=args.trial_config,
        ledger_schema_path=args.schema,
        campaign_lock_path=args.campaign_lock,
        trial_freeze_manifest_path=args.trial_freeze_manifest,
        raw_telemetry_path=args.raw_telemetry_file,
        signing_key=signing_key,
        campaign_id=args.campaign_id,
        acquisition_family_id=args.acquisition_family_id,
        session_id=args.session_id,
        release_phase=args.release_phase,
        model_sha256=args.model_sha256,
        final_test_unlock=unlock,
    )
    if args.raw_telemetry_sha256 != bindings.raw_telemetry_sha256:
        raise RuntimeError(
            "declared raw telemetry hash does not match the supplied file"
        )
    frozen = gate_report["frozen_artifacts"]
    frozen_binding_errors = {
        "route_config_sha256": (frozen.get("route_config_sha256"), bindings.route_config_sha256),
        "ledger_schema_sha256": (frozen.get("ledger_schema_sha256"), bindings.ledger_schema_sha256),
        "testbed_qualification_sha256": (
            frozen.get("testbed_qualification_sha256"),
            bindings.qualification_file_sha256,
        ),
        "campaign_registry_lock_sha256": (
            frozen.get("campaign_registry_lock_sha256"),
            bindings.campaign_lock_sha256,
        ),
    }
    frozen_binding_errors = {
        key: value for key, value in frozen_binding_errors.items() if value[0] != value[1]
    }
    if frozen_binding_errors:
        raise RuntimeError(
            f"signed gate report frozen artifacts do not match live inputs: {frozen_binding_errors}"
        )

    rollback = trial["rollback"]
    driver = CommandRouteDriver(
        snapshot_command=route["snapshot_command"],
        install_commands=route["install_commands"],
        first_packet_commands=route["first_packet_commands"],
        allow_live_mutation=True,
    )
    controller = ClosedLoopController(
        driver,
        maximum_total_loop_ms=rollback["maximum_single_decision_loop_ms"],
        first_packet_timeout_ms=rollback["first_packet_timeout_ms"],
    )
    context = DecisionContext(
        destination=route["destination"],
        baseline_path_id=route["baseline_path_id"],
        proposed_path_id=args.proposed_path_id,
        gate_state=gate_state,
        gate_reason=gate_reason,
        guard_state=args.guard_state,
        monitor_state=args.monitor_state,
        canary_state=canary_state,
        collect_ms=args.collect_ms,
        feature_ms=args.feature_ms,
        decision_ms=args.decision_ms,
        disseminate_ms=args.disseminate_ms,
    )
    result = controller.execute(context)
    fallback_triggered, fallback_reason = resolve_itt_fallback_status(
        gate_decision=gate_report["decision"],
        assigned_arm=assigned_arm,
        controller_fallback_triggered=result.fallback_triggered,
        controller_fallback_reason=result.fallback_reason,
    )
    installed = result.installed_path_id or "NONE"
    post_fib = result.post_fib_path_id or "UNAVAILABLE"
    first_packet_path = result.first_packet_path_id or "NONE"
    record = {
        "schema_version": "1.0.0",
        "record_type": "action",
        "recorded_wall_ns": result.action_end.wall_ns,
        "recorded_monotonic_ns": result.action_end.monotonic_ns,
        "study_id": args.study_id,
        "trial_id": args.trial_id,
        "protocol_version": trial["protocol_id"],
        "campaign_id": args.campaign_id,
        "acquisition_family_id": args.acquisition_family_id,
        "session_id": args.session_id,
        "task_id": args.task_id,
        "decision_id": args.decision_id,
        "model_sha256": args.model_sha256,
        "config_sha256": sha256_file(args.trial_config),
        "testbed_config_sha256": bindings.testbed_config_sha256,
        "route_config_sha256": bindings.route_config_sha256,
        "ledger_schema_sha256": bindings.ledger_schema_sha256,
        "qualification_artifact_sha256": bindings.qualification_artifact_sha256,
        "qualification_file_sha256": bindings.qualification_file_sha256,
        "campaign_lock_sha256": bindings.campaign_lock_sha256,
        "trial_freeze_manifest_sha256": bindings.trial_freeze_manifest_sha256,
        "campaign_role": bindings.campaign_role,
        "evidence_tier": args.evidence_tier,
        "block_id": args.block_id,
        "release_phase": args.release_phase,
        "gate_report_sha256": gate_report["artifact_sha256"],
        "assignment_plan_sha256": assignment_plan_sha256,
        "policy_arm": assigned_arm,
        "gate_state": gate_state,
        "gate_reason": gate_reason,
        "guard_state": args.guard_state,
        "monitor_state": args.monitor_state,
        "canary_state": canary_state,
        "available_path_ids": args.available_path_id,
        "proposed_path_id": args.proposed_path_id,
        "baseline_path_id": route["baseline_path_id"],
        "intended_path_id": result.intended_path_id,
        "pre_fib_path_id": result.pre_fib_path_id or "UNAVAILABLE",
        "install_requested": result.pre_fib_path_id != result.intended_path_id,
        "install_acknowledged": result.install_acknowledged,
        "install_ack_evidence_sha256": result.install_ack_evidence_sha256,
        "post_fib_path_id": post_fib,
        "installed_path_id": installed,
        "install_verified": result.install_verified,
        "first_packet_observed": result.first_packet_observed,
        "first_packet_path_id": first_packet_path,
        "first_packet_evidence_sha256": result.first_packet_evidence_sha256,
        "final_fib_path_id": result.final_fib_path_id or "UNAVAILABLE",
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "rollback_triggered": result.rollback_triggered,
        "rollback_reason": result.rollback_reason,
        "rollback_acknowledged": result.rollback_acknowledged,
        "rollback_install_verified": result.rollback_install_verified,
        "rollback_first_packet_observed": result.rollback_first_packet_observed,
        "rollback_first_packet_path_id": result.rollback_first_packet_path_id or "NONE",
        "rollback_ack_evidence_sha256": result.rollback_ack_evidence_sha256,
        "rollback_first_packet_evidence_sha256": result.rollback_first_packet_evidence_sha256,
        "rollback_verified": result.rollback_verified,
        "telemetry_age_ms": args.telemetry_age_ms,
        "collect_ms": args.collect_ms,
        "feature_ms": args.feature_ms,
        "decision_ms": args.decision_ms,
        "disseminate_ms": args.disseminate_ms,
        "install_ms": result.install_ms,
        "switch_ms": result.switch_ms,
        "total_loop_ms": result.total_loop_ms,
        "rollback_install_ms": result.rollback_install_ms,
        "rollback_switch_ms": result.rollback_switch_ms,
        "rollback_total_ms": result.rollback_total_ms,
        "action_total_ms": result.action_total_ms,
        "raw_telemetry_sha256": bindings.raw_telemetry_sha256,
    }
    appended = append_hash_chained_record(
        args.ledger, record, load_ledger_schema(args.schema)
    )
    print(json.dumps({"controller_result": asdict(result), "ledger_record": appended}, default=lambda value: asdict(value), indent=2))
    return 0 if not result.rollback_triggered else 2


if __name__ == "__main__":
    raise SystemExit(main())
