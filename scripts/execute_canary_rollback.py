#!/usr/bin/env python3
"""Execute and verify baseline rollback after a signed stopped canary checkpoint."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.live_actuation import verify_live_action_bindings  # noqa: E402
from leo_service_survival.prospective_release import (  # noqa: E402
    load_signing_key,
    sign_artifact,
    verify_gate_report,
    verify_signed_artifact,
)
from leo_service_survival.prospective_trial import CommandRouteDriver  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-report", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--canary-plan", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--qualification-artifact", type=Path, required=True)
    parser.add_argument("--testbed-config", type=Path, required=True)
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--trial-config", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--campaign-lock", type=Path, required=True)
    parser.add_argument("--trial-freeze-manifest", type=Path, required=True)
    parser.add_argument("--raw-telemetry-file", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--acquisition-family-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-live-route-change", action="store_true")
    return parser.parse_args()


def _hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    if not args.execute_live_route_change:
        raise RuntimeError(
            "live rollback is disabled; explicitly supply --execute-live-route-change"
        )
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite rollback evidence: {args.output}")

    key = load_signing_key(args.signing_key)
    checkpoint = json.loads(args.checkpoint_report.read_text(encoding="utf-8"))
    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    plan = json.loads(args.canary_plan.read_text(encoding="utf-8"))
    verify_signed_artifact(
        checkpoint, key, purpose="O4_CANARY_CHECKPOINT_V1"
    ).raise_for_errors()
    verify_gate_report(gate, key).raise_for_errors()
    verify_signed_artifact(
        plan, key, purpose="O4_FIXED_CANARY_PLAN_V1"
    ).raise_for_errors()
    if checkpoint.get("status") not in {
        "STOPPED_ROLLBACK_VERIFIED",
        "STOPPED_ROLLBACK_UNVERIFIED",
    }:
        raise RuntimeError("rollback execution requires a signed stopped checkpoint")
    if checkpoint.get("automatic_rollback_required") is not True:
        raise RuntimeError("checkpoint does not mandate rollback")
    if not (
        checkpoint.get("gate_report_sha256") == gate.get("artifact_sha256")
        and checkpoint.get("canary_plan_sha256") == plan.get("artifact_sha256")
        and plan.get("gate_report_sha256") == gate.get("artifact_sha256")
    ):
        raise RuntimeError("checkpoint, canary plan, and gate report are not one chain")
    freeze_manifest = json.loads(
        args.trial_freeze_manifest.read_text(encoding="utf-8")
    )
    if gate.get("trial_freeze_manifest_sha256") != freeze_manifest.get(
        "manifest_sha256"
    ):
        raise RuntimeError("gate report is not bound to the supplied trial freeze")
    if not (
        checkpoint.get("frozen_artifacts")
        == plan.get("frozen_artifacts")
        == gate.get("frozen_artifacts")
    ):
        raise RuntimeError("frozen artifacts changed before rollback")

    bindings = verify_live_action_bindings(
        qualification_path=args.qualification_artifact,
        testbed_config_path=args.testbed_config,
        route_config_path=args.route_config,
        trial_config_path=args.trial_config,
        ledger_schema_path=args.schema,
        campaign_lock_path=args.campaign_lock,
        trial_freeze_manifest_path=args.trial_freeze_manifest,
        raw_telemetry_path=args.raw_telemetry_file,
        signing_key=key,
        campaign_id=args.campaign_id,
        acquisition_family_id=args.acquisition_family_id,
        session_id=args.session_id,
        release_phase="canary",
        model_sha256=args.model_sha256,
    )

    route = json.loads(args.route_config.read_text(encoding="utf-8"))
    driver = CommandRouteDriver(
        snapshot_command=route["snapshot_command"],
        install_commands=route["install_commands"],
        first_packet_commands=route["first_packet_commands"],
        allow_live_mutation=True,
    )
    baseline = str(route["baseline_path_id"])
    destination = str(route["destination"])
    pre = driver.snapshot(destination)
    start_ns = time.monotonic_ns()
    acknowledgement = driver.install(baseline, destination)
    install_end_ns = time.monotonic_ns()
    post = driver.snapshot(destination)
    install_verified = bool(
        acknowledgement.acknowledged and post.path_id == baseline
    )
    packet_start_ns = time.monotonic_ns()
    packet = (
        driver.observe_first_packet(
            baseline,
            destination,
            float(
                json.loads(args.trial_config.read_text(encoding="utf-8"))[
                    "rollback"
                ]["first_packet_timeout_ms"]
            ),
        )
        if install_verified
        else None
    )
    end_ns = time.monotonic_ns()
    first_packet_verified = bool(
        packet is not None and packet.observed and packet.path_id == baseline
    )
    verified = bool(install_verified and first_packet_verified)
    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "canary_rollback_execution",
        "created_wall_ns": time.time_ns(),
        "checkpoint_report_sha256": checkpoint["artifact_sha256"],
        "gate_report_sha256": gate["artifact_sha256"],
        "canary_plan_sha256": plan["artifact_sha256"],
        "qualification_artifact_sha256": bindings.qualification_artifact_sha256,
        "route_config_sha256": bindings.route_config_sha256,
        "ledger_schema_sha256": bindings.ledger_schema_sha256,
        "campaign_lock_sha256": bindings.campaign_lock_sha256,
        "trial_freeze_manifest_sha256": bindings.trial_freeze_manifest_sha256,
        "raw_telemetry_sha256": bindings.raw_telemetry_sha256,
        "campaign_id": args.campaign_id,
        "acquisition_family_id": args.acquisition_family_id,
        "session_id": args.session_id,
        "destination": destination,
        "baseline_path_id": baseline,
        "pre_fib_path_id": pre.path_id,
        "rollback_install_acknowledged": acknowledgement.acknowledged,
        "rollback_ack_evidence_sha256": _hash_text(acknowledgement.raw),
        "post_fib_path_id": post.path_id,
        "rollback_install_verified": install_verified,
        "rollback_first_packet_observed": first_packet_verified,
        "rollback_first_packet_path_id": packet.path_id if packet else None,
        "rollback_first_packet_evidence_sha256": _hash_text(
            packet.raw if packet else "not_observed"
        ),
        "rollback_install_ms": (install_end_ns - start_ns) / 1_000_000.0,
        "rollback_switch_ms": (end_ns - packet_start_ns) / 1_000_000.0,
        "rollback_total_ms": (end_ns - start_ns) / 1_000_000.0,
        "rollback_verified": verified,
        "status": "BASELINE_ROLLBACK_VERIFIED" if verified else "BASELINE_ROLLBACK_FAILED",
        "claim_boundary": (
            "This artifact proves baseline rollback execution for the named stopped "
            "canary checkpoint; it does not establish candidate benefit."
        ),
    }
    signed = sign_artifact(
        payload, key, purpose="O4_CANARY_ROLLBACK_EXECUTION_V1"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(signed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": signed["status"],
                "artifact_sha256": signed["artifact_sha256"],
                "rollback_total_ms": signed["rollback_total_ms"],
            },
            indent=2,
        )
    )
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
