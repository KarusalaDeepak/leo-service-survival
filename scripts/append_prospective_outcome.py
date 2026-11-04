#!/usr/bin/env python3
"""Append a mature outcome linked to one prospective action record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_trial import (  # noqa: E402
    append_hash_chained_record,
    audit_decision_lifecycles,
    audit_hash_chain,
    load_ledger_schema,
)
from leo_service_survival.outcome_evidence import (  # noqa: E402
    derive_failure_stage,
    load_verified_outcome_evidence,
)
from leo_service_survival.prospective_release import load_signing_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument(
        "--outcome-evidence",
        type=Path,
        required=True,
        help=(
            "Raw application acknowledgement JSON. Live acknowledgements must "
            "be authenticated; controlled fake-driver acknowledgements must be "
            "explicitly labelled FAKE_DRIVER_TEST."
        ),
    )
    parser.add_argument(
        "--signing-key",
        type=Path,
        help="Operator HMAC key required for live acknowledgement verification.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = load_ledger_schema(args.schema)
    audit_hash_chain(args.ledger, schema).raise_for_errors()
    records = [json.loads(line) for line in args.ledger.read_text(encoding="utf-8").splitlines()]
    action = [
        row
        for row in records
        if row.get("record_type") == "action"
        and row.get("decision_id") == args.decision_id
    ]
    existing_outcome = [
        row
        for row in records
        if row.get("record_type") == "outcome"
        and row.get("decision_id") == args.decision_id
    ]
    if len(action) != 1:
        raise ValueError(f"expected exactly one linked action; found {len(action)}")
    if existing_outcome:
        raise ValueError("an outcome already exists for this decision")
    action = action[0]

    signing_key = load_signing_key(args.signing_key) if args.signing_key else None
    evidence = load_verified_outcome_evidence(
        args.outcome_evidence,
        action=action,
        signing_key=signing_key,
    )
    observed = evidence.payload
    met_deadline = evidence.met_deadline

    success = bool(
        action["install_verified"]
        and action["first_packet_observed"]
        and not action["rollback_triggered"]
        and met_deadline
        and not observed["no_action"]
        and not observed["controller_failure"]
        and not observed["late_install"]
    )
    failure_stage = derive_failure_stage(action=action, evidence=evidence)
    if success != (failure_stage == "none"):
        raise ValueError(
            "derived failure stage contradicts conservative completion; "
            "the acknowledgement or action is internally inconsistent"
        )
    recorded_wall_ns = time.time_ns()
    if recorded_wall_ns < observed["acknowledged_wall_ns"]:
        raise ValueError(
            "application acknowledgement time is in the future relative to the ledger logger"
        )

    common = {
        key: action[key]
        for key in (
            "schema_version",
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
            "evidence_tier",
        )
    }
    record = {
        **common,
        "record_type": "outcome",
        "recorded_wall_ns": recorded_wall_ns,
        "recorded_monotonic_ns": time.monotonic_ns(),
        "required_bytes": observed["required_bytes"],
        "delivered_bytes": observed["delivered_bytes"],
        "deadline_wall_ns": observed["deadline_wall_ns"],
        "completion_wall_ns": observed["completion_wall_ns"],
        "observation_valid": observed["observation_valid"],
        "application_complete": observed["application_complete"],
        "met_deadline": met_deadline,
        "no_action": observed["no_action"],
        "controller_failure": observed["controller_failure"],
        "late_install": observed["late_install"],
        "conservative_on_time_completion": success,
        "failure_stage": failure_stage,
        "missingness_reason": observed["missingness_reason"],
        "outcome_evidence_sha256": evidence.evidence_file_sha256,
        "outcome_evidence_status": observed["evidence_status"],
        "outcome_evidence_origin": observed["evidence_origin"],
        "application_acknowledgement_id": observed["acknowledgement_id"],
        "application_acknowledgement_source": observed["acknowledgement_source"],
        "application_acknowledgement_wall_ns": observed["acknowledged_wall_ns"],
        "application_acknowledgement_sha256": evidence.acknowledgement_sha256,
        "application_acknowledgement_authenticated": evidence.authenticated,
        "application_acknowledgement_key_id": evidence.authentication_key_id,
    }
    appended = append_hash_chained_record(args.ledger, record, schema)
    lifecycle = audit_decision_lifecycles(args.ledger, schema)
    lifecycle.raise_for_errors()
    print(json.dumps(appended, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
