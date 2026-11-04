#!/usr/bin/env python3
"""Emit one authenticated endpoint acknowledgement for a prospective action.

The command records supplied observations only.  It does not append an outcome
or calculate task success; use ``append_prospective_outcome.py`` after the
acknowledgement has been transferred to the ledger host.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.application_acknowledgement import (  # noqa: E402
    ApplicationOutcomeObservation,
    DEFAULT_CLAIM_BOUNDARY,
    emit_live_application_acknowledgement,
    load_application_acknowledgement_contract,
    load_bound_live_action,
)
from leo_service_survival.prospective_release import load_signing_key  # noqa: E402
from leo_service_survival.prospective_trial import sha256_file  # noqa: E402


def strict_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an immutable, purpose-bound HMAC acknowledgement containing "
            "explicit endpoint byte, deadline, and validity observations."
        )
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--ledger-schema", type=Path, required=True)
    parser.add_argument("--acknowledgement-schema", type=Path, required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledgement-id", required=True)
    parser.add_argument("--acknowledgement-source", required=True)
    parser.add_argument("--acknowledged-wall-ns", type=int)
    parser.add_argument("--required-bytes", type=int, required=True)
    parser.add_argument("--delivered-bytes", type=int, required=True)
    parser.add_argument("--deadline-wall-ns", type=int, required=True)
    parser.add_argument("--completion-wall-ns", type=int, required=True)
    parser.add_argument("--observation-valid", type=strict_bool, required=True)
    parser.add_argument("--application-complete", type=strict_bool, required=True)
    parser.add_argument("--no-action", type=strict_bool, required=True)
    parser.add_argument("--controller-failure", type=strict_bool, required=True)
    parser.add_argument("--late-install", type=strict_bool, required=True)
    parser.add_argument("--missingness-reason", default="")
    parser.add_argument("--claim-boundary", default=DEFAULT_CLAIM_BOUNDARY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    action = load_bound_live_action(
        args.ledger,
        args.ledger_schema,
        args.decision_id,
    )
    contract = load_application_acknowledgement_contract(args.acknowledgement_schema)
    key = load_signing_key(args.signing_key)
    observation = ApplicationOutcomeObservation(
        required_bytes=args.required_bytes,
        delivered_bytes=args.delivered_bytes,
        deadline_wall_ns=args.deadline_wall_ns,
        completion_wall_ns=args.completion_wall_ns,
        observation_valid=args.observation_valid,
        application_complete=args.application_complete,
        no_action=args.no_action,
        controller_failure=args.controller_failure,
        late_install=args.late_install,
        missingness_reason=args.missingness_reason,
    )
    payload = emit_live_application_acknowledgement(
        output_path=args.output,
        action=action,
        observation=observation,
        acknowledgement_id=args.acknowledgement_id,
        acknowledgement_source=args.acknowledgement_source,
        acknowledged_wall_ns=args.acknowledged_wall_ns,
        signing_key=key,
        contract=contract,
        claim_boundary=args.claim_boundary,
    )
    summary = {
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "artifact_sha256": payload["artifact_sha256"],
        "authentication_key_id": payload["authentication"]["key_id"],
        "authentication_purpose": payload["authentication"]["purpose"],
        "decision_id": payload["decision_id"],
        "evidence_status": payload["evidence_status"],
        "outcome_appended": False,
        "claim_boundary": payload["claim_boundary"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
