#!/usr/bin/env python3
"""Issue the signed no-retuning unlock for the untouched final-test family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_release import (  # noqa: E402
    load_signing_key,
    unlock_untouched_final_test,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--canary-plan", type=Path, required=True)
    parser.add_argument("--checkpoint-report", type=Path, required=True)
    parser.add_argument("--campaign-register", type=Path, required=True)
    parser.add_argument("--checkpoint-approval-id", required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attest-no-retuning", action="store_true")
    args = parser.parse_args()
    unlock = unlock_untouched_final_test(
        gate_report=json.loads(args.gate_report.read_text(encoding="utf-8")),
        canary_plan=json.loads(args.canary_plan.read_text(encoding="utf-8")),
        checkpoint_report=json.loads(args.checkpoint_report.read_text(encoding="utf-8")),
        register_path=args.campaign_register,
        signing_key=load_signing_key(args.signing_key),
        checkpoint_approval_id=args.checkpoint_approval_id,
        no_retuning_attestation=args.attest_no_retuning,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(unlock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": unlock["status"], "artifact_sha256": unlock["artifact_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
