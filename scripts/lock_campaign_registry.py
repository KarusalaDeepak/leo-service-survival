#!/usr/bin/env python3
"""Create or verify an immutable Objective 4 campaign-register snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.field_preparation import (  # noqa: E402
    build_campaign_registry_lock,
    verify_campaign_registry_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="Complete source register CSV")
    mode.add_argument("--verify", type=Path, help="Existing locked JSON snapshot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--qualification-sha256")
    parser.add_argument("--lock-timestamp-utc")
    parser.add_argument(
        "--allow-fake-driver-test",
        action="store_true",
        help="Create a FAKE_DRIVER_TEST lock that is never live evidence.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify:
        lock = json.loads(args.verify.read_text(encoding="utf-8"))
        report = verify_campaign_registry_lock(lock)
        print(
            json.dumps(
                {
                    "path": str(args.verify),
                    "evidence_status": report.evidence_status,
                    "live_eligible": report.live_eligible,
                    "checks": report.checks.to_dict(orient="records"),
                    "passed": report.passed,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report.passed else 1

    required = {
        "output": args.output,
        "protocol_sha256": args.protocol_sha256,
        "qualification_sha256": args.qualification_sha256,
        "lock_timestamp_utc": args.lock_timestamp_utc,
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise SystemExit(f"creation mode missing arguments: {missing}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite lock: {args.output}")

    frame = pd.read_csv(args.input, keep_default_na=False)
    lock = build_campaign_registry_lock(
        frame,
        protocol_sha256=args.protocol_sha256,
        qualification_sha256=args.qualification_sha256,
        lock_timestamp_utc=args.lock_timestamp_utc,
        allow_fake=args.allow_fake_driver_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evidence_status": lock["evidence_status"],
                "live_field_evidence": lock["live_field_evidence"],
                "registry_sha256": lock["registry_sha256"],
                "campaigns": len(lock["campaigns"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
