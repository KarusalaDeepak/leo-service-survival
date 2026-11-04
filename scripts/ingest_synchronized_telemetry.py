#!/usr/bin/env python3
"""Ingest a complete synchronized telemetry bundle into a new hash chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.field_preparation import ingest_telemetry_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-probe-schedule", type=Path, required=True)
    parser.add_argument("--campaign-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-pending-outcomes",
        action="store_true",
        help="Permit route decisions whose registered outcome has not matured.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schedule = pd.read_csv(args.expected_probe_schedule, keep_default_na=False)
    if "probe_id" not in schedule:
        raise SystemExit("expected probe schedule is missing probe_id")
    if schedule["probe_id"].duplicated().any():
        raise SystemExit("expected probe schedule contains duplicate probe_id values")
    lock = json.loads(args.campaign_lock.read_text(encoding="utf-8"))
    report = ingest_telemetry_bundle(
        records,
        args.output,
        expected_probe_ids=schedule["probe_id"],
        campaign_lock=lock,
        require_complete_action_outcome=not args.allow_pending_outcomes,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "evidence_status": report.evidence_status,
                "live_eligible": report.live_eligible,
                "checks": report.checks.to_dict(orient="records"),
                "passed": report.passed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
