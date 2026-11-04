#!/usr/bin/env python3
"""Evaluate completed canary block records against frozen stop rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_release import (  # noqa: E402
    evaluate_canary_checkpoint,
    load_signing_key,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-config", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--canary-plan", type=Path, required=True)
    parser.add_argument("--event-records", type=Path, required=True, help="JSON array of one event per planned block")
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.trial_config.read_text(encoding="utf-8"))
    if config.get("protocol_status") != "FROZEN_REGISTERED":
        raise RuntimeError("canary checkpoint requires a frozen registered protocol")
    checkpoint = evaluate_canary_checkpoint(
        gate_report=json.loads(args.gate_report.read_text(encoding="utf-8")),
        canary_plan=json.loads(args.canary_plan.read_text(encoding="utf-8")),
        event_records=json.loads(args.event_records.read_text(encoding="utf-8")),
        config=config,
        signing_key=load_signing_key(args.signing_key),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": checkpoint["status"], "triggered_stop_rules": checkpoint["triggered_stop_rules"], "artifact_sha256": checkpoint["artifact_sha256"]}, indent=2))
    return 0 if checkpoint["status"] == "CANARY_CHECKPOINT_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
