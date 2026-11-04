#!/usr/bin/env python3
"""Create a signed deterministic canary allocation after gate authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_release import (  # noqa: E402
    build_fixed_canary_plan,
    load_signing_key,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--block-register", type=Path, required=True, help="One unique block id per nonempty line")
    parser.add_argument("--allocation-seed-file", type=Path, required=True)
    parser.add_argument("--allocation-fraction", type=float, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-allocation", action="store_true")
    args = parser.parse_args()
    if not args.freeze_allocation:
        raise RuntimeError("allocation remains locked; pass --freeze-allocation")
    blocks = [line.strip() for line in args.block_register.read_text(encoding="utf-8").splitlines() if line.strip()]
    seed = args.allocation_seed_file.read_text(encoding="utf-8").strip()
    report = json.loads(args.gate_report.read_text(encoding="utf-8"))
    plan = build_fixed_canary_plan(
        gate_report=report,
        signing_key=load_signing_key(args.signing_key),
        block_ids=blocks,
        allocation_seed=seed,
        allocation_fraction=args.allocation_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "candidate_blocks": plan["candidate_blocks"], "artifact_sha256": plan["artifact_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
