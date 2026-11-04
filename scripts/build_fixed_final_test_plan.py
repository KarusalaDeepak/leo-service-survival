#!/usr/bin/env python3
"""Create the signed untouched final-test ITT assignment after unlock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_release import (  # noqa: E402
    build_fixed_final_test_plan,
    load_signing_key,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--final-test-unlock", type=Path, required=True)
    parser.add_argument("--block-register", type=Path, required=True)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--allocation-seed-file", type=Path, required=True)
    parser.add_argument("--candidate-allocation-fraction", type=float, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-final-test-allocation", action="store_true")
    args = parser.parse_args()
    if not args.freeze_final_test_allocation:
        raise RuntimeError("final-test allocation remains locked; pass --freeze-final-test-allocation")
    blocks = [line.strip() for line in args.block_register.read_text(encoding="utf-8").splitlines() if line.strip()]
    plan = build_fixed_final_test_plan(
        gate_report=json.loads(args.gate_report.read_text(encoding="utf-8")),
        final_test_unlock=json.loads(args.final_test_unlock.read_text(encoding="utf-8")),
        signing_key=load_signing_key(args.signing_key),
        analysis_spec_path=args.analysis_spec,
        block_ids=blocks,
        allocation_seed=args.allocation_seed_file.read_text(encoding="utf-8").strip(),
        candidate_allocation_fraction=args.candidate_allocation_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "candidate_blocks": plan["candidate_blocks"], "baseline_blocks": plan["baseline_blocks"], "artifact_sha256": plan["artifact_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
