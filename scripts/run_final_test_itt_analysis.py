#!/usr/bin/env python3
"""Run the signed prespecified Objective 4 final-test ITT analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_release import (  # noqa: E402
    load_signing_key,
    run_prespecified_final_test_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--campaign-register", type=Path, required=True)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--final-test-unlock", type=Path, required=True)
    parser.add_argument("--final-test-plan", type=Path, required=True)
    parser.add_argument("--end-to-end-audit", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-frozen-analysis", action="store_true")
    args = parser.parse_args()
    if not args.execute_frozen_analysis:
        raise RuntimeError("final-test analysis is locked; pass --execute-frozen-analysis")
    report = run_prespecified_final_test_analysis(
        ledger_path=args.ledger,
        schema_path=args.schema,
        register_path=args.campaign_register,
        analysis_spec_path=args.analysis_spec,
        gate_report=json.loads(args.gate_report.read_text(encoding="utf-8")),
        final_test_unlock=json.loads(args.final_test_unlock.read_text(encoding="utf-8")),
        final_test_plan=json.loads(args.final_test_plan.read_text(encoding="utf-8")),
        audit_report=json.loads(args.end_to_end_audit.read_text(encoding="utf-8")),
        signing_key=load_signing_key(args.signing_key),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "risk_difference": report["risk_difference_candidate_minus_baseline"], "interval": report["risk_difference_interval"], "benefit_claim_supported": report["benefit_claim_supported"], "artifact_sha256": report["artifact_sha256"]}, indent=2))
    return 0 if report["status"] == "ANALYSIS_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
