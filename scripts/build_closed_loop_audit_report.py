#!/usr/bin/env python3
"""Build a signed end-to-end action, timing, outcome, and claim-boundary audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_release import (  # noqa: E402
    build_end_to_end_audit_report,
    load_signing_key,
)


def _optional_json(path: Path | None) -> dict | None:
    return None if path is None else json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--campaign-register", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--checkpoint-report", type=Path)
    parser.add_argument("--final-test-unlock", type=Path)
    parser.add_argument("--final-test-plan", type=Path)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_end_to_end_audit_report(
        ledger_path=args.ledger,
        schema_path=args.schema,
        register_path=args.campaign_register,
        gate_report=json.loads(args.gate_report.read_text(encoding="utf-8")),
        checkpoint_report=_optional_json(args.checkpoint_report),
        final_test_unlock=_optional_json(args.final_test_unlock),
        final_test_plan=_optional_json(args.final_test_plan),
        signing_key=load_signing_key(args.signing_key),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"claim_level": report["claim_level"], "integrity_passed": report["integrity_passed"], "artifact_sha256": report["artifact_sha256"]}, indent=2))
    return 0 if report["integrity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
