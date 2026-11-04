#!/usr/bin/env python3
"""Run the registered Objective 4 evidence gate and write a signed report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_release import (  # noqa: E402
    load_signing_key,
    run_frozen_evidence_gate,
)
from leo_service_survival.gate_freeze import (  # noqa: E402
    CONFIRMATORY_LIVE,
    CONTROLLED_FIXTURE,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-config", type=Path, required=True)
    parser.add_argument("--campaign-register", type=Path, required=True)
    parser.add_argument("--gate-evidence", type=Path, required=True)
    parser.add_argument("--trial-freeze-manifest", type=Path)
    parser.add_argument("--opportunity-manifest", type=Path)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute-frozen-gate",
        action="store_true",
        help="Required acknowledgement that all inputs are frozen and registered.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--confirmatory-live",
        action="store_true",
        help=(
            "Run only from an authenticated freeze manifest, immutable registry "
            "lock, and metrics recomputed from hash-locked raw opportunities."
        ),
    )
    mode.add_argument(
        "--allow-fake-driver-test",
        action="store_true",
        help=(
            "Run an explicitly labelled FAKE_DRIVER_TEST software fixture. "
            "This can never be represented as live evidence."
        ),
    )
    args = parser.parse_args()
    if not args.execute_frozen_gate:
        raise RuntimeError("gate execution is locked; pass --execute-frozen-gate")
    execution_mode = CONFIRMATORY_LIVE if args.confirmatory_live else CONTROLLED_FIXTURE
    if args.confirmatory_live and (
        args.trial_freeze_manifest is None or args.opportunity_manifest is None
    ):
        raise RuntimeError(
            "--confirmatory-live requires --trial-freeze-manifest and "
            "--opportunity-manifest"
        )
    report = run_frozen_evidence_gate(
        config_path=args.trial_config,
        register_path=args.campaign_register,
        evidence_path=args.gate_evidence,
        signing_key=load_signing_key(args.signing_key),
        execution_mode=execution_mode,
        freeze_manifest_path=args.trial_freeze_manifest,
        opportunity_manifest_path=args.opportunity_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"decision": report["decision"], "report": str(args.output), "artifact_sha256": report["artifact_sha256"]}, indent=2))
    return 0 if report["decision"] == "AUTHORIZED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
