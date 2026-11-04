#!/usr/bin/env python3
"""Build or verify the authenticated Objective 4 trial-freeze manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.gate_freeze import (  # noqa: E402
    CONFIRMATORY_LIVE,
    CONTROLLED_FIXTURE,
    TrialFreezeManifest,
    build_trial_freeze_manifest,
    verify_trial_freeze_manifest,
)
from leo_service_survival.prospective_release import load_signing_key  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument(
        "--execution-mode",
        choices=(CONFIRMATORY_LIVE, CONTROLLED_FIXTURE),
    )
    parser.add_argument("--protocol-id")
    parser.add_argument("--freeze-timestamp-utc")
    parser.add_argument("--testbed-qualification", type=Path)
    parser.add_argument("--route-config", type=Path)
    parser.add_argument("--campaign-registry-lock", type=Path)
    parser.add_argument("--ledger-schema", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--trial-config", type=Path)
    parser.add_argument("--analysis-plan", type=Path)
    parser.add_argument("--opportunity-derivation", type=Path)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    key = load_signing_key(args.signing_key)
    if args.verify:
        value = json.loads(args.verify.read_text(encoding="utf-8"))
        manifest = TrialFreezeManifest.from_dict(value)
        verify_trial_freeze_manifest(
            manifest, signing_key=key, expected_mode=args.execution_mode
        )
        print(
            json.dumps(
                {
                    "verified": True,
                    "execution_mode": manifest.execution_mode,
                    "evidence_status": manifest.evidence_status,
                    "manifest_sha256": manifest.manifest_sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    required = {
        "execution_mode": args.execution_mode,
        "protocol_id": args.protocol_id,
        "freeze_timestamp_utc": args.freeze_timestamp_utc,
        "testbed_qualification": args.testbed_qualification,
        "route_config": args.route_config,
        "campaign_registry_lock": args.campaign_registry_lock,
        "ledger_schema": args.ledger_schema,
        "model": args.model,
        "trial_config": args.trial_config,
        "analysis_plan": args.analysis_plan,
        "opportunity_derivation": args.opportunity_derivation,
        "output": args.output,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise SystemExit(f"creation mode missing arguments: {missing}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite frozen manifest: {args.output}")
    artifact_paths = {
        name: value
        for name, value in required.items()
        if name
        in {
            "testbed_qualification",
            "route_config",
            "campaign_registry_lock",
            "ledger_schema",
            "model",
            "trial_config",
            "analysis_plan",
            "opportunity_derivation",
        }
    }
    manifest = build_trial_freeze_manifest(
        execution_mode=args.execution_mode,
        protocol_id=args.protocol_id,
        freeze_timestamp_utc=args.freeze_timestamp_utc,
        artifact_paths=artifact_paths,
        signing_key=key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "execution_mode": manifest.execution_mode,
                "evidence_status": manifest.evidence_status,
                "manifest_sha256": manifest.manifest_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

