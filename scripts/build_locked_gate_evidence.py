#!/usr/bin/env python3
"""Derive locked gate evidence from Step-2 hash-bound opportunity tables."""

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
    build_locked_gate_evidence,
)
from leo_service_survival.prospective_release import load_signing_key  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-freeze-manifest", type=Path, required=True)
    parser.add_argument("--opportunity-manifest", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument(
        "--execution-mode",
        choices=(CONFIRMATORY_LIVE, CONTROLLED_FIXTURE),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite locked gate evidence: {args.output}")
    manifest = TrialFreezeManifest.from_dict(
        json.loads(args.trial_freeze_manifest.read_text(encoding="utf-8"))
    )
    evidence = build_locked_gate_evidence(
        freeze_manifest=manifest,
        opportunity_manifest_path=args.opportunity_manifest,
        signing_key=load_signing_key(args.signing_key),
        execution_mode=args.execution_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "execution_mode": evidence["execution_mode"],
                "evidence_status": evidence["evidence_status"],
                "campaigns": len(evidence["campaigns"]),
                "paired_opportunity_groups": evidence["pooled_metrics"][
                    "paired_opportunity_groups"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

