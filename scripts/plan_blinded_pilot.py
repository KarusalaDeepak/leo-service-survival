#!/usr/bin/env python3
"""Create or verify a signed, treatment-label-free Objective 4 pilot plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.pilot_power import (  # noqa: E402
    BlindedPilotInputs,
    build_blinded_pilot_plan,
    verify_blinded_pilot_plan,
)
from leo_service_survival.prospective_release import load_signing_key  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan Objective 4 randomized blocks using blinded aggregate pilot "
            "quantities only; treatment labels are not accepted."
        )
    )
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--baseline-completion", type=float)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--icc", type=float)
    parser.add_argument("--missing-fraction", type=float)
    parser.add_argument("--installation-failure-fraction", type=float)
    parser.add_argument("--effect-margin", type=float)
    parser.add_argument("--target-power", type=float)
    parser.add_argument("--target-ci-half-width", type=float)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    key = load_signing_key(args.signing_key)
    if args.verify is not None:
        creation_only = {
            "baseline_completion": args.baseline_completion,
            "block_size": args.block_size,
            "icc": args.icc,
            "missing_fraction": args.missing_fraction,
            "installation_failure_fraction": args.installation_failure_fraction,
            "effect_margin": args.effect_margin,
            "target_power": args.target_power,
            "target_ci_half_width": args.target_ci_half_width,
            "output": args.output,
        }
        supplied = sorted(name for name, value in creation_only.items() if value is not None)
        if supplied:
            raise SystemExit(f"verification mode does not accept creation arguments: {supplied}")
        report = json.loads(args.verify.read_text(encoding="utf-8"))
        verify_blinded_pilot_plan(report, signing_key=key)
        print(
            json.dumps(
                {
                    "verified": True,
                    "artifact_sha256": report["artifact_sha256"],
                    "blocks_per_arm": report["selected_plan"]["blocks_per_arm"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    required = {
        "baseline_completion": args.baseline_completion,
        "block_size": args.block_size,
        "icc": args.icc,
        "missing_fraction": args.missing_fraction,
        "installation_failure_fraction": args.installation_failure_fraction,
        "effect_margin": args.effect_margin,
        "target_power": args.target_power,
        "target_ci_half_width": args.target_ci_half_width,
        "output": args.output,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise SystemExit(f"creation mode missing arguments: {missing}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite signed pilot plan: {args.output}")
    inputs = BlindedPilotInputs(
        baseline_completion_probability=args.baseline_completion,
        block_size=args.block_size,
        intracluster_correlation=args.icc,
        missing_outcome_fraction=args.missing_fraction,
        installation_failure_fraction=args.installation_failure_fraction,
        minimum_effect_absolute=args.effect_margin,
        target_power=args.target_power,
        target_ci_half_width=args.target_ci_half_width,
    )
    report = build_blinded_pilot_plan(inputs, signing_key=key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": report["artifact_sha256"],
                "binding_requirement": report["selected_plan"]["binding_requirement"],
                "blocks_per_arm": report["selected_plan"]["blocks_per_arm"],
                "total_blocks": report["selected_plan"]["total_blocks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

