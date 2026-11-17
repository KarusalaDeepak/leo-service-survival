#!/usr/bin/env python3
"""Build the hash-verified Paper 3 scheduler to Paper 4 path handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leo_service_survival.scheduler_path_bridge import (
    build_scheduler_path_bridge,
    write_scheduler_path_bridge,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = next(
    (parent for parent in ROOT.parents if (parent / "00_Research_Overview").is_dir()),
    ROOT.parent,
)
DEFAULT_SCENARIO = (
    ROOT
    / "data"
    / "external"
    / "eos_bench"
    / "extracted"
    / "output_re_s3"
    / "Sats3_M100_T7d_cities_08.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the frozen full_margin scheduler release at the Paper 4 path "
            "interface without combining incompatible synthetic outcomes."
        )
    )
    parser.add_argument(
        "--transfer-dir",
        type=Path,
        default=ROOT / "results" / "racs_eos_transfer_integration_v1",
    )
    parser.add_argument(
        "--paper3-root",
        type=Path,
        default=RESEARCH_ROOT / "03_Paper_RACS_EO_Scheduling" / "Code_and_Manuscript",
    )
    parser.add_argument(
        "--scheduler-release",
        type=Path,
        default=ROOT
        / "results"
        / "racs_scheduler_release_v1"
        / "scheduler_release_decision.json",
    )
    parser.add_argument(
        "--p4-full-chain-dir",
        type=Path,
        default=ROOT / "results" / "eos_full_chain_synthetic_v2",
    )
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--instance", default="output_re_s3/Sats3_M100_T7d_cities_08"
    )
    parser.add_argument("--simulation-seed", type=int, default=20261101)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "scheduler_path_bridge_v1",
    )
    args = parser.parse_args()

    artifacts = build_scheduler_path_bridge(
        transfer_dir=args.transfer_dir,
        paper3_root=args.paper3_root,
        scheduler_release_path=args.scheduler_release,
        p4_full_chain_dir=args.p4_full_chain_dir,
        scenario_path=args.scenario,
        instance=args.instance,
        simulation_seed=args.simulation_seed,
    )
    output = write_scheduler_path_bridge(artifacts, args.output_dir)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "installed_scheduler": artifacts.boundary["installed_scheduler"],
                "scheduled_plans": artifacts.boundary["scheduled_plans"],
                "plans_with_exact_geometry_mapping": artifacts.boundary[
                    "plans_with_exact_geometry_mapping"
                ],
                "continuous_outcome_experiment_completed": artifacts.boundary[
                    "continuous_outcome_experiment_completed"
                ],
                "evidence_tier": artifacts.boundary["evidence_tier"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
