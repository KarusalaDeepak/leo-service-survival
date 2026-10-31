#!/usr/bin/env python3
"""Run the audited EOS-Bench plus stochastic-network full-chain experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leo_service_survival.eos_full_chain import (
    EOSFullChainConfig,
    run_eos_full_chain_experiment,
    write_eos_full_chain_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "eos_bench"
    / "extracted"
    / "output_re_s3"
    / "Sats3_M100_T7d_cities_08.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "eos_full_chain_synthetic_v2"


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic full-chain experiment grounded in EOS-Bench geometry. "
            "The resulting evidence tier is always synthetic_simulation."
        )
    )
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=EOSFullChainConfig().seeds,
        help="comma-separated deterministic stochastic-replication seeds",
    )
    parser.add_argument(
        "--ns3-profile-csv",
        type=Path,
        default=None,
        help=(
            "optional aggregate ns-3 path-profile CSV; using it still produces "
            "synthetic_simulation evidence"
        ),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    args = parser.parse_args()

    config = EOSFullChainConfig(
        seeds=args.seeds,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    artifacts = run_eos_full_chain_experiment(
        args.scenario,
        config=config,
        network_profile_csv=args.ns3_profile_csv,
    )
    output = write_eos_full_chain_artifacts(artifacts, args.output)
    report = {
        "output": str(output.resolve()),
        "audit_passed": artifacts.audit.passed,
        "paired_inference": artifacts.paired_inference,
        "strict_completion_count": artifacts.completion_status["strict_completion_count"],
        "claim_boundary": artifacts.completion_status["claim_boundary"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
