#!/usr/bin/env python3
"""Run the matched synthetic EOS mechanism ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leo_service_survival.eos_full_chain import EOSFullChainConfig
from leo_service_survival.eos_mechanism_ablation import (
    EOSMechanismAblationConfig,
    run_eos_mechanism_ablation,
    write_eos_mechanism_ablation_artifacts,
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
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "eos_mechanism_ablation_v3"


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
            "Compare six matched path-ranking mechanisms under favorable, neutral and adverse "
            "synthetic SNR--reliability conditions. The fitted WetLinks model is not applied."
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
    parser.add_argument("--calibration-rows-per-profile", type=int, default=3_000)
    parser.add_argument("--predictor-rows-per-profile", type=int, default=3_000)
    parser.add_argument("--calibration-validation-rows-per-profile", type=int, default=1_500)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    args = parser.parse_args()

    full_chain = EOSFullChainConfig(seeds=args.seeds)
    config = EOSMechanismAblationConfig(
        full_chain=full_chain,
        predictor_rows_per_profile=args.predictor_rows_per_profile,
        calibration_rows_per_profile=args.calibration_rows_per_profile,
        calibration_validation_rows_per_profile=args.calibration_validation_rows_per_profile,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    artifacts = run_eos_mechanism_ablation(args.scenario, config=config)
    output = write_eos_mechanism_ablation_artifacts(artifacts, args.output)
    summary = artifacts.tables["policy_summary"]
    report = {
        "output": str(output.resolve()),
        "wetlinks_fitted_model_used": artifacts.run_manifest["wetlinks_fitted_model_used"],
        "conditions": artifacts.run_manifest["conditions"],
        "policy_arms": artifacts.run_manifest["policy_arms"],
        "best_observed_arm_by_condition": {
            condition: str(group.sort_values(["completion_rate", "policy_arm"], ascending=[False, True]).iloc[0]["policy_arm"])
            for condition, group in summary.groupby("condition")
        },
        "claim_boundary": artifacts.run_manifest["claim_boundary"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
