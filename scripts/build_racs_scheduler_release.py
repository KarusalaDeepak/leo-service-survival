#!/usr/bin/env python3
"""Build the fail-closed Paper 3 scheduling integration evidence for Paper 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leo_service_survival.scheduler_release import (
    SchedulerReleaseCriteria,
    evaluate_scheduler_release,
    write_scheduler_release_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER3_ROOT = PROJECT_ROOT.parent / "leo-eo-scheduling"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper3-root",
        type=Path,
        default=PAPER3_ROOT,
        help="authoritative Paper 3 repository containing frozen result tables",
    )
    parser.add_argument(
        "--transfer-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "racs_eos_transfer_integration_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "racs_scheduler_release_v1",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    args = parser.parse_args()

    robustness = args.paper3_root / "results" / "parameter_robustness_complete_2026-11-02"
    matched = args.paper3_root / "results" / "reviewer_revision"
    criteria = SchedulerReleaseCriteria(bootstrap_resamples=args.bootstrap_resamples)
    artifacts = evaluate_scheduler_release(
        score_outer_results=robustness / "score_outer_results.csv",
        matched_instance_results=matched / "instance_results.csv",
        paired_racs_rf_evidence=robustness / "paired_racs_rf_evidence.csv",
        transfer_summary=args.transfer_root / "comparison_summary.csv",
        transfer_paired_statistics=args.transfer_root / "paired_statistics.csv",
        transfer_instance_results=args.transfer_root / "instance_results.csv",
        transfer_manifest=args.transfer_root / "run_manifest.json",
        criteria=criteria,
    )
    output = write_scheduler_release_artifacts(artifacts, args.output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "decision": artifacts.decision["decision"],
                "failed_checks": artifacts.decision["failed_checks"],
                "field_claim": artifacts.decision["field_claim"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
