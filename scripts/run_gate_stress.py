#!/usr/bin/env python3
"""Run deterministic synthetic gate-logic operating-characteristic checks.

This script does not produce field validation.  It verifies that the exact code
path fails closed under injected feature and delayed-outcome shifts and reports
campaign-level operating characteristics with explicit evidence-tier labels.
The calibration, endpoint-effect, and timing values are a supplied logic-test
fixture; this script does not estimate statistical bounds from campaigns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leo_service_survival.gating import (
    GATE_LOGIC_FIXTURE,
    EvidenceGate,
    FrozenMahalanobisDetector,
    GateCriteria,
    GateEvidence,
    evaluate_calibration_monitor_campaigns,
    evaluate_gate_campaigns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--campaigns-per-class", type=int, default=30)
    parser.add_argument("--rows-per-campaign", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


def _stable_frame(rng: np.random.Generator, rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latency_ms": np.clip(rng.normal(42.0, 6.0, rows), 5.0, None),
            "loss_fraction": rng.beta(1.2, 45.0, rows),
            "throughput_mbps": np.clip(rng.lognormal(np.log(65.0), 0.25, rows), 1.0, None),
            "elevation_deg": rng.uniform(20.0, 80.0, rows),
        }
    )


def _shifted_frame(
    rng: np.random.Generator,
    rows: int,
    shift_kind: int,
) -> pd.DataFrame:
    frame = _stable_frame(rng, rows)
    if shift_kind == 0:
        frame["latency_ms"] += 80.0
    elif shift_kind == 1:
        frame["loss_fraction"] = rng.beta(8.0, 3.0, rows)
    else:
        frame["throughput_mbps"] *= 0.05
        frame.loc[rng.random(rows) < 0.35, "elevation_deg"] = np.nan
    return frame


def main() -> None:
    args = parse_args()
    if args.campaigns_per_class < 2:
        raise ValueError("at least two campaigns per class are required")
    if args.rows_per_campaign < args.batch_size:
        raise ValueError("rows per campaign must be at least one full batch")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(args.seed)
    features = (
        "latency_ms",
        "loss_fraction",
        "throughput_mbps",
        "elevation_deg",
    )
    development = _stable_frame(rng, 5000)
    detector = FrozenMahalanobisDetector(features, quantile=0.99).fit(development)
    criteria = GateCriteria(
        minimum_independent_campaigns=2,
        minimum_feature_coverage=0.90,
        maximum_ood_fraction=0.10,
        maximum_calibration_gap_upper=0.05,
        minimum_candidate_minus_baseline_endpoint_effect_lower=-0.02,
        maximum_control_loop_p95_ms=100.0,
    )
    evidence = GateEvidence(
        schema_passed=True,
        independent_campaigns=3,
        calibration_gap_upper=0.03,
        candidate_minus_baseline_endpoint_effect_lower=-0.01,
        control_loop_p95_ms=35.0,
        endpoint_name="task_completion_rate",
        evidence_basis=GATE_LOGIC_FIXTURE,
        multiple_testing_adjustment="not_applicable_supplied_fixture",
    )
    gate = EvidenceGate(detector, criteria)

    streams: dict[str, pd.DataFrame] = {}
    shifted: dict[str, bool] = {}
    loss_streams: dict[str, np.ndarray] = {}
    for index in range(args.campaigns_per_class):
        stable_id = f"stable-{index:03d}"
        shift_id = f"shift-{index:03d}"
        streams[stable_id] = _stable_frame(rng, args.rows_per_campaign)
        streams[shift_id] = _shifted_frame(
            rng,
            args.rows_per_campaign,
            index % 3,
        )
        shifted[stable_id] = False
        shifted[shift_id] = True
        loss_streams[stable_id] = rng.beta(1.0, 30.0, args.rows_per_campaign)
        loss_streams[shift_id] = rng.beta(8.0, 3.0, args.rows_per_campaign)

    immediate_detail, immediate_summary = evaluate_gate_campaigns(
        gate,
        streams,
        shifted,
        evidence,
        batch_size=args.batch_size,
    )
    delayed_detail, delayed_summary = evaluate_calibration_monitor_campaigns(
        loss_streams,
        shifted,
        null_mean=0.10,
        alpha=0.01,
        betting_fraction=1.0,
    )
    for frame in (immediate_detail, immediate_summary, delayed_detail, delayed_summary):
        frame.insert(0, "evidence_tier", "synthetic_simulation")

    artifacts = {
        "gate_immediate_detail.csv": immediate_detail,
        "gate_immediate_summary.csv": immediate_summary,
        "gate_delayed_detail.csv": delayed_detail,
        "gate_delayed_summary.csv": delayed_summary,
    }
    artifact_hashes: dict[str, str] = {}
    for name, frame in artifacts.items():
        path = args.output_dir / name
        frame.to_csv(path, index=False)
        artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "evidence_tier": "synthetic_simulation",
        "validation_level": "gate_logic_test",
        "statistical_evidence_generation": {
            "performed": False,
            "calibration_bound": "supplied_fixture_value",
            "candidate_minus_baseline_endpoint_effect_bound": (
                "supplied_fixture_value"
            ),
            "control_loop_p95": "supplied_fixture_value",
        },
        "claim_boundary": (
            "Injected-shift gate-logic operating-characteristic test only; it "
            "does not generate statistical evidence bounds and is not an "
            "independent field validation or policy authorization."
        ),
        "seed": args.seed,
        "campaigns_per_class": args.campaigns_per_class,
        "rows_per_campaign": args.rows_per_campaign,
        "batch_size": args.batch_size,
        "detector_features": list(features),
        "detector_quantile": detector.quantile,
        "detector_threshold": detector.threshold_,
        "gate_criteria": criteria.__dict__,
        "gate_evidence": evidence.__dict__,
        "artifact_sha256": artifact_hashes,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(immediate_summary.to_string(index=False))
    print(delayed_summary.to_string(index=False))


if __name__ == "__main__":
    main()
