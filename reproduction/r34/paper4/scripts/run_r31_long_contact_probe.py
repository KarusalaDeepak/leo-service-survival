#!/usr/bin/env python3
"""Execute the frozen R31 synthetic long-contact protocol.

The transformed 15/30/45-minute contacts are not native EOS-Bench or measured
LEO pass geometry. Recorded WetLinks QoS labels are assigned by hash to them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r30_wetlinks_eo_trace as r30

ROOT = Path(__file__).resolve().parents[1]
HORIZONS_MINUTES = (15, 30, 45)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for piece in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def synthetic_long_contact_instance(path: Path, minutes: int):
    """Apply the protocol's synthetic transformation without touching input."""
    if minutes not in HORIZONS_MINUTES:
        raise ValueError("horizon outside frozen protocol")
    base = r30.ms.load_instance(path, "combined")
    seconds = minutes * 60
    factor = seconds / 240.0
    for task in base["tasks"]:
        for atom in task["atoms"]:
            atom["volume"] *= factor
    for sat, rows in base["downlinks"].items():
        base["downlinks"][sat] = [(gs, b, b + seconds, rate)
                                   for gs, b, _e, rate in rows]
    base["name"] += f"_synthetic_continuous_{minutes}min"
    return base


def run(args) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("use an empty directory; no frozen output is overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = r30.load_frozen_predictions(args.predictions)
    updated, calibration, _ = r30.calibrate_and_score(frozen)
    rows = []
    for minutes in HORIZONS_MINUTES:
        h = minutes // 5
        eligible = updated[(updated["horizon_bins"] == h) &
                           updated["analysis_block"].isin(r30.EVALUATION_BLOCKS) &
                           updated["evaluable_at_horizon"].eq(True)]
        eligible = eligible.sort_values("test_row_index")
        if len(eligible) < 100:
            raise ValueError(f"insufficient complete-case traces at {minutes} min")
        instance = synthetic_long_contact_instance(args.instance, minutes)
        for seed in r30.SEEDS:
            for policy in r30.POLICIES:
                row = r30.run_one(instance, eligible, seed, policy,
                                  horizon_seconds=minutes * 60)
                row["synthetic_contact_minutes"] = minutes
                row["eligible_trace_episodes"] = len(eligible)
                row["volume_factor_over_native"] = minutes * 60 / 240
                rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(args.output_dir / "long_contact_runs.csv", index=False)
    summary = runs.groupby(["synthetic_contact_minutes", "policy"], as_index=False).agg(
        mean_qos_proxy_priority=("qos_proxy_priority_fraction", "mean"),
        min_qos_proxy_priority=("qos_proxy_priority_fraction", "min"),
        max_qos_proxy_priority=("qos_proxy_priority_fraction", "max"),
        mean_planned_priority=("planned_priority_fraction", "mean"),
        mean_wasted_downlink=("wasted_downlink_fraction", "mean"),
        eligible_trace_episodes=("eligible_trace_episodes", "first"),
    )
    summary.to_csv(args.output_dir / "long_contact_summary.csv", index=False)
    paired = []
    for minutes, group in runs.groupby("synthetic_contact_minutes"):
        wide = group.pivot(index="seed", columns="policy",
                           values="qos_proxy_priority_fraction")
        signatures = group.pivot(index="seed", columns="policy",
                                 values="plan_signature_sha256")
        for candidate, baseline in (("frozen_logistic", "earliest"),
                                    ("temporal_isotonic", "frozen_logistic"),
                                    ("lower_envelope", "frozen_logistic")):
            difference = wide[candidate] - wide[baseline]
            paired.append({"synthetic_contact_minutes": minutes,
                           "candidate": candidate, "baseline": baseline,
                           "mean_paired_priority_delta": difference.mean(),
                           "min_paired_delta": difference.min(),
                           "max_paired_delta": difference.max(),
                           "positive_assignments": int((difference > 1e-12).sum()),
                           "negative_assignments": int((difference < -1e-12).sum()),
                           "tied_assignments": int((difference.abs() <= 1e-12).sum()),
                           "different_plan_signatures": int((signatures[candidate] !=
                                                             signatures[baseline]).sum())})
    pd.DataFrame(paired).to_csv(args.output_dir / "paired_assignment_differences.csv",
                                index=False)
    calibration[calibration.horizon_bins.isin([3, 6, 9])].to_csv(
        args.output_dir / "long_horizon_calibration.csv", index=False)
    protocol = {"status": "exploratory controlled synthetic contact transformation",
                "frozen_protocol": str(args.protocol), "protocol_sha256": sha256(args.protocol),
                "input_prediction_sha256": sha256(args.predictions),
                "input_overlay_sha256": sha256(args.instance),
                "horizons_minutes": HORIZONS_MINUTES,
                "volume_factor": "H*60/240, applied to atomic request volumes",
                "contact_transformation": "all S2G ends become start+H; no physical orbit validation",
                "trace_assignment": "R30 SHA256 shared contact outcome; complete cases only",
                "evaluation_status": "previously inspected W09-W10 blocks, not confirmatory"}
    (args.output_dir / "protocol_and_inputs.json").write_text(
        json.dumps(protocol, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(pd.DataFrame(paired).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path,
        default=ROOT / "results/reviewer_prediction_release/wetlinks_all_pipeline_horizon_predictions.csv.gz")
    parser.add_argument("--instance", type=Path,
        default=r30.PAPER3 / "data/external/eos_bench/eos_s1_overlay.json")
    parser.add_argument("--protocol", type=Path,
        default=ROOT / "docs/r31_long_contact_protocol_2026-11-19.md")
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
