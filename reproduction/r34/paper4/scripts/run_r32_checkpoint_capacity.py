#!/usr/bin/env python3
"""Frozen R32 checkpoint/capacity sensitivity on R31 synthetic contacts.

No measured EO bytes, physical long LEO passes, or post-failure replanning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r31_long_contact_probe as r31

r30 = r31.r30
ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("earliest", "frozen_survival", "isotonic_survival",
            "frozen_goodput", "isotonic_goodput")


def add_frozen_curves(adjusted: pd.DataFrame, horizon_bins: int) -> pd.DataFrame:
    """Keep one evaluation episode per row with all earlier frozen predictions."""
    evaluation = adjusted[adjusted.analysis_block.isin(r30.EVALUATION_BLOCKS)]
    eligible = evaluation[(evaluation.horizon_bins == horizon_bins) &
                          evaluation.evaluable_at_horizon.eq(True)].copy()
    eligible = eligible.sort_values("test_row_index").reset_index(drop=True)
    if len(eligible) < 100:
        raise ValueError("too few evaluable held-out episodes")
    if eligible["observed_survival_target"].isna().any():
        raise ValueError("unknown endpoint in eligible pool")
    for name in ("frozen_logistic", "temporal_isotonic"):
        wide = evaluation.pivot(index="test_row_index", columns="horizon_bins",
                                values=name).reindex(columns=range(1, horizon_bins + 1))
        if wide.isna().any().any():
            raise ValueError("missing earlier survival forecast")
        curves = {int(index): tuple(float(x) for x in values)
                  for index, values in wide.iterrows()}
        eligible[f"curve_{name}"] = eligible.test_row_index.map(curves)
    return eligible


def expected_useful_fraction(row: pd.Series, name: str, bins: int) -> float:
    """Left-endpoint approximation to integral S(t) / reservation duration."""
    curve = row[f"curve_{name}"]
    if not 1 <= bins <= len(curve):
        raise ValueError("reservation exceeds available forecast")
    return (1.0 + sum(curve[:bins - 1])) / bins


def checkpoint_fraction(row: pd.Series, bins: int) -> float:
    if int(row["observed_survival_target"]) == 1:
        return 1.0
    if int(row["event"]) != 1:
        raise ValueError("known failure without confirmed event")
    onset = int(row["duration_bins"])
    return max(0.0, min((onset - 1) / bins, 1.0))


def choose_plan(plans, policy: str, outcomes, bins: int):
    if policy == "earliest":
        return min(plans, key=lambda p: (p.completion, p.variant))
    name = "frozen_logistic" if policy.startswith("frozen") else "temporal_isotonic"
    if policy.endswith("survival"):
        return max(plans, key=lambda p: (
            math.prod(float(row[f"curve_{name}"][bins - 1]) for row in outcomes(p).values()),
            -p.completion, p.variant))

    def score(plan):
        rows = outcomes(plan)
        reserved = 0.0
        useful = 0.0
        for sat, gs, b, e, _rate, _amount in plan.deliveries:
            duration = e - b
            key = (sat, gs, int(b), int(e))
            local_bins = max(1, math.ceil(duration / 300))
            fraction = expected_useful_fraction(rows[key], name, local_bins)
            reserved += duration
            useful += duration * fraction
        if reserved <= 0:
            raise ValueError("empty downlink reservation")
        # No fitted or test-tuned waste coefficient: efficiency and expected
        # unused reserved time are dimensionally explicit.
        return useful / reserved, -(reserved - useful), -plan.completion, plan.variant
    return max(plans, key=score)


def run_one(instance, eligible: pd.DataFrame, seed: int, policy: str,
            horizon_bins: int) -> dict:
    ledger, accepted = r30.ms.Ledger(), []
    total_priority = float(sum(t["priority"] for t in instance["tasks"]))
    requested_gb = float(sum(a["volume"] for t in instance["tasks"] for a in t["atoms"]))
    completed_priority = 0.0
    delivered_gb = 0.0
    unused_seconds = 0.0
    reserved_seconds = 0.0
    for task in r30.ms.ordered_tasks(instance):
        plans = r30.ms.candidate_plans(instance, task, ledger, single=False)
        if not plans:
            continue

        def outcomes(plan):
            keys = r30.plan_contact_keys(plan)
            if any((e - b) > 300 * horizon_bins for _, _, b, e in keys):
                raise ValueError("reservation exceeds forecast horizon")
            return {key: r30.keyed_episode(seed, key, eligible) for key in keys}

        chosen = choose_plan(plans, policy, outcomes, horizon_bins)
        records = outcomes(chosen)
        r30.ms.commit(chosen, ledger)
        accepted.append(chosen)
        if all(int(row["observed_survival_target"]) == 1 for row in records.values()):
            completed_priority += task["priority"]
        for sat, gs, b, e, _rate, amount in chosen.deliveries:
            key = (sat, gs, int(b), int(e))
            bins = max(1, math.ceil((e - b) / 300))
            fraction = checkpoint_fraction(records[key], bins)
            delivered_gb += float(amount) * fraction
            reserved_seconds += e - b
            unused_seconds += (e - b) * (1.0 - fraction)
    r30.ms.validate_schedule(instance, accepted)
    signature = hashlib.sha256(json.dumps(
        [(p.task_id, p.variant, r30.plan_contact_keys(p)) for p in accepted],
        separators=(",", ":"),
    ).encode()).hexdigest()
    return {"seed": seed, "policy": policy,
            "synthetic_contact_minutes": horizon_bins * 5,
            "strict_priority_fraction": completed_priority / total_priority,
            "checkpointed_goodput_fraction": delivered_gb / requested_gb,
            "unused_reserved_downlink_fraction": unused_seconds / max(reserved_seconds, 1),
            "planned_requests": len(accepted), "plan_signature_sha256": signature}


def run(args) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("frozen outputs are never overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adjusted, _calibration, _blocks = r30.calibrate_and_score(
        r30.load_frozen_predictions(args.predictions))
    rows = []
    for minutes in r31.HORIZONS_MINUTES:
        bins = minutes // 5
        eligible = add_frozen_curves(adjusted, bins)
        instance = r31.synthetic_long_contact_instance(args.instance, minutes)
        for seed in r30.SEEDS:
            for policy in POLICIES:
                rows.append(run_one(instance, eligible, seed, policy, bins))
    runs = pd.DataFrame(rows)
    runs.to_csv(args.output_dir / "runs.csv", index=False)
    summary = runs.groupby(["synthetic_contact_minutes", "policy"], as_index=False).agg(
        mean_strict_priority=("strict_priority_fraction", "mean"),
        mean_checkpointed_goodput=("checkpointed_goodput_fraction", "mean"),
        mean_unused_reserved_downlink=("unused_reserved_downlink_fraction", "mean"),
        min_strict_priority=("strict_priority_fraction", "min"),
        max_strict_priority=("strict_priority_fraction", "max"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired = []
    for minutes, group in runs.groupby("synthetic_contact_minutes"):
        for metric in ("strict_priority_fraction", "checkpointed_goodput_fraction",
                       "unused_reserved_downlink_fraction"):
            wide = group.pivot(index="seed", columns="policy", values=metric)
            for candidate, baseline in (("isotonic_survival", "frozen_survival"),
                                        ("isotonic_goodput", "frozen_goodput"),
                                        ("isotonic_goodput", "earliest")):
                difference = wide[candidate] - wide[baseline]
                paired.append({"synthetic_contact_minutes": minutes, "metric": metric,
                               "candidate": candidate, "baseline": baseline,
                               "mean_delta": difference.mean(), "min_delta": difference.min(),
                               "max_delta": difference.max(),
                               "positive_seeds": int((difference > 1e-12).sum()),
                               "negative_seeds": int((difference < -1e-12).sum()),
                               "tied_seeds": int((difference.abs() <= 1e-12).sum())})
    pd.DataFrame(paired).to_csv(args.output_dir / "paired.csv", index=False)
    manifest = {"protocol": str(args.protocol),
                "protocol_sha256": r31.sha256(args.protocol),
                "prediction_sha256": r31.sha256(args.predictions),
                "overlay_sha256": r31.sha256(args.instance),
                "outcome_rule": "ideal checkpoint at start of first violating 5-min bin",
                "planning_rule": "commit reservations before QoS realization; no reclaimed capacity",
                "status": "exploratory controlled simulation; previously inspected test blocks"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT /
                        "results/reviewer_prediction_release/wetlinks_all_pipeline_horizon_predictions.csv.gz")
    parser.add_argument("--instance", type=Path, default=r30.PAPER3 /
                        "data/external/eos_bench/eos_s1_overlay.json")
    parser.add_argument("--protocol", type=Path, default=ROOT /
                        "docs/r32_checkpoint_capacity_protocol_2026-11-19.md")
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
