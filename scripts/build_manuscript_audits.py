#!/usr/bin/env python3
"""Rebuild the focused scheduler and EOS consequence audits used in Paper 4.

The script reads only versioned Paper 4 result tables and the existing scheduler
implementation.  It does not refit a model or alter manuscript sources.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from leo_service_survival.decisions import run_trace_driven_scheduling  # noqa: E402
from leo_service_survival.data import build_wetlinks_episodes  # noqa: E402
from leo_service_survival.reporting import scheduler_two_level_inference  # noqa: E402


HORIZONS = np.asarray([1, 3, 6, 12], dtype=int)
METRICS = ("completion_rate_all", "priority_reward", "fairness")
CONDITION_ORDER = {"favorable": 0, "neutral": 1, "adverse": 2}


class PrecomputedSurvivalModel:
    """Small adapter that exposes saved per-episode survival predictions."""

    def __init__(self, stage_rows: pd.DataFrame, stage: str) -> None:
        selected = stage_rows.loc[stage_rows["stage"].eq(stage)]
        self.stage = stage
        self.table = selected.pivot(
            index="episode_id", columns="horizon_bins", values="predicted_survival"
        )

    def predict_survival(
        self, trace: pd.DataFrame, horizons: np.ndarray | list[int]
    ) -> np.ndarray:
        requested = [int(value) for value in horizons]
        return self.table.loc[trace["episode_id"], requested].to_numpy(float)


def as_bool(values: pd.Series) -> pd.Series:
    """Interpret persisted Boolean columns without treating the string 'False' as true."""

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin({"true", "false", "1", "0"})
    if unknown.any():
        examples = sorted(normalized.loc[unknown].unique())[:5]
        raise ValueError(f"unrecognized Boolean values: {examples}")
    return normalized.isin({"true", "1"})


def run_scheduler_audit(
    stage_predictions: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stages = pd.read_csv(stage_predictions)
    required_stages = {"raw_base", "final_monotone"}
    if not required_stages.issubset(set(stages["stage"])):
        raise ValueError(f"missing stages: {sorted(required_stages - set(stages['stage']))}")

    raw_h1 = stages.loc[
        stages["stage"].eq("raw_base") & stages["horizon_bins"].eq(1)
    ].reset_index(drop=True)
    trace = raw_h1[
        ["episode_id", "duration_bins", "event", "analysis_block"]
    ].rename(columns={"analysis_block": "calendar_block"})
    if trace["episode_id"].duplicated().any():
        raise ValueError("scheduler trace contains duplicate episode IDs")

    details: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    variants = (
        ("raw_base", "raw_base"),
        ("final_calibrated", "final_monotone"),
    )
    for policy_variant, prediction_stage in variants:
        model = PrecomputedSurvivalModel(stages, prediction_stage)
        detail, summary = run_trace_driven_scheduling(
            trace,
            model,
            scenarios=120,
            task_count=80,
            candidate_count=4,
            horizon_slots=48,
            satellite_count=4,
            ground_station_count=2,
            survival_congestion_weight=0.1,
            seed=17,
        )
        selected_detail = detail.loc[detail["policy"].eq("survival_aware")].copy()
        selected_detail["policy"] = policy_variant
        details.append(selected_detail)

        selected_summary = summary.loc[summary["policy"].eq("survival_aware")].iloc[0]
        summaries.append(
            {
                "policy_variant": policy_variant,
                "prediction_stage": prediction_stage,
                "completion_rate_all": float(selected_summary["completion_rate_all"]),
                "priority_reward": float(selected_summary["priority_reward"]),
                "fairness": float(selected_summary["fairness"]),
                "generated_scenarios": int(len(selected_detail)),
                "empirical_trace_blocks": int(selected_detail["source_block"].nunique()),
            }
        )

    paired_detail = pd.concat(details, ignore_index=True)
    effects, inference = scheduler_two_level_inference(
        paired_detail,
        reference="final_calibrated",
        metrics=METRICS,
        repetitions=10_000,
        seed=17,
    )
    inference.insert(0, "analysis", "final_calibrated_minus_raw_base")

    block_effects = (
        effects.groupby(["metric", "source_block", "reference", "baseline"])[
            "reference_minus_baseline"
        ]
        .agg(
            mean_difference="mean",
            generated_scenarios_in_block="count",
        )
        .reset_index()
    )
    block_effects.insert(0, "analysis", "final_calibrated_minus_raw_base")
    return pd.DataFrame(summaries), inference, block_effects


def paired_arm_rows(
    tasks: pd.DataFrame, left_arm: str, right_arm: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["condition", "simulation_seed", "logical_task_id"]
    left = tasks.loc[tasks["policy_arm"].eq(left_arm)].set_index(keys).sort_index()
    right = tasks.loc[tasks["policy_arm"].eq(right_arm)].set_index(keys).sort_index()
    if not left.index.is_unique or not right.index.is_unique or not left.index.equals(right.index):
        raise ValueError(f"unpaired EOS arms: {left_arm!r} and {right_arm!r}")
    return left, right


def run_eos_audit(
    eos_candidates: Path,
    eos_task_results: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(eos_candidates)
    candidates["input_eligible_bool"] = as_bool(candidates["input_eligible"])
    candidates["resource_feasible_bool"] = as_bool(candidates["resource_feasible"])
    candidates["resource_feasible_but_input_ineligible"] = (
        candidates["resource_feasible_bool"] & ~candidates["input_eligible_bool"]
    )
    candidate_audit = (
        candidates.groupby("condition", as_index=False)
        .agg(
            candidate_actions=("condition", "size"),
            input_ineligible_candidates=(
                "input_eligible_bool",
                lambda values: int((~values).sum()),
            ),
            resource_feasible_but_input_ineligible_candidates=(
                "resource_feasible_but_input_ineligible",
                "sum",
            ),
        )
    )

    tasks = pd.read_csv(eos_task_results)
    for column in (
        "action_selected",
        "selected_input_eligible",
        "application_complete",
    ):
        tasks[column] = as_bool(tasks[column])

    screened, unscreened = paired_arm_rows(
        tasks,
        "synthetic_calibrated_score",
        "synthetic_calibrated_no_input_screening",
    )
    raw, calibrated = paired_arm_rows(
        tasks,
        "synthetic_uncalibrated_score",
        "synthetic_calibrated_score",
    )

    consequence_rows: list[dict[str, object]] = []
    for condition in CONDITION_ORDER:
        screen = screened.xs(condition, level="condition")
        no_screen = unscreened.xs(condition, level="condition")
        raw_condition = raw.xs(condition, level="condition")
        calibrated_condition = calibrated.xs(condition, level="condition")

        changed_screen = screen["selected_path_id"].fillna("__NO_ACTION__").ne(
            no_screen["selected_path_id"].fillna("__NO_ACTION__")
        )
        selected_ineligible = no_screen["action_selected"] & ~no_screen[
            "selected_input_eligible"
        ]
        changed_calibration = raw_condition["selected_path_id"].fillna(
            "__NO_ACTION__"
        ).ne(calibrated_condition["selected_path_id"].fillna("__NO_ACTION__"))

        consequence_rows.append(
            {
                "condition": condition,
                "paired_tasks": int(len(screen)),
                "screened_completed": int(screen["application_complete"].sum()),
                "no_screen_completed": int(no_screen["application_complete"].sum()),
                "no_screen_minus_screen_completed": int(
                    no_screen["application_complete"].sum()
                    - screen["application_complete"].sum()
                ),
                "no_screen_selected_input_ineligible": int(selected_ineligible.sum()),
                "no_screen_selected_input_ineligible_completed": int(
                    (selected_ineligible & no_screen["application_complete"]).sum()
                ),
                "screen_vs_no_screen_changed_selection": int(changed_screen.sum()),
                "no_screen_improved_on_changed_selection": int(
                    (
                        changed_screen
                        & ~screen["application_complete"]
                        & no_screen["application_complete"]
                    ).sum()
                ),
                "no_screen_regressed_on_changed_selection": int(
                    (
                        changed_screen
                        & screen["application_complete"]
                        & ~no_screen["application_complete"]
                    ).sum()
                ),
                "raw_completed": int(raw_condition["application_complete"].sum()),
                "calibrated_completed": int(
                    calibrated_condition["application_complete"].sum()
                ),
                "calibrated_minus_raw_completed": int(
                    calibrated_condition["application_complete"].sum()
                    - raw_condition["application_complete"].sum()
                ),
                "raw_vs_calibrated_changed_selection": int(changed_calibration.sum()),
                "calibrated_improved_on_changed_selection": int(
                    (
                        changed_calibration
                        & ~raw_condition["application_complete"]
                        & calibrated_condition["application_complete"]
                    ).sum()
                ),
                "calibrated_regressed_on_changed_selection": int(
                    (
                        changed_calibration
                        & raw_condition["application_complete"]
                        & ~calibrated_condition["application_complete"]
                    ).sum()
                ),
            }
        )

    candidate_audit["_order"] = candidate_audit["condition"].map(CONDITION_ORDER)
    candidate_audit = candidate_audit.sort_values("_order").drop(columns="_order")
    consequences = pd.DataFrame(consequence_rows)
    return candidate_audit, consequences


def wetlinks_regular_grids(wetlinks_input: Path) -> dict[str, pd.DataFrame]:
    """Rebuild the five-minute RTT/loss grids used by the episode constructor."""

    paths = sorted(wetlinks_input.glob("analysis_data_*.csv"))
    if len(paths) < 2:
        raise FileNotFoundError(
            f"expected at least two WetLinks inputs in {wetlinks_input}"
        )
    grids: dict[str, pd.DataFrame] = {}
    for path in paths:
        raw = pd.read_csv(
            path,
            usecols=["site_name", "timestamp_start", "ping_avg", "ping_packet_loss"],
        )
        raw["timestamp"] = pd.to_datetime(
            raw["timestamp_start"], errors="coerce", utc=True
        )
        raw[["ping_avg", "ping_packet_loss"]] = raw[
            ["ping_avg", "ping_packet_loss"]
        ].apply(pd.to_numeric, errors="coerce")
        site = str(raw["site_name"].dropna().iloc[0])
        regular = (
            raw.set_index("timestamp")
            .resample("5min")
            .agg(
                latency_mean_ms=("ping_avg", "mean"),
                packet_loss_pct=("ping_packet_loss", "mean"),
            )
            .reset_index()
        )
        regular["observed"] = regular["latency_mean_ms"].notna() & regular[
            "packet_loss_pct"
        ].notna()
        grids[site] = regular
    return grids


def run_gap_censoring_audit(
    wetlinks_input: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grids = wetlinks_regular_grids(wetlinks_input)
    segment_rows: list[dict[str, object]] = []
    grid_counts: dict[str, dict[str, int]] = {}
    for site, regular in grids.items():
        observed = regular["observed"].astype(bool)
        grid_counts[site] = {
            "grid_rows": int(len(regular)),
            "observed_rows": int(observed.sum()),
            "missing_rows": int((~observed).sum()),
        }
        run_id = observed.ne(observed.shift(fill_value=False)).cumsum()
        for segment_number, (_, segment) in enumerate(
            regular.loc[observed].groupby(run_id.loc[observed], sort=True), start=1
        ):
            length_bins = int(len(segment))
            segment_rows.append(
                {
                    "site": site,
                    "segment_number_within_site": segment_number,
                    "start_time_utc": segment["timestamp"].iloc[0],
                    "end_time_utc": segment["timestamp"].iloc[-1],
                    "length_bins": length_bins,
                    "length_minutes": 5 * length_bins,
                }
            )
    segments = pd.DataFrame(segment_rows)

    summary_rows: list[dict[str, object]] = []
    scopes = ("overall", "uos-rz", "utwente")
    for scope in scopes:
        if scope == "overall":
            selected = segments
            counts = {
                key: sum(site_counts[key] for site_counts in grid_counts.values())
                for key in ("grid_rows", "observed_rows", "missing_rows")
            }
        else:
            selected = segments.loc[segments["site"].eq(scope)]
            counts = grid_counts[scope]
        lengths = selected["length_bins"].to_numpy(int)
        summary_rows.append(
            {
                "scope": scope,
                **counts,
                "observed_segments": int(len(lengths)),
                "segment_mean_bins": float(np.mean(lengths)),
                "segment_median_bins": float(np.median(lengths)),
                "segment_p95_bins": float(np.percentile(lengths, 95)),
                "segment_max_bins": int(np.max(lengths)),
                "segment_mean_minutes": float(5 * np.mean(lengths)),
                "segment_median_minutes": float(5 * np.median(lengths)),
                "segment_p95_minutes": float(5 * np.percentile(lengths, 95)),
                "segment_max_minutes": int(5 * np.max(lengths)),
            }
        )

    episodes = build_wetlinks_episodes(
        wetlinks_input,
        bin_minutes=5,
        latency_limit_ms=80.0,
        loss_limit_pct=1.0,
        sustained_bad_bins=2,
    ).frame.copy()
    episodes["start_utc_hour"] = pd.to_datetime(
        episodes["start_time"], errors="coerce", utc=True
    ).dt.hour
    reasons = (
        "internal_measurement_gap",
        "qualifying_event",
        "unresolved_bad_onset_missing_confirmation",
        "recording_ended_good",
    )
    episode_rows: list[dict[str, object]] = []
    for reason in reasons:
        selected = episodes.loc[episodes["censor_reason"].eq(reason)]
        site_counts = selected["site"].value_counts()
        episode_rows.append(
            {
                "endpoint_resolution": reason,
                "episodes": int(len(selected)),
                "uos_rz_episodes": int(site_counts.get("uos-rz", 0)),
                "utwente_episodes": int(site_counts.get("utwente", 0)),
                "duration_mean_bins": float(selected["duration_bins"].mean()),
                "duration_median_bins": float(selected["duration_bins"].median()),
                "current_rtt_mean_ms": float(selected["latency_mean_ms"].mean()),
                "current_rtt_median_ms": float(selected["latency_mean_ms"].median()),
                "current_loss_mean_pct": float(selected["packet_loss_pct"].mean()),
                "current_loss_median_pct": float(selected["packet_loss_pct"].median()),
                "start_utc_hour_mean": float(selected["start_utc_hour"].mean()),
                "start_utc_hour_median": float(selected["start_utc_hour"].median()),
            }
        )
    return pd.DataFrame(summary_rows), segments, pd.DataFrame(episode_rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(
    stage_predictions: Path,
    eos_candidates: Path,
    eos_task_results: Path,
    wetlinks_input: Path,
) -> pd.DataFrame:
    sources = [
        ("WetLinks saved calibration-stage predictions", stage_predictions),
        ("EOS candidate-level scores and eligibility", eos_candidates),
        ("EOS matched task-level decisions and outcomes", eos_task_results),
        (
            "WetLinks episode/grid construction implementation",
            SOURCE_ROOT / "leo_service_survival" / "data.py",
        ),
        ("scheduler/scenario implementation", SOURCE_ROOT / "leo_service_survival" / "decisions.py"),
        ("two-level scheduler inference implementation", SOURCE_ROOT / "leo_service_survival" / "reporting.py"),
    ]
    sources.extend(
        (f"WetLinks released analysis input: {path.name}", path)
        for path in sorted(wetlinks_input.glob("analysis_data_*.csv"))
    )
    rows = []
    for role, path in sources:
        rows.append(
            {
                "source_role": role,
                # Avoid leaking local home-directory paths into a public audit.
                "source_reference": (
                    str(path.relative_to(PROJECT_ROOT))
                    if path.is_relative_to(PROJECT_ROOT)
                    else path.name
                ),
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def configuration() -> pd.DataFrame:
    entries = (
        ("scheduler_raw_prediction_stage", "raw_base"),
        ("scheduler_calibrated_prediction_stage", "final_monotone"),
        ("scheduler_horizons_bins", "1,3,6,12"),
        ("scheduler_scenarios", "120"),
        ("scheduler_tasks_per_scenario", "80"),
        ("scheduler_candidates_per_task", "4"),
        ("scheduler_horizon_slots", "48"),
        ("scheduler_satellites", "4"),
        ("scheduler_ground_stations", "2"),
        ("scheduler_congestion_weight", "0.1"),
        ("scheduler_seed", "17"),
        ("conditional_bootstrap_repetitions", "10000"),
        ("conditional_bootstrap_seed", "17"),
        ("wetlinks_grid_bin_minutes", "5"),
        (
            "wetlinks_grid_observed_definition",
            "resampled mean RTT present AND resampled mean packet loss present",
        ),
        ("wetlinks_latency_limit_ms", "80"),
        ("wetlinks_packet_loss_limit_pct", "1"),
        ("wetlinks_sustained_bad_bins", "2"),
        ("wetlinks_internal_gap_max_contiguous_bin_multiplier", "2"),
        ("wetlinks_episode_start_hour", "integer UTC clock hour at episode start"),
        ("python", platform.python_version()),
        ("numpy", np.__version__),
        ("pandas", pd.__version__),
        (
            "scope",
            "controlled trace-driven and synthetic audits; not independent field validation",
        ),
    )
    return pd.DataFrame(entries, columns=["parameter", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the focused scheduler, EOS, and WetLinks audits used in "
            "the manuscript. Row-level third-party inputs are supplied by the "
            "user and are not distributed in this repository."
        )
    )
    parser.add_argument("--stage-predictions", type=Path, required=True)
    parser.add_argument("--eos-candidates", type=Path, required=True)
    parser.add_argument("--eos-task-results", type=Path, required=True)
    parser.add_argument("--wetlinks-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "review_audit",
    )
    parser.add_argument(
        "--include-observed-segments",
        action="store_true",
        help=(
            "also export the row-level observed-segment table; omitted by "
            "default because only aggregate derivatives are published"
        ),
    )
    args = parser.parse_args()

    stage_predictions = args.stage_predictions.resolve()
    eos_candidate_path = args.eos_candidates.resolve()
    eos_task_path = args.eos_task_results.resolve()
    wetlinks_input = args.wetlinks_dir.resolve()
    output_dir = args.output_dir.resolve()
    for path in (stage_predictions, eos_candidate_path, eos_task_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not wetlinks_input.is_dir():
        raise NotADirectoryError(wetlinks_input)

    scheduler_summary, scheduler_inference, scheduler_blocks = run_scheduler_audit(
        stage_predictions
    )
    eos_candidate_audit, eos_consequences = run_eos_audit(
        eos_candidate_path, eos_task_path
    )
    gap_summary, observed_segments, endpoint_descriptives = run_gap_censoring_audit(
        wetlinks_input
    )
    outputs = {
        "wetlinks_scheduler_raw_vs_calibrated_summary.csv": scheduler_summary,
        "wetlinks_scheduler_raw_vs_calibrated_inference.csv": scheduler_inference,
        "wetlinks_scheduler_raw_vs_calibrated_block_effects.csv": scheduler_blocks,
        "eos_candidate_screen_audit.csv": eos_candidate_audit,
        "eos_selection_consequence_audit.csv": eos_consequences,
        "wetlinks_gap_grid_summary.csv": gap_summary,
        "wetlinks_episode_endpoint_descriptives.csv": endpoint_descriptives,
        "audit_source_manifest.csv": source_manifest(
            stage_predictions,
            eos_candidate_path,
            eos_task_path,
            wetlinks_input,
        ),
        "audit_configuration.csv": configuration(),
    }
    if args.include_observed_segments:
        outputs["wetlinks_observed_segment_lengths.csv"] = observed_segments

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
        print(f"wrote {name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
