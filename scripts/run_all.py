#!/usr/bin/env python3
"""Run the complete Paper 4 empirical and trace-driven evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leo_service_survival.data import (  # noqa: E402
    CENSOR_REASON_ADMINISTRATIVE_HORIZON,
    CENSOR_REASON_QUALIFYING_EVENT,
    CENSOR_REASONS,
    build_commect_landmarks,
    build_lens_episodes,
    build_palo_alto_starlink_episodes,
    build_wetlinks_episodes,
    calendar_block_split,
    merge_sparse_terminal_inference_block,
    purged_epoch_split,
)
from leo_service_survival.decisions import (  # noqa: E402
    run_commect_replay,
    run_trace_driven_scheduling,
)
from leo_service_survival.diagnostics import (  # noqa: E402
    assert_fixed_c_index_orientation,
    assert_prediction_row_alignment,
    calibrated_prediction_stages,
    calibration_fit_audit,
    calibration_stage_prediction_table,
    rolling_origin_temporal_diagnostics,
    stage_horizon_diagnostics,
    stratified_stage_horizon_diagnostics,
)
from leo_service_survival.geometry import build_geometry_qos_fusion  # noqa: E402
from leo_service_survival.metrics import (  # noqa: E402
    block_bootstrap_prediction_inference,
    calibration_curve_block_intervals,
    circular_moving_block_paired_inference,
    evaluate_survival_predictions,
    censoring_support_table,
    censoring_weighted_brier_score,
    ipcw_survival_targets,
)
from leo_service_survival.models import (  # noqa: E402
    CalibratedSurvivalModel,
    CoxPHModel,
    DiscreteHazardModel,
    KaplanMeierModel,
    build_model_suite,
    select_focal_model_configuration,
    select_baseline_model_configurations,
)
from leo_service_survival.parameters import (  # noqa: E402
    parameter_profile_manifest,
    parameter_registry,
)
from leo_service_survival.reporting import (  # noqa: E402
    calibration_intercept_slope,
    competing_risk_cif_validation,
    competing_risk_event_adequacy,
    paired_proper_score_block_inference,
    proper_score_power_planning,
    scheduler_two_level_inference,
)


def bootstrap_cox_coefficients(
    train: pd.DataFrame,
    fitted: CoxPHModel,
    features: tuple[str, ...],
    *,
    repetitions: int,
    seed: int = 17,
) -> pd.DataFrame:
    """Calendar-block bootstrap intervals for standardized Cox coefficients.

    These intervals quantify serial-block uncertainty inside the two supplied
    acquisitions; they are not multi-campaign population intervals.
    """

    group_column = "calendar_block" if "calendar_block" in train else "block_id"
    labels = train[group_column].astype(str)
    groups = labels.unique()
    parts = {group: train.loc[labels.eq(group)] for group in groups}
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for iteration in range(max(1, int(repetitions))):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat(
            [parts[group].assign(_bootstrap_copy=index) for index, group in enumerate(selected)],
            ignore_index=True,
        )
        if int(sample["event"].sum()) < 2:
            continue
        try:
            model = CoxPHModel(features).fit(sample)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        if np.all(np.isfinite(model.beta_)):
            # Re-express every bootstrap slope on the original training-set
            # standard-deviation scale; each resample otherwise has a slightly
            # different encoder scale and the coefficients are not comparable.
            common_scale_beta = (
                model.beta_ / model.encoder.scale_ * fitted.encoder.scale_
            )
            draws.append(common_scale_beta)
    values = np.vstack(draws) if draws else np.empty((0, len(features)))
    rows = []
    for index, feature in enumerate(features):
        coefficient = float(fitted.beta_[index])
        low = float(np.percentile(values[:, index], 2.5)) if len(values) else np.nan
        high = float(np.percentile(values[:, index], 97.5)) if len(values) else np.nan
        rows.append(
            {
                "feature": feature,
                "standardized_log_hazard_coefficient": coefficient,
                "hazard_ratio_per_training_sd": float(np.exp(coefficient)),
                "ci_low_log_hazard": low,
                "ci_high_log_hazard": high,
                "ci_low_hazard_ratio": float(np.exp(low)) if np.isfinite(low) else np.nan,
                "ci_high_hazard_ratio": float(np.exp(high)) if np.isfinite(high) else np.nan,
                "resampling_unit": group_column,
                "groups": int(len(groups)),
                "successful_bootstrap_fits": int(len(values)),
                "inference_scope": "serial-block uncertainty within two supplied acquisitions",
            }
        )
    return pd.DataFrame(rows)


def censoring_extreme_case_sensitivity(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    horizon: int,
) -> pd.DataFrame:
    """Expose how unknown outcomes can change the horizon-specific Brier score.

    This is deliberately an extreme-case sensitivity analysis, not an
    alternative estimator.  The primary IPCW result assumes non-informative
    censoring for the marginal censoring Kaplan--Meier estimate.
    """

    known, observed, _ = ipcw_survival_targets(frame, horizon)
    unknown = ~known
    rows = []
    for name, survival in predictions.items():
        probability = np.asarray(survival[:, horizon - 1], dtype=float)
        scenarios = {
            "all_unknown_fail": np.zeros(int(unknown.sum()), dtype=float),
            "all_unknown_survive": np.ones(int(unknown.sum()), dtype=float),
            "prediction_favorable": (probability[unknown] >= 0.5).astype(float),
            "prediction_adverse": (probability[unknown] < 0.5).astype(float),
        }
        for scenario, unknown_outcome in scenarios.items():
            outcome = observed.copy()
            outcome[unknown] = unknown_outcome
            rows.append(
                {
                    "model": name,
                    "horizon_bins": int(horizon),
                    "scenario": scenario,
                    "unknown_episodes": int(unknown.sum()),
                    "unknown_fraction": float(unknown.mean()),
                    "complete_data_brier_under_scenario": float(
                        np.mean(np.square(outcome - probability))
                    ),
                    "assumption": "extreme assignment of otherwise unobserved horizon outcomes",
                }
            )
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truncate(frame: pd.DataFrame, max_horizon: int) -> pd.DataFrame:
    out = frame.copy()
    original_duration = out["duration_bins"].astype(int).copy()
    original_event = out["event"].astype(int).copy()
    if {"start_time", "bin_seconds"}.issubset(out.columns):
        analysis_resolution = []
        for index, row in out.iterrows():
            raw_resolution = row.get("label_resolution_time", pd.NaT)
            if original_duration.iloc[index] <= max_horizon:
                if pd.notna(raw_resolution):
                    resolution = raw_resolution
                elif original_event.iloc[index] == 1:
                    resolution = row.get("event_confirmation_time", pd.NaT)
                else:
                    resolution = row.get("censoring_time", pd.NaT)
            else:
                # With a K-bin persistence endpoint, ruling out an onset at H
                # requires observations through H+K-1.  One-bin endpoints such
                # as COMMECT default to K=1 and therefore retain the old H
                # boundary.  This is label metadata only; it does not change
                # duration, event, cause, features, or model inputs.
                sustained_bad_bins = row.get("sustained_bad_bins", 1)
                sustained_bad_bins = (
                    int(sustained_bad_bins)
                    if pd.notna(sustained_bad_bins)
                    else 1
                )
                resolution = pd.to_datetime(row["start_time"], utc=True) + pd.Timedelta(
                    seconds=int(row["bin_seconds"])
                    * (max_horizon + max(sustained_bad_bins - 1, 0))
                )
            analysis_resolution.append(resolution)
        out["analysis_label_resolution_time"] = pd.to_datetime(
            analysis_resolution, utc=True
        )
        # Backward-compatible alias for existing downstream readers.  The
        # value is now confirmation-aware rather than onset/boundary-only.
        out["analysis_label_end_time"] = out["analysis_label_resolution_time"]
    beyond = out["duration_bins"].gt(max_horizon)
    # Encode administrative censoring just after the final analysed bin so it
    # contributes all H at-risk rows and is a known survivor at H.
    out.loc[beyond, "duration_bins"] = max_horizon + 1
    out.loc[beyond, "event"] = 0
    out.loc[beyond, "cause"] = "censored"
    out.loc[beyond, "censor_reason"] = CENSOR_REASON_ADMINISTRATIVE_HORIZON
    # Events observed on or before H remain events and retain their endpoint
    # provenance.  The assignment is intentionally limited to rows that were
    # already qualifying events; it never turns a censored row into an event.
    retained_events = original_event.eq(1) & ~beyond
    out.loc[retained_events, "censor_reason"] = CENSOR_REASON_QUALIFYING_EVENT
    return out.reset_index(drop=True)


def wetlinks_censor_reason_breakdown(
    partitions: dict[str, pd.DataFrame],
    *,
    horizon_bins: tuple[int, ...] = (3, 6, 9, 12),
    bin_seconds: int = 300,
) -> pd.DataFrame:
    """Return an explicit endpoint-resolution audit for retained WetLinks rows.

    Each requested horizon is evaluated under the same convention used by the
    survival analysis: outcomes beyond H are administratively censored at H+1,
    while events and other censoring mechanisms resolved on or before H retain
    their recorded reason.  Counts are reported by partition and site, with an
    additional pooled-site row.  Split embargoes are deliberately absent:
    they are exclusions from the analysis sample, not endpoint censoring.
    """

    horizon_bins = tuple(sorted({int(horizon) for horizon in horizon_bins}))
    if not horizon_bins or min(horizon_bins) < 1:
        raise ValueError("censor-reason audit horizons must be positive")
    if int(bin_seconds) < 1:
        raise ValueError("bin_seconds must be positive")
    rows: list[dict] = []
    allowed_reasons = set(CENSOR_REASONS)
    for partition, frame in partitions.items():
        required = {
            "site",
            "duration_bins",
            "event",
            "cause",
            "censor_reason",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{partition} WetLinks partition lacks endpoint audit columns: "
                + ", ".join(sorted(missing))
            )
        reasons = set(frame["censor_reason"].dropna().astype(str).unique())
        unknown_reasons = sorted(reasons.difference(allowed_reasons))
        if unknown_reasons or frame["censor_reason"].isna().any():
            raise ValueError(
                f"{partition} WetLinks partition has unrecognized censor reasons: "
                f"{unknown_reasons or ['<missing>']}"
            )
        event_reason_mismatch = frame["event"].astype(int).eq(1) != frame[
            "censor_reason"
        ].eq(CENSOR_REASON_QUALIFYING_EVENT)
        if event_reason_mismatch.any():
            raise ValueError(
                f"{partition} WetLinks partition has event/censor-reason mismatch"
            )

        for horizon in horizon_bins:
            horizon_frame = truncate(frame, horizon)
            for site_scope, scoped in [
                ("all_sites", horizon_frame),
                *[
                    (
                        str(site),
                        horizon_frame.loc[
                            horizon_frame["site"].astype(str).eq(str(site))
                        ],
                    )
                    for site in sorted(
                        horizon_frame["site"].dropna().astype(str).unique()
                    )
                ],
            ]:
                denominator = int(len(scoped))
                counts = scoped["censor_reason"].astype(str).value_counts()
                for reason in CENSOR_REASONS:
                    count = int(counts.get(reason, 0))
                    rows.append(
                        {
                            "partition": str(partition),
                            "horizon_bins": int(horizon),
                            "horizon_minutes": float(
                                horizon * int(bin_seconds) / 60.0
                            ),
                            "site_scope": site_scope,
                            "censor_reason": reason,
                            "episode_count": count,
                            "partition_horizon_site_episode_count": denominator,
                            "fraction_within_partition_horizon_site": (
                                float(count / denominator)
                                if denominator
                                else np.nan
                            ),
                            "endpoint_event": bool(
                                reason == CENSOR_REASON_QUALIFYING_EVENT
                            ),
                            "administrative_censoring_rule": (
                                "outcomes beyond H are censored at H+1"
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def write_wetlinks_endpoint_audits(
    partitions: dict[str, pd.DataFrame],
    output_dir: Path,
    *,
    horizon_bins: tuple[int, ...] = (3, 6, 9, 12),
    bin_seconds: int = 300,
) -> tuple[pd.DataFrame, dict]:
    """Serialize endpoint-censoring counts and split exclusions separately."""

    output_dir.mkdir(parents=True, exist_ok=True)
    breakdown = wetlinks_censor_reason_breakdown(
        partitions,
        horizon_bins=horizon_bins,
        bin_seconds=bin_seconds,
    )
    breakdown.to_csv(
        output_dir / "wetlinks_censor_reason_breakdown.csv", index=False
    )

    partition_audits = {
        str(name): frame.attrs.get("split_exclusion_audit")
        for name, frame in partitions.items()
    }
    if any(audit is None for audit in partition_audits.values()):
        missing = sorted(
            name for name, audit in partition_audits.items() if audit is None
        )
        raise ValueError(
            "WetLinks split exclusion audit is missing for partition(s): "
            + ", ".join(missing)
        )
    serialized = {
        "scope": "analysis-sample exclusions; not endpoint censoring",
        "consistent_across_partitions": bool(
            len(
                {
                    json.dumps(audit, sort_keys=True)
                    for audit in partition_audits.values()
                }
            )
            == 1
        ),
        "partition_audits": partition_audits,
    }
    if not serialized["consistent_across_partitions"]:
        raise ValueError("WetLinks split exclusion audits disagree across partitions")
    (output_dir / "wetlinks_split_exclusion_audit.json").write_text(
        json.dumps(serialized, indent=2), encoding="utf-8"
    )
    return breakdown, serialized


def fit_suite(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    features: tuple[str, ...],
    max_horizon: int,
    *,
    focal_hyperparameters: dict[str, float | int] | None = None,
    calibration_c: float = 0.1,
    baseline_hyperparameters: dict[str, object] | None = None,
):
    suite = build_model_suite(
        features,
        max_horizon=max_horizon,
        focal_hyperparameters=focal_hyperparameters,
        calibration_c=calibration_c,
        baseline_hyperparameters=baseline_hyperparameters,
    )
    fit_seconds: dict[str, float] = {}
    fit_peak_python_mb: dict[str, float] = {}
    for name, model in suite.models.items():
        tracemalloc.start()
        start = perf_counter()
        model.fit(train)
        if isinstance(model, CalibratedSurvivalModel):
            model.calibrate(calibration)
        fit_seconds[name] = perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fit_peak_python_mb[name] = float(peak / (1024.0**2))
    suite.fit_seconds = fit_seconds
    suite.fit_peak_python_mb = fit_peak_python_mb
    return suite


def development_tripartition(
    frame: pd.DataFrame,
    *,
    group_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological inner split used only for parameter selection."""

    groups = sorted(frame[group_column].astype(str).unique())
    if len(groups) < 5:
        raise ValueError("at least five development groups are required for inner selection")
    first = max(1, int(np.floor(0.6 * len(groups))))
    second = max(first + 1, int(np.floor(0.8 * len(groups))))
    second = min(second, len(groups) - 1)
    fit_groups = set(groups[:first])
    calibration_groups = set(groups[first:second])
    validation_groups = set(groups[second:])
    labels = frame[group_column].astype(str)
    parts = (
        frame.loc[labels.isin(fit_groups)].copy(),
        frame.loc[labels.isin(calibration_groups)].copy(),
        frame.loc[labels.isin(validation_groups)].copy(),
    )
    if any(part.empty for part in parts):
        raise ValueError("inner development partition produced an empty split")
    return parts


def fit_proposed(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    features: tuple[str, ...],
    max_horizon: int,
    *,
    random_state: int = 17,
    focal_hyperparameters: dict[str, float | int] | None = None,
    calibration_c: float = 0.1,
) -> CalibratedSurvivalModel:
    model = CalibratedSurvivalModel(
        DiscreteHazardModel(
            features,
            max_horizon=max_horizon,
            nonlinear=True,
            competing=True,
            random_state=random_state,
            hyperparameters=focal_hyperparameters,
        ),
        range(1, max_horizon + 1),
        calibration_c=calibration_c,
    )
    model.fit(train)
    model.calibrate(calibration)
    return model


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def switching_penalty_sensitivity(
    decisions: pd.DataFrame,
    scheduled_epochs: np.ndarray,
    *,
    costs_seconds: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0),
) -> pd.DataFrame:
    """Assumed switching-blackout sensitivity on the one-second replay grid.

    The metric is an effective successful-service fraction, not binary task
    completion and not a measured handover delay.  Fractional costs are kept
    because the source data have only one-second resolution.
    """

    rows = []
    grid = pd.DataFrame({"decision_epoch": scheduled_epochs})
    for policy, policy_rows in decisions.groupby("policy", sort=True):
        aligned = grid.merge(
            policy_rows[["decision_epoch", "success", "chosen_path"]],
            on="decision_epoch",
            how="left",
            validate="one_to_one",
        )
        success = aligned["success"].fillna(0.0).to_numpy(float)
        path = aligned["chosen_path"].fillna("no_action").astype(str).to_numpy()
        switches = np.zeros(len(aligned), dtype=bool)
        if len(aligned) > 1:
            switches[1:] = (
                (path[1:] != "no_action")
                & (path[:-1] != "no_action")
                & (path[1:] != path[:-1])
            )
        switch_locations = np.flatnonzero(switches)
        for cost in costs_seconds:
            blackout = np.zeros(len(aligned), dtype=float)
            for location in switch_locations:
                remaining = float(cost)
                cursor = int(location)
                while remaining > 0 and cursor < len(blackout):
                    allocation = min(1.0, remaining)
                    blackout[cursor] = min(1.0, blackout[cursor] + allocation)
                    remaining -= allocation
                    cursor += 1
            effective = success * (1.0 - blackout)
            rows.append(
                {
                    "policy": policy,
                    "assumed_switch_cost_seconds": float(cost),
                    "scheduled_seconds": int(len(success)),
                    "switches": int(switches.sum()),
                    "observed_conservative_success": float(success.mean()),
                    "effective_successful_service_fraction": float(effective.mean()),
                    "interpretation": (
                        "assumed blackout sensitivity at one-second trace resolution; "
                        "not measured handover cost or binary task completion"
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_survival_bands(
    frame: pd.DataFrame,
    survival: np.ndarray,
    horizons: tuple[int, ...],
    *,
    repetitions: int,
    seed: int = 17,
) -> pd.DataFrame:
    group_column = (
        "analysis_block"
        if "analysis_block" in frame
        else (
            "calendar_block"
            if "calendar_block" in frame
            else ("block_id" if "block_id" in frame else "campaign_id")
        )
    )
    groups = frame[group_column].astype(str).unique()
    indices = {
        group: np.flatnonzero(frame[group_column].astype(str).to_numpy() == group)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    means = []
    rmst = []
    for _ in range(repetitions):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sample = np.concatenate([indices[group] for group in selected])
        curve = survival[sample].mean(axis=0)
        means.append(curve)
        rmst.append(float(1.0 + curve[:-1].sum()))
    values = np.asarray(means)
    rows = []
    for column, horizon in enumerate(horizons):
        rows.append(
            {
                "horizon_bins": horizon,
                "mean_survival": float(survival[:, column].mean()),
                "ci_low": float(np.percentile(values[:, column], 2.5)),
                "ci_high": float(np.percentile(values[:, column], 97.5)),
                "resampling_unit": group_column,
                "groups": int(len(groups)),
            }
        )
    rows.append(
        {
            "horizon_bins": 0,
            "mean_survival": float(1.0 + survival[:, :-1].mean(axis=0).sum()),
            "ci_low": float(np.percentile(rmst, 2.5)),
            "ci_high": float(np.percentile(rmst, 97.5)),
            "resampling_unit": f"{group_column}:restricted_mean_bins",
            "groups": int(len(groups)),
        }
    )
    return pd.DataFrame(rows)


def benchmark_scoring(
    model: CalibratedSurvivalModel,
    frame: pd.DataFrame,
    *,
    sizes: tuple[int, ...] = (100, 1000, 10000, 50000),
) -> pd.DataFrame:
    rows = []
    warmup = frame.iloc[np.arange(256) % len(frame)].reset_index(drop=True)
    model.predict_survival(warmup, [1, 6, 12])
    for size in sizes:
        indices = np.arange(size) % len(frame)
        sample = frame.iloc[indices].reset_index(drop=True)
        timings = []
        for _ in range(5):
            start = perf_counter()
            model.predict_survival(sample, [1, 6, 12])
            timings.append((perf_counter() - start) * 1000.0)
        median = float(np.median(timings))
        rows.append(
            {
                "candidates": size,
                "median_scoring_ms": median,
                "candidates_per_second": float(size / max(median / 1000.0, 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def summarize_geometry_fusion(fusion: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare ranking adapters in the controlled geometry/QoS pairing."""

    decisions = []
    for start_time, group in fusion.groupby("start_time_utc", sort=True):
        if len(group) < 2:
            continue
        selections = {
            "geometry_longest": group.sort_values("geometry_duration_s", ascending=False).iloc[0],
            "paper1_engineering_utility": group.sort_values(
                "paper1_engineering_utility", ascending=False
            ).iloc[0],
            "fused_empirical_survival": group.sort_values(
                ["task_completion_probability", "fused_empirical_eust_s"],
                ascending=[False, False],
            ).iloc[0],
        }
        for policy, row in selections.items():
            decisions.append(
                {
                    "start_time_utc": start_time,
                    "policy": policy,
                    "known_outcome": int(row["measured_outcome_known"]),
                    "completion": int(row["controlled_task_completion"]),
                    "chosen_satellite_id": row["satellite_id"],
                }
            )
    detail = pd.DataFrame(decisions)
    summaries = []
    for policy, group in detail.groupby("policy", sort=True):
        known = group["known_outcome"].eq(1)
        summaries.append(
            {
                "policy": policy,
                "choice_sets": int(len(group)),
                "known_outcomes": int(known.sum()),
                "complete_case_completion": float(group.loc[known, "completion"].mean())
                if known.any()
                else np.nan,
                "conservative_completion": float(group["completion"].mean()),
            }
        )
    return detail, pd.DataFrame(summaries)


def save_prediction_figure(metrics: pd.DataFrame, output: Path) -> None:
    order = metrics.sort_values("integrated_brier")["model"].tolist()
    frame = metrics.set_index("model").loc[order]
    labels = [label.replace("_", "\n") for label in order]
    colors = ["#315E8A" if label == "calibrated_competing_risk" else "#A7B5C4" for label in order]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(np.arange(len(frame)), frame["integrated_brier"], color=colors)
    ax.set_xticks(np.arange(len(frame)), labels, fontsize=8)
    ax.set_ylabel("IPCW integrated Brier score (lower is better)")
    ax.set_title("Held-out WetLinks service-survival accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_calibration_figure(
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    output: Path,
    *,
    proposed_curve: pd.DataFrame | None = None,
    horizon_index: int = 5,
) -> None:
    horizon = horizon_index + 1
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    if proposed_curve is not None and len(proposed_curve):
        curve = proposed_curve.loc[
            proposed_curve["horizon_bins"].eq(horizon)
        ].sort_values("bin")
        lower = np.clip(
            curve["observed_survival_ipcw"] - curve["observed_ci_low"],
            0.0,
            None,
        )
        upper = np.clip(
            curve["observed_ci_high"] - curve["observed_survival_ipcw"],
            0.0,
            None,
        )
        ax.errorbar(
            curve["predicted_survival"],
            curve["observed_survival_ipcw"],
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color="#315E8A",
            label="calibrated competing risk (descriptive 95% block interval)",
        )
    else:
        raise ValueError("a calibration-partition-frozen proposed curve is required")
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="ideal")
    ax.set(xlabel="Predicted 30-minute survival", ylabel="Observed survival", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Probability calibration on held-out episodes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_policy_figure(summary: pd.DataFrame, output: Path, title: str, metric: str) -> None:
    ordered = summary.sort_values(metric, ascending=False)
    colors = ["#315E8A" if "survival" in policy else "#A7B5C4" for policy in ordered["policy"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(np.arange(len(ordered)), ordered[metric], color=colors)
    ax.set_xticks(np.arange(len(ordered)), [value.replace("_", "\n") for value in ordered["policy"]], fontsize=8)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--figure-dir", default=str(ROOT / "figures"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=400)
    parser.add_argument("--scheduling-scenarios", type=int, default=120)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow writing into non-empty output/figure directories (engineering checks only)",
    )
    parser.add_argument(
        "--wetlinks-dir",
        default=str(ROOT / "upstream/paper2/data/external/wetlinks_dataset/Preprocessed_Data"),
        help="Directory containing the two WetLinks analysis_data_*.csv tables",
    )
    parser.add_argument(
        "--palo-alto-dir",
        default=str(ROOT / "data/external/palo_alto_starlink"),
        help=(
            "Directory containing the 12 Palo Alto Starlink daily ping tables used "
            "for the exploratory fixed-model transport-stress evaluation"
        ),
    )
    parser.add_argument(
        "--commect-path",
        default=str(ROOT / "data/external/commect_multiaccess_1s.csv"),
        help="Processed concurrent COMMECT one-second trace",
    )
    parser.add_argument(
        "--lens-path",
        default=str(ROOT / "upstream/paper2/results/temporal_forecasting/temporal_forecast_predictions.csv"),
        help="Released LENS-derived prediction artifact used only for diagnostics",
    )
    parser.add_argument(
        "--paper1-windows-path",
        default=str(ROOT / "upstream/paper1/regional_starlink_risk_aware_visibility.csv"),
        help="Paper 1 TLE-derived geometry-window table",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    for label, directory in (("output", output_dir), ("figure", figure_dir)):
        if directory.exists() and any(directory.iterdir()) and not args.allow_overwrite:
            raise FileExistsError(
                f"Refusing to mix a new run with non-empty {label} directory {directory}. "
                "Choose a fresh directory, or use --allow-overwrite for a disposable engineering check."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    parameter_registry().to_csv(output_dir / "parameter_registry.csv", index=False)

    wetlinks_dir = Path(args.wetlinks_dir).expanduser().resolve()
    palo_alto_dir = Path(args.palo_alto_dir).expanduser().resolve()
    commect_path = Path(args.commect_path).expanduser().resolve()
    lens_path = Path(args.lens_path).expanduser().resolve()
    paper1_windows_path = Path(args.paper1_windows_path).expanduser().resolve()
    required_inputs = [commect_path, lens_path, paper1_windows_path]
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if not wetlinks_dir.is_dir():
        missing.append(str(wetlinks_dir))
    if not palo_alto_dir.is_dir():
        missing.append(str(palo_alto_dir))
    if missing:
        raise FileNotFoundError(
            "Required input(s) not found. Supply the corresponding command-line path(s): "
            + ", ".join(missing)
        )
    palo_alto_files = sorted(palo_alto_dir.glob("ping_metrics_2024-05-*.*"))
    if len(palo_alto_files) != 12:
        raise FileNotFoundError(
            "The exploratory Palo Alto transport-stress input requires exactly 12 "
            f"daily ping tables matching ping_metrics_2024-05-*.* in {palo_alto_dir}; "
            f"found {len(palo_alto_files)}. Supply a complete --palo-alto-dir."
        )
    max_horizon = 12
    horizons = tuple(range(1, max_horizon + 1))

    wetlinks = build_wetlinks_episodes(wetlinks_dir)
    wet_frame = truncate(wetlinks.frame, max_horizon)
    wet_train, wet_calibration, wet_test = calendar_block_split(wet_frame)
    write_wetlinks_endpoint_audits(
        {
            "train": wet_train,
            "calibration": wet_calibration,
            "test": wet_test,
        },
        output_dir,
        horizon_bins=(3, 6, 9, 12),
        bin_seconds=wetlinks.bin_seconds,
    )
    wet_test, wet_analysis_block_audit = merge_sparse_terminal_inference_block(
        wet_test,
        group_column="calendar_block",
        output_column="analysis_block",
        minimum_fraction_of_prior_median=0.25,
    )
    # Capture the immutable test-row key before any model is called.  Every
    # exported prediction matrix is checked against this exact order below.
    wet_test_episode_order = wet_test["episode_id"].astype(str).to_numpy(copy=True)
    wet_split_audit = {
        "train_blocks": sorted(wet_train["calendar_block"].unique().tolist()),
        "calibration_blocks": sorted(wet_calibration["calendar_block"].unique().tolist()),
        "test_blocks": sorted(wet_test["calendar_block"].unique().tolist()),
        "train_end": wet_train["start_time"].max().isoformat(),
        "calibration_start": wet_calibration["start_time"].min().isoformat(),
        "calibration_end": wet_calibration["start_time"].max().isoformat(),
        "test_start": wet_test["start_time"].min().isoformat(),
        "embargo_blocks_per_boundary": 1,
        "train_label_resolution_end": wet_train[
            "analysis_label_resolution_time"
        ].max().isoformat(),
        "calibration_label_resolution_end": wet_calibration[
            "analysis_label_resolution_time"
        ].max().isoformat(),
        # Retain the historical keys for machine readers; they now carry the
        # same confirmation-aware values as the explicit resolution fields.
        "train_analysis_label_end": wet_train[
            "analysis_label_resolution_time"
        ].max().isoformat(),
        "calibration_analysis_label_end": wet_calibration[
            "analysis_label_resolution_time"
        ].max().isoformat(),
        "label_lookahead_disjoint": bool(
            wet_train["analysis_label_resolution_time"].max()
            < wet_calibration["start_time"].min()
            and wet_calibration["analysis_label_resolution_time"].max()
            < wet_test["start_time"].min()
        ),
        "inference_block_rule": wet_analysis_block_audit,
    }
    if (
        set(wet_split_audit["train_blocks"]) & set(wet_split_audit["calibration_blocks"])
        or set(wet_split_audit["calibration_blocks"]) & set(wet_split_audit["test_blocks"])
    ):
        raise AssertionError("WetLinks calendar blocks overlap across partitions")
    if not wet_split_audit["label_lookahead_disjoint"]:
        raise AssertionError("WetLinks label look-ahead crosses a retained split boundary")
    (output_dir / "wetlinks_split_audit.json").write_text(
        json.dumps(wet_split_audit, indent=2), encoding="utf-8"
    )
    inner_fit, inner_calibration, inner_validation = development_tripartition(
        wet_train,
        group_column="calendar_block",
    )
    focal_hyperparameters, focal_calibration_c, focal_selection = (
        select_focal_model_configuration(
            inner_fit,
            inner_calibration,
            inner_validation,
            wetlinks.feature_columns,
            max_horizon=max_horizon,
        )
    )
    focal_selection.to_csv(
        output_dir / "wetlinks_focal_hyperparameter_selection.csv", index=False
    )
    baseline_hyperparameters, baseline_selection = (
        select_baseline_model_configurations(
            inner_fit,
            inner_validation,
            wetlinks.feature_columns,
            max_horizon=max_horizon,
        )
    )
    baseline_selection.to_csv(
        output_dir / "wetlinks_baseline_hyperparameter_selection.csv", index=False
    )
    suite = fit_suite(
        wet_train,
        wet_calibration,
        wetlinks.feature_columns,
        max_horizon,
        focal_hyperparameters=focal_hyperparameters,
        calibration_c=focal_calibration_c,
        baseline_hyperparameters=baseline_hyperparameters,
    )
    calibration_fit_audit(
        suite.models[suite.proposed_name],
        wet_calibration,
        horizons,
        bin_seconds=wetlinks.bin_seconds,
    ).to_csv(
        output_dir / "wetlinks_calibration_fit_audit.csv",
        index=False,
    )
    bootstrap_cox_coefficients(
        wet_train,
        suite.models["cox_ph"],
        tuple(wetlinks.feature_columns),
        repetitions=min(args.bootstrap_repetitions, 200),
    ).to_csv(output_dir / "wetlinks_cox_coefficients.csv", index=False)
    metric_rows = []
    runtime_rows = []
    predictions: dict[str, np.ndarray] = {}
    for name, model in suite.models.items():
        tracemalloc.start()
        prediction_start = perf_counter()
        prediction = model.predict_survival(wet_test, horizons)
        prediction_seconds = perf_counter() - prediction_start
        _, prediction_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        predictions[name] = prediction
        metrics = evaluate_survival_predictions(wet_test, prediction, horizons)
        metric_rows.append({"model": name, **metrics})
        runtime_rows.append(
            {
                "model": name,
                "fit_seconds": suite.fit_seconds[name],
                "fit_peak_python_tracemalloc_mb": suite.fit_peak_python_mb[name],
                "test_prediction_ms": prediction_seconds * 1000.0,
                "prediction_peak_python_tracemalloc_mb": float(
                    prediction_peak / (1024.0**2)
                ),
                "serialized_model_mb": float(
                    len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
                    / (1024.0**2)
                ),
                "test_episodes": len(wet_test),
                "memory_scope": (
                    "Python allocator peak and serialized object size; native-library and "
                    "whole-process RSS are not included"
                ),
            }
        )
    metrics_frame = pd.DataFrame(metric_rows).sort_values("integrated_brier")
    metrics_frame.to_csv(output_dir / "wetlinks_model_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output_dir / "wetlinks_model_runtime.csv", index=False)

    prediction_alignment_audit = assert_prediction_row_alignment(
        wet_test,
        predictions,
        horizons,
        expected_episode_ids=wet_test_episode_order,
    )
    (output_dir / "wetlinks_prediction_alignment_audit.json").write_text(
        json.dumps(prediction_alignment_audit, indent=2),
        encoding="utf-8",
    )
    c_index_audits = []
    metric_lookup = metrics_frame.set_index("model")
    for name, prediction in predictions.items():
        audit = assert_fixed_c_index_orientation(
            wet_test,
            prediction,
            reported_c_index=float(metric_lookup.loc[name, "c_index"]),
        )
        c_index_audits.append({"model": name, **audit})
    pd.DataFrame(c_index_audits).to_csv(
        output_dir / "wetlinks_c_index_orientation_audit.csv",
        index=False,
    )

    proposed_model = suite.models[suite.proposed_name]
    proposed_stages = calibrated_prediction_stages(
        proposed_model,
        wet_test,
        horizons,
    )
    # These identities ensure the stage audit refers to the actual primary
    # run, not a separately reconstructed or reordered prediction.
    if not np.allclose(
        proposed_stages["raw_base"],
        predictions["competing_risk_uncalibrated"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("raw proposed stage differs from uncalibrated base export")
    if not np.allclose(
        proposed_stages["final_monotone"],
        predictions[suite.proposed_name],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("final proposed stage differs from primary prediction export")
    calibration_stage_prediction_table(
        wet_test,
        proposed_stages,
        horizons,
        bin_seconds=wetlinks.bin_seconds,
    ).to_csv(
        output_dir / "wetlinks_calibration_stage_predictions.csv",
        index=False,
    )
    stage_horizon_diagnostics(
        wet_test,
        proposed_stages,
        horizons,
    ).to_csv(
        output_dir / "wetlinks_calibration_stage_diagnostics.csv",
        index=False,
    )
    stratified_stage_horizon_diagnostics(
        wet_test,
        proposed_stages,
        horizons,
        stratum_columns=("site", "analysis_block"),
    ).to_csv(
        output_dir / "wetlinks_calibration_stage_site_block_diagnostics.csv",
        index=False,
    )

    rolling_diagnostics, rolling_folds = rolling_origin_temporal_diagnostics(
        wet_frame,
        wetlinks.feature_columns,
        max_horizon=max_horizon,
        block_column="calendar_block",
        minimum_train_blocks=8,
        calibration_blocks=4,
        embargo_blocks=1,
        test_blocks=1,
        random_state=17,
    )
    rolling_diagnostics.to_csv(
        output_dir / "wetlinks_rolling_origin_stage_diagnostics.csv",
        index=False,
    )
    rolling_folds.to_csv(
        output_dir / "wetlinks_rolling_origin_fold_definitions.csv",
        index=False,
    )

    support = pd.concat(
        [
            censoring_support_table(wet_test, (1, 3, 6, 9, 12)),
            censoring_support_table(
                wet_test, (1, 3, 6, 9, 12), stratum_column="site"
            ),
            censoring_support_table(
                wet_test, (1, 3, 6, 9, 12), stratum_column="analysis_block"
            ),
        ],
        ignore_index=True,
    )
    support.to_csv(output_dir / "wetlinks_censoring_support.csv", index=False)

    ipcw_sensitivity_rows = []
    for model_name in (
        suite.proposed_name,
        "competing_risk_uncalibrated",
        "cox_ph",
    ):
        prediction = predictions[model_name]
        for maximum_horizon in (3, 6, 9, 12):
            for weighting, stratum_column in (
                ("marginal", None),
                ("site_stratified", "site"),
                ("calendar_block_stratified", "analysis_block"),
            ):
                scores = [
                    censoring_weighted_brier_score(
                        wet_test,
                        prediction[:, horizon - 1],
                        horizon,
                        stratum_column=stratum_column,
                    )
                    for horizon in range(1, maximum_horizon + 1)
                ]
                ipcw_sensitivity_rows.append(
                    {
                        "model": model_name,
                        "maximum_horizon_bins": maximum_horizon,
                        "maximum_horizon_minutes": maximum_horizon * 5,
                        "censoring_weighting": weighting,
                        "integrated_brier": float(np.mean(scores)),
                        "assumption": (
                            "independent censoring marginally"
                            if stratum_column is None
                            else f"independent censoring conditional on {stratum_column}"
                        ),
                    }
                )
    pd.DataFrame(ipcw_sensitivity_rows).to_csv(
        output_dir / "wetlinks_ipcw_sensitivity.csv", index=False
    )

    # Apply the already fitted WetLinks models, including any WetLinks-only
    # calibration, to a separate Palo Alto ping record.  No Palo Alto outcome or
    # covariate is used to refit, recalibrate, select, or tune a model.  Because
    # this record lacks most WetLinks covariates and has only one short vantage
    # point, these artifacts are an exploratory transport-stress evaluation,
    # not an independent confirmatory validation or a superiority experiment.
    palo_alto = build_palo_alto_starlink_episodes(palo_alto_dir)
    palo_alto_untruncated = palo_alto.frame.copy()
    palo_alto_frame = truncate(palo_alto_untruncated, max_horizon)
    palo_alto_predictions: dict[str, np.ndarray] = {}
    palo_alto_metric_rows = []
    for name, model in suite.models.items():
        prediction = model.predict_survival(palo_alto_frame, horizons)
        palo_alto_predictions[name] = prediction
        palo_alto_metric_rows.append(
            {
                "model": name,
                **evaluate_survival_predictions(
                    palo_alto_frame,
                    prediction,
                    horizons,
                ),
                "training_source": "WetLinks training partition only",
                "calibration_source": (
                    "WetLinks calibration partition only for calibrated models"
                ),
                "palo_alto_refit_or_recalibration": False,
                "evidence_role": "exploratory external transport-stress evaluation",
                "interpretation": (
                    "descriptive under strong covariate and event-rate shift; no "
                    "confirmatory or model-superiority inference"
                ),
            }
        )
    pd.DataFrame(palo_alto_metric_rows).sort_values("integrated_brier").to_csv(
        output_dir / "palo_alto_transport_stress_metrics.csv",
        index=False,
    )
    palo_alto_metric_uncertainty, _ = block_bootstrap_prediction_inference(
        palo_alto_frame,
        palo_alto_predictions,
        horizons,
        reference=suite.proposed_name,
        repetitions=max(args.bootstrap_repetitions, 400),
    )
    palo_alto_metric_uncertainty["evidence_role"] = (
        "descriptive three-week transport-stress variation; not confirmatory"
    )
    palo_alto_metric_uncertainty.to_csv(
        output_dir / "palo_alto_transport_stress_metric_uncertainty.csv",
        index=False,
    )

    palo_alto_prediction_metadata = palo_alto_frame[
        [
            "episode_id",
            "site",
            "campaign_id",
            "block_id",
            "calendar_block",
            "start_time",
            "duration_bins",
            "event",
            "cause",
        ]
    ].reset_index(drop=True)
    palo_alto_prediction_columns = {
        f"{name}_s{horizon}": prediction[:, column]
        for name, prediction in palo_alto_predictions.items()
        for column, horizon in enumerate(horizons)
    }
    palo_alto_prediction_export = pd.concat(
        [
            palo_alto_prediction_metadata,
            pd.DataFrame(palo_alto_prediction_columns),
        ],
        axis=1,
    )
    palo_alto_prediction_export.to_csv(
        output_dir / "palo_alto_transport_stress_predictions.csv",
        index=False,
    )

    recorded_features = {
        "latency_mean_ms",
        "latency_std_ms",
        "latency_max_ms",
        "packet_loss_pct",
    }
    derived_features = {"latency_roll3", "loss_roll3"}
    palo_alto_feature_rows = []
    for feature in palo_alto.feature_columns:
        values = pd.to_numeric(palo_alto_frame[feature], errors="coerce")
        finite = np.isfinite(values.to_numpy(float))
        nonmissing = int(finite.sum())
        if feature in recorded_features:
            origin = "recorded ping field aggregated to five-minute bin"
        elif feature in derived_features:
            origin = "derived from recorded ping history"
        else:
            origin = "unavailable in Palo Alto record"
        palo_alto_feature_rows.append(
            {
                "feature": feature,
                "origin": origin,
                "nonmissing_episodes": nonmissing,
                "missing_episodes": int(len(palo_alto_frame) - nonmissing),
                "availability_fraction": float(nonmissing / len(palo_alto_frame)),
                "coverage_status": (
                    "complete"
                    if nonmissing == len(palo_alto_frame)
                    else "unavailable"
                    if nonmissing == 0
                    else "partial"
                ),
                "covariate_model_handling": (
                    "WetLinks-training median; no Palo Alto fit"
                    if nonmissing == 0
                    else "observed value where finite; WetLinks-training median otherwise"
                ),
            }
        )
    palo_alto_feature_coverage = pd.DataFrame(palo_alto_feature_rows)
    palo_alto_feature_coverage.to_csv(
        output_dir / "palo_alto_transport_stress_feature_coverage.csv",
        index=False,
    )

    palo_alto_evaluable = {
        str(horizon): int(ipcw_survival_targets(palo_alto_frame, horizon)[0].sum())
        for horizon in horizons
    }
    palo_alto_counts = {
        "evidence_role": "exploratory external transport-stress evaluation",
        "source_record": "Zenodo 10.5281/zenodo.14987305",
        "source_daily_files": int(len(palo_alto_files)),
        "episodes": int(len(palo_alto_frame)),
        "events_in_full_episode_record": int(palo_alto_untruncated["event"].sum()),
        "events_within_60_minute_analysis_horizon": int(palo_alto_frame["event"].sum()),
        "non_event_records_after_horizon_truncation": int(
            palo_alto_frame["event"].eq(0).sum()
        ),
        "known_survivors_through_60_minute_horizon": int(
            palo_alto_frame["duration_bins"].gt(max_horizon).sum()
        ),
        "censored_with_unknown_status_at_or_before_horizon": int(
            (
                palo_alto_frame["event"].eq(0)
                & palo_alto_frame["duration_bins"].le(max_horizon)
            ).sum()
        ),
        "analysis_event_fraction": float(palo_alto_frame["event"].mean()),
        "event_categories_within_horizon": {
            str(key): int(value)
            for key, value in palo_alto_frame.loc[
                palo_alto_frame["event"].eq(1), "cause"
            ].value_counts().items()
        },
        "sites": int(palo_alto_frame["site"].nunique()),
        "calendar_blocks": int(palo_alto_frame["calendar_block"].nunique()),
        "first_episode_timestamp_as_parsed": palo_alto_frame["start_time"].min().isoformat(),
        "last_episode_timestamp_as_parsed": palo_alto_frame["start_time"].max().isoformat(),
        "timestamp_note": (
            "source timestamps have no stated time zone; UTC attachment is a parsing "
            "convention, and cyclical clock-time features are intentionally unavailable"
        ),
        "analysis_horizon_bins": int(max_horizon),
        "bin_seconds": int(palo_alto.bin_seconds),
        "evaluable_episodes_by_horizon": palo_alto_evaluable,
        "features_total": int(len(palo_alto.feature_columns)),
        "features_unavailable": int(
            palo_alto_feature_coverage["coverage_status"].eq("unavailable").sum()
        ),
        "features_partial": int(
            palo_alto_feature_coverage["coverage_status"].eq("partial").sum()
        ),
        "features_complete": int(
            palo_alto_feature_coverage["coverage_status"].eq("complete").sum()
        ),
        "model_application": (
            "all models fixed after WetLinks training/calibration; no Palo Alto refit, "
            "recalibration, tuning, or model selection"
        ),
        "interpretation": (
            "descriptive performance under cross-geography transport and covariate "
            "shift; not confirmatory and not evidence of model superiority"
        ),
    }
    (output_dir / "palo_alto_transport_stress_counts.json").write_text(
        json.dumps(palo_alto_counts, indent=2),
        encoding="utf-8",
    )

    metric_uncertainty, paired_metric_inference = block_bootstrap_prediction_inference(
        wet_test,
        predictions,
        horizons,
        reference=suite.proposed_name,
        repetitions=args.bootstrap_repetitions,
        group_column="analysis_block",
    )
    metric_uncertainty.to_csv(
        output_dir / "wetlinks_model_metric_uncertainty.csv",
        index=False,
    )
    adjusted_parts = []
    for _, group in paired_metric_inference.groupby("metric", sort=False):
        group = group.copy()
        group["p_holm_within_metric"] = holm_adjust(
            group["p_exact_block_swap"].tolist()
        )
        adjusted_parts.append(group)
    paired_metric_inference = pd.concat(adjusted_parts, ignore_index=True)
    paired_metric_inference.to_csv(
        output_dir / "wetlinks_paired_metric_inference.csv",
        index=False,
    )
    proper_score_blocks, proper_score_inference = paired_proper_score_block_inference(
        wet_test,
        predictions,
        horizons,
        reference=suite.proposed_name,
        group_column="analysis_block",
    )
    proper_score_blocks.to_csv(
        output_dir / "wetlinks_paired_score_blocks.csv",
        index=False,
    )
    proper_score_inference.to_csv(
        output_dir / "wetlinks_paired_score_inference.csv",
        index=False,
    )
    proper_score_power_planning(proper_score_inference).to_csv(
        output_dir / "proper_score_power_planning.csv",
        index=False,
    )
    proposed_calibration_curves = []
    for calibration_horizon in (1, 6, 12):
        proposed_calibration_curves.append(
            calibration_curve_block_intervals(
                wet_test,
                predictions[suite.proposed_name][:, calibration_horizon - 1],
                calibration_horizon,
                calibration_probability=suite.models[
                    suite.proposed_name
                ].predict_survival(
                    wet_calibration,
                    [calibration_horizon],
                )[:, 0],
                bins=6,
                repetitions=args.bootstrap_repetitions,
                group_column="analysis_block",
            )
        )
    proposed_calibration_curve = pd.concat(
        proposed_calibration_curves,
        ignore_index=True,
    )
    proposed_calibration_curve.to_csv(
        output_dir / "wetlinks_calibration_curve_uncertainty.csv",
        index=False,
    )
    calibration_intercept_slope(
        wet_test,
        predictions[suite.proposed_name][:, [0, 5, 11]],
        (1, 6, 12),
        repetitions=max(args.bootstrap_repetitions, 400),
        group_column="analysis_block",
    ).to_csv(
        output_dir / "wetlinks_calibration_parameters.csv",
        index=False,
    )

    event_counts = competing_risk_event_adequacy(
        {
            "train": wet_train,
            "calibration": wet_calibration,
            "test": wet_test,
        },
        design_columns=len(wetlinks.feature_columns) + 3,
    )
    event_counts.to_csv(
        output_dir / "competing_risk_event_adequacy.csv",
        index=False,
    )
    censoring_extreme_case_sensitivity(
        wet_test,
        predictions,
        horizon=max_horizon,
    ).to_csv(output_dir / "wetlinks_censoring_sensitivity.csv", index=False)

    bootstrap_survival_bands(
        wet_test,
        predictions[suite.proposed_name],
        horizons,
        repetitions=args.bootstrap_repetitions,
    ).to_csv(output_dir / "wetlinks_survival_uncertainty_bands.csv", index=False)

    weather_features = {
        "temperature_c",
        "humidity_pct",
        "rain_mm",
        "wind_mps",
        "pressure_hpa",
        "rain_latency_interaction",
    }
    interaction_features = {"rain_latency_interaction"}
    ablation_specs = {
        "full_calibrated_competing_risk": tuple(wetlinks.feature_columns),
        "without_weather": tuple(column for column in wetlinks.feature_columns if column not in weather_features),
        "without_rain_latency_interaction": tuple(
            column for column in wetlinks.feature_columns if column not in interaction_features
        ),
    }
    ablation_rows = []
    for ablation_name, features in ablation_specs.items():
        if ablation_name == "full_calibrated_competing_risk":
            prediction = predictions[suite.proposed_name]
        else:
            ablation_model = fit_proposed(
                wet_train,
                wet_calibration,
                features,
                max_horizon,
                focal_hyperparameters=focal_hyperparameters,
                calibration_c=focal_calibration_c,
            )
            prediction = ablation_model.predict_survival(wet_test, horizons)
        ablation_rows.append(
            {
                "ablation": ablation_name,
                "feature_count": len(features),
                **evaluate_survival_predictions(wet_test, prediction, horizons),
            }
        )
    pd.DataFrame(ablation_rows).to_csv(output_dir / "wetlinks_feature_ablation.csv", index=False)

    sensitivity_rows = []
    if not args.skip_sensitivity:
        for latency_limit in (70.0, 80.0, 90.0):
            for sustained_bins in (1, 2):
                if latency_limit == 80.0 and sustained_bins == 2:
                    sensitivity_frame = wet_frame
                    sensitivity_test = wet_test
                    proposed_prediction = predictions[suite.proposed_name]
                    km_prediction = predictions["kaplan_meier"]
                else:
                    sensitivity_data = build_wetlinks_episodes(
                        wetlinks_dir,
                        latency_limit_ms=latency_limit,
                        sustained_bad_bins=sustained_bins,
                    )
                    sensitivity_frame = truncate(sensitivity_data.frame, max_horizon)
                    sensitivity_train, sensitivity_calibration, sensitivity_test = calendar_block_split(
                        sensitivity_frame
                    )
                    km_model = KaplanMeierModel().fit(sensitivity_train)
                    km_prediction = km_model.predict_survival(sensitivity_test, horizons)
                    try:
                        sensitivity_model = fit_proposed(
                            sensitivity_train,
                            sensitivity_calibration,
                            sensitivity_data.feature_columns,
                            max_horizon,
                            focal_hyperparameters=focal_hyperparameters,
                            calibration_c=focal_calibration_c,
                        )
                        proposed_prediction = sensitivity_model.predict_survival(
                            sensitivity_test, horizons
                        )
                        proposed_status = "estimable"
                    except ValueError as error:
                        # A changed endpoint can make a cause class too sparse
                        # for the multiclass learner.  Preserve that negative
                        # result instead of changing the endpoint or dropping
                        # the condition until the model fits.
                        proposed_prediction = None
                        proposed_status = f"not_estimable: {error}"
                for model_name, prediction in (
                    ("calibrated_competing_risk", proposed_prediction),
                    ("kaplan_meier", km_prediction),
                ):
                    if prediction is None:
                        metrics = {
                            "integrated_brier": np.nan,
                            "mean_calibration_error": np.nan,
                            "binary_cross_entropy": np.nan,
                            "restricted_mean_time_mae_bins": np.nan,
                            "c_index": np.nan,
                            "n_test_episodes": int(len(sensitivity_test)),
                            "event_fraction": float(sensitivity_test["event"].mean()),
                        }
                        status = proposed_status
                    else:
                        metrics = evaluate_survival_predictions(
                            sensitivity_test,
                            prediction,
                            horizons,
                        )
                        status = "estimable"
                    sensitivity_rows.append(
                        {
                            "latency_limit_ms": latency_limit,
                            "sustained_bad_bins": sustained_bins,
                            "episodes": int(len(sensitivity_frame)),
                            "overall_event_fraction": float(sensitivity_frame["event"].mean()),
                            "model": model_name,
                            "estimation_status": status,
                            **metrics,
                        }
                    )
    pd.DataFrame(sensitivity_rows).to_csv(
        output_dir / "event_definition_sensitivity.csv", index=False
    )

    proposed = suite.models[suite.proposed_name]
    incidence = proposed.predict_cumulative_incidence(wet_test, max_horizon)
    incidence_frame = pd.DataFrame(
        [{"cause": cause, "mean_cumulative_incidence_h12": float(values.mean())} for cause, values in incidence.items()]
    )
    incidence_frame.to_csv(output_dir / "wetlinks_cause_incidence.csv", index=False)
    predicted_cif_by_horizon = {
        horizon: proposed.predict_cumulative_incidence(wet_test, horizon)
        for horizon in (1, 6, 12)
    }
    competing_risk_cif_validation(
        wet_test,
        predicted_cif_by_horizon,
        repetitions=max(args.bootstrap_repetitions, 400),
        group_column="analysis_block",
    ).to_csv(
        output_dir / "competing_risk_cif_validation.csv",
        index=False,
    )

    paper1_windows = pd.read_csv(paper1_windows_path)
    geometry_fusion = build_geometry_qos_fusion(
        paper1_windows,
        wet_test,
        proposed,
        horizons=horizons,
        bin_seconds=wetlinks.bin_seconds,
        task_seconds=wetlinks.bin_seconds,
    )
    geometry_fusion.to_csv(output_dir / "paper1_geometry_qos_fusion.csv", index=False)
    fusion_detail, fusion_summary = summarize_geometry_fusion(geometry_fusion)
    fusion_detail.to_csv(output_dir / "paper1_geometry_qos_decisions.csv", index=False)
    fusion_summary.to_csv(output_dir / "paper1_geometry_qos_summary.csv", index=False)

    prediction_export = wet_test[["episode_id", "site", "campaign_id", "duration_bins", "event", "cause"]].copy()
    for name, prediction in predictions.items():
        prediction_export[f"{name}_s6"] = prediction[:, 5]
        prediction_export[f"{name}_s12"] = prediction[:, 11]
    prediction_export.to_csv(output_dir / "wetlinks_test_predictions.csv", index=False)

    site_rows = []
    sites = sorted(wet_frame["site"].unique())
    for held_site in sites:
        source = wet_frame[wet_frame["site"].ne(held_site)].sort_values("start_time")
        split = max(1, int(len(source) * 0.8))
        source_train, source_calibration = source.iloc[:split], source.iloc[split:]
        held_test = wet_frame[wet_frame["site"].eq(held_site)].copy()
        site_suite = fit_suite(
            source_train,
            source_calibration,
            wetlinks.feature_columns,
            max_horizon,
            focal_hyperparameters=focal_hyperparameters,
            calibration_c=focal_calibration_c,
            baseline_hyperparameters=baseline_hyperparameters,
        )
        for name in ("kaplan_meier", "cox_ph", suite.proposed_name):
            prediction = site_suite.models[name].predict_survival(held_test, horizons)
            site_rows.append(
                {
                    "held_out_site": held_site,
                    "model": name,
                    **evaluate_survival_predictions(held_test, prediction, horizons),
                }
            )
    pd.DataFrame(site_rows).to_csv(output_dir / "wetlinks_leave_one_site_out.csv", index=False)

    lens = build_lens_episodes(lens_path, latency_limit_ms=60.0)
    lens_frame = truncate(lens.frame, max_horizon)
    # Consecutive LENS-derived landmarks share up to 30 future bins, while the
    # small released artifact has sessions too short to support a 30-bin purge
    # plus disjoint train/calibration/test partitions.  Report provenance and
    # label diagnostics only; do not present a leakage-prone survival score.
    pd.DataFrame(
        [
            {
                "records": int(len(lens_frame)),
                "events": int(lens_frame["event"].sum()),
                "sites": int(lens_frame["site"].nunique()),
                "sessions": int(lens_frame["campaign_id"].nunique()),
                "survival_evaluation_performed": False,
                "reason": "overlapping 30-bin landmarks and insufficient session length for a purged three-way split",
                "evidence_role": "released derived-artifact provenance diagnostic only",
            }
        ]
    ).to_csv(output_dir / "lens_derived_artifact_diagnostic.csv", index=False)

    commect = build_commect_landmarks(commect_path)
    commect_frame = truncate(commect.frame, 10)
    commect_train, commect_calibration, commect_test = purged_epoch_split(
        commect_frame,
        purge_bins=30,
    )
    split_audit = {
        "train_end_epoch": int(commect_train["decision_epoch"].max()),
        "calibration_start_epoch": int(commect_calibration["decision_epoch"].min()),
        "calibration_end_epoch": int(commect_calibration["decision_epoch"].max()),
        "test_start_epoch": int(commect_test["decision_epoch"].min()),
        "label_lookahead_bins": 30,
        "train_calibration_gap": int(
            commect_calibration["decision_epoch"].min() - commect_train["decision_epoch"].max()
        ),
        "calibration_test_gap": int(
            commect_test["decision_epoch"].min() - commect_calibration["decision_epoch"].max()
        ),
    }
    if split_audit["train_calibration_gap"] <= 30 or split_audit["calibration_test_gap"] <= 30:
        raise AssertionError("COMMECT temporal purge is shorter than label look-ahead")
    (output_dir / "commect_split_audit.json").write_text(
        json.dumps(split_audit, indent=2), encoding="utf-8"
    )
    commect_suite = fit_suite(commect_train, commect_calibration, commect.feature_columns, 10)
    commect_proposed = commect_suite.models[commect_suite.proposed_name]
    decisions, commect_summary = run_commect_replay(commect_test, commect_proposed)
    decisions.to_csv(output_dir / "commect_decisions.csv", index=False)
    commect_summary.to_csv(output_dir / "commect_policy_summary.csv", index=False)
    # Put every policy on the same complete scheduled one-second grid.  Missing
    # actions and unevaluable futures receive zero only for this explicitly
    # conservative outcome, matching the summary-table denominator.
    scheduled_epochs = np.arange(
        int(commect_test["decision_epoch"].min()),
        int(commect_test["decision_epoch"].max()) + 1,
    )
    policy_names = sorted(decisions["policy"].unique())
    commect_comparison_input = (
        pd.MultiIndex.from_product(
            [scheduled_epochs, policy_names],
            names=["decision_epoch", "policy"],
        )
        .to_frame(index=False)
        .merge(
            decisions[["decision_epoch", "policy", "success"]],
            on=["decision_epoch", "policy"],
            how="left",
            validate="one_to_one",
        )
    )
    commect_comparison_input["conservative_success"] = commect_comparison_input[
        "success"
    ].fillna(0.0)
    commect_grid = commect_comparison_input.pivot(
        index="decision_epoch",
        columns="policy",
        values="conservative_success",
    ).sort_index()
    commect_effect_rows = []
    commect_inference_rows = []
    commect_reference = "survival_candidate"
    time_series_repetitions = max(10_000, args.bootstrap_repetitions * 25)
    for baseline in sorted(
        policy for policy in commect_grid.columns if policy != commect_reference
    ):
        difference = (
            commect_grid[commect_reference] - commect_grid[baseline]
        ).to_numpy(float)
        for epoch, value in zip(commect_grid.index, difference):
            commect_effect_rows.append(
                {
                    "decision_epoch": int(epoch),
                    "reference": commect_reference,
                    "baseline": baseline,
                    "metric": "conservative_success",
                    "reference_minus_baseline": float(value),
                }
            )
        inference = circular_moving_block_paired_inference(
            commect_grid[commect_reference].to_numpy(float),
            commect_grid[baseline].to_numpy(float),
            block_lengths=(30, 60, 120),
            repetitions=time_series_repetitions,
            seed=17,
        )
        inference.insert(0, "metric", "conservative_success")
        inference.insert(0, "baseline", baseline)
        inference.insert(0, "reference", commect_reference)
        commect_inference_rows.append(inference)
    pd.DataFrame(commect_effect_rows).to_csv(
        output_dir / "commect_paired_epoch_effects.csv",
        index=False,
    )
    pd.concat(commect_inference_rows, ignore_index=True).to_csv(
        output_dir / "commect_time_series_inference.csv",
        index=False,
    )
    switching_penalty_sensitivity(
        decisions,
        scheduled_epochs,
    ).to_csv(output_dir / "commect_switching_penalty_sensitivity.csv", index=False)

    scheduler_tuning_rows = []
    tuning_scenarios = max(20, min(40, args.scheduling_scenarios // 3 or 20))
    for congestion_weight in (0.0, 0.1, 0.25, 0.5, 1.0):
        _, tuning_summary = run_trace_driven_scheduling(
            wet_calibration,
            proposed,
            scenarios=tuning_scenarios,
            survival_congestion_weight=congestion_weight,
            seed=1017,
        )
        survival_row = tuning_summary[
            tuning_summary["policy"].eq("survival_aware")
        ].iloc[0]
        scheduler_tuning_rows.append(
            {
                "survival_congestion_weight": congestion_weight,
                "calibration_completion_rate": float(survival_row["completion_rate_all"]),
                "calibration_priority_reward": float(survival_row["priority_reward"]),
            }
        )
    scheduler_tuning = pd.DataFrame(scheduler_tuning_rows)
    selected_weight = float(
        scheduler_tuning.sort_values(
            ["calibration_completion_rate", "survival_congestion_weight"],
            ascending=[False, True],
        ).iloc[0]["survival_congestion_weight"]
    )
    scheduler_tuning["selected"] = scheduler_tuning["survival_congestion_weight"].eq(
        selected_weight
    )
    scheduler_tuning.to_csv(output_dir / "scheduler_calibration_tuning.csv", index=False)

    scheduler_test = wet_test.copy()
    scheduler_test["calendar_block"] = scheduler_test["analysis_block"]
    schedule_detail, schedule_summary = run_trace_driven_scheduling(
        scheduler_test,
        proposed,
        scenarios=args.scheduling_scenarios,
        survival_congestion_weight=selected_weight,
    )
    schedule_detail.to_csv(output_dir / "trace_driven_scheduling_detail.csv", index=False)
    schedule_summary.to_csv(output_dir / "trace_driven_scheduling_summary.csv", index=False)
    scheduler_effects, scheduler_inference = scheduler_two_level_inference(
        schedule_detail,
        reference="survival_aware",
        metrics=("completion_rate_all", "priority_reward", "fairness"),
        repetitions=max(10_000, args.bootstrap_repetitions * 25),
        seed=17,
    )
    scheduler_effects.to_csv(
        output_dir / "scheduler_paired_scenario_effects.csv",
        index=False,
    )
    scheduler_inference.to_csv(
        output_dir / "scheduler_inference.csv",
        index=False,
    )

    benchmark_scoring(proposed, wet_test).to_csv(output_dir / "online_scoring_scalability.csv", index=False)

    save_prediction_figure(metrics_frame, figure_dir / "wetlinks_model_comparison.png")
    save_calibration_figure(
        wet_test,
        predictions,
        figure_dir / "wetlinks_calibration.png",
        proposed_curve=proposed_calibration_curve,
    )
    save_policy_figure(
        commect_summary,
        figure_dir / "commect_policy_comparison.png",
        "COMMECT five-second task completion replay",
        "complete_case_success",
    )
    save_policy_figure(
        schedule_summary,
        figure_dir / "scheduling_comparison.png",
        "Trace-driven multi-user scheduling stress test",
        "completion_rate_all",
    )

    pd.DataFrame(
        [
            {
                "dataset": "WetLinks empirical",
                "records": len(wet_frame),
                "events": int(wet_frame["event"].sum()),
                "sites_or_paths": int(wet_frame["site"].nunique()),
                "role": "primary service-survival fit/calibration/test",
            },
            {
                "dataset": "Palo Alto Starlink ping record",
                "records": len(palo_alto_frame),
                "events": int(palo_alto_frame["event"].sum()),
                "sites_or_paths": int(palo_alto_frame["site"].nunique()),
                "role": (
                    "exploratory external transport-stress evaluation with fixed "
                    "WetLinks-trained models"
                ),
            },
            {
                "dataset": "COMMECT concurrent multi-access",
                "records": len(commect_frame),
                "events": int(commect_frame["event"].sum()),
                "sites_or_paths": int(commect_frame["relative_path"].nunique()),
                "role": "single-campaign purged shadow replay",
            },
            {
                "dataset": "LENS-derived released prediction artifact",
                "records": len(lens_frame),
                "events": int(lens_frame["event"].sum()),
                "sites_or_paths": int(lens_frame["site"].nunique()),
                "role": "exploratory diagnostic only",
            },
            {
                "dataset": "Paper 1 TLE-derived windows",
                "records": len(paper1_windows),
                "events": np.nan,
                "sites_or_paths": int(paper1_windows["satellite_id"].nunique()),
                "role": "controlled deterministic geometry bound",
            },
        ]
    ).to_csv(output_dir / "dataset_roles_and_counts.csv", index=False)

    source_files = (
        sorted(wetlinks_dir.glob("analysis_data_*.csv"))
        + palo_alto_files
        + [
            commect_path,
            lens_path,
            paper1_windows_path,
        ]
    )
    manifest = {
        "title": "From Visibility to Completion",
        "run_kind": (
            "manuscript_exploratory_snapshot"
            if args.bootstrap_repetitions >= 400
            and args.scheduling_scenarios >= 120
            and not args.skip_sensitivity
            else "engineering_check"
        ),
        "arguments": {
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "scheduling_scenarios": int(args.scheduling_scenarios),
            "skip_sensitivity": bool(args.skip_sensitivity),
            "allow_overwrite": bool(args.allow_overwrite),
            "wetlinks_dir": str(wetlinks_dir),
            "palo_alto_dir": str(palo_alto_dir),
            "commect_path": str(commect_path),
            "lens_path": str(lens_path),
            "paper1_windows_path": str(paper1_windows_path),
        },
        "parameter_profiles": parameter_profile_manifest(),
        "focal_parameter_selection": {
            "source": "chronological inner tripartition of WetLinks training blocks only",
            "selected_hazard_hyperparameters": focal_hyperparameters,
            "selected_calibration_c": focal_calibration_c,
            "selected_baseline_hyperparameters": baseline_hyperparameters,
            "test_outcomes_used": False,
            "artifacts": [
                "wetlinks_focal_hyperparameter_selection.csv",
                "wetlinks_baseline_hyperparameter_selection.csv",
            ],
        },
        "random_seeds": {
            "models_and_bootstraps": 17,
            "scheduler_tuning": 1017,
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "xgboost": xgboost.__version__,
        "wetlinks_episodes": int(len(wet_frame)),
        "wetlinks_train": int(len(wet_train)),
        "wetlinks_calibration": int(len(wet_calibration)),
        "wetlinks_test": int(len(wet_test)),
        "wetlinks_events": int(wet_frame["event"].sum()),
        "wetlinks_sites": sorted(wet_frame["site"].astype(str).unique().tolist()),
        "wetlinks_independent_acquisitions": int(wet_frame["campaign_id"].nunique()),
        "wetlinks_serial_blocks": int(wet_frame["block_id"].nunique()),
        "wetlinks_calendar_blocks": int(wet_frame["calendar_block"].nunique()),
        "wetlinks_split_audit": wet_split_audit,
        "wetlinks_prediction_alignment_audit": prediction_alignment_audit,
        "wetlinks_calibration_stage_exports": [
            "wetlinks_calibration_fit_audit.csv",
            "wetlinks_calibration_stage_predictions.csv",
            "wetlinks_calibration_stage_diagnostics.csv",
            "wetlinks_calibration_stage_site_block_diagnostics.csv",
        ],
        "wetlinks_rolling_origin_design": {
            "minimum_train_blocks": 8,
            "calibration_blocks": 4,
            "embargo_blocks_at_each_boundary": 1,
            "test_blocks_per_fold": 1,
            "folds": int(len(rolling_folds)),
            "risk_orientation": "1 - survival at maximum horizon",
            "test_informed_inversion": False,
        },
        "palo_alto_source_record": "10.5281/zenodo.14987305",
        "palo_alto_source_files": int(len(palo_alto_files)),
        "palo_alto_episodes": int(len(palo_alto_frame)),
        "palo_alto_events_within_horizon": int(palo_alto_frame["event"].sum()),
        "palo_alto_sites": sorted(
            palo_alto_frame["site"].astype(str).unique().tolist()
        ),
        "palo_alto_evidence_role": (
            "exploratory external transport-stress evaluation; all predictions use "
            "fixed WetLinks-trained/calibrated models with no Palo Alto refit"
        ),
        "commect_landmarks": int(len(commect_frame)),
        "commect_split_audit": split_audit,
        "lens_episodes": int(len(lens_frame)),
        "event_definition": "first onset of two contiguous observed bins that violate the same QoS limit (RTT>80ms or packet loss>1%); mixed latency-only/loss-only pairs do not qualify; unconfirmed candidate onsets and measurement gaps are administratively censored",
        "event_time_convention": "duration is measured from a usable landmark to the first violating bin; the following bin confirms persistence and is recorded separately",
        "censoring_assumption": "primary IPCW scores use a marginal Kaplan-Meier censoring model and assume non-informative censoring; site- and calendar-block-stratified weighting plus extreme missing-outcome assignments are exported separately",
        "validation_status": "exploratory: held-out calendar weeks were inspected during method development; independent external confirmation remains required",
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted(
                [*SRC.rglob("*.py"), *Path(ROOT / "scripts").rglob("*.py"), *Path(ROOT / "tests").rglob("*.py")]
            )
        },
        "limitations": [
            "WetLinks does not identify the serving satellite or handover cause.",
            "WetLinks contains two nearby temperate-European sites, not polar, tropical, and arid independent campaigns.",
            "Calendar-week block bootstrap intervals describe serial-block uncertainty and are not evidence from independent operator campaigns.",
            "The Palo Alto record is one short ICMP-only vantage point with no throughput, weather, serving-satellite, beam, or stated timestamp-time-zone fields; its outputs are an exploratory transport-stress evaluation, not confirmatory validation.",
            "Palo Alto covariates unavailable from the source record are replaced only by medians learned from WetLinks training data; this severe covariate and event-rate shift limits metric interpretation.",
            "COMMECT contains one acquisition campaign and supports shadow replay only.",
            "The LENS table is a small released prediction artifact, not raw multi-site survival telemetry.",
            "The multi-user scheduler is a trace-driven controlled stress benchmark, not an operational EO deployment.",
            "Paper 1 TLE windows are fused only as deterministic geometry bounds; no RF or QoS event is inferred from TLE data.",
        ],
    }
    artifact_paths = [
        *((f"results/{path.name}", path) for path in sorted(output_dir.glob("*.csv"))),
        *((f"results/{path.name}", path) for path in sorted(output_dir.glob("*.json"))),
        *((f"figures/{path.name}", path) for path in sorted(figure_dir.glob("*.png"))),
    ]
    manifest["artifact_sha256_before_manifest"] = {
        label: sha256(path)
        for label, path in artifact_paths
        if path.name != "run_manifest.json"
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(metrics_frame.to_string(index=False))
    print(commect_summary.to_string(index=False))
    print(schedule_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
