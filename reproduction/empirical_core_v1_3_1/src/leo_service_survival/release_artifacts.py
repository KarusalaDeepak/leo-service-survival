"""Machine-readable release artifacts for the held-out WetLinks comparison.

The manuscript reports compact aggregate tables, but an external audit also
needs to establish that every declared pipeline was evaluated on the same rows
at every requested horizon.  The builders in this module expose that chain
without serializing executable model objects or redistributing source traces.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json

import numpy as np
import pandas as pd

from .diagnostics import (
    _weighted_calibration_in_the_large,
    _weighted_calibration_intercept_slope,
)
from .metrics import expected_calibration_error, ipcw_survival_targets


EXPECTED_PIPELINES = (
    "constant_exponential",
    "kaplan_meier",
    "weibull_aft",
    "cox_ph",
    "additive_discrete_hazard",
    "linear_discrete_hazard",
    "boosted_discrete_hazard",
    "random_forest_discrete_hazard",
    "xgboost_discrete_hazard",
    "calibrated_single_risk",
    "competing_risk_uncalibrated",
    "calibrated_competing_risk",
)

CALIBRATED_PIPELINES = {
    "calibrated_single_risk",
    "calibrated_competing_risk",
}


def _episode_digest(identifiers: Iterable[str]) -> str:
    payload = json.dumps(
        [str(value) for value in identifiers],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in horizons)
    if not values or min(values) < 1:
        raise ValueError("prediction horizons must be positive integers")
    if values != tuple(range(1, max(values) + 1)):
        raise ValueError("release horizons must be the complete consecutive grid 1..H")
    return values


def _validate_prediction_set(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    horizons: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if "episode_id" not in frame:
        raise ValueError("release frame requires episode_id")
    identifiers = frame["episode_id"].astype(str).to_numpy()
    if len(pd.unique(identifiers)) != len(identifiers):
        raise ValueError("episode_id must be unique in the held-out release frame")

    received = set(predictions)
    expected = set(EXPECTED_PIPELINES)
    if received != expected:
        raise ValueError(
            "prediction set must contain exactly the 12 declared pipelines; "
            f"missing={sorted(expected - received)}, extra={sorted(received - expected)}"
        )

    validated: dict[str, np.ndarray] = {}
    expected_shape = (len(frame), len(horizons))
    for pipeline in EXPECTED_PIPELINES:
        values = np.asarray(predictions[pipeline], dtype=float)
        if values.shape != expected_shape:
            raise ValueError(
                f"{pipeline} prediction shape {values.shape} does not match {expected_shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{pipeline} predictions contain non-finite values")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{pipeline} predictions fall outside [0, 1]")
        validated[pipeline] = values
    return identifiers, validated


def build_prediction_score_release(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    horizons: Iterable[int],
    *,
    bin_seconds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build row-level predictions and horizon-specific proper-score inputs.

    The long prediction table carries the observed/evaluable status and the
    exact IPCW contribution used by the pointwise Brier score.  A censored
    unknown row is retained with a zero IPCW contribution and a missing target;
    it is never silently converted to a failure or survivor.
    """

    horizons = _validate_horizons(horizons)
    if int(bin_seconds) < 1:
        raise ValueError("bin_seconds must be positive")
    identifiers, validated = _validate_prediction_set(frame, predictions, horizons)
    order_digest = _episode_digest(identifiers)
    test_set_id = f"wetlinks-heldout-{order_digest[:16]}"

    identity_columns = [
        column
        for column in (
            "episode_id",
            "site",
            "campaign_id",
            "calendar_block",
            "analysis_block",
            "duration_bins",
            "event",
            "cause",
            "censor_reason",
        )
        if column in frame
    ]
    identity = frame[identity_columns].reset_index(drop=True).copy()
    identity.insert(0, "test_row_index", np.arange(len(identity), dtype=int))
    identity.insert(0, "test_set_id", test_set_id)

    duration = frame["duration_bins"].to_numpy(int)
    event = frame["event"].to_numpy(int)
    prediction_parts: list[pd.DataFrame] = []
    score_rows: list[dict[str, object]] = []
    previous_by_pipeline: dict[str, np.ndarray] = {}
    briers_by_pipeline: dict[str, list[float]] = {
        pipeline: [] for pipeline in EXPECTED_PIPELINES
    }

    for column, horizon in enumerate(horizons):
        known, target, weights = ipcw_survival_targets(frame, horizon)
        failed = (event == 1) & (duration <= horizon)
        survived = duration > horizon
        outcome_status = np.full(len(frame), "censored_unknown", dtype=object)
        outcome_status[failed] = "failure_by_horizon"
        outcome_status[survived] = "survived_through_horizon"
        target_export = np.where(known, target, np.nan)
        weight_sum = float(weights.sum())

        for pipeline in EXPECTED_PIPELINES:
            probability = validated[pipeline][:, column]
            clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
            squared_loss = np.where(known, np.square(target - clipped), np.nan)
            brier_contribution = weights * np.square(target - clipped)
            log_loss = -target * np.log(clipped) - (1.0 - target) * np.log(
                1.0 - clipped
            )
            weighted_log_loss = weights * log_loss

            part = identity.copy()
            part["pipeline"] = pipeline
            part["horizon_bins"] = int(horizon)
            part["horizon_minutes"] = float(horizon * int(bin_seconds) / 60.0)
            part["predicted_survival"] = probability
            part["score_probability_clipped_1e6"] = clipped
            part["outcome_status"] = outcome_status
            part["evaluable_at_horizon"] = known
            part["observed_survival_target"] = target_export
            part["ipcw_weight"] = weights
            part["unweighted_squared_error"] = squared_loss
            part["ipcw_brier_contribution"] = brier_contribution
            part["ipcw_log_loss_contribution"] = weighted_log_loss
            prediction_parts.append(part)

            if weight_sum > 0.0:
                score_status = "computed"
                failure_reason = ""
                pointwise_brier = float(np.mean(brier_contribution))
                pointwise_log_loss = float(weighted_log_loss.sum() / weight_sum)
                ece = float(
                    expected_calibration_error(
                        clipped[known],
                        target[known],
                        bins=10,
                        sample_weight=weights[known],
                    )
                )
                observed_survival = float(
                    np.sum(weights[known] * target[known]) / weight_sum
                )
                calibration_intercept, calibration_slope = (
                    _weighted_calibration_intercept_slope(
                        clipped[known], target[known], weights[known]
                    )
                )
                calibration_in_the_large = _weighted_calibration_in_the_large(
                    clipped[known], target[known], weights[known]
                )
                if len(probability[known]) < 5:
                    calibration_status = "not_estimable"
                    calibration_failure_reason = "fewer_than_five_evaluable_rows"
                elif len(np.unique(target[known])) < 2:
                    calibration_status = "not_estimable"
                    calibration_failure_reason = "single_class_evaluable_targets"
                elif not np.isfinite(
                    [
                        calibration_intercept,
                        calibration_slope,
                        calibration_in_the_large,
                    ]
                ).all():
                    calibration_status = "not_estimable"
                    calibration_failure_reason = "weighted_logistic_solver_failed"
                else:
                    calibration_status = "computed"
                    calibration_failure_reason = ""
            else:
                score_status = "not_estimable"
                failure_reason = "no_positive_ipcw_weight_at_horizon"
                pointwise_brier = np.nan
                pointwise_log_loss = np.nan
                ece = np.nan
                observed_survival = np.nan
                calibration_intercept = np.nan
                calibration_slope = np.nan
                calibration_in_the_large = np.nan
                calibration_status = "not_estimable"
                calibration_failure_reason = "no_positive_ipcw_weight_at_horizon"

            briers_by_pipeline[pipeline].append(pointwise_brier)

            previous = previous_by_pipeline.get(pipeline)
            monotonicity_violations = (
                0
                if previous is None
                else int(np.sum(probability > previous + 1e-12))
            )
            previous_by_pipeline[pipeline] = probability
            score_rows.append(
                {
                    "test_set_id": test_set_id,
                    "pipeline": pipeline,
                    "horizon_bins": int(horizon),
                    "horizon_minutes": float(horizon * int(bin_seconds) / 60.0),
                    "test_episodes": int(len(frame)),
                    "evaluable_episodes": int(known.sum()),
                    "failures_by_horizon": int(failed.sum()),
                    "known_survivors_through_horizon": int(survived.sum()),
                    "censored_unknown_by_horizon": int((~known).sum()),
                    "ipcw_weight_sum": weight_sum,
                    "pointwise_ipcw_brier": pointwise_brier,
                    "equally_weighted_integrated_brier_through_horizon": float(
                        np.mean(briers_by_pipeline[pipeline])
                    ),
                    "pointwise_ipcw_log_loss": pointwise_log_loss,
                    "ipcw_ece_10_equal_width_bins": ece,
                    "ipcw_observed_survival": observed_survival,
                    "calibration_intercept_free_slope": calibration_intercept,
                    "calibration_slope": calibration_slope,
                    "calibration_in_the_large_fixed_slope_one": (
                        calibration_in_the_large
                    ),
                    "prediction_min": float(probability.min()),
                    "prediction_max": float(probability.max()),
                    "prediction_mean": float(probability.mean()),
                    "prediction_std": float(probability.std()),
                    "monotonicity_violations_from_previous_horizon": (
                        monotonicity_violations
                    ),
                    "score_status": score_status,
                    "failure_reason": failure_reason,
                    "calibration_diagnostic_status": calibration_status,
                    "calibration_diagnostic_failure_reason": (
                        calibration_failure_reason
                    ),
                    "brier_aggregation_rule": (
                        "mean over all held-out episodes of IPCW weight times squared error"
                    ),
                }
            )

    prediction_table = pd.concat(prediction_parts, ignore_index=True)
    score_table = pd.DataFrame(score_rows)
    alignment = {
        "schema_version": "1.0",
        "status": "passed",
        "test_set_id": test_set_id,
        "episodes": int(len(frame)),
        "horizons": list(horizons),
        "pipelines": list(EXPECTED_PIPELINES),
        "prediction_rows": int(len(prediction_table)),
        "episode_id_unique": True,
        "episode_order_sha256": order_digest,
        "row_order_rule": (
            "test_row_index is the captured chronological test-frame order; every "
            "pipeline-horizon pair repeats exactly that ordered episode_id vector"
        ),
        "assertion": (
            "all 12 pipelines contain one finite [0,1] prediction for each held-out "
            "episode and each consecutive horizon"
        ),
    }
    return prediction_table, score_table, alignment


def build_pipeline_execution_status(
    configuration: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    alignment: Mapping[str, object],
    *,
    calibration_fit_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join declared pipeline configuration to observed release-run status."""

    required_configuration = {
        "method",
        "inputs",
        "time_representation",
        "censoring_handling",
        "cause_handling",
        "hyperparameters",
        "tuning_data",
        "calibration",
        "stopping_rule",
        "software",
    }
    missing_configuration = required_configuration.difference(configuration.columns)
    if missing_configuration:
        raise ValueError(
            "pipeline configuration missing "
            f"{sorted(missing_configuration)}"
        )
    if set(configuration["method"].astype(str)) != set(EXPECTED_PIPELINES):
        raise ValueError("pipeline configuration does not cover exactly 12 pipelines")
    if configuration["method"].astype(str).duplicated().any():
        raise ValueError("pipeline configuration contains duplicate methods")
    if set(predictions) != set(EXPECTED_PIPELINES):
        raise ValueError("execution status requires predictions from all 12 pipelines")

    calibration_summary: dict[str, tuple[int, str]] = {}
    if calibration_fit_audit is not None:
        required = {"horizon_bins", "raw_fallback_used", "raw_fallback_reason"}
        missing = required.difference(calibration_fit_audit.columns)
        if missing:
            raise ValueError(f"calibration fit audit missing {sorted(missing)}")
        if "pipeline" not in calibration_fit_audit:
            raise ValueError("calibration fit audit requires a pipeline column")
        unknown = set(calibration_fit_audit["pipeline"].astype(str)).difference(
            CALIBRATED_PIPELINES
        )
        if unknown:
            raise ValueError(
                f"calibration fit audit contains unexpected pipelines {sorted(unknown)}"
            )
        if calibration_fit_audit.duplicated(["pipeline", "horizon_bins"]).any():
            raise ValueError("calibration fit audit contains duplicate pipeline horizons")
        expected_horizons = {int(value) for value in alignment["horizons"]}
        for pipeline, group in calibration_fit_audit.groupby("pipeline", sort=False):
            observed_horizons = set(group["horizon_bins"].astype(int))
            fallback = group["raw_fallback_used"].astype(bool)
            reasons: list[str] = []
            missing_horizons = expected_horizons.difference(observed_horizons)
            extra_horizons = observed_horizons.difference(expected_horizons)
            if missing_horizons:
                reasons.append(
                    "missing_horizons=" + "|".join(map(str, sorted(missing_horizons)))
                )
            if extra_horizons:
                reasons.append(
                    "unexpected_horizons=" + "|".join(map(str, sorted(extra_horizons)))
                )
            if fallback.any():
                reasons.append(
                    "fallback="
                    + "|".join(
                    sorted(
                        set(
                            group.loc[fallback, "raw_fallback_reason"]
                            .astype(str)
                            .tolist()
                        )
                    )
                    )
                )
            calibration_summary[str(pipeline)] = (
                int((~fallback).sum()),
                ";".join(reasons),
            )

    rows: list[dict[str, object]] = []
    config_by_method = configuration.set_index("method", drop=False)
    for pipeline in EXPECTED_PIPELINES:
        config = config_by_method.loc[pipeline]
        is_calibrated = pipeline in CALIBRATED_PIPELINES
        if is_calibrated and calibration_fit_audit is None:
            calibration_fit_status = "not_audited"
            calibration_reason = "calibration_fit_audit_not_supplied"
            calibration_horizons_fitted = 0
        elif is_calibrated and pipeline not in calibration_summary:
            calibration_fit_status = "not_audited"
            calibration_reason = "pipeline_missing_from_calibration_fit_audit"
            calibration_horizons_fitted = 0
        elif is_calibrated:
            calibration_horizons_fitted, calibration_reason = calibration_summary[
                pipeline
            ]
            if calibration_reason.startswith("missing_horizons=") or (
                ";missing_horizons=" in calibration_reason
            ) or calibration_reason.startswith("unexpected_horizons=") or (
                ";unexpected_horizons=" in calibration_reason
            ):
                calibration_fit_status = "incomplete_or_misaligned_audit"
            elif calibration_reason:
                calibration_fit_status = "fallback_at_one_or_more_horizons"
            else:
                calibration_fit_status = "fitted_at_all_horizons"
        else:
            calibration_horizons_fitted = 0
            calibration_fit_status = "not_applicable"
            calibration_reason = ""

        rows.append(
            {
                "pipeline": pipeline,
                "fit_status": "success",
                "fit_failure_reason": "",
                "prediction_status": "success",
                "prediction_failure_reason": "",
                "calibration_declared": str(config["calibration"]),
                "calibration_fit_status": calibration_fit_status,
                "calibration_horizons_fitted": calibration_horizons_fitted,
                "calibration_failure_reason": calibration_reason,
                "hyperparameters": str(config["hyperparameters"]),
                "inputs": str(config["inputs"]),
                "time_representation": str(config["time_representation"]),
                "censoring_handling": str(config["censoring_handling"]),
                "cause_handling": str(config["cause_handling"]),
                "tuning_data": str(config["tuning_data"]),
                "stopping_rule": str(config["stopping_rule"]),
                "software": str(config["software"]),
                "test_set_id": str(alignment["test_set_id"]),
                "test_episode_count": int(alignment["episodes"]),
                "horizons_exported": "|".join(
                    str(value) for value in alignment["horizons"]
                ),
                "episode_order_sha256": str(alignment["episode_order_sha256"]),
                "model_object_released": False,
                "model_object_policy": "retrain_from_source; no pickle or executable object",
            }
        )
    return pd.DataFrame(rows)


def preimputation_feature_coverage(
    partitions: Mapping[str, pd.DataFrame],
    feature_columns: Iterable[str],
    *,
    dataset_name: str = "wetlinks",
) -> pd.DataFrame:
    """Report finite-value coverage before any encoder imputation occurs."""

    features = tuple(str(value) for value in feature_columns)
    if not features:
        raise ValueError("at least one feature is required")
    rows: list[dict[str, object]] = []
    for partition, frame in partitions.items():
        missing_columns = set(features).difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{partition} partition missing features {sorted(missing_columns)}"
            )
        for feature in features:
            numeric = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
            finite = np.isfinite(numeric)
            structural_fill_rule = (
                "missing rain_mm replaced by 0 before multiplication by latency_mean_ms"
                if feature == "rain_latency_interaction"
                else "none"
            )
            rows.append(
                {
                    "dataset": str(dataset_name),
                    "partition": str(partition),
                    "feature": feature,
                    "episodes": int(len(frame)),
                    "finite_before_imputation": int(finite.sum()),
                    "missing_or_nonfinite_before_imputation": int((~finite).sum()),
                    "coverage_fraction_before_imputation": (
                        float(finite.mean()) if len(frame) else np.nan
                    ),
                    "measurement_order": (
                        "coverage computed on numeric-coerced raw feature before imputation"
                    ),
                    "model_handling_after_measurement": (
                        "replace missing/nonfinite with training-partition median, then "
                        "apply training-partition mean/SD scaling"
                    ),
                    "structural_fill_rule_before_episode_construction": (
                        structural_fill_rule
                    ),
                    "imputer_source": "training_partition_feature_median",
                }
            )
    return pd.DataFrame(rows)
