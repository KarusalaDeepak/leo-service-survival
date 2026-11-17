"""Auditable calibration-stage and chronological validation diagnostics.

The primary model deliberately remains unchanged.  This module exposes the
three prediction states that already exist inside :class:`CalibratedSurvivalModel`
and evaluates them with one fixed, documented risk orientation.  No held-out
outcome is used to reverse, rescale, select, or otherwise repair a prediction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .metrics import concordance_index, ipcw_survival_targets
from .models import CalibratedSurvivalModel, DiscreteHazardModel


CALIBRATION_STAGE_ORDER = (
    "raw_base",
    "horizon_calibrated_pre_projection",
    "final_monotone",
)

CALIBRATION_BETA_CONSTRAINT_NOTE = (
    "beta is unconstrained and may be negative, so horizon-wise calibration "
    "may reverse cross-episode ordering"
)


def calibrated_prediction_stages(
    model: CalibratedSurvivalModel,
    frame: pd.DataFrame,
    horizons: Iterable[int],
) -> dict[str, np.ndarray]:
    """Return raw, horizon-calibrated, and final monotone survival values.

    Calibrators are evaluated on the complete integer grid through the largest
    requested horizon before requested columns are selected.  This exactly
    mirrors ``CalibratedSurvivalModel.predict_survival`` and prevents a subset
    of requested horizons from changing the monotone projection.
    """

    requested = tuple(int(value) for value in horizons)
    if not requested:
        raise ValueError("at least one prediction horizon is required")
    if min(requested) < 1 or len(set(requested)) != len(requested):
        raise ValueError("prediction horizons must be distinct positive integers")
    if tuple(sorted(requested)) != requested:
        raise ValueError("prediction horizons must be in increasing order")

    full_horizons = tuple(range(1, max(requested) + 1))
    raw_full = np.asarray(
        model.base_model.predict_survival(frame, full_horizons), dtype=float
    )
    calibrated_full = raw_full.copy()
    for column, horizon in enumerate(full_horizons):
        calibrator = model.calibrators_.get(horizon)
        if calibrator is None:
            continue
        clipped = np.clip(raw_full[:, column], 1e-6, 1.0 - 1e-6)
        calibrated_full[:, column] = calibrator.predict_proba(
            logit(clipped).reshape(-1, 1)
        )[:, 1]
    final_full = np.minimum.accumulate(calibrated_full, axis=1)
    selected = np.asarray(requested, dtype=int) - 1
    stages = {
        "raw_base": np.clip(raw_full[:, selected], 0.0, 1.0),
        "horizon_calibrated_pre_projection": np.clip(
            calibrated_full[:, selected], 0.0, 1.0
        ),
        "final_monotone": np.clip(final_full[:, selected], 0.0, 1.0),
    }
    direct_final = model.predict_survival(frame, requested)
    if not np.allclose(stages["final_monotone"], direct_final, rtol=1e-12, atol=1e-12):
        raise AssertionError("diagnostic final stage does not match model prediction")
    return stages


def calibration_fit_audit(
    model: CalibratedSurvivalModel,
    frame: pd.DataFrame,
    horizons: Iterable[int],
    *,
    bin_seconds: int = 300,
) -> pd.DataFrame:
    """Audit every horizon-specific calibration fit on its fit partition.

    The table is deliberately reconstructed from the exact calibration frame
    and fitted estimator state.  It therefore exposes both the support that
    entered each fit and the unprojected probability map applied before the
    across-horizon cumulative-minimum projection.  The logistic coefficient is
    not sign-constrained: a negative slope is retained and reported rather than
    silently reversed using evaluation outcomes.
    """

    requested = tuple(int(value) for value in horizons)
    if not requested:
        raise ValueError("at least one calibration-audit horizon is required")
    if min(requested) < 1 or len(set(requested)) != len(requested):
        raise ValueError(
            "calibration-audit horizons must be distinct positive integers"
        )
    if tuple(sorted(requested)) != requested:
        raise ValueError("calibration-audit horizons must be in increasing order")
    if int(bin_seconds) < 1:
        raise ValueError("bin_seconds must be positive")

    raw = np.asarray(model.base_model.predict_survival(frame, requested), dtype=float)
    expected_shape = (len(frame), len(requested))
    if raw.shape != expected_shape:
        raise AssertionError(
            f"raw calibration predictions have shape {raw.shape}, expected {expected_shape}"
        )
    if not np.isfinite(raw).all() or np.any((raw < 0.0) | (raw > 1.0)):
        raise AssertionError("raw calibration predictions must be finite probabilities")

    duration = frame["duration_bins"].to_numpy(int)
    event = frame["event"].to_numpy(int)
    rows: list[dict[str, object]] = []
    for column, horizon in enumerate(requested):
        known, target, _weights = ipcw_survival_targets(frame, horizon)
        evaluable_count = int(known.sum())
        failure_count = int(((event == 1) & (duration <= horizon)).sum())
        survivor_count = int((duration > horizon).sum())
        if evaluable_count != failure_count + survivor_count:
            raise AssertionError(
                "calibration evaluability does not equal failures plus survivors"
            )

        fallback_reasons: list[str] = []
        if evaluable_count < 25:
            fallback_reasons.append("evaluable_count_below_25")
        if np.unique(target[known]).size < 2:
            fallback_reasons.append("single_class_evaluable_targets")

        calibrator = model.calibrators_.get(horizon)
        raw_probability = raw[:, column]
        if calibrator is None:
            if not fallback_reasons:
                raise AssertionError(
                    f"horizon {horizon} lacks a calibrator despite sufficient support"
                )
            calibrated_probability = raw_probability.copy()
            alpha = np.nan
            beta = np.nan
            fallback_used = True
            fallback_reason = ";".join(fallback_reasons)
        else:
            if fallback_reasons:
                raise AssertionError(
                    f"horizon {horizon} has a calibrator despite insufficient support"
                )
            clipped = np.clip(raw_probability, 1e-6, 1.0 - 1e-6)
            calibrated_probability = np.asarray(
                calibrator.predict_proba(logit(clipped).reshape(-1, 1))[:, 1],
                dtype=float,
            )
            alpha = float(np.asarray(calibrator.intercept_).reshape(-1)[0])
            beta = float(np.asarray(calibrator.coef_).reshape(-1)[0])
            fallback_used = False
            fallback_reason = "not_used_calibrator_fitted"

        if not np.isfinite(calibrated_probability).all() or np.any(
            (calibrated_probability < 0.0) | (calibrated_probability > 1.0)
        ):
            raise AssertionError(
                "pre-projection calibrated predictions must be finite probabilities"
            )
        rows.append(
            {
                "horizon_bins": int(horizon),
                "horizon_minutes": float(horizon * int(bin_seconds) / 60.0),
                "calibration_partition_episodes": int(len(frame)),
                "calibration_partition_evaluable_count": evaluable_count,
                "failure_event_count_by_horizon": failure_count,
                "survivor_count_through_horizon": survivor_count,
                "censored_unknown_count_by_horizon": int(len(frame) - evaluable_count),
                "raw_fallback_used": fallback_used,
                "raw_fallback_reason": fallback_reason,
                "fitted_alpha_intercept": alpha,
                "fitted_beta_slope": beta,
                "regularization_c": float(model.calibration_c),
                "raw_probability_min": float(raw_probability.min()),
                "raw_probability_max": float(raw_probability.max()),
                "calibrated_pre_projection_probability_min": float(
                    calibrated_probability.min()
                ),
                "calibrated_pre_projection_probability_max": float(
                    calibrated_probability.max()
                ),
                "beta_constraint_note": CALIBRATION_BETA_CONSTRAINT_NOTE,
            }
        )
    return pd.DataFrame(rows)


def _episode_id_array(frame: pd.DataFrame) -> np.ndarray:
    if "episode_id" not in frame:
        raise ValueError("row-alignment audit requires an episode_id column")
    identifiers = frame["episode_id"].astype(str).to_numpy()
    if len(np.unique(identifiers)) != len(identifiers):
        raise AssertionError("episode_id must be unique for row-alignment auditing")
    return identifiers


def _identifier_digest(identifiers: Sequence[str]) -> str:
    payload = json.dumps(list(identifiers), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_prediction_row_alignment(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    horizons: Iterable[int],
    *,
    expected_episode_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Fail if prediction rows or columns cannot map one-to-one to episodes."""

    horizons = tuple(int(value) for value in horizons)
    identifiers = _episode_id_array(frame)
    if expected_episode_ids is not None:
        expected = np.asarray(list(expected_episode_ids), dtype=str)
        if not np.array_equal(identifiers, expected):
            raise AssertionError("episode order changed between capture and prediction")
    for name, values in predictions.items():
        array = np.asarray(values, dtype=float)
        if array.shape != (len(frame), len(horizons)):
            raise AssertionError(
                f"{name} prediction shape {array.shape} does not match "
                f"{(len(frame), len(horizons))}"
            )
        if not np.isfinite(array).all():
            raise AssertionError(f"{name} predictions contain non-finite values")
        if np.any((array < 0.0) | (array > 1.0)):
            raise AssertionError(f"{name} predictions fall outside [0, 1]")
    return {
        "status": "passed",
        "episodes": int(len(frame)),
        "horizons": list(horizons),
        "prediction_sets": sorted(predictions),
        "episode_id_unique": True,
        "episode_order_sha256": _identifier_digest(identifiers.tolist()),
        "assertion": (
            "all prediction matrices have one finite [0,1] value per captured "
            "episode_id and requested horizon, in unchanged row order"
        ),
    }


def assert_fixed_c_index_orientation(
    frame: pd.DataFrame,
    survival: np.ndarray,
    *,
    reported_c_index: float | None = None,
) -> dict[str, object]:
    """Audit the prespecified risk orientation ``risk = 1 - S(H)``.

    The inverse orientation is exported only as a diagnostic.  It is never
    chosen based on which orientation performs better on held-out outcomes.
    """

    survival = np.asarray(survival, dtype=float)
    if survival.ndim != 2 or survival.shape[0] != len(frame):
        raise AssertionError("survival rows do not align with the evaluation frame")
    fixed = concordance_index(frame, 1.0 - survival[:, -1])
    inverse = concordance_index(frame, survival[:, -1])
    if reported_c_index is not None and np.isfinite(reported_c_index):
        if not np.isclose(fixed, float(reported_c_index), rtol=1e-12, atol=1e-12):
            raise AssertionError(
                "reported concordance does not use fixed risk = 1 - survival"
            )
    return {
        "status": "passed",
        "fixed_risk_definition": "1 - predicted_survival_at_maximum_horizon",
        "fixed_orientation_c_index": float(fixed),
        "inverse_orientation_c_index_diagnostic_only": float(inverse),
        "test_informed_inversion_performed": False,
        "interpretation": (
            "the fixed orientation is retained even when its held-out c-index is "
            "below the inverse-orientation diagnostic"
        ),
    }


def _weighted_calibration_intercept_slope(
    probability: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1.0 - 1e-5)
    outcome = np.asarray(outcome, dtype=float)
    weight = np.asarray(weight, dtype=float)
    selected = (
        np.isfinite(probability)
        & np.isfinite(outcome)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    probability, outcome, weight = (
        probability[selected],
        outcome[selected],
        weight[selected],
    )
    if len(probability) < 5 or len(np.unique(outcome)) < 2:
        return float("nan"), float("nan")
    design = np.column_stack([np.ones(len(probability)), logit(probability)])
    coefficients = np.asarray([0.0, 1.0], dtype=float)
    for _ in range(100):
        fitted = expit(design @ coefficients)
        variance_weight = weight * np.maximum(fitted * (1.0 - fitted), 1e-8)
        information = design.T @ (variance_weight[:, None] * design)
        score = design.T @ (weight * (outcome - fitted))
        try:
            step = np.linalg.solve(information + np.eye(2) * 1e-10, score)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        coefficients += step
        if np.max(np.abs(step)) < 1e-8:
            break
        if not np.isfinite(coefficients).all() or np.max(np.abs(coefficients)) > 100:
            return float("nan"), float("nan")
    return float(coefficients[0]), float(coefficients[1])


def _weighted_calibration_in_the_large(
    probability: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
) -> float:
    """Weighted calibration intercept with the prediction slope fixed at one."""

    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1.0 - 1e-5)
    outcome = np.asarray(outcome, dtype=float)
    weight = np.asarray(weight, dtype=float)
    selected = (
        np.isfinite(probability)
        & np.isfinite(outcome)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    probability, outcome, weight = (
        probability[selected],
        outcome[selected],
        weight[selected],
    )
    if len(probability) < 5 or len(np.unique(outcome)) < 2:
        return float("nan")
    offset = logit(probability)
    intercept = 0.0
    for _ in range(100):
        fitted = expit(offset + intercept)
        score = float(np.sum(weight * (outcome - fitted)))
        information = float(
            np.sum(weight * np.maximum(fitted * (1.0 - fitted), 1e-8))
        )
        if information <= 0.0:
            return float("nan")
        step = score / information
        intercept += step
        if abs(step) < 1e-8:
            break
        if not np.isfinite(intercept) or abs(intercept) > 100:
            return float("nan")
    return float(intercept)


def stage_horizon_diagnostics(
    frame: pd.DataFrame,
    stage_predictions: Mapping[str, np.ndarray],
    horizons: Iterable[int],
    *,
    scope_kind: str = "pooled",
    scope_value: str = "all",
) -> pd.DataFrame:
    """Compute pointwise and through-horizon diagnostics for every stage."""

    horizons = tuple(int(value) for value in horizons)
    assert_prediction_row_alignment(frame, stage_predictions, horizons)
    duration = frame["duration_bins"].to_numpy(int)
    event = frame["event"].to_numpy(int)
    rows: list[dict[str, object]] = []
    for stage in CALIBRATION_STAGE_ORDER:
        if stage not in stage_predictions:
            raise KeyError(f"missing calibration stage {stage!r}")
        values = np.asarray(stage_predictions[stage], dtype=float)
        pointwise_briers: list[float] = []
        for column, horizon in enumerate(horizons):
            probability = np.clip(values[:, column], 1e-6, 1.0 - 1e-6)
            known, outcome, weights = ipcw_survival_targets(frame, horizon)
            weighted_brier = weights * np.square(outcome - probability)
            pointwise_brier = float(np.mean(weighted_brier))
            pointwise_briers.append(pointwise_brier)
            positive = weights > 0.0
            log_loss = -outcome * np.log(probability) - (
                1.0 - outcome
            ) * np.log(1.0 - probability)
            weight_sum = float(weights[positive].sum())
            weighted_log_loss = (
                float(np.sum(weights[positive] * log_loss[positive]) / weight_sum)
                if weight_sum > 0.0
                else np.nan
            )
            calibration_intercept, calibration_slope = (
                _weighted_calibration_intercept_slope(
                    probability[known], outcome[known], weights[known]
                )
            )
            calibration_in_the_large = _weighted_calibration_in_the_large(
                probability[known], outcome[known], weights[known]
            )
            observed_survival = (
                float(np.sum(weights[positive] * outcome[positive]) / weight_sum)
                if weight_sum > 0.0
                else np.nan
            )
            monotonicity_violations = (
                0
                if column == 0
                else int(np.sum(values[:, column] > values[:, column - 1] + 1e-12))
            )
            rows.append(
                {
                    "scope_kind": str(scope_kind),
                    "scope_value": str(scope_value),
                    "stage": stage,
                    "horizon_bins": int(horizon),
                    "episodes": int(len(frame)),
                    "evaluable_count": int(known.sum()),
                    "events_by_horizon": int(
                        ((event == 1) & (duration <= horizon)).sum()
                    ),
                    "known_survivors_through_horizon": int((duration > horizon).sum()),
                    "censored_unknown_count": int((~known).sum()),
                    "ipcw_observed_survival": observed_survival,
                    "calibration_intercept_free_slope": calibration_intercept,
                    "calibration_slope": calibration_slope,
                    "calibration_in_the_large_fixed_slope_one": calibration_in_the_large,
                    "pointwise_brier_bs_t": pointwise_brier,
                    "equally_weighted_integrated_brier_through_h": float(
                        np.mean(pointwise_briers)
                    ),
                    "ipcw_log_loss": weighted_log_loss,
                    "prediction_min": float(probability.min()),
                    "prediction_max": float(probability.max()),
                    "prediction_mean": float(probability.mean()),
                    "prediction_std": float(probability.std()),
                    "monotonicity_violations_from_previous_horizon": (
                        monotonicity_violations
                    ),
                    "censoring_weight_scope": (
                        "Kaplan-Meier censoring distribution estimated within this "
                        "diagnostic scope; weights clipped through G(t)>=0.001"
                    ),
                    "ibs_definition": (
                        "arithmetic mean of pointwise IPCW Brier scores from the first "
                        "requested horizon through this horizon"
                    ),
                }
            )
    return pd.DataFrame(rows)


def stratified_stage_horizon_diagnostics(
    frame: pd.DataFrame,
    stage_predictions: Mapping[str, np.ndarray],
    horizons: Iterable[int],
    *,
    stratum_columns: Iterable[str],
) -> pd.DataFrame:
    """Apply the same diagnostics separately within each site or time block."""

    horizons = tuple(int(value) for value in horizons)
    parts: list[pd.DataFrame] = []
    for column in stratum_columns:
        if column not in frame:
            raise KeyError(column)
        labels = frame[column].astype(str).to_numpy()
        for label in pd.unique(labels):
            selected = np.flatnonzero(labels == label)
            local_frame = frame.iloc[selected].reset_index(drop=True)
            local_predictions = {
                stage: np.asarray(values)[selected]
                for stage, values in stage_predictions.items()
            }
            parts.append(
                stage_horizon_diagnostics(
                    local_frame,
                    local_predictions,
                    horizons,
                    scope_kind=column,
                    scope_value=str(label),
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def calibration_stage_prediction_table(
    frame: pd.DataFrame,
    stage_predictions: Mapping[str, np.ndarray],
    horizons: Iterable[int],
    *,
    bin_seconds: int,
) -> pd.DataFrame:
    """Build a tidy, row-identifiable export of all three prediction stages."""

    horizons = tuple(int(value) for value in horizons)
    assert_prediction_row_alignment(frame, stage_predictions, horizons)
    identity_columns = [
        column
        for column in (
            "episode_id",
            "site",
            "campaign_id",
            "calendar_block",
            "analysis_block",
            "start_time",
            "duration_bins",
            "event",
            "cause",
        )
        if column in frame
    ]
    identity = frame[identity_columns].reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for stage in CALIBRATION_STAGE_ORDER:
        values = np.asarray(stage_predictions[stage], dtype=float)
        for column, horizon in enumerate(horizons):
            part = identity.copy()
            part["stage"] = stage
            part["horizon_bins"] = int(horizon)
            part["horizon_minutes"] = float(horizon * bin_seconds / 60.0)
            part["predicted_survival"] = values[:, column]
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def rolling_origin_temporal_diagnostics(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    max_horizon: int,
    block_column: str = "calendar_block",
    minimum_train_blocks: int = 8,
    calibration_blocks: int = 4,
    embargo_blocks: int = 1,
    test_blocks: int = 1,
    random_state: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window prequential diagnostic with frozen chronological folds.

    Fold boundaries depend only on calendar order.  Each fold uses all eligible
    past blocks for fitting, a fixed number of later past blocks for
    calibration, one embargo block at each boundary, and the next block(s) for
    testing.  Predictions retain ``risk = 1-S(H)`` regardless of test results.
    """

    if block_column not in frame:
        raise KeyError(block_column)
    if "start_time" not in frame:
        raise KeyError("start_time")
    chronology = (
        frame.groupby(block_column, sort=False)["start_time"]
        .min()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    earliest_test = (
        int(minimum_train_blocks)
        + int(embargo_blocks)
        + int(calibration_blocks)
        + int(embargo_blocks)
    )
    if earliest_test + int(test_blocks) > len(chronology):
        raise ValueError("too few chronological blocks for the frozen rolling design")

    horizons = tuple(range(1, int(max_horizon) + 1))
    diagnostic_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    labels = frame[block_column].astype(str)
    for fold_number, test_start in enumerate(
        range(earliest_test, len(chronology) - int(test_blocks) + 1), start=1
    ):
        calibration_end = test_start - int(embargo_blocks)
        calibration_start = calibration_end - int(calibration_blocks)
        train_end = calibration_start - int(embargo_blocks)
        train_names = chronology[:train_end]
        calibration_names = chronology[calibration_start:calibration_end]
        test_names = chronology[test_start : test_start + int(test_blocks)]
        train = frame.loc[labels.isin(train_names)].sort_values("start_time").reset_index(drop=True)
        calibration = (
            frame.loc[labels.isin(calibration_names)]
            .sort_values("start_time")
            .reset_index(drop=True)
        )
        test = frame.loc[labels.isin(test_names)].sort_values("start_time").reset_index(drop=True)
        if min(len(train), len(calibration), len(test)) == 0:
            raise AssertionError("frozen rolling fold contains an empty partition")
        if "analysis_label_resolution_time" in frame:
            label_disjoint = bool(
                train["analysis_label_resolution_time"].max()
                < calibration["start_time"].min()
                and calibration["analysis_label_resolution_time"].max()
                < test["start_time"].min()
            )
            if not label_disjoint:
                raise AssertionError(
                    f"label resolution crosses rolling fold {fold_number} boundary"
                )
        else:
            label_disjoint = True

        model = CalibratedSurvivalModel(
            DiscreteHazardModel(
                tuple(feature_columns),
                max_horizon=max_horizon,
                nonlinear=True,
                competing=True,
                random_state=random_state,
            ),
            horizons,
        )
        model.fit(train)
        model.calibrate(calibration)
        stages = calibrated_prediction_stages(model, test, horizons)
        alignment = assert_prediction_row_alignment(test, stages, horizons)
        fixed_orientation = assert_fixed_c_index_orientation(
            test, stages["final_monotone"]
        )
        diagnostics = stage_horizon_diagnostics(
            test,
            stages,
            horizons,
            scope_kind="rolling_fold",
            scope_value=str(fold_number),
        )
        diagnostics.insert(0, "fold", fold_number)
        diagnostics["test_blocks"] = "|".join(test_names)
        diagnostics["fixed_risk_definition"] = (
            "1 - predicted_survival_at_maximum_horizon"
        )
        diagnostics["test_informed_inversion_performed"] = False
        diagnostic_parts.append(diagnostics)
        fold_rows.append(
            {
                "fold": fold_number,
                "train_blocks": "|".join(train_names),
                "calibration_blocks": "|".join(calibration_names),
                "embargo_before_calibration": chronology[train_end],
                "embargo_before_test": chronology[calibration_end],
                "test_blocks": "|".join(test_names),
                "train_episodes": int(len(train)),
                "calibration_episodes": int(len(calibration)),
                "test_episodes": int(len(test)),
                "test_events": int(test["event"].sum()),
                "label_resolution_disjoint": label_disjoint,
                "test_episode_order_sha256": alignment["episode_order_sha256"],
                "fixed_orientation_c_index_h_max": fixed_orientation[
                    "fixed_orientation_c_index"
                ],
                "inverse_orientation_c_index_diagnostic_only_h_max": (
                    fixed_orientation["inverse_orientation_c_index_diagnostic_only"]
                ),
                "test_informed_inversion_performed": False,
                "fold_rule": (
                    "expanding past-only train; fixed four-block calibration; one "
                    "calendar-block embargo at each boundary; next block test"
                ),
            }
        )
    return pd.concat(diagnostic_parts, ignore_index=True), pd.DataFrame(fold_rows)
