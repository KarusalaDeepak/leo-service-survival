from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leo_service_survival.diagnostics import (
    CALIBRATION_BETA_CONSTRAINT_NOTE,
    assert_fixed_c_index_orientation,
    assert_prediction_row_alignment,
    calibrated_prediction_stages,
    calibration_fit_audit,
    calibration_stage_prediction_table,
    rolling_origin_temporal_diagnostics,
    stage_horizon_diagnostics,
)
from leo_service_survival.metrics import evaluate_survival_predictions
from leo_service_survival.models import CalibratedSurvivalModel


class _StubBaseModel:
    def predict_survival(self, frame, horizons):
        values = np.asarray([0.90, 0.80, 0.70], dtype=float)
        columns = np.asarray(list(horizons), dtype=int) - 1
        return np.tile(values[columns], (len(frame), 1))


class _ConstantCalibrator:
    def __init__(self, probability):
        self.probability = float(probability)

    def predict_proba(self, design):
        positive = np.full(len(design), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class _FrameProbabilityBaseModel:
    def predict_survival(self, frame, horizons):
        return np.column_stack(
            [frame[f"raw_s{int(horizon)}"].to_numpy(float) for horizon in horizons]
        )


def _diagnostic_frame(rows: int = 30) -> pd.DataFrame:
    duration = np.tile([1, 2, 3, 4, 4], rows // 5 + 1)[:rows]
    event = np.tile([1, 1, 1, 0, 0], rows // 5 + 1)[:rows]
    return pd.DataFrame(
        {
            "episode_id": [f"episode-{index}" for index in range(rows)],
            "site": np.where(np.arange(rows) % 2, "a", "b"),
            "calendar_block": np.where(np.arange(rows) < rows / 2, "w1", "w2"),
            "duration_bins": duration,
            "event": event,
            "cause": np.where(event == 1, "loss", "censored"),
        }
    )


def test_stage_export_exposes_pre_projection_and_final_monotone_values():
    frame = _diagnostic_frame()
    model = CalibratedSurvivalModel(_StubBaseModel(), range(1, 4))
    model.calibrators_ = {
        1: _ConstantCalibrator(0.85),
        2: _ConstantCalibrator(0.92),
        3: _ConstantCalibrator(0.60),
    }
    stages = calibrated_prediction_stages(model, frame, (1, 2, 3))
    assert np.allclose(stages["raw_base"][0], [0.90, 0.80, 0.70])
    assert np.allclose(
        stages["horizon_calibrated_pre_projection"][0], [0.85, 0.92, 0.60]
    )
    assert np.allclose(stages["final_monotone"][0], [0.85, 0.85, 0.60])

    diagnostics = stage_horizon_diagnostics(frame, stages, (1, 2, 3))
    pre = diagnostics.loc[
        diagnostics["stage"].eq("horizon_calibrated_pre_projection")
    ].sort_values("horizon_bins")
    final = diagnostics.loc[diagnostics["stage"].eq("final_monotone")]
    assert pre["monotonicity_violations_from_previous_horizon"].tolist() == [0, 30, 0]
    assert final["monotonicity_violations_from_previous_horizon"].eq(0).all()
    expected_ibs_h2 = pre["pointwise_brier_bs_t"].iloc[:2].mean()
    assert np.isclose(
        pre["equally_weighted_integrated_brier_through_h"].iloc[1],
        expected_ibs_h2,
    )
    assert {
        "calibration_intercept_free_slope",
        "calibration_slope",
        "calibration_in_the_large_fixed_slope_one",
        "prediction_min",
        "prediction_max",
        "evaluable_count",
        "events_by_horizon",
    }.issubset(diagnostics.columns)

    table = calibration_stage_prediction_table(
        frame, stages, (1, 2, 3), bin_seconds=300
    )
    assert len(table) == len(frame) * 3 * 3
    assert table["episode_id"].notna().all()
    assert sorted(table["horizon_minutes"].unique()) == [5.0, 10.0, 15.0]


def test_calibration_fit_audit_reports_unconstrained_slope_and_raw_fallback():
    rows = 40
    frame = pd.DataFrame(
        {
            "duration_bins": np.r_[np.ones(20, dtype=int), np.full(20, 3)],
            "event": np.r_[np.ones(20, dtype=int), np.zeros(20, dtype=int)],
            # Reverse the expected ordering so the unconstrained fitted slope
            # is negative and the audit must preserve, not repair, it.
            "raw_s1": np.r_[np.linspace(0.75, 0.95, 20), np.linspace(0.05, 0.25, 20)],
            "raw_s2": np.r_[np.linspace(0.70, 0.90, 20), np.linspace(0.10, 0.30, 20)],
            "raw_s3": np.linspace(0.15, 0.85, rows),
        }
    )
    model = CalibratedSurvivalModel(
        _FrameProbabilityBaseModel(),
        range(1, 4),
        calibration_c=0.2,
    ).calibrate(frame)

    audit = calibration_fit_audit(model, frame, (1, 2, 3), bin_seconds=300)
    assert audit["horizon_bins"].tolist() == [1, 2, 3]
    assert audit["horizon_minutes"].tolist() == [5.0, 10.0, 15.0]
    assert audit["calibration_partition_evaluable_count"].tolist() == [40, 40, 20]
    assert audit["failure_event_count_by_horizon"].tolist() == [20, 20, 20]
    assert audit["survivor_count_through_horizon"].tolist() == [20, 20, 0]
    assert audit["raw_fallback_used"].tolist() == [False, False, True]
    assert audit.loc[2, "raw_fallback_reason"] == (
        "evaluable_count_below_25;single_class_evaluable_targets"
    )
    assert np.isnan(audit.loc[2, "fitted_alpha_intercept"])
    assert np.isnan(audit.loc[2, "fitted_beta_slope"])
    assert audit.loc[0, "fitted_beta_slope"] < 0.0
    assert audit["regularization_c"].eq(0.2).all()
    assert audit["beta_constraint_note"].eq(CALIBRATION_BETA_CONSTRAINT_NOTE).all()
    assert np.isclose(audit.loc[0, "raw_probability_min"], 0.05)
    assert np.isclose(audit.loc[0, "raw_probability_max"], 0.95)
    assert (
        audit.loc[2, "calibrated_pre_projection_probability_min"]
        == audit.loc[2, "raw_probability_min"]
    )
    assert (
        audit.loc[2, "calibrated_pre_projection_probability_max"]
        == audit.loc[2, "raw_probability_max"]
    )


def test_alignment_and_c_index_orientation_are_hard_assertions():
    frame = _diagnostic_frame()
    prediction = np.tile([0.9, 0.8, 0.7], (len(frame), 1))
    captured = frame["episode_id"].to_numpy(copy=True)
    audit = assert_prediction_row_alignment(
        frame,
        {"candidate": prediction},
        (1, 2, 3),
        expected_episode_ids=captured,
    )
    assert audit["status"] == "passed"
    with pytest.raises(AssertionError, match="episode order changed"):
        assert_prediction_row_alignment(
            frame.iloc[::-1].reset_index(drop=True),
            {"candidate": prediction},
            (1, 2, 3),
            expected_episode_ids=captured,
        )
    with pytest.raises(AssertionError, match="prediction shape"):
        assert_prediction_row_alignment(
            frame,
            {"candidate": prediction[:, :2]},
            (1, 2, 3),
        )

    reported = evaluate_survival_predictions(frame, prediction, (1, 2, 3))["c_index"]
    orientation = assert_fixed_c_index_orientation(
        frame, prediction, reported_c_index=reported
    )
    assert orientation["test_informed_inversion_performed"] is False
    with pytest.raises(AssertionError, match="fixed risk"):
        assert_fixed_c_index_orientation(
            frame, prediction, reported_c_index=reported + 0.1
        )


def test_rolling_origin_folds_are_past_only_and_keep_fixed_orientation():
    rng = np.random.default_rng(11)
    rows = []
    start = pd.Timestamp("2026-01-05", tz="UTC")
    for block in range(9):
        block_start = start + pd.Timedelta(weeks=block)
        for offset in range(12):
            duration = int(rng.integers(1, 5))
            event = int((offset + block) % 4 != 0)
            cause = "censored" if event == 0 else ("loss" if offset % 2 else "latency")
            timestamp = block_start + pd.Timedelta(minutes=10 * offset)
            rows.append(
                {
                    "episode_id": f"b{block}-e{offset}",
                    "calendar_block": f"block-{block:02d}",
                    "site": "a" if offset % 2 else "b",
                    "start_time": timestamp,
                    "analysis_label_resolution_time": timestamp + pd.Timedelta(minutes=15),
                    "duration_bins": duration,
                    "event": event,
                    "cause": cause,
                    "x1": rng.normal(),
                    "x2": rng.normal(),
                }
            )
    frame = pd.DataFrame(rows)
    diagnostics, folds = rolling_origin_temporal_diagnostics(
        frame,
        ("x1", "x2"),
        max_horizon=3,
        minimum_train_blocks=4,
        calibration_blocks=2,
        embargo_blocks=1,
        test_blocks=1,
    )
    assert len(folds) == 1
    assert folds.loc[0, "train_blocks"].split("|") == [
        "block-00",
        "block-01",
        "block-02",
        "block-03",
    ]
    assert folds.loc[0, "embargo_before_calibration"] == "block-04"
    assert folds.loc[0, "calibration_blocks"] == "block-05|block-06"
    assert folds.loc[0, "embargo_before_test"] == "block-07"
    assert folds.loc[0, "test_blocks"] == "block-08"
    assert bool(folds.loc[0, "label_resolution_disjoint"])
    assert not bool(folds.loc[0, "test_informed_inversion_performed"])
    assert len(diagnostics) == 3 * 3
    assert diagnostics["test_informed_inversion_performed"].eq(False).all()
