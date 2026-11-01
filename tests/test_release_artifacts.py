from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leo_service_survival.release_artifacts import (
    EXPECTED_PIPELINES,
    build_pipeline_execution_status,
    build_prediction_score_release,
    preimputation_feature_coverage,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": ["a", "b", "c"],
            "site": ["x", "x", "y"],
            "campaign_id": ["x", "x", "y"],
            "calendar_block": ["w1", "w1", "w2"],
            "analysis_block": ["w1", "w1", "w2"],
            "duration_bins": [1, 2, 3],
            "event": [1, 0, 0],
            "cause": ["loss", "censored", "censored"],
        }
    )


def _predictions() -> dict[str, np.ndarray]:
    values = np.asarray(
        [
            [0.8, 0.7],
            [0.9, 0.8],
            [0.95, 0.9],
        ],
        dtype=float,
    )
    return {name: values.copy() for name in EXPECTED_PIPELINES}


def _configuration() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": name,
                "inputs": "fixture",
                "time_representation": "fixture",
                "censoring_handling": "fixture",
                "cause_handling": "fixture",
                "calibration": "IPCW logistic" if name.startswith("calibrated_") else "none",
                "hyperparameters": "frozen",
                "tuning_data": "none",
                "stopping_rule": "frozen",
                "software": "test",
            }
            for name in EXPECTED_PIPELINES
        ]
    )


def test_release_contains_every_pipeline_horizon_and_same_row_ids() -> None:
    prediction, score, alignment = build_prediction_score_release(
        _frame(),
        _predictions(),
        (1, 2),
        bin_seconds=300,
    )

    assert len(prediction) == 3 * 12 * 2
    assert len(score) == 12 * 2
    assert set(prediction["pipeline"]) == set(EXPECTED_PIPELINES)
    assert set(prediction["horizon_bins"]) == {1, 2}
    assert alignment["status"] == "passed"
    assert alignment["prediction_rows"] == len(prediction)
    assert alignment["episodes"] == 3

    for (_pipeline, _horizon), group in prediction.groupby(
        ["pipeline", "horizon_bins"], sort=False
    ):
        assert group["episode_id"].tolist() == ["a", "b", "c"]
        assert group["test_row_index"].tolist() == [0, 1, 2]

    horizon_two = prediction.loc[
        prediction["pipeline"].eq(EXPECTED_PIPELINES[0])
        & prediction["horizon_bins"].eq(2)
    ]
    assert horizon_two["outcome_status"].tolist() == [
        "failure_by_horizon",
        "censored_unknown",
        "survived_through_horizon",
    ]
    assert horizon_two["evaluable_at_horizon"].tolist() == [True, False, True]
    assert np.isnan(horizon_two.iloc[1]["observed_survival_target"])
    assert horizon_two.iloc[1]["ipcw_weight"] == 0.0
    assert score["calibration_diagnostic_status"].isin(
        ["computed", "not_estimable"]
    ).all()
    assert score["score_status"].eq("computed").all()


def test_release_fails_closed_for_missing_pipeline_or_row_mismatch() -> None:
    predictions = _predictions()
    predictions.pop(EXPECTED_PIPELINES[-1])
    with pytest.raises(ValueError, match="exactly the 12 declared pipelines"):
        build_prediction_score_release(_frame(), predictions, (1, 2), bin_seconds=300)

    predictions = _predictions()
    predictions[EXPECTED_PIPELINES[-1]] = np.ones((2, 2))
    with pytest.raises(ValueError, match="prediction shape"):
        build_prediction_score_release(_frame(), predictions, (1, 2), bin_seconds=300)


def test_pipeline_status_records_calibration_and_no_model_object() -> None:
    _, _, alignment = build_prediction_score_release(
        _frame(), _predictions(), (1, 2), bin_seconds=300
    )
    fit_audit = pd.DataFrame(
        [
            {
                "pipeline": pipeline,
                "horizon_bins": horizon,
                "raw_fallback_used": False,
                "raw_fallback_reason": "not_used_calibrator_fitted",
            }
            for pipeline in ("calibrated_single_risk", "calibrated_competing_risk")
            for horizon in (1, 2)
        ]
    )
    status = build_pipeline_execution_status(
        _configuration(),
        _predictions(),
        alignment,
        calibration_fit_audit=fit_audit,
    )

    assert len(status) == 12
    assert status["fit_status"].eq("success").all()
    assert status["prediction_status"].eq("success").all()
    assert ~status["model_object_released"].any()
    calibrated = status["pipeline"].isin(
        ["calibrated_single_risk", "calibrated_competing_risk"]
    )
    assert status.loc[calibrated, "calibration_fit_status"].eq(
        "fitted_at_all_horizons"
    ).all()
    assert status.loc[calibrated, "calibration_horizons_fitted"].eq(2).all()


def test_partial_calibration_audit_is_not_labeled_complete() -> None:
    _, _, alignment = build_prediction_score_release(
        _frame(), _predictions(), (1, 2), bin_seconds=300
    )
    fit_audit = pd.DataFrame(
        [
            {
                "pipeline": pipeline,
                "horizon_bins": 1,
                "raw_fallback_used": False,
                "raw_fallback_reason": "not_used_calibrator_fitted",
            }
            for pipeline in ("calibrated_single_risk", "calibrated_competing_risk")
        ]
    )
    status = build_pipeline_execution_status(
        _configuration(),
        _predictions(),
        alignment,
        calibration_fit_audit=fit_audit,
    ).set_index("pipeline")

    for pipeline in ("calibrated_single_risk", "calibrated_competing_risk"):
        assert (
            status.loc[pipeline, "calibration_fit_status"]
            == "incomplete_or_misaligned_audit"
        )
        assert "missing_horizons=2" in status.loc[
            pipeline, "calibration_failure_reason"
        ]


def test_preimputation_coverage_is_measured_before_fill() -> None:
    source = pd.DataFrame(
        {
            "f1": [1.0, np.nan, np.inf],
            "f2": ["2", "missing", 4.0],
        }
    )
    original = source.copy(deep=True)
    coverage = preimputation_feature_coverage(
        {"test": source},
        ("f1", "f2"),
    ).set_index("feature")

    assert coverage.loc["f1", "finite_before_imputation"] == 1
    assert coverage.loc["f1", "missing_or_nonfinite_before_imputation"] == 2
    assert coverage.loc["f2", "finite_before_imputation"] == 2
    assert coverage.loc["f2", "missing_or_nonfinite_before_imputation"] == 1
    pd.testing.assert_frame_equal(source, original)
