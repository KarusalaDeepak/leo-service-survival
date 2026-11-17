"""Tests for numerical-choice audits that must not tune on held-out outcomes."""

import numpy as np
import pandas as pd
import pytest

from leo_service_survival.models import CalibratedSurvivalModel, DiscreteHazardModel
from leo_service_survival.parameter_defensibility import (
    EpisodeSeparatedHazardModel, calibrated_prediction, feature_guard_sensitivity,
    coverage_budget_sensitivity, governance_logic_sensitivity,
    hazard_cap_sensitivity, monitor_logic_sensitivity, nested_development_folds, select_grid,
)


def _episodes(blocks=16, per_block=8):
    rows = []
    rng = np.random.default_rng(17)
    for block in range(blocks):
        for index in range(per_block):
            start = pd.Timestamp("2023-01-02", tz="UTC") + pd.Timedelta(weeks=block, hours=index)
            rows.append({
                "episode_id": f"{block}_{index}", "site": "a" if index % 2 else "b",
                "calendar_block": f"block_{block:02d}", "start_time": start,
                "analysis_label_resolution_time": start + pd.Timedelta(hours=2),
                "duration_bins": 2 + index % 4, "event": int(index % 3 == 0),
                "cause": "loss" if index % 3 == 0 else "censored",
                "x": rng.normal(), "y": rng.normal(),
            })
    return pd.DataFrame(rows)


def test_nested_folds_are_past_only_with_confirmation_purge():
    frame = _episodes()
    folds = nested_development_folds(frame)
    assert len(folds) == 2
    for fold in folds:
        assert fold.inner_fit.start_time.max() < fold.inner_calibration.start_time.min()
        assert fold.inner_calibration.analysis_label_resolution_time.max() < fold.inner_validation.start_time.min()
        assert fold.outer_calibration.analysis_label_resolution_time.max() < fold.outer_validation.start_time.min()
        assert set(fold.inner_validation.calendar_block).isdisjoint(fold.outer_validation.calendar_block)
    # A very long future-resolved label must be excluded, not truncated into an event.
    frame.loc[0, "analysis_label_resolution_time"] = frame.start_time.max() + pd.Timedelta(days=2)
    assert "0_0" not in set(nested_development_folds(frame)[0].inner_fit.episode_id)


def test_too_few_blocks_fail_closed():
    with pytest.raises(ValueError, match="not enough"):
        nested_development_folds(_episodes(10))


def test_encoder_is_fitted_only_on_model_fitting_rows_and_no_random_early_stop():
    frame = _episodes(3)
    config = {"learning_rate": 0.03, "max_iter": 4, "max_leaf_nodes": 3, "min_samples_leaf": 2, "l2_regularization": 1.0}
    frame.loc[0, "x"] = np.nan
    model = EpisodeSeparatedHazardModel(("x", "y"), max_horizon=4, competing=True, hyperparameters=config).fit(frame)
    before = model.encoder.mean_.copy()
    future = frame.copy()
    future["x"] = 1e9
    model.predict_survival(future, (1, 2, 3, 4))
    np.testing.assert_array_equal(model.encoder.mean_, before)
    assert model.model_.early_stopping is False
    assert model.encoder.median_[0] == pytest.approx(frame.x.median())


def test_calibrator_support_clip_and_monotonicity_do_not_take_evaluation_outcomes():
    frame = _episodes(5, 12)
    raw_cal = np.tile([0.99, 0.80, 0.7], (len(frame), 1))
    raw_eval = np.asarray([[1.0, 0.0, 1.0], [0.6, 0.8, 0.9]])
    for minimum in (25, 50, 100):
        for clip in (1e-8, 1e-6, 1e-4):
            output, count = calibrated_prediction(raw_cal, frame, raw_eval, (1, 2, 3), regularization_c=0.03, minimum_support=minimum, probability_clip=clip)
            assert np.isfinite(output).all()
            assert (np.diff(output, axis=1) <= 0).all()
            assert 0 <= count <= 3
    _, count = calibrated_prediction(raw_cal, frame, raw_eval, (1, 2, 3), regularization_c=0.03, minimum_support=100)
    assert count == 0


def test_search_original_and_expanded_are_distinct_and_deterministic():
    table = pd.DataFrame([
        {"configuration_index": 0, "calibration_c": 0.03, "inner_validation_integrated_brier": 0.10},
        {"configuration_index": 3, "calibration_c": 0.003, "inner_validation_integrated_brier": 0.09},
    ])
    assert select_grid(table, expanded=False).configuration_index == 0
    assert select_grid(table, expanded=True).configuration_index == 3


def test_feature_guard_audit_is_invariant_to_outcomes():
    train = _episodes(4)
    later = _episodes(2)
    first, _ = feature_guard_sensitivity(train, {"test": later}, ("x", "y"))
    later["event"] = 1 - later.event
    later["duration_bins"] = 999
    second, _ = feature_guard_sensitivity(train, {"test": later}, ("x", "y"))
    pd.testing.assert_frame_equal(first, second)
    assert first.evidence_role.str.contains("descriptive").all()


def test_governance_grid_is_only_logic_fixture_and_contains_pass_and_fail():
    table = governance_logic_sensitivity(_episodes(4), ("x", "y"))
    assert len(table) == 81
    assert table.evidence_role.eq("gate_logic_fixture_not_statistical_evidence").all()
    assert table.release.any() and (~table.release).any()
    assert not table.statistical_authorization_supported.any()


def test_coverage_budget_grid_is_not_a_full_gate_decision():
    train = _episodes(4)
    later = _episodes(2)
    later.loc[0, "x"] = np.nan
    guard, _ = feature_guard_sensitivity(train, {"test": later}, ("x", "y"))
    table = coverage_budget_sensitivity(guard)
    assert len(table) == 3 * len(guard)
    assert table.not_a_complete_release_gate_decision.all()
    assert table.loc[table.minimum_feature_coverage.eq(1)].coverage_budget_failed.any()


def test_monitor_grid_is_controlled_logic_with_sensible_null_response():
    table = monitor_logic_sensitivity()
    assert len(table) == 108
    assert not table.loc[table.sequence.eq("zero_loss"), "alarm"].any()
    assert table.loc[table.sequence.eq("controlled_abrupt_shift"), "alarm"].any()
    assert table.evidence_role.str.contains("not_field").all()


def test_cap_audit_reports_substantive_effects_without_selecting_cap(monkeypatch):
    class StubHazard:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, frame):
            return self

        def _step_probabilities(self, frame, horizon):
            hazard = np.tile([0.2, 0.96, 0.99999999, 0.0], (len(frame), 1))[:, :horizon]
            return np.stack([1 - hazard, hazard], axis=2)

    monkeypatch.setattr("leo_service_survival.parameter_defensibility.EpisodeSeparatedHazardModel", StubHazard)
    selection = [{"fold": fold, "search": "expanded_grid", "configuration": {}, "calibration_c": 0.03} for fold in range(2)]
    table, hits = hazard_cap_sensitivity(_episodes(), ("x", "y"), selection, max_horizon=4)
    assert len(table) == 8 and len(hits) == 16
    assert not table.cap_selected_using_validation.any()
    assert table.maximum_raw_probability_change_from_cap_095.max() > 0
    assert hits.loc[hits.cap.eq(.95), "raw_hazard_cells_above_cap"].gt(0).all()
    assert hits.loc[hits.cap.eq(.95), "maximum_unadjusted_cif_mass_discrepancy"].gt(0).all()
    no_cap = hits.loc[hits.cap.eq(1)]
    assert no_cap.raw_hazard_cells_above_cap.eq(0).all()
    assert no_cap.raw_hazard_cells_below_floor.eq(0).all()
    assert no_cap.maximum_unadjusted_cif_mass_discrepancy.max() < 1e-12


@pytest.mark.parametrize("cause_probabilities", [(0.60, 0.39), (0.0, 0.0), (1e-12, 5e-13), (0.02, 0.01)])
def test_active_discrete_and_calibrated_cif_preserve_total_mass_at_bounds(cause_probabilities):
    class FixedHazard(DiscreteHazardModel):
        def __init__(self):
            self.cause_to_class_ = {"loss": 1, "latency": 2}

        def _step_probabilities(self, frame, horizon):
            loss, latency = cause_probabilities
            return np.tile([1 - loss - latency, loss, latency], (len(frame), horizon, 1))

    frame = pd.DataFrame({"unused": [1, 2]})
    base = FixedHazard()
    calibrated = CalibratedSurvivalModel(base, (1, 2, 3, 4))
    raw_total = sum(cause_probabilities)
    for model in (base, calibrated):
        survival = model.predict_survival(frame, (1, 2, 3, 4))
        for horizon in range(1, 5):
            incidence = model.predict_cumulative_incidence(frame, horizon)
            total_incidence = sum(incidence.values())
            np.testing.assert_allclose(total_incidence + survival[:, horizon - 1], 1.0, atol=1e-14)
            if raw_total > 0:
                np.testing.assert_allclose(incidence["loss"] / total_incidence, cause_probabilities[0] / raw_total)
        if raw_total == 0:
            np.testing.assert_array_equal(survival, np.ones((len(frame), 4)))
        else:
            # Survival is unchanged for every positive learned event hazard.
            old_survival = np.cumprod(np.full((len(frame), 4), 1 - np.clip(raw_total, 1e-8, .95)), axis=1)
            np.testing.assert_array_equal(survival, old_survival)


def test_active_model_no_event_causes_means_survival_one_and_empty_cif():
    class NoEventHazard(DiscreteHazardModel):
        def __init__(self):
            self.cause_to_class_ = {}

        def _step_probabilities(self, frame, horizon):
            return np.tile([1.0, 0.0], (len(frame), horizon, 1))

    frame = pd.DataFrame({"unused": [1, 2]})
    base = NoEventHazard()
    for model in (base, CalibratedSurvivalModel(base, (1, 2, 3))):
        np.testing.assert_array_equal(model.predict_survival(frame, (1, 2, 3)), np.ones((2, 3)))
        assert model.predict_cumulative_incidence(frame, 3) == {}
