from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leo_service_survival.data import (
    _episodes_from_regular_series,
    build_commect_landmarks,
    calendar_block_split,
    merge_sparse_terminal_inference_block,
    purged_epoch_split,
)
from leo_service_survival.geometry import restricted_eust_seconds
from leo_service_survival.metrics import (
    block_bootstrap_prediction_inference,
    calibration_curve_block_intervals,
    circular_moving_block_paired_inference,
    evaluate_survival_predictions,
    censoring_support_table,
    censoring_weighted_brier_score,
    ipcw_brier_score,
    known_survival_outcome,
)
from leo_service_survival.models import (
    CalibratedSurvivalModel,
    ExponentialModel,
    KaplanMeierModel,
    build_model_suite,
    select_focal_model_configuration,
)
from leo_service_survival.gating import (
    CalibrationEProcess,
    EvidenceGate,
    FrozenMahalanobisDetector,
    GateCriteria,
    GateEvidence,
    evaluate_calibration_monitor_campaigns,
    evaluate_gate_campaigns,
)
from leo_service_survival.integration import (
    assess_research_completion,
    build_full_chain_ledger,
    summarize_full_chain_completion,
    validate_full_chain_bundle,
)
from leo_service_survival.reporting import (
    aalen_johansen_cif,
    calibration_intercept_slope,
    competing_risk_event_adequacy,
    paired_proper_score_block_inference,
    proper_score_power_planning,
    scheduler_two_level_inference,
)
from scripts.run_all import truncate


def sample_frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    duration = rng.integers(1, 7, size=rows)
    event = rng.binomial(1, 0.7, size=rows)
    cause = np.where(event == 0, "censored", np.where(rng.random(rows) < 0.5, "latency", "loss"))
    return pd.DataFrame(
        {
            "duration_bins": duration,
            "event": event,
            "cause": cause,
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "campaign_id": np.where(np.arange(rows) % 2, "a", "b"),
        }
    )


def test_kaplan_meier_is_monotone():
    frame = sample_frame()
    model = KaplanMeierModel().fit(frame)
    survival = model.predict_survival(frame.iloc[:3], range(1, 7))
    assert survival.shape == (3, 6)
    assert np.all(np.diff(survival, axis=1) <= 1e-12)


def test_focal_hyperparameters_are_selected_without_test_rows():
    frame = sample_frame(180)
    fit, calibration, validation = frame.iloc[:100], frame.iloc[100:140], frame.iloc[140:]
    candidates = (
        {
            "learning_rate": 0.05,
            "max_iter": 30,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 10,
            "l2_regularization": 0.5,
        },
        {
            "learning_rate": 0.08,
            "max_iter": 30,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 10,
            "l2_regularization": 1.0,
        },
    )
    selected, calibration_c, audit = select_focal_model_configuration(
        fit,
        calibration,
        validation,
        ("x1", "x2"),
        max_horizon=6,
        candidate_configs=candidates,
        calibration_c_grid=(0.1, 1.0),
    )
    assert selected in candidates
    assert calibration_c in {0.1, 1.0}
    assert audit["selected"].sum() == 1
    assert audit["fit_rows"].eq(len(fit)).all()
    assert audit["calibration_rows"].eq(len(calibration)).all()
    assert audit["validation_rows"].eq(len(validation)).all()


def _endpoint_fixture(causes, observed=None):
    causes = list(causes)
    observed = [True] * len(causes) if observed is None else list(observed)
    usable = [cause == "unknown" for cause in causes]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=len(causes), freq="5min", tz="UTC"
            ),
            "observed": observed,
            "usable": usable,
            "cause": causes,
            "x": np.arange(len(causes), dtype=float),
        }
    )
    return _episodes_from_regular_series(
        frame,
        site="fixture",
        timestamp_column="timestamp",
        observed_column="observed",
        usable_column="usable",
        cause_column="cause",
        feature_columns=("x",),
        bin_seconds=300,
        sustained_bad_bins=2,
        max_gap_bins=2,
    )[0]


@pytest.mark.parametrize(
    ("first_bad", "second_bad", "expected_event", "expected_cause"),
    [
        ("latency", "unknown", 0, "censored"),
        ("loss", "unknown", 0, "censored"),
        ("latency", "latency", 1, "latency"),
        ("loss", "loss", 1, "loss"),
        ("joint", "joint", 1, "joint"),
        ("latency", "joint", 1, "latency"),
        ("joint", "latency", 1, "latency"),
        ("loss", "joint", 1, "loss"),
        ("joint", "loss", 1, "loss"),
        ("latency", "loss", 0, "censored"),
        ("loss", "latency", 0, "censored"),
    ],
)
def test_sustained_endpoint_truth_table_and_timing(
    first_bad, second_bad, expected_event, expected_cause
):
    episode = _endpoint_fixture(
        ["unknown", first_bad, second_bad, "unknown"]
    )
    assert episode["event"] == expected_event
    assert episode["cause"] == expected_cause
    if expected_event:
        assert episode["duration_bins"] == 1
        assert episode["event_onset_time"] == pd.Timestamp(
            "2026-01-01 00:05:00Z"
        )
        assert episode["event_confirmation_time"] == pd.Timestamp(
            "2026-01-01 00:10:00Z"
        )
        assert episode["label_resolution_time"] == pd.Timestamp(
            "2026-01-01 00:10:00Z"
        )


@pytest.mark.parametrize("cause", ["latency", "loss", "joint"])
def test_unconfirmed_final_violation_is_censored_at_possible_onset(cause):
    final_bad = _endpoint_fixture(["unknown", "unknown", cause])
    assert final_bad["event"] == 0
    assert final_bad["duration_bins"] == 2
    assert final_bad["censoring_time"] == pd.Timestamp("2026-01-01 00:10:00Z")
    assert final_bad["label_resolution_time"] == pd.Timestamp(
        "2026-01-01 00:15:00Z"
    )

    missing_confirmation = _endpoint_fixture(
        ["unknown", cause, "unknown"], observed=[True, True, False]
    )
    assert missing_confirmation["event"] == 0
    assert missing_confirmation["duration_bins"] == 1
    assert missing_confirmation["censoring_time"] == pd.Timestamp(
        "2026-01-01 00:05:00Z"
    )
    assert missing_confirmation["label_resolution_time"] == pd.Timestamp(
        "2026-01-01 00:10:00Z"
    )


def test_confirmation_lookahead_crossing_split_is_detected():
    # The candidate onset is at H=1, but recovery at H+1 is needed to establish
    # that no sustained event began at H.  A split at 00:07 would therefore
    # leak the later label resolution even though the onset/boundary 00:05 is
    # before the split.
    episode = _endpoint_fixture(
        ["unknown", "latency", "unknown", "unknown"]
    )
    truncated = truncate(pd.DataFrame([episode]), max_horizon=1)
    assert truncated.loc[0, "event"] == 0
    assert truncated.loc[0, "duration_bins"] == 2
    assert truncated.loc[0, "analysis_label_resolution_time"] == pd.Timestamp(
        "2026-01-01 00:10:00Z"
    )
    assert truncated.loc[0, "analysis_label_end_time"] == truncated.loc[
        0, "analysis_label_resolution_time"
    ]
    next_partition_start = pd.Timestamp("2026-01-01 00:07:00Z")
    assert not (
        truncated["analysis_label_resolution_time"].max()
        < next_partition_start
    )


def test_kaplan_meier_removes_tied_censor_before_event():
    frame = pd.DataFrame(
        {
            "duration_bins": [2, 2],
            "event": [1, 0],
            "cause": ["latency", "censored"],
        }
    )
    survival = KaplanMeierModel().fit(frame).predict_survival(frame, [2])
    assert np.allclose(survival, 0.0)


def test_exponential_exposure_uses_only_bins_before_censor_boundary():
    frame = pd.DataFrame(
        {
            "duration_bins": [2, 2],
            "event": [1, 0],
            "cause": ["latency", "censored"],
        }
    )
    model = ExponentialModel().fit(frame)
    assert np.isclose(model.hazard_, 1.0 / 3.0)


def test_commect_full_horizon_survivor_is_encoded_after_horizon(tmp_path):
    path = tmp_path / "commect.csv"
    pd.DataFrame(
        {
            "relative_path": ["sat", "sat", "sat"],
            "bin_start_utc": pd.date_range("2026-01-01", periods=3, freq="s", tz="UTC"),
            "bin_epoch": [100, 101, 102],
            "latency_mean_ms": [40.0, 41.0, 42.0],
            "latency_std_ms": [1.0, 1.0, 1.0],
            "latency_max_ms": [45.0, 46.0, 47.0],
            "observed_replies": [1.0, 1.0, 1.0],
            "observation_age_ms": [0.0, 0.0, 0.0],
            "rsrp_mean_dbm": [np.nan, np.nan, np.nan],
            "rsrq_mean_db": [np.nan, np.nan, np.nan],
            "snr_mean_db": [np.nan, np.nan, np.nan],
            "access_technology": ["satellite", "satellite", "satellite"],
        }
    ).to_csv(path, index=False)
    frame = build_commect_landmarks(path, max_horizon_seconds=2).frame
    first = frame.loc[frame["decision_epoch"].eq(100)].iloc[0]
    assert first["event"] == 0
    assert first["duration_bins"] == 3


def test_proposed_model_outputs_probabilities():
    frame = sample_frame(120)
    train, calibration, test = frame.iloc[:70], frame.iloc[70:95], frame.iloc[95:]
    suite = build_model_suite(("x1", "x2"), max_horizon=6)
    model = suite.models[suite.proposed_name]
    model.fit(train)
    model.calibrate(calibration)
    survival = model.predict_survival(test, range(1, 7))
    assert np.all((survival >= 0.0) & (survival <= 1.0))
    assert np.all(np.diff(survival, axis=1) <= 1e-10)
    assert np.allclose(model.predict_survival(test, [6])[:, 0], survival[:, 5])
    assert np.allclose(
        model.predict_survival(test, [1, 3, 6]),
        survival[:, [0, 2, 5]],
    )
    incidence = model.predict_cumulative_incidence(test, 6)
    assert incidence
    total_incidence = np.column_stack(list(incidence.values())).sum(axis=1)
    assert np.allclose(total_incidence, 1.0 - survival[:, 5])
    metrics = evaluate_survival_predictions(test, survival, range(1, 7))
    assert np.isfinite(metrics["integrated_brier"])


def test_calibrated_survival_subset_invariance_and_cif_monotonicity():
    class StubBaseModel:
        cause_to_class_ = {"loss": 1, "latency": 2}

        def predict_survival(self, frame, horizons):
            values = np.asarray([0.92, 0.84, 0.76, 0.68], dtype=float)
            columns = np.asarray(list(horizons), dtype=int) - 1
            return np.tile(values[columns], (len(frame), 1))

        def _step_probabilities(self, frame, max_horizon):
            probabilities = np.zeros((len(frame), int(max_horizon), 3), dtype=float)
            probabilities[:, :, 0] = 0.8
            probabilities[:, :, 1] = np.asarray([0.15, 0.04, 0.16, 0.03])[:max_horizon]
            probabilities[:, :, 2] = np.asarray([0.05, 0.16, 0.04, 0.17])[:max_horizon]
            return probabilities

    class ConstantCalibrator:
        def __init__(self, probability):
            self.probability = float(probability)

        def predict_proba(self, design):
            positive = np.full(len(design), self.probability, dtype=float)
            return np.column_stack([1.0 - positive, positive])

    frame = pd.DataFrame(index=range(4))
    model = CalibratedSurvivalModel(StubBaseModel(), range(1, 5))
    # The third horizon rises before projection, making the old subset-dependent
    # implementation return 0.90 for [1, 3] but 0.80 on the complete grid.
    model.calibrators_ = {
        1: ConstantCalibrator(0.90),
        2: ConstantCalibrator(0.80),
        3: ConstantCalibrator(0.95),
        4: ConstantCalibrator(0.70),
    }

    full = model.predict_survival(frame, range(1, 5))
    assert np.allclose(full[0], [0.90, 0.80, 0.80, 0.70])
    assert np.allclose(model.predict_survival(frame, [3])[:, 0], full[:, 2])
    assert np.allclose(model.predict_survival(frame, [1, 3]), full[:, [0, 2]])

    incidence_by_horizon = {
        horizon: model.predict_cumulative_incidence(frame, horizon)
        for horizon in range(1, 5)
    }
    for cause in ("loss", "latency"):
        curve = np.column_stack(
            [incidence_by_horizon[horizon][cause] for horizon in range(1, 5)]
        )
        assert np.all(np.diff(curve, axis=1) >= -1e-12)
    for horizon in range(1, 5):
        total = np.column_stack(
            list(incidence_by_horizon[horizon].values())
        ).sum(axis=1)
        assert np.allclose(total, 1.0 - full[:, horizon - 1])


def test_xgboost_discrete_hazard_outputs_monotone_survival():
    frame = sample_frame(120)
    model = build_model_suite(("x1", "x2"), max_horizon=6).models[
        "xgboost_discrete_hazard"
    ]
    model.fit(frame.iloc[:90])
    survival = model.predict_survival(frame.iloc[90:], range(1, 7))
    assert survival.shape == (30, 6)
    assert np.all((survival >= 0.0) & (survival <= 1.0))
    assert np.all(np.diff(survival, axis=1) <= 1e-10)


def test_block_inference_and_calibration_bands_are_schema_valid():
    frame = sample_frame(80)
    horizons = tuple(range(1, 7))
    first = np.tile(np.linspace(0.95, 0.65, 6), (len(frame), 1))
    second = np.tile(np.linspace(0.92, 0.62, 6), (len(frame), 1))
    intervals, comparisons = block_bootstrap_prediction_inference(
        frame,
        {"first": first, "second": second},
        horizons,
        reference="first",
        repetitions=20,
        seed=3,
    )
    assert len(intervals) == 10
    assert len(comparisons) == 2
    assert set(comparisons["metric"]) == {
        "integrated_brier",
        "binary_cross_entropy",
    }
    assert comparisons["p_exact_block_swap"].between(0.0, 1.0).all()
    assert comparisons["block_label_swap_assumption"].str.contains(
        "exchangeability"
    ).all()
    curve = calibration_curve_block_intervals(
        frame,
        first[:, 2],
        3,
        calibration_probability=np.linspace(0.1, 0.9, 40),
        bins=3,
        repetitions=20,
        seed=3,
    )
    assert {
        "observed_ci_low",
        "observed_ci_high",
        "gap_ci_low",
        "gap_ci_high",
        "ipcw_effective_sample_size",
        "groups",
    }.issubset(curve.columns)
    assert len(curve) == 3
    assert curve["bin"].tolist() == [1, 2, 3]
    nonempty = curve["evaluable_episodes"].gt(0)
    assert (
        curve.loc[nonempty, "ipcw_effective_sample_size"]
        <= curve.loc[nonempty, "evaluable_episodes"] + 1e-12
    ).all()


def test_calibration_bins_are_frozen_from_external_predictions():
    frame = sample_frame(80)
    heldout_probability = np.linspace(0.8, 0.99, len(frame))
    calibration_probability = np.linspace(0.1, 0.7, 60)
    curve = calibration_curve_block_intervals(
        frame,
        heldout_probability,
        3,
        calibration_probability=calibration_probability,
        bins=3,
        repetitions=10,
        seed=9,
    )
    expected_inner = np.quantile(calibration_probability, [1.0 / 3.0, 2.0 / 3.0])
    assert len(curve) == 3
    assert np.allclose(curve["bin_upper"].iloc[:2], expected_inner)
    assert curve["episodes"].tolist() == [0, 0, len(frame)]
    assert curve["binning_source"].eq(
        "external calibration-partition predictions"
    ).all()


def test_circular_moving_block_inference_preserves_pairing_and_mean():
    baseline = np.asarray([0.0, 1.0, 0.0, 1.0] * 20)
    candidate = baseline + 0.25
    result = circular_moving_block_paired_inference(
        candidate,
        baseline,
        block_lengths=(4, 10),
        repetitions=100,
        seed=5,
    )
    assert result["block_length"].tolist() == [4, 10]
    assert np.allclose(result["mean_difference"], 0.25)
    assert np.allclose(result["ci_low"], 0.25)
    assert np.allclose(result["ci_high"], 0.25)
    assert result["inference_scope"].str.contains("not independent-campaign").all()


def test_censoring_at_horizon_is_unknown_and_horizon_plus_one_survives():
    frame = pd.DataFrame(
        {
            "duration_bins": [3, 4, 3, 2],
            "event": [0, 0, 1, 0],
        }
    )
    known, outcome = known_survival_outcome(frame, 3)
    assert known.tolist() == [False, True, True, False]
    assert outcome.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_ipcw_perfect_prediction_with_horizon_censoring():
    frame = pd.DataFrame(
        {
            "duration_bins": [4, 3, 2, 1],
            "event": [0, 1, 0, 1],
        }
    )
    # At h=3: first row is known to survive, second/last are failures, and the
    # row censored at 2 contributes zero IPCW weight.
    prediction = np.asarray([1.0, 0.0, 0.5, 0.0])
    assert ipcw_brier_score(frame, prediction, 3) < 1e-12


def test_ipcw_failure_weight_respects_discrete_censoring_boundary_ties():
    frame = pd.DataFrame(
        {
            "duration_bins": [2, 2, 4],
            "event": [1, 0, 0],
        }
    )
    # A censoring boundary at t=2 means status at t=2 is unknown.  Observing a
    # failure at t=2 therefore requires C>2, so the discrete IPCW denominator
    # is the post-boundary censoring survival 2/3 rather than G(2-)=1.
    from leo_service_survival.metrics import ipcw_survival_targets

    _known, _target, weights = ipcw_survival_targets(frame, 3)
    assert np.isclose(weights[0], 1.5)


def test_restricted_duration_cases_use_observed_follow_up_not_latent_time():
    from leo_service_survival.metrics import ipcw_restricted_duration_targets

    frame = pd.DataFrame(
        {
            # event before H; known survival beyond H; censor before H; censor at H
            "duration_bins": [2, 4, 2, 3],
            "event": [1, 0, 0, 0],
        }
    )
    known, reference, weights = ipcw_restricted_duration_targets(frame, horizon=3)

    assert known.tolist() == [True, True, False, False]
    assert np.allclose(reference[:2], [2.0, 3.0])
    assert np.isnan(reference[2:]).all()
    # With this hand-checkable risk set, G(2)=3/4 and G(3)=3/8.
    assert np.allclose(weights, [4.0 / 3.0, 8.0 / 3.0, 0.0, 0.0])


def test_censoring_support_and_stratified_brier_are_explicit():
    frame = pd.DataFrame(
        {
            "duration_bins": [1, 2, 4, 4, 1, 3],
            "event": [1, 0, 0, 1, 0, 1],
            "site": ["a", "a", "a", "a", "b", "b"],
        }
    )
    probability = np.asarray([0.1, 0.5, 0.9, 0.2, 0.5, 0.1])
    marginal = censoring_weighted_brier_score(frame, probability, 3)
    stratified = censoring_weighted_brier_score(
        frame, probability, 3, stratum_column="site"
    )
    assert np.isfinite(marginal)
    assert np.isfinite(stratified)
    support = censoring_support_table(frame, [1, 3], stratum_column="site")
    assert len(support) == 4
    assert {
        "events_by_horizon",
        "censored_unknown_by_horizon",
        "censoring_survival_at_horizon",
        "ipcw_effective_sample_size",
    }.issubset(support.columns)


def test_calendar_split_has_common_nonoverlapping_weeks():
    rows = []
    for week in range(12):
        for site in ("a", "b"):
            rows.append(
                {
                    "calendar_block": f"2026-W{week:02d}",
                    "start_time": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(days=7 * week),
                    "site": site,
                }
            )
    train, calibration, test = calendar_block_split(pd.DataFrame(rows), embargo_blocks=1)
    assert set(train["calendar_block"]).isdisjoint(calibration["calendar_block"])
    assert set(calibration["calendar_block"]).isdisjoint(test["calendar_block"])
    assert train["start_time"].max() < calibration["start_time"].min() < test["start_time"].min()


def test_purged_split_uses_common_boundaries():
    rows = []
    for epoch in range(300):
        for path in ("a", "b"):
            rows.append({"decision_epoch": epoch, "site": path})
    frame = pd.DataFrame(rows)
    train, calibration, test = purged_epoch_split(frame, purge_bins=10)
    assert calibration["decision_epoch"].min() - train["decision_epoch"].max() > 10
    assert test["decision_epoch"].min() - calibration["decision_epoch"].max() > 10
    assert set(train.groupby("site")["decision_epoch"].max()) == {train["decision_epoch"].max()}


def test_sparse_terminal_inference_block_is_merged_without_outcomes():
    frame = pd.DataFrame(
        {
            "calendar_block": ["w1"] * 100 + ["w2"] * 100 + ["w3"] * 3,
            "start_time": pd.date_range("2026-01-01", periods=203, freq="h", tz="UTC"),
        }
    )
    merged, audit = merge_sparse_terminal_inference_block(frame)
    assert audit["merged"] is True
    assert audit["merged_terminal_block"] == "w3"
    assert audit["merged_into_block"] == "w2"
    assert merged["analysis_block"].nunique() == 2
    assert (merged.loc[merged["calendar_block"].eq("w3"), "analysis_block"] == "w2").all()


def test_geometry_fusion_integral_is_bounded():
    survival = np.asarray([0.9, 0.8, 0.7])
    value = restricted_eust_seconds(survival, 450.0, bin_seconds=300)
    assert 0.0 <= value <= 450.0


def test_scheduler_inference_separates_generator_and_trace_layers():
    rows = []
    for block in ("w1", "w2", "w3"):
        for scenario in range(5):
            for policy, offset in (("survival_aware", 0.1), ("edf", 0.0)):
                rows.append(
                    {
                        "source_block": block,
                        "scenario": scenario,
                        "policy": policy,
                        "completion_rate_all": 0.5 + offset,
                        "priority_reward": 0.4 + offset,
                        "fairness": 0.8 + offset,
                    }
                )
    effects, inference = scheduler_two_level_inference(
        pd.DataFrame(rows), repetitions=50, seed=3
    )
    assert len(effects) == 45
    assert set(inference["uncertainty_layer"]) == {
        "conditional_generator",
        "empirical_trace_block",
    }
    assert np.allclose(inference["mean_difference"], 0.1)


def test_competing_risk_reporting_keeps_sparse_status_explicit():
    frame = pd.DataFrame(
        {
            "duration_bins": [1, 2, 3, 4, 5, 6],
            "event": [1, 1, 0, 1, 0, 0],
            "cause": ["loss", "latency", "censored", "loss", "censored", "censored"],
            "site": ["a", "a", "a", "b", "b", "b"],
            "calendar_block": ["w1", "w1", "w1", "w2", "w2", "w2"],
        }
    )
    table = competing_risk_event_adequacy(
        {"train": frame, "calibration": frame, "test": frame},
        design_columns=4,
    )
    joint_test = table.loc[
        table["partition"].eq("test") & table["cause"].eq("joint")
    ].iloc[0]
    assert joint_test["adequacy_status"] == "not_estimable"
    assert 0.0 <= aalen_johansen_cif(frame, "loss", 6) <= 1.0


def test_calibration_intercept_slope_exports_block_intervals():
    frame = sample_frame(90)
    frame["calendar_block"] = np.repeat(["w1", "w2", "w3"], 30)
    survival = np.tile(np.linspace(0.9, 0.5, 6), (len(frame), 1))
    table = calibration_intercept_slope(
        frame,
        survival[:, [0, 2, 5]],
        (1, 3, 6),
        repetitions=20,
        seed=4,
    )
    assert len(table) == 6
    assert set(table["parameter"]) == {"intercept", "slope"}
    assert table["scope"].str.contains("not an ordinary likelihood-ratio test").all()


def test_proper_score_blocks_use_fixed_weights_and_support_planning():
    frame = sample_frame(90)
    frame["calendar_block"] = np.repeat(["w1", "w2", "w3"], 30)
    horizons = tuple(range(1, 7))
    first = np.tile(np.linspace(0.9, 0.5, 6), (len(frame), 1))
    second = np.tile(np.linspace(0.8, 0.4, 6), (len(frame), 1))
    blocks, summary = paired_proper_score_block_inference(
        frame,
        {"proposed": first, "cox_ph": second},
        horizons,
        reference="proposed",
    )
    assert len(blocks) == 6
    assert set(summary["metric"]) == {"integrated_brier", "binary_cross_entropy"}
    assert blocks["weight_source"].str.contains("fixed full-test").all()
    planning = proper_score_power_planning(summary)
    assert len(planning) == 12
    assert planning["minimum_detectable_absolute_difference"].gt(0).all()


def full_chain_fixture() -> dict[str, pd.DataFrame]:
    campaign = "campaign-a"
    decision = "decision-1"
    task = "task-1"
    return {
        "campaign_manifest": pd.DataFrame(
            {
                "campaign_id": [campaign],
                "independent_cluster_id": ["site-a:2026-01-01"],
                "site_id": ["site-a"],
                "collection_date": ["2026-01-01"],
                "randomized_policy_arm": ["integrated"],
                "protocol_version": ["1.0"],
                "model_sha256": ["a" * 64],
                "config_sha256": ["b" * 64],
                "evidence_tier": ["field_live_randomized"],
            }
        ),
        "candidate_path_observation": pd.DataFrame(
            {
                "campaign_id": [campaign, campaign],
                "decision_id": [decision, decision],
                "candidate_path_id": ["leo", "5g"],
                "interface_type": ["leo", "5g"],
                "operator_id": ["leo-op", "cell-op"],
                "path_available": [True, True],
                "telemetry_sample_ns": [1_000_000_000, 1_000_000_000],
                "telemetry_cutoff_ns": [1_010_000_000, 1_010_000_000],
                "observation_age_ms": [10.0, 10.0],
                "satellite_id": ["sat-1", np.nan],
                "beam_id": ["beam-1", np.nan],
                "gateway_id": ["gw-1", "cell-gw"],
                "handover_state": ["stable", "stable"],
                "rf_snr_db": [12.0, 18.0],
                "geometry_allows_task": [True, True],
                "predicted_completion_probability": [0.9, 0.7],
                "missingness_reason": ["", "not_applicable_terrestrial"],
            }
        ),
        "task": pd.DataFrame(
            {
                "campaign_id": [campaign],
                "task_id": [task],
                "decision_id": [decision],
                "release_time_ns": [1_020_000_000],
                "required_duration_s": [5.0],
                "required_bytes": [1000],
                "priority": [1],
                "deadline_ns": [7_000_000_000],
                "task_type": ["eo_product"],
            }
        ),
        "decision_action": pd.DataFrame(
            {
                "campaign_id": [campaign],
                "task_id": [task],
                "decision_id": [decision],
                "policy_arm": ["integrated"],
                "selected_path_id": ["leo"],
                "baseline_path_id": ["5g"],
                "gate_state": ["released"],
                "gate_reason": ["all_checks_passed"],
                "shift_alarm_state": [False],
                "decision_start_ns": [1_020_000_000],
                "decision_end_ns": [1_025_000_000],
                "install_ack_ns": [1_030_000_000],
                "installed_path_id": ["leo"],
                "install_failed": [False],
                "late_install": [False],
                "rollback": [False],
            }
        ),
        "eo_resource_state": pd.DataFrame(
            {
                "campaign_id": [campaign],
                "task_id": [task],
                "decision_id": [decision],
                "satellite_id": ["sat-1"],
                "ground_station_id": ["gs-1"],
                "contact_id": ["contact-1"],
                "observation_feasible": [True],
                "storage_available_bytes": [10_000],
                "downlink_capacity_bytes": [10_000],
                "predicted_congestion": [0.1],
            }
        ),
        "outcome": pd.DataFrame(
            {
                "campaign_id": [campaign],
                "task_id": [task],
                "decision_id": [decision],
                "action_executed": [True],
                "observation_acquired": [True],
                "product_valid": [True],
                "bytes_delivered": [1000],
                "application_complete": [True],
                "completion_time_ns": [6_000_000_000],
                "met_deadline": [True],
                "no_action": [False],
                "failure_stage": ["none"],
            }
        ),
    }


def test_full_chain_contract_builds_conservative_task_ledger():
    tables = full_chain_fixture()
    audit = validate_full_chain_bundle(tables)
    assert audit.passed, audit.checks.loc[~audit.checks["passed"]].to_dict("records")
    ledger = build_full_chain_ledger(tables)
    assert len(ledger) == 1
    assert ledger.loc[0, "conservative_on_time_completion"] == 1
    assert np.isclose(ledger.loc[0, "control_loop_ms"], 10.0)
    summary = summarize_full_chain_completion(ledger)
    assert summary.loc[0, "completion_rate"] == 1.0
    assert summary.loc[0, "independent_clusters"] == 1


def test_full_chain_contract_rejects_unverified_installed_path():
    tables = full_chain_fixture()
    tables["decision_action"].loc[0, "installed_path_id"] = "5g"
    audit = validate_full_chain_bundle(tables)
    assert not audit.passed
    failed = audit.checks.loc[~audit.checks["passed"], "check"].tolist()
    assert "decision_action.installed_action_verified" in failed


def test_research_completion_audit_separates_engineering_chain_from_thesis_completion():
    tables = full_chain_fixture()
    audit = assess_research_completion(tables)
    result = audit.checks.set_index("criterion")["passed"]
    assert bool(result["continuous_field_chain"])
    assert not bool(result["independent_multisite_multioperator_validation"])
    assert bool(result["matched_satellite_beam_gateway_handover_rf"])
    assert not bool(result["operational_eo_workload_validation"])
    assert not bool(result["validated_data_driven_gate"])
    assert not bool(result["replicated_primary_endpoint_benefit"])
    assert not audit.completed
    assert audit.completed_count == 2


def test_research_completion_audit_does_not_promote_simulation_to_field_evidence():
    tables = full_chain_fixture()
    tables["campaign_manifest"].loc[0, "evidence_tier"] = "synthetic_simulation"
    audit = assess_research_completion(tables)
    assert not audit.checks["passed"].any()


def test_frozen_evidence_gate_releases_in_domain_and_fails_closed_on_shift():
    rng = np.random.default_rng(23)
    development = pd.DataFrame(
        {"latency": rng.normal(40.0, 3.0, 500), "loss": rng.beta(1.0, 30.0, 500)}
    )
    detector = FrozenMahalanobisDetector(("latency", "loss"), quantile=0.99).fit(
        development
    )
    gate = EvidenceGate(
        detector,
        GateCriteria(maximum_ood_fraction=0.20),
    )
    evidence = GateEvidence(
        schema_passed=True,
        independent_campaigns=5,
        independent_families=3,
        paired_opportunity_groups=5,
        calibration_gap_upper=0.03,
        safety_effect_lower=-0.01,
        control_loop_p95_ms=30.0,
        endpoint_effect_ci_half_width=0.03,
        validation_tier="confirmatory_release",
    )
    stable = development.iloc[:100].copy()
    released = gate.evaluate(stable, evidence)
    assert released.release_candidate
    shifted = pd.DataFrame({"latency": np.full(100, 180.0), "loss": np.full(100, 0.8)})
    blocked = gate.evaluate(shifted, evidence)
    assert not blocked.release_candidate
    assert "out_of_domain_fraction_exceeded" in blocked.reasons


def test_calibration_e_process_triggers_on_sustained_excess_loss():
    monitor = CalibrationEProcess(0.10, alpha=0.01, betting_fraction=1.0)
    for _ in range(20):
        monitor.update(1.0)
    assert monitor.alarm
    assert monitor.e_value >= 100.0


def test_gate_campaign_validation_uses_campaign_units_and_reports_delay():
    rng = np.random.default_rng(31)
    development = pd.DataFrame(
        {"latency": rng.normal(40.0, 2.0, 1000), "loss": rng.beta(1.0, 40.0, 1000)}
    )
    detector = FrozenMahalanobisDetector(("latency", "loss"), quantile=0.995).fit(
        development
    )
    gate = EvidenceGate(detector, GateCriteria(maximum_ood_fraction=0.20))
    evidence = GateEvidence(
        True,
        5,
        0.03,
        -0.01,
        20.0,
        independent_families=3,
        paired_opportunity_groups=5,
        endpoint_effect_ci_half_width=0.03,
        validation_tier="confirmatory_release",
    )
    streams = {
        "stable-a": development.iloc[:200].reset_index(drop=True),
        "stable-b": development.iloc[200:400].reset_index(drop=True),
        "shift-a": pd.DataFrame(
            {"latency": np.full(200, 180.0), "loss": np.full(200, 0.8)}
        ),
        "shift-b": pd.DataFrame(
            {"latency": np.full(200, 140.0), "loss": np.full(200, 0.6)}
        ),
    }
    labels = {key: key.startswith("shift") for key in streams}
    detail, summary = evaluate_gate_campaigns(
        gate, streams, labels, evidence, batch_size=50
    )
    assert detail["campaign_id"].nunique() == 4
    assert summary.loc[0, "inference_unit"] == "campaign"
    assert summary.loc[0, "detected_shift_campaigns"] == 2
    assert summary.loc[0, "detection_rate"] == 1.0


def test_delayed_calibration_monitor_validation_resets_each_campaign():
    losses = {
        "stable-a": np.full(40, 0.05),
        "stable-b": np.full(40, 0.08),
        "shift-a": np.full(40, 0.90),
        "shift-b": np.full(40, 1.00),
    }
    labels = {key: key.startswith("shift") for key in losses}
    detail, summary = evaluate_calibration_monitor_campaigns(
        losses,
        labels,
        null_mean=0.10,
        alpha=0.01,
        betting_fraction=1.0,
    )
    assert detail["campaign_id"].nunique() == 4
    assert summary.loc[0, "false_alarm_campaigns"] == 0
    assert summary.loc[0, "detected_shift_campaigns"] == 2
