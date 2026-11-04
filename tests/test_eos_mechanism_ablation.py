from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from leo_service_survival.eos_full_chain import EOSFullChainConfig
from leo_service_survival.eos_mechanism_ablation import (
    CONDITIONS,
    POLICY_ARMS,
    EOSMechanismAblationConfig,
    ARM_CALIBRATED,
    ARM_EDF_SNR,
    condition_network_profiles,
    fit_development_completion_model,
    fit_monotone_logit_calibrator,
    run_eos_mechanism_ablation,
    write_eos_mechanism_ablation_artifacts,
)


def _satellite(satellite_id: str) -> dict:
    return {
        "id": satellite_id,
        "orbital_type": "LEO",
        "orbital_params": {"semi_major_axis_km": 7000.0, "inclination_deg": 98.0},
        "satellite_specs": {"max_data_storage_GB": 1000.0},
        "observation_capability": {
            "sensors": [
                {
                    "sensor_id": f"{satellite_id}_sensor",
                    "sensor_mode": "visible",
                    "data_rate_Mbps": 0.1,
                }
            ]
        },
        "maneuverability_capability": {"stabilization_time_s": 5.0},
    }


def _window(satellite: str, mission: str, start: str, end: str, orbit: int) -> dict:
    return {
        "satellite_id": satellite,
        "sensor_id": f"{satellite}_sensor",
        "mission_id": mission,
        "time_windows": [
            {
                "start_time": start,
                "end_time": end,
                "orbit_number": orbit,
                "agile_data": {},
                "non_agile_data": {},
            }
        ],
    }


def _write_fixture(path) -> None:
    payload = {
        "scenario_id": "AblationFixture",
        "scenario_type": "test",
        "metadata": {
            "name": "AblationFixture",
            "creation_time": "2025-01-01T00:00:00+00:00",
            "duration": 172800.0,
            "time_step": 1.0,
        },
        "satellites": [_satellite("SAT_A"), _satellite("SAT_B")],
        "missions": [
            {
                "id": "M1",
                "target_location": {"latitude": 1.0, "longitude": 2.0},
                "priority": 10.0,
                "observation_requirement": {"duration_s": 10.0},
            },
            {
                "id": "M2",
                "target_location": {"latitude": 3.0, "longitude": 4.0},
                "priority": 8.0,
                "observation_requirement": {"duration_s": 12.0},
            },
        ],
        "observation_windows": [
            _window("SAT_A", "M1", "2025-01-01T01:00:00+00:00", "2025-01-01T01:02:00+00:00", 1),
            _window("SAT_B", "M1", "2025-01-01T02:00:00+00:00", "2025-01-01T02:02:00+00:00", 1),
            _window("SAT_A", "M2", "2025-01-01T03:00:00+00:00", "2025-01-01T03:02:00+00:00", 2),
            _window("SAT_B", "M2", "2025-01-01T04:00:00+00:00", "2025-01-01T04:02:00+00:00", 2),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_conditions_encode_requested_snr_reliability_relations():
    proxy = {}
    for condition in CONDITIONS:
        profiles = condition_network_profiles(condition)
        high, low = profiles
        assert high.mean_snr_db > low.mean_snr_db
        assert high.mean_rate_mbps == low.mean_rate_mbps
        proxy[condition] = {
            profile.path_profile_id: (1 - profile.outage_probability)
            * (1 - profile.handover_probability)
            for profile in profiles
        }
    assert proxy["favorable"]["high_snr_path"] > proxy["favorable"]["lower_snr_path"]
    assert proxy["neutral"]["high_snr_path"] == pytest.approx(
        proxy["neutral"]["lower_snr_path"]
    )
    assert proxy["adverse"]["high_snr_path"] < proxy["adverse"]["lower_snr_path"]


def test_monotone_calibrator_is_bounded_and_non_decreasing():
    reference = pd.DataFrame(
        {
            "raw_synthetic_score": np.linspace(0.05, 0.95, 200),
            "technical_completion": np.r_[np.zeros(80), np.ones(120)],
        }
    )
    calibrator = fit_monotone_logit_calibrator(reference, l2=0.01)
    grid = np.linspace(0.001, 0.999, 500)
    predicted = calibrator.predict(grid)
    assert np.all((predicted >= 0.0) & (predicted <= 1.0))
    assert np.all(np.diff(predicted) >= -1e-12)
    assert calibrator.slope >= 0.0


def test_synthetic_completion_coefficients_are_development_fitted():
    rng = np.random.default_rng(44)
    rows = 500
    frame = pd.DataFrame(
        {
            "forecast_capacity_ratio": rng.uniform(1.0, 3.0, rows),
            "forecast_jitter_ms": rng.uniform(2.0, 30.0, rows),
            "profile_outage_probability": rng.uniform(0.02, 0.25, rows),
            "forecast_handover_probability": rng.uniform(0.02, 0.30, rows),
        }
    )
    latent = (
        1.4 * np.log(frame["forecast_capacity_ratio"])
        - 0.8 * np.log1p(frame["forecast_jitter_ms"])
        - 2.0 * frame["profile_outage_probability"]
        - 1.5 * frame["forecast_handover_probability"]
    )
    frame["technical_completion"] = (
        rng.random(rows) < 1.0 / (1.0 + np.exp(-latent))
    ).astype(int)
    model, selection = fit_development_completion_model(
        frame, c_grid=(0.1, 1.0)
    )
    probability = model.predict(frame)
    assert np.isfinite(probability).all()
    assert ((probability > 0.0) & (probability < 1.0)).all()
    assert selection["selected"].sum() == 1
    assert len(model.coefficients) == 4


def test_matched_ablation_runs_all_arms_and_records_model_boundary(tmp_path):
    scenario_path = tmp_path / "fixture.json"
    _write_fixture(scenario_path)
    full_chain = EOSFullChainConfig(
        seeds=(101, 202),
        release_spread_hours=0.0,
        deadline_jitter_hours=0.0,
        storage_stress_cap_bytes=500_000,
        gate_reference_rows_per_profile=200,
        bootstrap_resamples=20,
    )
    config = EOSMechanismAblationConfig(
        full_chain=full_chain,
        predictor_rows_per_profile=120,
        calibration_rows_per_profile=120,
        calibration_validation_rows_per_profile=120,
        bootstrap_resamples=20,
    )
    artifacts = run_eos_mechanism_ablation(scenario_path, config=config)
    results = artifacts.tables["task_results"]

    expected = len(CONDITIONS) * len(full_chain.seeds) * 2 * len(POLICY_ARMS)
    assert len(results) == expected
    counts = results.groupby(["condition", "simulation_seed", "logical_task_id"])[
        "policy_arm"
    ].nunique()
    assert counts.eq(len(POLICY_ARMS)).all()
    assert results["evidence_tier"].eq("synthetic_simulation").all()
    assert artifacts.run_manifest["wetlinks_fitted_model_used"] is False
    assert "No validated feature, unit or horizon adapter" in artifacts.run_manifest[
        "wetlinks_nonapplication_reason"
    ]
    assert set(artifacts.tables["policy_summary"]["policy_arm"]) == set(POLICY_ARMS)
    assert {ARM_EDF_SNR, ARM_CALIBRATED}.issubset(
        set(artifacts.tables["pairwise_inference"]["comparison_arm"])
    )
    assert set(artifacts.tables["mechanism_contrasts"]["contrast"]) == {
        "reliability_information_vs_current_snr",
        "synthetic_calibration_effect",
        "remove_input_screening",
        "remove_congestion_penalty",
    }
    assert artifacts.tables["predictor_regularization_selection"][
        "selected"
    ].groupby(artifacts.tables["predictor_regularization_selection"]["condition"]).sum().eq(1).all()
    assert artifacts.tables["calibration_regularization_selection"][
        "selected"
    ].groupby(artifacts.tables["calibration_regularization_selection"]["condition"]).sum().eq(1).all()
    assert artifacts.tables["congestion_penalty_selection"][
        "selected"
    ].groupby(artifacts.tables["congestion_penalty_selection"]["condition"]).sum().eq(1).all()
    assert np.isfinite(
        artifacts.tables["predictor_coefficients"][
            "standardized_log_odds_coefficient"
        ]
    ).all()

    audit = artifacts.tables["action_set_audit"]
    assert audit["policy_arms_using_same_action_set"].eq(len(POLICY_ARMS)).all()
    for _, group in results.groupby(["condition", "simulation_seed", "logical_task_id"]):
        assert group["action_set_sha256"].nunique() == 1

    candidates = artifacts.tables["candidate_scores"]
    expected_input_eligible = candidates["gate_feature_coverage"].eq(1.0) & candidates[
        "gate_ood_fraction"
    ].eq(0.0)
    assert candidates["input_eligible"].equals(expected_input_eligible)
    assert candidates["evidence_type"].eq("synthetic_simulation").all()

    assert results["action_selected"].equals(results["selected_path_id"].notna())
    assert (results["action_installed"] <= results["action_selected"]).all()
    assert not results["baseline_fallback"].any()
    assert results["evidence_type"].eq("synthetic_simulation").all()
    assert not results.loc[~results["action_selected"], "policy_authorized"].any()
    screened_selected = results["input_screening"] & results["action_selected"]
    assert results.loc[screened_selected, "policy_authorized"].equals(
        results.loc[screened_selected, "selected_input_eligible"]
    )
    unscreened_selected = ~results["input_screening"] & results["action_selected"]
    assert results.loc[unscreened_selected, "policy_authorized"].all()
    definitions = artifacts.tables["normalized_field_definitions"]
    assert set(definitions["field"]) == {
        "input_eligible",
        "policy_authorized",
        "baseline_fallback",
        "action_selected",
        "action_installed",
        "evidence_type",
    }

    output = write_eos_mechanism_ablation_artifacts(artifacts, tmp_path / "results")
    for name in artifacts.tables:
        assert (output / f"{name}.csv").exists()
    assert (output / "RESULTS.md").exists()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["wetlinks_fitted_model_used"] is False
    assert "task_results.csv" in manifest["artifact_sha256_before_manifest"]

    with pytest.raises(FileExistsError, match="not empty"):
        write_eos_mechanism_ablation_artifacts(artifacts, output)
