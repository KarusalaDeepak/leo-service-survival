from __future__ import annotations

import json

import pandas as pd
import pytest

from leo_service_survival.eos_full_chain import (
    EOSFullChainConfig,
    POLICY_BASELINE,
    POLICY_INTEGRATED,
    parse_eos_bench_scenario,
    run_eos_full_chain_experiment,
    write_eos_full_chain_artifacts,
)
from leo_service_survival.integration import REQUIRED_COLUMNS, validate_full_chain_bundle


def _satellite(satellite_id: str) -> dict:
    return {
        "id": satellite_id,
        "orbital_type": "LEO",
        "orbital_params": {
            "semi_major_axis_km": 7000.0,
            "inclination_deg": 98.0,
        },
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
        "scenario_id": "Fixture_S2",
        "scenario_type": "test",
        "metadata": {
            "name": "Fixture_S2",
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
            {
                "id": "M3",
                "target_location": {"latitude": 5.0, "longitude": 6.0},
                "priority": 5.0,
                "observation_requirement": {"duration_s": 8.0},
            },
            {
                "id": "M_NO_WINDOW",
                "target_location": {"latitude": 7.0, "longitude": 8.0},
                "priority": 1.0,
                "observation_requirement": {"duration_s": 9.0},
            },
        ],
        "observation_windows": [
            _window(
                "SAT_A",
                "M1",
                "2025-01-01T01:00:00+00:00",
                "2025-01-01T01:01:00+00:00",
                1,
            ),
            _window(
                "SAT_A",
                "M2",
                "2025-01-01T01:00:05+00:00",
                "2025-01-01T01:01:10+00:00",
                1,
            ),
            _window(
                "SAT_B",
                "M2",
                "2025-01-01T02:00:00+00:00",
                "2025-01-01T02:01:30+00:00",
                1,
            ),
            _window(
                "SAT_B",
                "M3",
                "2025-01-01T03:00:00+00:00",
                "2025-01-01T03:01:00+00:00",
                1,
            ),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parser_reads_actual_fixture_entities(tmp_path):
    scenario_path = tmp_path / "fixture.json"
    _write_fixture(scenario_path)
    scenario = parse_eos_bench_scenario(scenario_path)

    assert scenario.scenario_id == "Fixture_S2"
    assert set(scenario.tasks["mission_id"]) == {"M1", "M2", "M3", "M_NO_WINDOW"}
    assert set(scenario.satellites["satellite_id"]) == {"SAT_A", "SAT_B"}
    assert len(scenario.contacts) == 4
    assert scenario.contacts["source_geometry"].eq("EOS-Bench observation_windows").all()


def test_full_chain_fixture_passes_strict_audit_and_is_paired(tmp_path):
    scenario_path = tmp_path / "fixture.json"
    _write_fixture(scenario_path)
    config = EOSFullChainConfig(
        seeds=(101, 202),
        release_spread_hours=0.0,
        deadline_jitter_hours=0.0,
        storage_stress_cap_bytes=500_000,
        bootstrap_resamples=50,
        bootstrap_seed=303,
    )
    artifacts = run_eos_full_chain_experiment(scenario_path, config=config)

    audit = validate_full_chain_bundle(artifacts.tables)
    assert audit.passed, audit.checks.to_dict(orient="records")
    assert artifacts.audit.passed
    assert set(artifacts.tables) == set(REQUIRED_COLUMNS)
    for frame in artifacts.tables.values():
        assert "evidence_tier" in frame
        assert frame["evidence_tier"].eq("synthetic_simulation").all()

    task_ids = set(artifacts.tables["task"]["task_id"])
    assert task_ids == set(artifacts.tables["decision_action"]["task_id"])
    assert task_ids == set(artifacts.tables["outcome"]["task_id"])
    assert task_ids == set(artifacts.tables["eo_resource_state"]["task_id"])

    campaigns = artifacts.tables["campaign_manifest"].set_index("campaign_id")
    candidates = artifacts.tables["candidate_path_observation"].copy()
    candidates["policy_arm"] = candidates["campaign_id"].map(
        campaigns["randomized_policy_arm"]
    )
    action_hashes = candidates.groupby(
        ["simulation_seed", "logical_task_id", "policy_arm"]
    )["common_action_set_sha256"].first().unstack("policy_arm")
    assert action_hashes[POLICY_BASELINE].equals(action_hashes[POLICY_INTEGRATED])
    assert candidates["gate_method"].eq("frozen_ledoit_wolf_mahalanobis").all()
    assert candidates["gate_scope"].eq(
        "simulation_release_only_not_field_authorization"
    ).all()
    assert candidates["gate_feature_coverage"].eq(1.0).all()
    assert candidates["gate_ood_fraction"].isin([0.0, 1.0]).all()

    logical_counts = artifacts.ledger.groupby(
        ["simulation_seed", "logical_task_id"]
    )["policy_arm"].nunique()
    assert logical_counts.eq(2).all()
    assert len(artifacts.paired_seed_results) == 2
    assert artifacts.completion_status["strict_completion_count"] == 1
    criteria = artifacts.completion_status["criteria"]
    assert criteria["continuous_geometry_path_scheduling_actual_completion"]["satisfied"]
    assert not criteria["operational_eo_workload_validation"]["satisfied"]
    assert "Mahalanobis" in criteria[
        "data_driven_gate_or_distribution_shift_detector"
    ]["reason"]

    output = write_eos_full_chain_artifacts(artifacts, tmp_path / "results")
    for table_name in REQUIRED_COLUMNS:
        assert (output / f"{table_name}.csv").exists()
    assert (output / "task_ledger.csv").exists()
    assert (output / "completion_status.json").exists()
    written = pd.read_csv(output / "full_chain_audit.csv")
    assert written["passed"].all()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert "task_ledger.csv" in manifest["artifact_sha256_before_manifest"]

    with pytest.raises(FileExistsError, match="not empty"):
        write_eos_full_chain_artifacts(artifacts, output)
