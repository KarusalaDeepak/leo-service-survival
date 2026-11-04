import json

import pandas as pd
import pytest

from leo_service_survival.scheduler_path_bridge import (
    find_schedule_entry,
    map_schedule_to_path_candidates,
    validate_scheduler_release,
)


def test_release_requires_fail_closed_full_margin():
    valid = {
        "decision": "RETAIN_MATCHED_SIMPLE_BASELINE",
        "installed_scheduler": "full_margin",
        "authorized": False,
        "shadow_scheduler": "cross_fitted_racs",
    }
    validate_scheduler_release(valid)
    invalid = dict(valid, installed_scheduler="cross_fitted_racs", authorized=True)
    with pytest.raises(ValueError):
        validate_scheduler_release(invalid)


def test_find_schedule_entry_requires_unique_match(tmp_path):
    path = tmp_path / "schedules.jsonl"
    rows = [
        {"instance": "a", "policy": "full_margin", "plans": []},
        {"instance": "a", "policy": "full_racs_candidate6", "plans": []},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert find_schedule_entry(path, instance="a")["policy"] == "full_margin"
    with pytest.raises(ValueError):
        find_schedule_entry(path, instance="missing")


def test_exact_geometry_handoff_collapses_policy_arm_duplicates():
    schedule = {
        "plans": [
            {
                "task_id": "req::City",
                "observations": [["obs::City", "SAT", 10, 16, 0.001]],
                "deliveries": [["SAT", "SIM_GS_0", 100, 160, 0.01, 0.001]],
                "variant": "earliest-early",
                "margin": 0.8,
                "completion": 160,
            }
        ]
    }
    candidate = {
        "mission_id": "City",
        "simulation_seed": 7,
        "candidate_path_id": "contact::path::alpha",
        "path_profile_id": "alpha",
        "satellite_id": "SAT",
        "geometry_contact_id": "contact",
        "window_start_ns": 10_000_000_000,
        "window_end_ns": 20_000_000_000,
        "resource_feasible": True,
        "gate_release": True,
        "predicted_completion_probability": 0.75,
        "rf_snr_db": 12.0,
        "gateway_id": "gw",
        "evidence_tier": "synthetic_simulation",
    }
    candidates = pd.DataFrame([candidate, dict(candidate)])
    handoff, summary = map_schedule_to_path_candidates(
        schedule, candidates, scenario_creation_ns=0, simulation_seed=7
    )
    assert len(handoff) == 1
    assert bool(handoff.iloc[0]["geometry_handoff_valid"])
    assert summary.iloc[0]["unique_path_candidates_exposed"] == 1
    assert bool(summary.iloc[0]["all_observations_geometry_mapped"])
    assert not bool(summary.iloc[0]["outcome_inference_permitted"])


def test_unmapped_observation_fails_closed_in_summary():
    schedule = {
        "plans": [
            {
                "task_id": "req::City",
                "observations": [["obs::City", "OTHER", 10, 16, 0.001]],
                "deliveries": [["OTHER", "SIM_GS_0", 100, 160, 0.01, 0.001]],
                "margin": 0.8,
                "completion": 160,
            }
        ]
    }
    candidates = pd.DataFrame(
        [
            {
                "mission_id": "City",
                "simulation_seed": 7,
                "candidate_path_id": "contact::path::alpha",
                "path_profile_id": "alpha",
                "satellite_id": "SAT",
                "geometry_contact_id": "contact",
                "window_start_ns": 10_000_000_000,
                "window_end_ns": 20_000_000_000,
                "resource_feasible": True,
                "gate_release": True,
                "predicted_completion_probability": 0.75,
                "rf_snr_db": 12.0,
                "gateway_id": "gw",
                "evidence_tier": "synthetic_simulation",
            }
        ]
    )
    handoff, summary = map_schedule_to_path_candidates(
        schedule, candidates, scenario_creation_ns=0, simulation_seed=7
    )
    assert handoff.iloc[0]["handoff_status"] == "NO_EXACT_GEOMETRY_MAPPING"
    assert summary.iloc[0]["bridge_action"] == "FAIL_CLOSED_NO_PATH_HANDOFF"
