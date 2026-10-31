from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from leo_service_survival.evidence_audit import (
    audit_commect_paper2,
    audit_commect_paper4,
    audit_eos_candidate_fallback,
    audit_eos_effect,
    describe_gate_mechanisms,
)


ROOT = Path(__file__).resolve().parents[1]


def _compact_eos_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows = []
    decision_rows = []
    definitions = [
        ("task-1", [("a", True, False), ("b", True, True)], "a", "released", "pass"),
        ("task-2", [("c", False, True)], None, "no_action", "no_resource_feasible_action"),
    ]
    for arm in ("edf_reactive", "integrated_risk_gated"):
        campaign = f"campaign::{arm}"
        for logical_task, choices, selected, state, reason in definitions:
            decision = f"{arm}::{logical_task}"
            decision_rows.append(
                {
                    "campaign_id": campaign,
                    "decision_id": decision,
                    "simulation_seed": 7,
                    "logical_task_id": logical_task,
                    "policy_arm": arm,
                    "selected_path_id": selected,
                    "gate_state": "baseline" if arm == "edf_reactive" else state,
                    "gate_reason": "baseline" if arm == "edf_reactive" else reason,
                    "shift_alarm_state": False,
                }
            )
            for candidate_id, feasible, ood in choices:
                candidate_rows.append(
                    {
                        "campaign_id": campaign,
                        "decision_id": decision,
                        "simulation_seed": 7,
                        "logical_task_id": logical_task,
                        "candidate_path_id": candidate_id,
                        "resource_feasible": feasible,
                        "gate_release": bool(feasible and not ood),
                        "gate_feature_coverage": 1.0,
                        "gate_ood_fraction": float(ood),
                        "gate_reasons": (
                            "out_of_domain_fraction_exceeded" if ood else ""
                        ),
                    }
                )
    return pd.DataFrame(candidate_rows), pd.DataFrame(decision_rows)


def test_candidate_ood_is_not_equated_with_decision_fallback() -> None:
    candidates, decisions = _compact_eos_tables()
    audit = audit_eos_candidate_fallback(candidates, decisions)

    assert audit["exported_candidate_rows"] == 6
    assert audit["exported_ood_candidate_rows"] == 4
    assert audit["unique_common_action_records"] == 3
    assert audit["unique_common_action_ood_records"] == 2
    assert audit["release_rule_violations"] == 0
    assert audit["choice_sets_with_any_ood_candidate"] == 2
    assert audit["choice_sets_with_any_resource_feasible_ood_candidate"] == 1
    assert audit["selected_released_actions"] == 1
    assert audit["selected_ood_actions"] == 0
    assert audit["integrated_fallback_decisions"] == 0
    assert audit["candidate_ood_implies_decision_fallback"] is False


def _endpoint_row(policy: str, task: str, success: bool) -> dict:
    return {
        "policy_arm": policy,
        "simulation_seed": 11,
        "logical_task_id": task,
        "conservative_on_time_completion": int(success),
        "action_executed": success,
        "no_action": not success,
        "install_failed": False,
        "late_install": False,
        "geometry_allows_task": success,
        "observation_acquired": success,
        "product_valid": success,
        "bytes_delivered": 10 if success else 0,
        "required_bytes": 10,
        "application_complete": success,
        "met_deadline": success,
    }


def test_eos_effect_audit_reconstructs_endpoint_and_seed_interval() -> None:
    ledger = pd.DataFrame(
        [
            _endpoint_row("edf_reactive", "t1", True),
            _endpoint_row("edf_reactive", "t2", False),
            _endpoint_row("integrated_risk_gated", "t1", True),
            _endpoint_row("integrated_risk_gated", "t2", True),
        ]
    )
    seed = pd.DataFrame(
        [{"simulation_seed": 11, "paired_tasks": 2, "paired_difference": 0.5}]
    )
    audit = audit_eos_effect(
        ledger,
        seed,
        {"cluster_bootstrap_95_ci": [0.5, 0.5]},
        bootstrap_seed=4,
        bootstrap_resamples=20,
    )

    assert audit["endpoint_rule_violations"] == 0
    assert audit["paired_task_denominator"] == 2
    assert audit["paired_completion_difference"] == 0.5
    assert audit["paired_improved_tasks"] == 1
    assert audit["paired_tied_tasks"] == 1
    assert audit["interval_reconstruction_matches_recorded"]


def test_commect_protocols_keep_horizons_and_denominators_distinct() -> None:
    paper4_summary = pd.DataFrame(
        [
            {
                "policy": "survival_candidate",
                "scheduled_epochs": 1225,
                "action_epochs": 1107,
                "evaluable_epochs": 959,
                "complete_case_success": 0.808133472367049,
                "conservative_scheduled_success": 775 / 1225,
            },
            {
                "policy": "reactive",
                "scheduled_epochs": 1225,
                "action_epochs": 1107,
                "evaluable_epochs": 975,
                "complete_case_success": 0.72,
                "conservative_scheduled_success": 702 / 1225,
            },
            {
                "policy": "evidence_gated_survival",
                "scheduled_epochs": 1225,
                "action_epochs": 1107,
                "evaluable_epochs": 975,
                "complete_case_success": 0.72,
                "conservative_scheduled_success": 702 / 1225,
            },
        ]
    )
    paper4_inference = pd.DataFrame(
        [
            {
                "reference": "survival_candidate",
                "baseline": "reactive",
                "block_length": 60,
                "ci_low": 0.031020408163265307,
                "ci_high": 0.09142857142857143,
            }
        ]
    )
    paper4 = audit_commect_paper4(
        paper4_summary,
        paper4_inference,
        {
            "train_end_epoch": 1,
            "calibration_start_epoch": 32,
            "calibration_end_epoch": 64,
            "test_start_epoch": 96,
            "label_lookahead_bins": 30,
            "train_calibration_gap": 31,
            "calibration_test_gap": 32,
        },
    )

    paper2_summary = pd.DataFrame(
        [
            {
                "policy_name": "qos_filter_then_ensemble_selector",
                "decision_count": 812,
                "success_rate_under_60ms": 734 / 812,
            },
            {
                "policy_name": "reactive_greedy",
                "decision_count": 812,
                "success_rate_under_60ms": 717 / 812,
            },
            {
                "policy_name": "validation_gated_qos_selector",
                "decision_count": 812,
                "success_rate_under_60ms": 717 / 812,
            },
        ]
    )
    paper2 = audit_commect_paper2(
        paper2_summary,
        {
            "scheduled_test_epoch_count": 948,
            "split_boundary_audit": {
                "ratios": {
                    "train": 0.55,
                    "calibration": 0.15,
                    "selection": 0.15,
                    "test": 0.15,
                },
                "partitions": {
                    "test": {"first_bin_epoch": 100, "last_bin_epoch": 1047}
                },
            },
        },
    )

    assert paper4["experiment_id"] == "P4-COMMECT-5S-CS"
    assert paper4["selection_horizon_seconds"] == 5
    assert paper4["scheduled_test_seconds"] == 1225
    assert paper4["candidate_successes"] == 775
    assert paper4["reactive_successes"] == 702
    assert paper2["experiment_id"] == "P2-COMMECT-1S-CC-FTR"
    assert paper2["prediction_horizon_seconds"] == 1
    assert paper2["complete_case_decisions"] == 812
    assert paper2["candidate_successes"] == 734
    assert paper2["baseline_successes"] == 717


def test_gate_mechanisms_have_nonoverlapping_units() -> None:
    immediate = pd.DataFrame(
        [
            {
                "evidence_tier": "synthetic_simulation",
                "stable_campaigns": 30,
                "false_alarm_campaigns": 0,
                "shift_campaigns": 30,
                "detected_shift_campaigns": 30,
                "median_detection_delay_rows": 0.0,
            }
        ]
    )
    delayed = pd.DataFrame(
        [
            {
                "evidence_tier": "synthetic_simulation",
                "stable_campaigns": 30,
                "false_alarm_campaigns": 0,
                "shift_campaigns": 30,
                "detected_shift_campaigns": 30,
                "median_detection_delay_outcomes": 9.0,
            }
        ]
    )
    mechanisms = describe_gate_mechanisms(immediate, delayed)

    assert [item["mechanism_id"] for item in mechanisms] == [
        "input_ood_guard",
        "delayed_stream_monitor",
        "offline_campaign_admission_gate",
    ]
    assert len({item["unit"] for item in mechanisms}) == 3


def test_frozen_eos_v2_exports_have_audited_counts() -> None:
    eos = ROOT / "results" / "eos_full_chain_synthetic_v2"
    if not (eos / "candidate_path_observation.csv").is_file():
        pytest.skip(
            "requires the private row-level EOS evidence bundle; "
            "redistribution-safe aggregate outputs are provided instead"
        )
    candidate_columns = [
        "campaign_id",
        "decision_id",
        "simulation_seed",
        "logical_task_id",
        "candidate_path_id",
        "resource_feasible",
        "gate_release",
        "gate_feature_coverage",
        "gate_ood_fraction",
        "gate_reasons",
    ]
    decision_columns = [
        "campaign_id",
        "decision_id",
        "simulation_seed",
        "logical_task_id",
        "policy_arm",
        "selected_path_id",
        "gate_state",
        "gate_reason",
        "shift_alarm_state",
    ]
    audit = audit_eos_candidate_fallback(
        pd.read_csv(eos / "candidate_path_observation.csv", usecols=candidate_columns),
        pd.read_csv(eos / "decision_action.csv", usecols=decision_columns),
    )

    assert audit["exported_candidate_rows"] == 59640
    assert audit["exported_ood_candidate_rows"] == 714
    assert audit["unique_common_action_records"] == 29820
    assert audit["unique_common_action_ood_records"] == 357
    assert audit["release_rule_compliant_rows"] == 59640
    assert audit["choice_sets_with_any_ood_candidate"] == 283
    assert audit["choice_sets_with_any_resource_feasible_ood_candidate"] == 127
    assert audit["integrated_decision_gate_states"] == {
        "no_action": 181,
        "released": 819,
    }
    assert audit["selected_released_actions"] == 819
    assert audit["selected_ood_actions"] == 0
    assert audit["integrated_fallback_decisions"] == 0

    ledger_columns = [
        "policy_arm",
        "simulation_seed",
        "logical_task_id",
        "conservative_on_time_completion",
        "action_executed",
        "no_action",
        "install_failed",
        "late_install",
        "geometry_allows_task",
        "observation_acquired",
        "product_valid",
        "bytes_delivered",
        "required_bytes",
        "application_complete",
        "met_deadline",
    ]
    effect = audit_eos_effect(
        pd.read_csv(eos / "task_ledger.csv", usecols=ledger_columns),
        pd.read_csv(eos / "paired_seed_results.csv"),
        json.loads((eos / "paired_inference.json").read_text(encoding="utf-8")),
    )
    assert effect["endpoint_rule_violations"] == 0
    assert effect["policy_results"]["edf_reactive"]["successful_tasks"] == 557
    assert effect["policy_results"]["integrated_risk_gated"]["successful_tasks"] == 736
    assert effect["paired_completion_difference"] == 0.179
    assert effect["recorded_cluster_bootstrap_95_ci"] == [0.139, 0.225]
    assert effect["interval_reconstruction_matches_recorded"]
