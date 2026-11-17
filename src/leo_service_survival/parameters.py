"""Authoritative numerical-parameter definitions for Paper 4.

The project previously scattered numerical choices across endpoint builders,
models, gates and synthetic experiments.  This module gives every primary
number a category and a declared selection rule.  A value in this file is not
automatically a physical constant: ``application_requirement`` and
``controlled_factor`` values define the evaluated use case, while
``development_selected`` values must be selected without test outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ServiceEndpointProfile:
    profile_id: str = "interactive_service_v1"
    bin_minutes: int = 5
    latency_limit_ms: float = 80.0
    loss_limit_pct: float = 1.0
    sustained_bad_bins: int = 2
    maximum_gap_bins: int = 2
    analysis_horizon_bins: int = 12


@dataclass(frozen=True)
class GateRequirementProfile:
    profile_id: str = "confirmatory_release_v1"
    validation_tier: str = "confirmatory_release"
    minimum_independent_campaigns: int = 5
    minimum_feature_coverage: float = 0.95
    maximum_ood_fraction: float = 0.10
    maximum_calibration_gap_upper: float = 0.05
    minimum_candidate_minus_baseline_effect_lower: float = -0.02
    maximum_control_loop_p95_ms: float = 100.0
    confidence_level: float = 0.95
    minimum_independent_families: int = 3
    minimum_paired_opportunity_groups: int = 5
    maximum_endpoint_effect_ci_half_width: float = 0.05


PRIMARY_ENDPOINT = ServiceEndpointProfile()
PROOF_OF_METHOD_GATE_REQUIREMENTS = GateRequirementProfile(
    profile_id="proof_of_method_v1",
    validation_tier="proof_of_method",
    minimum_independent_campaigns=1,
    minimum_independent_families=1,
    minimum_paired_opportunity_groups=1,
    maximum_endpoint_effect_ci_half_width=1.0,
)
EXTERNAL_VALIDATION_GATE_REQUIREMENTS = GateRequirementProfile(
    profile_id="external_validation_v1",
    validation_tier="external_validation",
    minimum_independent_campaigns=2,
    minimum_independent_families=2,
    minimum_paired_opportunity_groups=2,
    maximum_endpoint_effect_ci_half_width=0.10,
)
CONFIRMATORY_GATE_REQUIREMENTS = GateRequirementProfile()
PRIMARY_GATE_REQUIREMENTS = CONFIRMATORY_GATE_REQUIREMENTS

# These are bounded development-only search sets, not claims that one value is
# universally correct.  The selected value must be frozen before test scoring.
FOCAL_HAZARD_SEARCH_GRID: tuple[dict[str, Any], ...] = (
    {
        "learning_rate": 0.03,
        "max_iter": 240,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.06,
        "max_iter": 180,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 0.5,
    },
    {
        "learning_rate": 0.04,
        "max_iter": 240,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 50,
        "l2_regularization": 2.0,
    },
)
CALIBRATION_C_GRID: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)
SCHEDULER_CONGESTION_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
SYNTHETIC_PREDICTOR_C_GRID: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0)
SYNTHETIC_CALIBRATION_L2_GRID: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
SYNTHETIC_CONGESTION_WEIGHT_GRID: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5)
BASELINE_SEARCH_GRIDS: dict[str, tuple[Any, ...]] = {
    "weibull_aft_l2": (1e-4, 1e-3, 1e-2),
    "cox_ph_l2": (1e-3, 5e-3, 2e-2),
    "additive_alpha": (1.0, 5.0, 20.0),
    "linear_hazard_c": (0.1, 0.5, 2.0),
    "random_forest": (
        {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 10},
        {"n_estimators": 120, "max_depth": 10, "min_samples_leaf": 20},
        {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 40},
    ),
    "xgboost": (
        {"n_estimators": 160, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 12.0},
        {"n_estimators": 240, "learning_rate": 0.04, "max_depth": 4, "min_child_weight": 12.0},
        {"n_estimators": 240, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 20.0},
    ),
}


def parameter_registry() -> pd.DataFrame:
    """Return the machine-readable primary parameter provenance table."""

    rows = [
        ("endpoint.bin_minutes", 5, "min", "application_requirement", "Declared aggregation of irregular burst summaries; separate rebuilt-endpoint audit evaluates 5/10 minutes and marks 1 minute unsupported by source cadence without interpolation.", "1, 5, 10"),
        ("endpoint.latency_limit_ms", 80.0, "ms", "application_requirement", "Illustrative QoS profile chosen by the authors, not a supplied operator SLA or universal LEO failure constant; alternatives are reported without selecting a favourable test endpoint.", "70, 80, 90"),
        ("endpoint.loss_limit_pct", 1.0, "%", "application_requirement", "Declared packet-loss budget for the investigated service profile.", "0.5, 1, 2"),
        ("endpoint.sustained_bad_bins", 2, "bins", "application_requirement", "Requires persistence so one noisy aggregate is not treated as failure.", "1, 2, 3"),
        ("endpoint.maximum_gap_bins", 2, "bins", "measurement_rule", "The complete WetLinks grid immediately censors any missing bin, so this gap bound is structurally inactive there; identical rebuilt-label fingerprints are not validation of gap spanning.", "1, 2, 3"),
        ("endpoint.analysis_horizon_bins", 12, "bins", "application_requirement", "Covers tasks from 5 through 60 minutes on the declared five-minute grid.", "6, 9, 12"),
        ("model.focal_hazard", "grid", "configuration", "development_selected", "Choose the lowest development IPCW Brier score from a bounded equal-budget grid; freeze before test.", str(FOCAL_HAZARD_SEARCH_GRID)),
        ("model.calibration_C", "grid", "inverse L2", "development_selected", "Choose only within development data and retain raw prediction when support is inadequate.", str(CALIBRATION_C_GRID)),
        ("model.baseline_hyperparameters", "family-specific grids", "configuration", "development_selected", "Each trainable baseline receives a bounded family-specific development grid and is frozen before test scoring.", str(BASELINE_SEARCH_GRIDS)),
        ("calibration.minimum_evaluable", 25, "episodes", "estimation_guard", "Prevents fitting a two-parameter calibration map to extremely small support; event counts are reported separately.", "25, 50, 100"),
        ("probability.clip", 1e-6, "probability", "numerical_safeguard", "Avoids infinite log odds; predictions are restored to [0,1] after calibration.", "1e-8, 1e-6, 1e-4"),
        ("model.total_hazard_bounds", "[1e-8, 0.95] in preserved empirical core", "conditional probability", "model_safeguard", "A historical probability floor/cap, not an empirically identified constant; separate development-only bound sensitivity reports activation and prediction changes without replacing the frozen fit.", "upper 0.95, 0.99, 1-1e-8, 1; lower 1e-8 or 0"),
        ("gate.detector_quantile", 0.99, "development quantile", "application_requirement", "Candidate-level coverage budget for a development-fitted input-domain guard.", "0.975, 0.99, 0.995"),
        ("gate.runtime_minimum_feature_coverage", 0.90, "fraction", "application_requirement", "Illustrative runtime batch alarm distinct from the stricter release coverage budget; measured feature-only sweeps do not calibrate field false-alarm rates.", "0.90, 0.95, 1.00"),
        ("monitor.reference_mean_squared_error", 0.10, "squared probability error", "monitor_null_requirement", "Defines the conditional error null being monitored, not a measured universal reference; synthetic sequence sweeps are logic checks only.", "0.05, 0.10, 0.20"),
        ("monitor.betting_fraction", 1.0, "unitless", "monitor_design_choice", "Fixed before monitoring; changes alarm response under the declared bounded-loss null, not physical hazard strength.", "0.5, 1, 2"),
        ("monitor.alpha", 0.01, "probability budget", "monitor_risk_budget", "Declared alarm-risk budget under the monitored conditional null; not learned to obtain favourable detections.", "0.005, 0.01, 0.05"),
        ("gate.evidence_tiers", "1 proof; 2 external; 5 confirmatory", "campaign groups", "study_design_profiles", "One campaign supports internal proof only; two independent campaign families can support a frozen external test; five groups and three families are the conservative deployment-authorization profile, subject to precision and power.", "power/precision may require more"),
        ("gate.minimum_independent_campaigns", 5, "campaigns", "confirmatory_eligibility_floor", "Five is not an IEEE or statistical constant. It is the conservative profile's eligibility floor and permits one complete leave-one-campaign-out cycle.", "1 proof; 2 external; 5 confirmatory"),
        ("gate.minimum_independent_families", 3, "acquisition families", "confirmatory_eligibility_floor", "Limits authorization based on several campaigns collected by one instrument or study family.", "1 proof; 2 external; 3 confirmatory"),
        ("gate.minimum_paired_opportunity_groups", 5, "groups", "confirmatory_eligibility_floor", "Only groups with paired candidate and baseline opportunities contribute to the policy-effect bound.", "1 proof; 2 external; 5 confirmatory"),
        ("gate.minimum_feature_coverage", 0.95, "fraction", "application_requirement", "Allows at most 5% missing required feature cells before fallback.", "0.90, 0.95, 1.00"),
        ("gate.maximum_ood_fraction", 0.10, "fraction", "application_requirement", "Batch fallback budget for rows outside the frozen development domain.", "0.05, 0.10, 0.20"),
        ("gate.maximum_calibration_gap_upper", 0.05, "probability", "application_requirement", "Largest acceptable upper confidence bound on absolute probability error.", "0.025, 0.05, 0.10"),
        ("gate.minimum_candidate_minus_baseline_effect_lower", -0.02, "completion fraction", "noninferiority_requirement", "Illustrative tolerated-deficit budget of two percentage points; no supplied stakeholder cost study establishes operational acceptability. Risk-budget sweeps test logic, not field authorization.", "-0.01, -0.02, -0.05"),
        ("gate.maximum_control_loop_p95_ms", 100.0, "ms", "system_requirement", "Sub-100-ms decision-loop budget; deployment hardware must measure it.", "50, 100, 200"),
        ("scheduler.congestion_weight", "grid", "unitless", "development_selected", "Maximize completion on calibration scenarios; ties select the smaller penalty; test scenarios remain untouched.", str(SCHEDULER_CONGESTION_WEIGHT_GRID)),
        ("scheduler.end_slot_penalty", 0.0, "unitless", "removed", "The unsupported additive coefficient was removed; earliest end time is now a deterministic tie-break only.", "not applicable"),
        ("synthetic.predictor_regularization", "grid", "inverse L2", "development_selected", "Synthetic outcome coefficients are learned from disjoint development draws rather than written into a reliability equation.", str(SYNTHETIC_PREDICTOR_C_GRID)),
        ("synthetic.congestion_weight", "grid", "unitless", "development_selected", "Selected on disjoint tuning seeds and frozen for evaluation seeds.", str(SYNTHETIC_CONGESTION_WEIGHT_GRID)),
        ("synthetic.condition_levels", "favorable, neutral, adverse", "factor levels", "controlled_factor", "A three-level mechanism grid, not measured operator prevalence.", "report each level separately"),
    ]
    return pd.DataFrame(
        rows,
        columns=(
            "parameter",
            "primary_value_or_rule",
            "unit",
            "category",
            "rationale_or_selection_rule",
            "required_sensitivity",
        ),
    )


def parameter_profile_manifest() -> dict[str, Any]:
    return {
        "endpoint": asdict(PRIMARY_ENDPOINT),
        "gate_requirements": asdict(PRIMARY_GATE_REQUIREMENTS),
        "gate_requirement_profiles": {
            "proof_of_method": asdict(PROOF_OF_METHOD_GATE_REQUIREMENTS),
            "external_validation": asdict(EXTERNAL_VALIDATION_GATE_REQUIREMENTS),
            "confirmatory_release": asdict(CONFIRMATORY_GATE_REQUIREMENTS),
        },
        "focal_hazard_search_grid": list(FOCAL_HAZARD_SEARCH_GRID),
        "calibration_c_grid": list(CALIBRATION_C_GRID),
        "scheduler_congestion_weight_grid": list(SCHEDULER_CONGESTION_WEIGHT_GRID),
        "synthetic_predictor_c_grid": list(SYNTHETIC_PREDICTOR_C_GRID),
        "synthetic_calibration_l2_grid": list(SYNTHETIC_CALIBRATION_L2_GRID),
        "synthetic_congestion_weight_grid": list(SYNTHETIC_CONGESTION_WEIGHT_GRID),
        "baseline_search_grids": BASELINE_SEARCH_GRIDS,
    }
