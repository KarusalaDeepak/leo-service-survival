"""Matched EOS-Bench mechanism ablation with explicit synthetic-evidence labels.

This module asks a narrow implementation question: when geometry, tasks,
candidate actions, resource constraints and potential outcomes are held fixed,
which part of a *synthetic* path-ranking mechanism changes EO completion?

The experiment deliberately does not call any arm the fitted WetLinks survival
model.  EOS-Bench plus the stochastic overlay exposes RF SNR, a forecast byte
rate, jitter, handover risk and profile outage risk.  The fitted WetLinks model
expects a different 16-variable, five-minute transport/weather feature vector
(``WETLINKS_FEATURES``).  There is no validated mapping between those schemas,
units or prediction horizons.  Applying that fitted model here would therefore
be scientifically misleading.  Instead, this module evaluates plainly named
synthetic scores and a Platt calibrator trained on disjoint synthetic reference
draws.  All exported evidence remains ``synthetic_simulation``.

All policy arms use the same EDF task order, the same feasible action set, the
same resource-conflict checks and hash-keyed potential outcomes.  Only candidate
eligibility and ranking differ.  This isolates mechanisms without tuning an arm
to win and preserves negative or tied results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .data import WETLINKS_FEATURES
from .eos_full_chain import (
    EVIDENCE_TIER,
    EOSFullChainConfig,
    NetworkPathProfile,
    ParsedEOSScenario,
    _build_common_actions,
    _exact_mcnemar_p,
    _fit_simulation_domain_gate,
    _hash_json,
    _interval_is_free,
    _prediction_probability,
    _realize_outcome,
    _reserve_interval,
    _rng_for,
    _seed_task_table,
    parse_eos_bench_scenario,
)


PROTOCOL_VERSION = "eos-mechanism-ablation-v2"

ARM_EDF_SNR = "edf_current_snr"
ARM_EDF_RELIABILITY = "edf_simple_reliability_heuristic"
ARM_UNCALIBRATED = "synthetic_uncalibrated_score"
ARM_CALIBRATED = "synthetic_calibrated_score"
ARM_NO_SCREENING = "synthetic_calibrated_no_input_screening"
ARM_NO_CONGESTION = "synthetic_calibrated_no_congestion_penalty"

POLICY_ARMS: tuple[str, ...] = (
    ARM_EDF_SNR,
    ARM_EDF_RELIABILITY,
    ARM_UNCALIBRATED,
    ARM_CALIBRATED,
    ARM_NO_SCREENING,
    ARM_NO_CONGESTION,
)

CONDITIONS: tuple[str, ...] = ("favorable", "neutral", "adverse")


@dataclass(frozen=True)
class EOSMechanismAblationConfig:
    """Frozen settings for the targeted mechanism experiment."""

    full_chain: EOSFullChainConfig = field(default_factory=EOSFullChainConfig)
    calibration_rows_per_profile: int = 3_000
    calibration_validation_rows_per_profile: int = 1_500
    calibration_seed: int = 2026102801
    calibration_validation_seed: int = 2026102802
    calibration_l2: float = 0.01
    congestion_penalty_weight: float = 0.10
    bootstrap_resamples: int = 2_000
    bootstrap_seed: int = 2026102803

    def __post_init__(self) -> None:
        if self.calibration_rows_per_profile < 100:
            raise ValueError("calibration_rows_per_profile must be at least 100")
        if self.calibration_validation_rows_per_profile < 100:
            raise ValueError("calibration_validation_rows_per_profile must be at least 100")
        if self.calibration_l2 < 0:
            raise ValueError("calibration_l2 cannot be negative")
        if self.congestion_penalty_weight < 0:
            raise ValueError("congestion_penalty_weight cannot be negative")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")


@dataclass(frozen=True)
class MonotoneLogitCalibrator:
    """Logistic calibration with a non-negative slope on raw-score log odds."""

    intercept: float
    slope: float
    l2: float
    training_rows: int
    training_prevalence: float

    def predict(self, raw_score: np.ndarray | pd.Series | float) -> np.ndarray:
        values = np.asarray(raw_score, dtype=float)
        clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
        logit = np.log(clipped / (1.0 - clipped))
        return expit(self.intercept + self.slope * logit)

    @property
    def sha256(self) -> str:
        return _hash_json(asdict(self))


@dataclass(frozen=True)
class EOSMechanismAblationArtifacts:
    tables: Mapping[str, pd.DataFrame]
    run_manifest: Mapping[str, Any]


def condition_network_profiles(condition: str) -> tuple[NetworkPathProfile, ...]:
    """Return two paths differing only in declared SNR--reliability alignment.

    Forecast and realized-rate distributions are kept equal across paths.  The
    high-SNR path is more reliable in ``favorable``, equally reliable in
    ``neutral`` and less reliable in ``adverse``.  This avoids silently making
    the SNR arm win through a higher mean byte rate.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"unknown SNR--reliability condition: {condition}")
    reliable = {"jitter_ms": 8.0, "outage_probability": 0.045, "handover_probability": 0.045}
    neutral = {"jitter_ms": 15.0, "outage_probability": 0.12, "handover_probability": 0.145}
    fragile = {"jitter_ms": 22.0, "outage_probability": 0.20, "handover_probability": 0.25}
    if condition == "favorable":
        high_risk, low_risk = reliable, fragile
    elif condition == "neutral":
        high_risk, low_risk = neutral, neutral
    else:
        high_risk, low_risk = fragile, reliable

    shared = {
        "interface_type": "synthetic_ntn",
        "mean_rate_mbps": 0.032,
        "rate_log_sd": 0.18,
        "source": f"declared_{condition}_snr_reliability_overlay",
    }
    return (
        NetworkPathProfile(
            path_profile_id="high_snr_path",
            operator_id="synthetic_operator_high_snr",
            gateway_id="synthetic_gateway_high_snr",
            mean_snr_db=14.0,
            snr_std_db=1.8,
            **shared,
            **high_risk,
        ),
        NetworkPathProfile(
            path_profile_id="lower_snr_path",
            operator_id="synthetic_operator_lower_snr",
            gateway_id="synthetic_gateway_lower_snr",
            mean_snr_db=10.5,
            snr_std_db=1.4,
            **shared,
            **low_risk,
        ),
    )


def _sample_synthetic_completion_reference(
    scenario: ParsedEOSScenario,
    profiles: tuple[NetworkPathProfile, ...],
    *,
    condition: str,
    partition: str,
    rows_per_profile: int,
    seed: int,
    full_chain: EOSFullChainConfig,
) -> pd.DataFrame:
    """Generate disjoint score/outcome rows for synthetic calibration only.

    The endpoint is successful technical delivery within the declared downlink
    contact, conditional on forecast path availability and forecast capacity.
    It is not an empirical LEO service-survival endpoint.
    """

    required_values = scenario.tasks["required_bytes"].to_numpy(dtype=int)
    if not len(required_values):
        raise ValueError("scenario has no tasks from which to sample required bytes")
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        rng = _rng_for(seed, condition, partition, profile.path_profile_id, "calibration_reference")
        accepted = 0
        attempts = 0
        while accepted < rows_per_profile:
            attempts += 1
            if attempts > rows_per_profile * 100:
                raise RuntimeError("could not generate enough forecast-feasible calibration rows")
            required_bytes = int(rng.choice(required_values))
            path_available = bool(rng.random() >= 0.015)
            forecast_rate = float(
                profile.mean_rate_mbps * math.exp(rng.normal(0.0, profile.rate_log_sd))
            )
            forecast_jitter = float(
                max(0.1, profile.jitter_ms * math.exp(rng.normal(0.0, 0.18)))
            )
            handover_forecast = float(
                np.clip(profile.handover_probability + rng.normal(0.0, 0.015), 0.0, 0.95)
            )
            capacity_bytes = int(
                max(0.0, forecast_rate * 1_000_000 / 8.0 * full_chain.downlink_contact_s)
            )
            if not path_available or capacity_bytes < required_bytes:
                continue
            capacity_ratio = capacity_bytes / max(required_bytes, 1)
            raw_score = _prediction_probability(
                path_available=True,
                capacity_ratio=capacity_ratio,
                jitter_ms=forecast_jitter,
                outage_probability=profile.outage_probability,
                handover_probability=handover_forecast,
            )

            install_ok = bool(rng.random() >= full_chain.install_failure_probability)
            acquired = bool(install_ok and rng.random() >= full_chain.acquisition_failure_probability)
            product_valid = bool(
                acquired and rng.random() >= full_chain.product_invalid_probability
            )
            handover = bool(rng.random() < handover_forecast)
            outage = bool(rng.random() < profile.outage_probability)
            realized_rate = float(
                profile.mean_rate_mbps * math.exp(rng.normal(0.0, profile.rate_log_sd))
            )
            if handover:
                realized_rate *= 0.20
            if outage or not product_valid:
                realized_rate = 0.0
            delivered = realized_rate * 1_000_000 / 8.0 * full_chain.downlink_contact_s
            outcome = int(product_valid and delivered >= required_bytes)
            rows.append(
                {
                    "condition": condition,
                    "partition": partition,
                    "path_profile_id": profile.path_profile_id,
                    "required_bytes": required_bytes,
                    "raw_synthetic_score": raw_score,
                    "technical_completion": outcome,
                    "forecast_rate_mbps": forecast_rate,
                    "forecast_jitter_ms": forecast_jitter,
                    "forecast_handover_probability": handover_forecast,
                    "profile_outage_probability": profile.outage_probability,
                    "reference_seed": int(seed),
                    "evidence_tier": EVIDENCE_TIER,
                }
            )
            accepted += 1
    return pd.DataFrame(rows)


def fit_monotone_logit_calibrator(
    reference: pd.DataFrame, *, l2: float
) -> MonotoneLogitCalibrator:
    """Fit a regularized, non-decreasing Platt map without test outcomes."""

    required = {"raw_synthetic_score", "technical_completion"}
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(f"calibration reference missing columns: {sorted(missing)}")
    raw = np.clip(reference["raw_synthetic_score"].to_numpy(float), 1e-6, 1 - 1e-6)
    target = reference["technical_completion"].to_numpy(float)
    if len(np.unique(target)) < 2:
        raise ValueError("calibration reference must contain both endpoint classes")
    x = np.log(raw / (1.0 - raw))

    prevalence = float(target.mean())
    initial = np.array([math.log(prevalence / (1.0 - prevalence)), 1.0], dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = beta[0] + beta[1] * x
        probability = expit(linear)
        epsilon = 1e-12
        loss = -np.sum(
            target * np.log(probability + epsilon)
            + (1.0 - target) * np.log(1.0 - probability + epsilon)
        )
        loss += 0.5 * l2 * float(beta[1] ** 2)
        residual = probability - target
        gradient = np.array(
            [residual.sum(), np.dot(residual, x) + l2 * beta[1]], dtype=float
        )
        return float(loss), gradient

    fitted = minimize(
        lambda beta: objective(beta)[0],
        initial,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        bounds=((None, None), (0.0, None)),
    )
    if not fitted.success:
        raise RuntimeError(f"synthetic calibration failed: {fitted.message}")
    return MonotoneLogitCalibrator(
        intercept=float(fitted.x[0]),
        slope=float(fitted.x[1]),
        l2=float(l2),
        training_rows=int(len(reference)),
        training_prevalence=prevalence,
    )


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    total = len(target)
    return float(
        sum(
            (np.sum(indices == index) / total)
            * abs(float(target[indices == index].mean()) - float(probability[indices == index].mean()))
            for index in range(bins)
            if np.any(indices == index)
        )
    )


def _calibration_diagnostics(
    validation: pd.DataFrame,
    calibrator: MonotoneLogitCalibrator,
    *,
    condition: str,
) -> dict[str, Any]:
    target = validation["technical_completion"].to_numpy(float)
    raw = validation["raw_synthetic_score"].to_numpy(float)
    calibrated = calibrator.predict(raw)
    epsilon = 1e-12

    def log_loss(probability: np.ndarray) -> float:
        clipped = np.clip(probability, epsilon, 1.0 - epsilon)
        return float(
            -np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
        )

    return {
        "condition": condition,
        "validation_rows": int(len(validation)),
        "validation_prevalence": float(target.mean()),
        "raw_brier": float(np.mean((raw - target) ** 2)),
        "calibrated_brier": float(np.mean((calibrated - target) ** 2)),
        "raw_log_loss": log_loss(raw),
        "calibrated_log_loss": log_loss(calibrated),
        "raw_ece_10bin": _ece(target, raw),
        "calibrated_ece_10bin": _ece(target, calibrated),
        "calibrator_sha256": calibrator.sha256,
        "evidence_tier": EVIDENCE_TIER,
    }


def _score_actions(
    actions: pd.DataFrame,
    calibrator: MonotoneLogitCalibrator,
    *,
    congestion_penalty_weight: float,
) -> pd.DataFrame:
    scored = actions.copy()
    scored["simple_reliability_heuristic"] = (
        (1.0 - scored["profile_outage_probability"].astype(float))
        * (1.0 - scored["forecast_handover_probability"].astype(float))
        * np.exp(-scored["forecast_jitter_ms"].astype(float) / 250.0)
    ).clip(0.0, 1.0)
    scored["raw_synthetic_score"] = scored["predicted_completion_probability"].astype(float)
    scored["calibrated_synthetic_score"] = calibrator.predict(
        scored["raw_synthetic_score"].to_numpy(float)
    )
    scored["normalized_congestion"] = scored["predicted_congestion"].astype(float).clip(0.0, 1.0)
    scored["uncalibrated_penalized_utility"] = (
        scored["raw_synthetic_score"]
        - congestion_penalty_weight * scored["normalized_congestion"]
    )
    scored["calibrated_penalized_utility"] = (
        scored["calibrated_synthetic_score"]
        - congestion_penalty_weight * scored["normalized_congestion"]
    )
    # ``input_eligible`` is deliberately independent of geometry and resource
    # feasibility. It records only whether the complete input vector is inside
    # the frozen synthetic Mahalanobis reference domain. ``gate_release`` keeps
    # its older, stricter meaning: input eligible *and* resource feasible.
    scored["input_eligible"] = (
        scored["gate_feature_coverage"].astype(float).eq(1.0)
        & scored["gate_ood_fraction"].astype(float).eq(0.0)
    )
    scored["evidence_type"] = EVIDENCE_TIER
    return scored


def arm_definitions() -> pd.DataFrame:
    """Machine-readable intervention definitions for the six matched arms."""

    rows = [
        {
            "policy_arm": ARM_EDF_SNR,
            "ranking_field": "rf_snr_db",
            "input_screening": False,
            "congestion_penalty": False,
            "predictor_kind": "current_synthetic_telemetry_baseline",
            "description": "EDF task order; highest current synthetic RF SNR first.",
        },
        {
            "policy_arm": ARM_EDF_RELIABILITY,
            "ranking_field": "simple_reliability_heuristic",
            "input_screening": False,
            "congestion_penalty": False,
            "predictor_kind": "fixed_simple_heuristic",
            "description": "EDF task order; fixed outage/handover/jitter reliability heuristic.",
        },
        {
            "policy_arm": ARM_UNCALIBRATED,
            "ranking_field": "uncalibrated_penalized_utility",
            "input_screening": True,
            "congestion_penalty": True,
            "predictor_kind": "hand_designed_synthetic_score",
            "description": "EDF; screened hand-designed synthetic score minus frozen congestion penalty.",
        },
        {
            "policy_arm": ARM_CALIBRATED,
            "ranking_field": "calibrated_penalized_utility",
            "input_screening": True,
            "congestion_penalty": True,
            "predictor_kind": "synthetically_calibrated_hand_designed_score",
            "description": "EDF; screened synthetic Platt-calibrated score minus frozen congestion penalty.",
        },
        {
            "policy_arm": ARM_NO_SCREENING,
            "ranking_field": "calibrated_penalized_utility",
            "input_screening": False,
            "congestion_penalty": True,
            "predictor_kind": "synthetically_calibrated_hand_designed_score",
            "description": "As calibrated arm, but the synthetic Mahalanobis input screen is disabled.",
        },
        {
            "policy_arm": ARM_NO_CONGESTION,
            "ranking_field": "calibrated_synthetic_score",
            "input_screening": True,
            "congestion_penalty": False,
            "predictor_kind": "synthetically_calibrated_hand_designed_score",
            "description": "As calibrated arm, but the explicit congestion penalty is removed.",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["task_order"] = "earliest_deadline_first_then_priority"
    frame["evidence_tier"] = EVIDENCE_TIER
    frame["wetlinks_fitted_model_used"] = False
    return frame


def normalized_field_definitions() -> pd.DataFrame:
    """Exact semantics for normalized decision-chain status fields."""

    rows = [
        {
            "table": "candidate_scores",
            "field": "input_eligible",
            "type": "boolean",
            "semantics": (
                "True exactly when the candidate has full gate-feature coverage and zero "
                "out-of-domain fraction under the frozen synthetic Mahalanobis input screen. "
                "It does not imply geometry, storage, capacity, policy or field eligibility."
            ),
        },
        {
            "table": "candidate_scores",
            "field": "evidence_type",
            "type": "string",
            "semantics": (
                "Evidence provenance for the row; always synthetic_simulation in this experiment."
            ),
        },
        {
            "table": "task_results",
            "field": "policy_authorized",
            "type": "boolean",
            "semantics": (
                "True exactly when an action was selected and that action satisfies the arm's "
                "declared simulation eligibility rule. Unscreened arms accept any resource-feasible "
                "selected action; screened arms additionally require input_eligible=True. This is "
                "simulation-scope authorization and is never field-release authorization."
            ),
        },
        {
            "table": "task_results",
            "field": "baseline_fallback",
            "type": "boolean",
            "semantics": (
                "True only if a non-baseline arm substitutes the EDF-current-SNR baseline after its "
                "own candidate is unauthorized. It is always False in v2 because screened arms "
                "abstain rather than fall back."
            ),
        },
        {
            "table": "task_results",
            "field": "action_selected",
            "type": "boolean",
            "semantics": (
                "True exactly when the scheduler returns a non-null selected_path_id before "
                "installation and outcome realization."
            ),
        },
        {
            "table": "task_results",
            "field": "action_installed",
            "type": "boolean",
            "semantics": (
                "True exactly when an action was selected and the synthetic installation step "
                "did not fail. It does not imply acquisition, byte delivery or task completion."
            ),
        },
        {
            "table": "task_results",
            "field": "evidence_type",
            "type": "string",
            "semantics": (
                "Evidence provenance for the row; always synthetic_simulation in this experiment."
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["protocol_version"] = PROTOCOL_VERSION
    return frame


def _schedule_arm(
    tasks: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    arm: Mapping[str, Any],
) -> tuple[dict[str, str | None], dict[str, str]]:
    """Schedule one arm with identical EDF/resource logic and arm-specific rank."""

    eligible = actions.loc[actions["resource_feasible"]].copy()
    if bool(arm["input_screening"]):
        eligible = eligible.loc[eligible["gate_release"]].copy()
    ordering = tasks.sort_values(
        ["deadline_ns", "priority", "logical_task_id"],
        ascending=[True, False, True],
    )
    selected: dict[str, str | None] = {}
    reasons: dict[str, str] = {}
    occupied_satellite: dict[str, list[tuple[int, int]]] = {}
    occupied_gateway: dict[str, list[tuple[int, int]]] = {}
    ranking_field = str(arm["ranking_field"])

    for task in ordering.to_dict(orient="records"):
        logical_id = str(task["logical_task_id"])
        candidates = eligible.loc[eligible["logical_task_id"] == logical_id].sort_values(
            [ranking_field, "window_start_ns", "candidate_path_id"],
            ascending=[False, True, True],
        )
        if candidates.empty:
            selected[logical_id] = None
            reasons[logical_id] = (
                "no_screened_resource_feasible_action"
                if bool(arm["input_screening"])
                else "no_resource_feasible_action"
            )
            continue
        choice: dict[str, Any] | None = None
        for candidate in candidates.to_dict(orient="records"):
            guard_ns = int(float(candidate["stabilization_time_s"]) * 1e9)
            observation_start = int(candidate["window_start_ns"])
            observation_end = int(candidate["observation_end_ns"])
            downlink_end = observation_end + int(candidate["planned_downlink_duration_s"] * 1e9)
            if not _interval_is_free(
                occupied_satellite,
                str(candidate["satellite_id"]),
                observation_start - guard_ns,
                observation_end + guard_ns,
            ):
                continue
            if not _interval_is_free(
                occupied_gateway,
                str(candidate["gateway_id"]),
                observation_end,
                downlink_end,
            ):
                continue
            choice = candidate
            _reserve_interval(
                occupied_satellite,
                str(candidate["satellite_id"]),
                observation_start - guard_ns,
                observation_end + guard_ns,
            )
            _reserve_interval(
                occupied_gateway,
                str(candidate["gateway_id"]),
                observation_end,
                downlink_end,
            )
            break
        selected[logical_id] = None if choice is None else str(choice["candidate_path_id"])
        reasons[logical_id] = "selected" if choice is not None else "all_eligible_actions_conflicted"
    return selected, reasons


def _run_arm(
    tasks: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    condition: str,
    seed: int,
    arm: Mapping[str, Any],
    full_chain: EOSFullChainConfig,
) -> pd.DataFrame:
    selected, reasons = _schedule_arm(tasks, actions, arm=arm)
    lookup = actions.set_index(["logical_task_id", "candidate_path_id"])
    rows: list[dict[str, Any]] = []
    for task in tasks.to_dict(orient="records"):
        logical_id = str(task["logical_task_id"])
        selected_id = selected[logical_id]
        candidate: dict[str, Any] | None = None
        if selected_id is not None:
            candidate = lookup.loc[(logical_id, selected_id)].to_dict()
            candidate["candidate_path_id"] = selected_id
        outcome = _realize_outcome(
            selected=candidate,
            task=task,
            seed=seed,
            config=full_chain,
        )
        action_selected = candidate is not None
        selected_input_eligible = bool(
            candidate is not None and candidate["input_eligible"]
        )
        policy_authorized = bool(
            action_selected
            and (
                not bool(arm["input_screening"])
                or selected_input_eligible
            )
        )
        rows.append(
            {
                "condition": condition,
                "simulation_seed": int(seed),
                "logical_task_id": logical_id,
                "policy_arm": str(arm["policy_arm"]),
                "selected_path_id": selected_id,
                "selected_path_profile_id": None if candidate is None else candidate["path_profile_id"],
                "selected_rf_snr_db": np.nan if candidate is None else candidate["rf_snr_db"],
                "selected_simple_reliability_heuristic": (
                    np.nan if candidate is None else candidate["simple_reliability_heuristic"]
                ),
                "selected_raw_synthetic_score": (
                    np.nan if candidate is None else candidate["raw_synthetic_score"]
                ),
                "selected_calibrated_synthetic_score": (
                    np.nan if candidate is None else candidate["calibrated_synthetic_score"]
                ),
                "selected_normalized_congestion": (
                    np.nan if candidate is None else candidate["normalized_congestion"]
                ),
                "selection_reason": reasons[logical_id],
                "input_screening": bool(arm["input_screening"]),
                "congestion_penalty": bool(arm["congestion_penalty"]),
                "predictor_kind": str(arm["predictor_kind"]),
                "selected_input_eligible": selected_input_eligible,
                "policy_authorized": policy_authorized,
                "baseline_fallback": False,
                "action_selected": action_selected,
                "action_installed": bool(
                    action_selected and outcome["action_executed"]
                ),
                "action_set_sha256": actions.loc[
                    actions["logical_task_id"] == logical_id, "common_action_set_sha256"
                ].iloc[0],
                "application_complete": bool(outcome["application_complete"]),
                "failure_stage": str(outcome["failure_stage"]),
                "bytes_delivered": int(outcome["bytes_delivered"]),
                "required_bytes": int(task["required_bytes"]),
                "realized_rate_mbps": float(outcome["realized_rate_mbps"]),
                "realized_outage": bool(outcome["realized_outage"]),
                "realized_handover": bool(outcome["realized_handover"]),
                "evidence_tier": EVIDENCE_TIER,
                "evidence_type": EVIDENCE_TIER,
            }
        )
    return pd.DataFrame(rows)


def _cluster_bootstrap_interval(
    frame: pd.DataFrame,
    value_column: str,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    clusters = frame["simulation_seed"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = [frame.loc[frame["simulation_seed"] == cluster, value_column] for cluster in sampled]
        values.append(float(pd.concat(pieces, ignore_index=True).mean()))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _summaries(
    task_results: pd.DataFrame,
    *,
    config: EOSMechanismAblationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_seed = (
        task_results.groupby(["condition", "simulation_seed", "policy_arm"], as_index=False)
        .agg(
            tasks=("logical_task_id", "size"),
            completed=("application_complete", "sum"),
            completion_rate=("application_complete", "mean"),
            no_action_rate=("selected_path_id", lambda value: float(value.isna().mean())),
            high_snr_path_selection_rate=(
                "selected_path_profile_id",
                lambda value: float((value == "high_snr_path").mean()),
            ),
        )
        .sort_values(["condition", "simulation_seed", "policy_arm"])
        .reset_index(drop=True)
    )

    summary_rows: list[dict[str, Any]] = []
    for (condition, arm), group in task_results.groupby(["condition", "policy_arm"], sort=True):
        low, high = _cluster_bootstrap_interval(
            group,
            "application_complete",
            resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed + POLICY_ARMS.index(str(arm)) + 100 * CONDITIONS.index(str(condition)),
        )
        summary_rows.append(
            {
                "condition": condition,
                "policy_arm": arm,
                "tasks": int(len(group)),
                "synthetic_replications": int(group["simulation_seed"].nunique()),
                "completion_rate": float(group["application_complete"].mean()),
                "seed_cluster_bootstrap_95_ci_low": low,
                "seed_cluster_bootstrap_95_ci_high": high,
                "no_action_rate": float(group["selected_path_id"].isna().mean()),
                "high_snr_path_selection_rate": float(
                    (group["selected_path_profile_id"] == "high_snr_path").mean()
                ),
                "evidence_tier": EVIDENCE_TIER,
            }
        )
    policy_summary = pd.DataFrame(summary_rows).sort_values(["condition", "policy_arm"])

    inference_rows: list[dict[str, Any]] = []
    key = ["condition", "simulation_seed", "logical_task_id"]
    pivot = task_results.pivot(index=key, columns="policy_arm", values="application_complete").reset_index()
    for condition in CONDITIONS:
        condition_pivot = pivot.loc[pivot["condition"] == condition].copy()
        for arm_index, arm in enumerate(POLICY_ARMS):
            condition_pivot["paired_difference"] = (
                condition_pivot[arm].astype(int) - condition_pivot[ARM_EDF_SNR].astype(int)
            )
            low, high = _cluster_bootstrap_interval(
                condition_pivot,
                "paired_difference",
                resamples=config.bootstrap_resamples,
                seed=config.bootstrap_seed + 1_000 + arm_index + 100 * CONDITIONS.index(condition),
            )
            improved = int((condition_pivot["paired_difference"] > 0).sum())
            regressed = int((condition_pivot["paired_difference"] < 0).sum())
            inference_rows.append(
                {
                    "condition": condition,
                    "reference_arm": ARM_EDF_SNR,
                    "comparison_arm": arm,
                    "paired_tasks": int(len(condition_pivot)),
                    "reference_completion_rate": float(condition_pivot[ARM_EDF_SNR].mean()),
                    "comparison_completion_rate": float(condition_pivot[arm].mean()),
                    "paired_completion_difference": float(condition_pivot["paired_difference"].mean()),
                    "seed_cluster_bootstrap_95_ci_low": low,
                    "seed_cluster_bootstrap_95_ci_high": high,
                    "improved_task_pairs": improved,
                    "regressed_task_pairs": regressed,
                    "tied_task_pairs": int((condition_pivot["paired_difference"] == 0).sum()),
                    "conditional_task_pair_mcnemar_p": _exact_mcnemar_p(improved, regressed),
                    "evidence_tier": EVIDENCE_TIER,
                    "inference_scope": (
                        "Controlled synthetic mechanism comparison. Seed-cluster intervals reuse one "
                        "EOS-Bench scenario and are not independent-campaign confidence intervals."
                    ),
                }
            )
    contrast_definitions = (
        (
            "reliability_information_vs_current_snr",
            ARM_EDF_SNR,
            ARM_EDF_RELIABILITY,
        ),
        (
            "synthetic_calibration_effect",
            ARM_UNCALIBRATED,
            ARM_CALIBRATED,
        ),
        (
            "remove_input_screening",
            ARM_CALIBRATED,
            ARM_NO_SCREENING,
        ),
        (
            "remove_congestion_penalty",
            ARM_CALIBRATED,
            ARM_NO_CONGESTION,
        ),
    )
    contrast_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_pivot = pivot.loc[pivot["condition"] == condition].copy()
        for contrast_index, (contrast, reference_arm, comparison_arm) in enumerate(
            contrast_definitions
        ):
            condition_pivot["paired_difference"] = (
                condition_pivot[comparison_arm].astype(int)
                - condition_pivot[reference_arm].astype(int)
            )
            low, high = _cluster_bootstrap_interval(
                condition_pivot,
                "paired_difference",
                resamples=config.bootstrap_resamples,
                seed=(
                    config.bootstrap_seed
                    + 2_000
                    + contrast_index
                    + 100 * CONDITIONS.index(condition)
                ),
            )
            improved = int((condition_pivot["paired_difference"] > 0).sum())
            regressed = int((condition_pivot["paired_difference"] < 0).sum())
            contrast_rows.append(
                {
                    "condition": condition,
                    "contrast": contrast,
                    "reference_arm": reference_arm,
                    "comparison_arm": comparison_arm,
                    "paired_tasks": int(len(condition_pivot)),
                    "reference_completion_rate": float(condition_pivot[reference_arm].mean()),
                    "comparison_completion_rate": float(condition_pivot[comparison_arm].mean()),
                    "paired_completion_difference": float(
                        condition_pivot["paired_difference"].mean()
                    ),
                    "seed_cluster_bootstrap_95_ci_low": low,
                    "seed_cluster_bootstrap_95_ci_high": high,
                    "improved_task_pairs": improved,
                    "regressed_task_pairs": regressed,
                    "tied_task_pairs": int(
                        (condition_pivot["paired_difference"] == 0).sum()
                    ),
                    "conditional_task_pair_mcnemar_p": _exact_mcnemar_p(
                        improved, regressed
                    ),
                    "evidence_tier": EVIDENCE_TIER,
                    "inference_scope": (
                        "Controlled synthetic mechanism contrast. Seed-cluster intervals reuse "
                        "one EOS-Bench scenario and are not independent-campaign confidence intervals."
                    ),
                }
            )
    return (
        per_seed,
        policy_summary.reset_index(drop=True),
        pd.DataFrame(inference_rows),
        pd.DataFrame(contrast_rows),
    )


def _condition_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for profile in condition_network_profiles(condition):
            reliability_proxy = (
                (1.0 - profile.outage_probability)
                * (1.0 - profile.handover_probability)
                * math.exp(-profile.jitter_ms / 250.0)
            )
            rows.append(
                {
                    "condition": condition,
                    "path_profile_id": profile.path_profile_id,
                    "mean_snr_db": profile.mean_snr_db,
                    "mean_rate_mbps": profile.mean_rate_mbps,
                    "jitter_ms": profile.jitter_ms,
                    "outage_probability": profile.outage_probability,
                    "handover_probability": profile.handover_probability,
                    "declared_reliability_proxy": reliability_proxy,
                    "evidence_tier": EVIDENCE_TIER,
                }
            )
    return pd.DataFrame(rows)


def run_eos_mechanism_ablation(
    scenario_path: str | Path,
    *,
    config: EOSMechanismAblationConfig | None = None,
) -> EOSMechanismAblationArtifacts:
    """Run all matched arms under favorable, neutral and adverse conditions."""

    config = config or EOSMechanismAblationConfig()
    scenario = parse_eos_bench_scenario(scenario_path)
    arm_table = arm_definitions()
    calibration_rows: list[pd.DataFrame] = []
    calibration_fit_rows: list[dict[str, Any]] = []
    calibration_diagnostic_rows: list[dict[str, Any]] = []
    candidate_tables: list[pd.DataFrame] = []
    result_tables: list[pd.DataFrame] = []
    action_audit_rows: list[dict[str, Any]] = []
    condition_model_hashes: dict[str, str] = {}

    for condition in CONDITIONS:
        profiles = condition_network_profiles(condition)
        development = _sample_synthetic_completion_reference(
            scenario,
            profiles,
            condition=condition,
            partition="calibration_development",
            rows_per_profile=config.calibration_rows_per_profile,
            seed=config.calibration_seed,
            full_chain=config.full_chain,
        )
        validation = _sample_synthetic_completion_reference(
            scenario,
            profiles,
            condition=condition,
            partition="calibration_validation",
            rows_per_profile=config.calibration_validation_rows_per_profile,
            seed=config.calibration_validation_seed,
            full_chain=config.full_chain,
        )
        calibrator = fit_monotone_logit_calibrator(development, l2=config.calibration_l2)
        development = development.copy()
        validation = validation.copy()
        development["calibrated_synthetic_score"] = calibrator.predict(
            development["raw_synthetic_score"]
        )
        validation["calibrated_synthetic_score"] = calibrator.predict(
            validation["raw_synthetic_score"]
        )
        calibration_rows.extend([development, validation])
        calibration_fit_rows.append(
            {
                "condition": condition,
                "intercept": calibrator.intercept,
                "slope": calibrator.slope,
                "l2": calibrator.l2,
                "training_rows": calibrator.training_rows,
                "training_prevalence": calibrator.training_prevalence,
                "calibrator_sha256": calibrator.sha256,
                "calibration_source": "disjoint_synthetic_reference_draws",
                "evidence_tier": EVIDENCE_TIER,
            }
        )
        calibration_diagnostic_rows.append(
            _calibration_diagnostics(validation, calibrator, condition=condition)
        )
        condition_model_hashes[condition] = calibrator.sha256

        domain_gate, gate_evidence, _ = _fit_simulation_domain_gate(profiles, config.full_chain)
        for seed in config.full_chain.seeds:
            tasks = _seed_task_table(scenario, seed, config.full_chain)
            common = _build_common_actions(
                scenario,
                tasks,
                profiles,
                seed,
                config.full_chain,
                domain_gate,
                gate_evidence,
            )
            scored = _score_actions(
                common,
                calibrator,
                congestion_penalty_weight=config.congestion_penalty_weight,
            )
            scored.insert(0, "condition", condition)
            candidate_tables.append(scored)
            for logical_id, group in scored.groupby("logical_task_id", sort=False):
                hashes = group["common_action_set_sha256"].drop_duplicates().tolist()
                if len(hashes) != 1:
                    raise AssertionError("one task has inconsistent common-action hashes")
                action_audit_rows.append(
                    {
                        "condition": condition,
                        "simulation_seed": int(seed),
                        "logical_task_id": logical_id,
                        "candidate_actions": int(len(group)),
                        "resource_feasible_actions": int(group["resource_feasible"].sum()),
                        "screened_eligible_actions": int(group["gate_release"].sum()),
                        "common_action_set_sha256": hashes[0],
                        "policy_arms_using_same_action_set": len(POLICY_ARMS),
                        "evidence_tier": EVIDENCE_TIER,
                    }
                )
            for arm in arm_table.to_dict(orient="records"):
                result_tables.append(
                    _run_arm(
                        tasks,
                        scored,
                        condition=condition,
                        seed=seed,
                        arm=arm,
                        full_chain=config.full_chain,
                    )
                )

    task_results = pd.concat(result_tables, ignore_index=True)
    per_seed, policy_summary, inference, mechanism_contrasts = _summaries(
        task_results, config=config
    )
    candidates = pd.concat(candidate_tables, ignore_index=True)
    calibration_reference = pd.concat(calibration_rows, ignore_index=True)
    tables: dict[str, pd.DataFrame] = {
        "normalized_field_definitions": normalized_field_definitions(),
        "arm_definitions": arm_table,
        "condition_definitions": _condition_table(),
        "calibration_fit": pd.DataFrame(calibration_fit_rows),
        "calibration_validation": pd.DataFrame(calibration_diagnostic_rows),
        "calibration_reference": calibration_reference,
        "candidate_scores": candidates,
        "action_set_audit": pd.DataFrame(action_audit_rows),
        "task_results": task_results,
        "paired_seed_results": per_seed,
        "policy_summary": policy_summary,
        "pairwise_inference": inference,
        "mechanism_contrasts": mechanism_contrasts,
    }

    expected_rows = (
        len(CONDITIONS)
        * len(config.full_chain.seeds)
        * len(scenario.tasks)
        * len(POLICY_ARMS)
    )
    if len(task_results) != expected_rows:
        raise AssertionError(
            f"incomplete matched experiment: expected {expected_rows} task rows, got {len(task_results)}"
        )
    arm_counts = task_results.groupby(
        ["condition", "simulation_seed", "logical_task_id"]
    )["policy_arm"].nunique()
    if not arm_counts.eq(len(POLICY_ARMS)).all():
        raise AssertionError("not every condition/seed/task has all matched policy arms")

    wetlinks_feature_units = {
        "latency_mean_ms": "milliseconds over five-minute bin",
        "latency_std_ms": "milliseconds over five-minute bin",
        "latency_max_ms": "milliseconds over five-minute bin",
        "packet_loss_pct": "percent over five-minute bin",
        "download_mbps": "megabits per second over five-minute bin",
        "upload_mbps": "megabits per second over five-minute bin",
        "temperature_c": "degrees Celsius",
        "humidity_pct": "percent",
        "rain_mm": "millimetres",
        "wind_mps": "metres per second",
        "pressure_hpa": "hectopascals",
        "latency_roll3": "rolling milliseconds",
        "loss_roll3": "rolling percent",
        "rain_latency_interaction": "millimetre-millisecond product",
        "hour_sin": "unitless",
        "hour_cos": "unitless",
    }
    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_tier": EVIDENCE_TIER,
        "scenario_path": str(Path(scenario_path).resolve()),
        "scenario_id": scenario.scenario_id,
        "source_scenario_sha256": scenario.source_sha256,
        "config": asdict(config),
        "config_sha256": _hash_json(asdict(config)),
        "conditions": list(CONDITIONS),
        "policy_arms": list(POLICY_ARMS),
        "condition_calibrator_sha256": condition_model_hashes,
        "rows": {name: int(len(frame)) for name, frame in tables.items()},
        "matched_design": (
            "Within each condition/seed/task, all six arms share the EOS geometry, synthetic "
            "telemetry, forecast-feasible actions, resource constraints and hash-keyed potential "
            "outcomes. Every arm uses EDF task order."
        ),
        "wetlinks_fitted_model_used": False,
        "wetlinks_nonapplication_reason": (
            "The EOS overlay supplies RF SNR, forecast rate, jitter, handover probability and "
            "profile outage probability. The fitted WetLinks survival model requires a 16-field "
            "five-minute transport/weather vector and predicts sustained QoS survival on a different "
            "time grid. No validated feature, unit or horizon adapter exists; applying it here would "
            "not be a valid model evaluation."
        ),
        "wetlinks_required_features": list(WETLINKS_FEATURES),
        "wetlinks_feature_units": wetlinks_feature_units,
        "eos_score_inputs": [
            "path_available",
            "forecast_capacity_ratio",
            "forecast_jitter_ms",
            "profile_outage_probability",
            "forecast_handover_probability",
        ],
        "calibration_scope": (
            "Monotone logistic calibration is fitted only on disjoint synthetic reference draws "
            "for the technical-delivery endpoint. It is not empirical WetLinks calibration and "
            "does not calibrate a field service-survival probability."
        ),
        "claim_boundary": (
            "Controlled synthetic mechanism ablation over one generated EOS-Bench scenario. "
            "Seeds are stochastic replications, not independent campaigns, operators or sites."
        ),
        "negative_results_preserved": True,
        "normalized_field_semantics": {
            row["field"]: row["semantics"]
            for row in normalized_field_definitions().to_dict(orient="records")
        },
    }
    return EOSMechanismAblationArtifacts(tables=tables, run_manifest=manifest)


def _summary_markdown(artifacts: EOSMechanismAblationArtifacts) -> str:
    summary = artifacts.tables["policy_summary"]
    diagnostics = artifacts.tables["calibration_validation"]
    lines = [
        "# EOS mechanism ablation (synthetic simulation)",
        "",
        "The fitted WetLinks survival model was **not** applied because the EOS overlay does not "
        "provide its 16 transport/weather features, five-minute units, or validated horizon adapter.",
        "",
        "All six arms use matched actions, EDF task order, shared resource constraints and common "
        "hash-keyed potential outcomes. Results below are controlled synthetic evidence only.",
        "",
        "## Completion summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Synthetic calibration validation",
        "",
        diagnostics.to_markdown(index=False),
        "",
        "See `pairwise_inference.csv` for paired differences and seed-cluster intervals. Those "
        "intervals reuse one generated scenario and are not field-inference confidence intervals.",
        "Direct calibration, input-screening and congestion-penalty ablations are in "
        "`mechanism_contrasts.csv`.",
        "",
    ]
    return "\n".join(lines)


def write_eos_mechanism_ablation_artifacts(
    artifacts: EOSMechanismAblationArtifacts,
    output_directory: str | Path,
) -> Path:
    """Write a fresh result bundle and refuse to mix it with a prior run."""

    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; choose a fresh run path: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in artifacts.tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    (destination / "RESULTS.md").write_text(
        _summary_markdown(artifacts), encoding="utf-8"
    )
    artifact_hashes = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = dict(artifacts.run_manifest)
    manifest["artifact_sha256_before_manifest"] = artifact_hashes
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return destination
