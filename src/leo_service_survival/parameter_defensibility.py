"""Retrospective, development-only numerical-choice robustness diagnostics.

This add-on does not replace the frozen empirical model.  Nested chronological
folds are constructed exclusively inside its original training partition.
Held-out calibration/test rows are used only for a feature-distribution audit;
their outcomes cannot select search settings or guard risk budgets.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .gating import (
    CalibrationEProcess,
    EvidenceGate,
    FrozenMahalanobisDetector,
    GateCriteria,
    GateEvidence,
    GATE_LOGIC_FIXTURE,
)
from .metrics import censoring_weighted_brier_score, ipcw_survival_targets
from .models import (
    CoxPHModel,
    DiscreteHazardModel,
    ExponentialModel,
    KaplanMeierModel,
    _expand_person_period,
)
from .parameters import CALIBRATION_C_GRID, FOCAL_HAZARD_SEARCH_GRID


EXPANDED_CALIBRATION_C_GRID = (0.003, 0.01, *CALIBRATION_C_GRID)
EXPANDED_FOCAL_GRID = (
    *tuple(deepcopy(config) for config in FOCAL_HAZARD_SEARCH_GRID),
    {**FOCAL_HAZARD_SEARCH_GRID[0], "learning_rate": 0.015},
    {**FOCAL_HAZARD_SEARCH_GRID[0], "learning_rate": 0.0075},
)
GUARD_QUANTILES = (0.975, 0.99, 0.995)
OOD_FRACTION_BUDGETS = (0.05, 0.10, 0.20)
FEATURE_COVERAGE_BUDGETS = (0.90, 0.95, 1.00)
CALIBRATION_MINIMUM_SUPPORT = (25, 50, 100)
PROBABILITY_CLIPS = (1e-8, 1e-6, 1e-4)
HAZARD_CAPS = (0.95, 0.99, 1 - 1e-8, 1.0)


class EpisodeSeparatedHazardModel(DiscreteHazardModel):
    """Disable random expanded-row early stopping in this temporal audit.

    The old frozen model is not silently changed.  A fixed declared iteration
    budget makes every fitted candidate use only its past fitting episodes.
    """

    def fit(self, frame: pd.DataFrame) -> "EpisodeSeparatedHazardModel":
        self.encoder.fit(frame)
        x, y, weights, cause_to_class = _expand_person_period(
            frame, self.encoder, self.max_horizon, competing=self.competing
        )
        self.cause_to_class_ = cause_to_class
        if np.unique(y).size < 2:
            raise ValueError("training prefix lacks two hazard classes")
        self.model_ = HistGradientBoostingClassifier(
            **self.hyperparameters,
            random_state=self.random_state,
            early_stopping=False,
        )
        self.model_.fit(x, y, sample_weight=weights)
        self.classes_ = np.asarray(self.model_.classes_, dtype=int)
        return self


@dataclass
class TemporalAuditFold:
    fold: int
    inner_fit: pd.DataFrame
    inner_calibration: pd.DataFrame
    inner_validation: pd.DataFrame
    outer_fit: pd.DataFrame
    outer_calibration: pd.DataFrame
    outer_validation: pd.DataFrame
    audit: dict[str, Any]


def _partition(
    frame: pd.DataFrame,
    blocks: list[str],
    *,
    next_start: pd.Timestamp | None,
) -> tuple[pd.DataFrame, int]:
    part = frame.loc[frame["calendar_block"].astype(str).isin(blocks)].copy()
    removed = 0
    if next_start is not None:
        if "analysis_label_resolution_time" not in part:
            raise ValueError("confirmation-aware label resolution metadata is required")
        keep = pd.to_datetime(part["analysis_label_resolution_time"], utc=True).lt(next_start)
        removed = int((~keep).sum())
        part = part.loc[keep].copy()
    if part.empty:
        raise ValueError("empty temporal audit partition after look-ahead purge")
    return part.sort_values(["start_time", "site"]).reset_index(drop=True), removed


def nested_development_folds(
    training_frame: pd.DataFrame,
    *,
    outer_validation_blocks: int = 2,
    folds: int = 2,
) -> list[TemporalAuditFold]:
    """Expanding past-only nested folds, with one whole block at boundaries.

    Inner fit/calibration/validation select settings; the next outer validation
    is used once to describe their stability.  Both inner and outer calibration
    use two complete blocks.  A confirmation-time purge is applied in addition
    to the block embargo.  Fold overlap is retained and explicitly descriptive.
    """

    if folds < 1 or outer_validation_blocks < 1:
        raise ValueError("folds and validation block counts must be positive")
    frame = training_frame.copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], utc=True)
    chronology = (
        frame.groupby("calendar_block")["start_time"].min().sort_values().index.astype(str).tolist()
    )
    first_outer = len(chronology) - folds * outer_validation_blocks
    if first_outer < 10:
        raise ValueError("not enough original training blocks for embargoed nested folds")
    output = []
    for fold_index in range(folds):
        outer_start = first_outer + fold_index * outer_validation_blocks
        prefix = chronology[: outer_start - 1]
        outer_validation_groups = chronology[outer_start : outer_start + outer_validation_blocks]
        # Prefix's final two blocks are calibration; the preceding block is an
        # embargo.  The same shape gives a separate inner selection validation.
        outer_fit_groups, outer_cal_groups = prefix[:-3], prefix[-2:]
        inner_fit_groups = prefix[:-6]
        inner_cal_groups = prefix[-5:-3]
        inner_val_groups = prefix[-2:]

        def start(groups: list[str]) -> pd.Timestamp:
            return frame.loc[frame["calendar_block"].astype(str).isin(groups), "start_time"].min()

        specifications = {
            "inner_fit": (inner_fit_groups, start(inner_cal_groups)),
            "inner_calibration": (inner_cal_groups, start(inner_val_groups)),
            "inner_validation": (inner_val_groups, start(outer_validation_groups)),
            "outer_fit": (outer_fit_groups, start(outer_cal_groups)),
            "outer_calibration": (outer_cal_groups, start(outer_validation_groups)),
            "outer_validation": (outer_validation_groups, None),
        }
        parts: dict[str, pd.DataFrame] = {}
        audit: dict[str, Any] = {"fold": fold_index, "scope": "original_training_only"}
        for role, (groups, next_start) in specifications.items():
            part, removed = _partition(frame, groups, next_start=next_start)
            parts[role] = part
            audit[role] = {
                "blocks": groups,
                "rows": int(len(part)),
                "events": int(part["event"].sum()),
                "purged_confirmation_lookahead_rows": removed,
                "start": part["start_time"].min().isoformat(),
                "end": part["start_time"].max().isoformat(),
                "label_resolution_end": pd.to_datetime(
                    part["analysis_label_resolution_time"], utc=True
                ).max().isoformat(),
            }
        for earlier, later in (
            ("inner_fit", "inner_calibration"),
            ("inner_calibration", "inner_validation"),
            ("outer_fit", "outer_calibration"),
            ("outer_calibration", "outer_validation"),
        ):
            if not (
                pd.to_datetime(parts[earlier]["analysis_label_resolution_time"], utc=True).max()
                < parts[later]["start_time"].min()
            ):
                raise AssertionError("label look-ahead crosses temporal audit boundary")
        output.append(TemporalAuditFold(fold=fold_index, **parts, audit=audit))
    return output


def calibrated_prediction(
    raw_calibration: np.ndarray,
    calibration: pd.DataFrame,
    raw_prediction: np.ndarray,
    horizons: Iterable[int],
    *,
    regularization_c: float,
    minimum_support: int = 25,
    probability_clip: float = 1e-6,
) -> tuple[np.ndarray, int]:
    """Calibrate on past calibration rows only; never inspect evaluation labels."""

    horizons = tuple(horizons)
    if raw_calibration.shape != (len(calibration), len(horizons)):
        raise ValueError("calibration shape mismatch")
    if raw_prediction.ndim != 2 or raw_prediction.shape[1] != len(horizons):
        raise ValueError("prediction shape mismatch")
    if regularization_c <= 0 or minimum_support < 1 or not 0 < probability_clip < 0.5:
        raise ValueError("invalid calibration configuration")
    prediction = np.asarray(raw_prediction, dtype=float).copy()
    fitted = 0
    for column, horizon in enumerate(horizons):
        known, target, weights = ipcw_survival_targets(calibration, int(horizon))
        if known.sum() < minimum_support or np.unique(target[known]).size < 2:
            continue
        raw = np.clip(raw_calibration[known, column], probability_clip, 1 - probability_clip)
        design = np.log(raw / (1 - raw)).reshape(-1, 1)
        model = LogisticRegression(C=float(regularization_c), max_iter=1000, random_state=17)
        model.fit(design, target[known], sample_weight=weights[known])
        current = np.clip(prediction[:, column], probability_clip, 1 - probability_clip)
        prediction[:, column] = model.predict_proba(np.log(current / (1 - current)).reshape(-1, 1))[:, 1]
        fitted += 1
    return np.minimum.accumulate(np.clip(prediction, 0, 1), axis=1), fitted


def integrated_brier(frame: pd.DataFrame, prediction: np.ndarray, horizons: Iterable[int]) -> float:
    return float(np.mean([
        censoring_weighted_brier_score(frame, prediction[:, column], int(horizon))
        for column, horizon in enumerate(horizons)
    ]))


def select_grid(table: pd.DataFrame, *, expanded: bool) -> pd.Series:
    eligible = table if expanded else table.loc[
        table["configuration_index"].lt(len(FOCAL_HAZARD_SEARCH_GRID))
        & table["calibration_c"].isin(CALIBRATION_C_GRID)
    ]
    if eligible.empty:
        raise ValueError("search grid contains no eligible original candidates")
    return eligible.sort_values([
        "inner_validation_integrated_brier", "configuration_index", "calibration_c"
    ], kind="stable").iloc[0]


def run_search_boundary_audit(
    training: pd.DataFrame,
    features: tuple[str, ...],
    *,
    max_horizon: int = 12,
    configs: tuple[Mapping[str, float | int], ...] = EXPANDED_FOCAL_GRID,
    c_grid: tuple[float, ...] = EXPANDED_CALIBRATION_C_GRID,
) -> dict[str, Any]:
    """Fit expanded and original candidates on identical past-only rows."""

    horizons = tuple(range(1, max_horizon + 1))
    all_search, scores, sensitivities, freeze_rows, folds = [], [], [], [], []
    for fold in nested_development_folds(training):
        folds.append(fold.audit)
        search_rows = []
        for config_index, config in enumerate(configs):
            base = EpisodeSeparatedHazardModel(
                features, max_horizon=max_horizon, competing=True, hyperparameters=config
            ).fit(fold.inner_fit)
            raw_cal = base.predict_survival(fold.inner_calibration, horizons)
            raw_val = base.predict_survival(fold.inner_validation, horizons)
            for c in c_grid:
                prediction, fitted = calibrated_prediction(
                    raw_cal, fold.inner_calibration, raw_val, horizons, regularization_c=c
                )
                search_rows.append({
                    "fold": fold.fold,
                    "configuration_index": config_index,
                    **config,
                    "calibration_c": c,
                    "fitted_calibration_horizons": fitted,
                    "inner_validation_integrated_brier": integrated_brier(fold.inner_validation, prediction, horizons),
                    "evidence_role": "inner_development_selection",
                })
        search = pd.DataFrame(search_rows)
        all_search.extend(search_rows)
        for expanded, label in ((False, "original_grid"), (True, "expanded_grid")):
            winner = select_grid(search, expanded=expanded)
            config = dict(configs[int(winner["configuration_index"])])
            chosen_c = float(winner["calibration_c"])
            freeze_rows.append({
                "fold": fold.fold,
                "search": label,
                "configuration": config,
                "calibration_c": chosen_c,
                "selected_without_outer_validation_outcomes": True,
                "learning_rate_at_expanded_lower_boundary": bool(config["learning_rate"] == min(x["learning_rate"] for x in configs)),
                "calibration_c_at_expanded_lower_boundary": bool(chosen_c == min(c_grid)),
                "calibration_c_at_upper_boundary": bool(chosen_c == max(c_grid)),
            })
            base = EpisodeSeparatedHazardModel(
                features, max_horizon=max_horizon, competing=True, hyperparameters=config
            ).fit(fold.outer_fit)
            raw_cal = base.predict_survival(fold.outer_calibration, horizons)
            raw_val = base.predict_survival(fold.outer_validation, horizons)
            prediction, fitted = calibrated_prediction(
                raw_cal, fold.outer_calibration, raw_val, horizons, regularization_c=chosen_c
            )
            scores.append({
                "fold": fold.fold, "method": label,
                "integrated_brier": integrated_brier(fold.outer_validation, prediction, horizons),
                "evaluation_rows": len(fold.outer_validation),
                "fitted_calibration_horizons": fitted,
                "evidence_role": "outer_development_validation_descriptive",
            })
            if expanded:
                for minimum in CALIBRATION_MINIMUM_SUPPORT:
                    for clip in PROBABILITY_CLIPS:
                        varied, fitted = calibrated_prediction(
                            raw_cal, fold.outer_calibration, raw_val, horizons,
                            regularization_c=chosen_c, minimum_support=minimum, probability_clip=clip,
                        )
                        sensitivities.append({
                            "fold": fold.fold, "minimum_evaluable": minimum,
                            "probability_clip": clip, "frozen_calibration_c": chosen_c,
                            "integrated_brier": integrated_brier(fold.outer_validation, varied, horizons),
                            "fitted_calibration_horizons": fitted,
                            "maximum_probability_change_from_primary": float(np.max(np.abs(varied - prediction))),
                            "evidence_role": "outer_development_sensitivity_not_retuning",
                        })
        # Transparent low-complexity references, not an all-family superiority
        # search.  All use the same fit rows; no validation-derived coefficients.
        for name, model in (
            ("kaplan_meier", KaplanMeierModel()),
            ("constant_exponential", ExponentialModel()),
            ("cox_ph_reference_l2_0.005", CoxPHModel(features, l2=0.005)),
        ):
            model.fit(fold.outer_fit)
            prediction = model.predict_survival(fold.outer_validation, horizons)
            scores.append({
                "fold": fold.fold, "method": name,
                "integrated_brier": integrated_brier(fold.outer_validation, prediction, horizons),
                "evaluation_rows": len(fold.outer_validation),
                "fitted_calibration_horizons": 0,
                "evidence_role": "outer_development_validation_descriptive",
            })
    score_table = pd.DataFrame(scores)
    summary = score_table.groupby("method", sort=False).agg(
        mean_outer_development_integrated_brier=("integrated_brier", "mean"),
        minimum_outer_development_integrated_brier=("integrated_brier", "min"),
        maximum_outer_development_integrated_brier=("integrated_brier", "max"),
        folds=("fold", "nunique"),
    ).reset_index()
    paired = score_table.pivot(index="fold", columns="method", values="integrated_brier").reset_index()
    paired["expanded_minus_original_integrated_brier"] = paired["expanded_grid"] - paired["original_grid"]
    return {
        "search": pd.DataFrame(all_search), "scores": score_table,
        "summary": summary, "paired": paired,
        "calibration_sensitivity": pd.DataFrame(sensitivities),
        "selection_freeze": freeze_rows, "fold_audit": folds,
    }


def feature_guard_sensitivity(
    training: pd.DataFrame,
    partitions: Mapping[str, pd.DataFrame],
    features: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature-only descriptive guard audit, never a field false-alarm estimate."""

    rows, batches = [], []
    # Explicitly restrict the guard's inputs to features; outcome columns are
    # not accepted by the fitting/scoring path below.
    development_features = training.loc[:, features].copy()
    for quantile in GUARD_QUANTILES:
        detector = FrozenMahalanobisDetector(features, quantile=quantile).fit(development_features)
        for role, frame in partitions.items():
            groups = [("pooled", "all", frame)]
            groups += [("site", str(site), local) for site, local in frame.groupby("site")]
            for group_kind, group_id, local in groups:
                current = local.loc[:, features].copy()
                distances = detector.score(current)
                outliers = distances > detector.threshold_
                for cutoff in OOD_FRACTION_BUDGETS:
                    rows.append({
                        "partition": role, "group_kind": group_kind, "group": group_id,
                        "detector_quantile": quantile,
                        "maximum_ood_fraction": cutoff,
                        "training_distance_threshold": detector.threshold_,
                        "rows": len(current), "outlier_rows": int(outliers.sum()),
                        "observed_ood_fraction": float(outliers.mean()),
                        "feature_coverage": detector.feature_coverage(current),
                        "primary_feature_coverage_requirement": 0.95,
                        "primary_feature_coverage_failed": bool(detector.feature_coverage(current) < 0.95),
                        "ood_budget_exceeded": bool(outliers.mean() > cutoff),
                        "evidence_role": "feature_only_descriptive_not_authorization",
                    })
            for (site, block), local in frame.groupby(["site", "calendar_block"]):
                current = local.loc[:, features]
                fraction = float(detector.outlier_mask(current).mean())
                for cutoff in OOD_FRACTION_BUDGETS:
                    batches.append({
                        "partition": role, "site": str(site), "calendar_block": str(block),
                        "detector_quantile": quantile,
                        "maximum_ood_fraction": cutoff, "rows": len(current),
                        "observed_ood_fraction": fraction,
                        "ood_budget_exceeded": bool(fraction > cutoff),
                        "evidence_role": "site_calendar_batch_descriptive",
                    })
    return pd.DataFrame(rows), pd.DataFrame(batches)


def governance_logic_sensitivity(training: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Exercise declared risk-budget choices using an explicitly labelled fixture."""

    from itertools import product

    detector = FrozenMahalanobisDetector(features).fit(training.loc[:, features])
    # This is an expressly constructed complete-feature fixture, not a
    # measured authorization case.  Otherwise missingness in the measured
    # development table would make every budget row fail the same coverage
    # check and obscure the logic boundaries being exercised here.
    current = training.loc[:, features].fillna(detector.medians_).copy()
    evidence = GateEvidence(
        schema_passed=True, independent_campaigns=5, independent_families=3,
        paired_opportunity_groups=5, calibration_gap_upper=0.04,
        candidate_minus_baseline_endpoint_effect_lower=-0.015,
        control_loop_p95_ms=75.0, endpoint_effect_ci_half_width=0.04,
        evidence_basis=GATE_LOGIC_FIXTURE, validation_tier="confirmatory_release",
    )
    rows = []
    for gap, effect, latency, precision in product(
        (0.025, 0.05, 0.10), (-0.01, -0.02, -0.05), (50.0, 100.0, 200.0), (0.025, 0.05, 0.10)
    ):
        criteria = GateCriteria(
            maximum_calibration_gap_upper=gap,
            minimum_candidate_minus_baseline_endpoint_effect_lower=effect,
            maximum_control_loop_p95_ms=latency,
            maximum_endpoint_effect_ci_half_width=precision,
        )
        decision = EvidenceGate(detector, criteria).evaluate(current, evidence)
        rows.append({
            "maximum_calibration_gap_upper": gap,
            "minimum_candidate_minus_baseline_endpoint_effect_lower": effect,
            "maximum_control_loop_p95_ms": latency,
            "maximum_endpoint_effect_ci_half_width": precision,
            "release": decision.release_candidate,
            "statistical_authorization_supported": decision.statistical_authorization_supported,
            "reasons": ";".join(decision.reasons),
            "evidence_role": "gate_logic_fixture_not_statistical_evidence",
        })
    return pd.DataFrame(rows)


def coverage_budget_sensitivity(guard: pd.DataFrame) -> pd.DataFrame:
    """Report the full coverage/quantile/OOD-budget grid without selection."""

    rows = []
    for row in guard.to_dict(orient="records"):
        for budget in FEATURE_COVERAGE_BUDGETS:
            rows.append({
                **row,
                "minimum_feature_coverage": budget,
                "coverage_budget_failed": bool(row["feature_coverage"] < budget),
                "input_budgets_pass": bool(
                    row["feature_coverage"] >= budget and not row["ood_budget_exceeded"]
                ),
                "not_a_complete_release_gate_decision": True,
            })
    return pd.DataFrame(rows)


def hazard_cap_sensitivity(
    training: pd.DataFrame,
    features: tuple[str, ...],
    selection_freeze: list[dict[str, Any]],
    *,
    max_horizon: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit the substantive total-hazard cap without selecting a new cap.

    Each fold reuses its previously selected development configuration.  Every
    cap receives its own probability calibration on past calibration episodes;
    outer validation labels only describe sensitivity.  The 1.0 alternative
    uses the classifier's valid [0,1] probabilities with neither an artificial
    positive floor nor an upper cap below one.  Other alternatives retain the
    frozen implementation's 1e-8 lower floor.  No frozen model is changed.
    """

    horizons = tuple(range(1, max_horizon + 1))
    rows, hits = [], []
    selected = {int(row["fold"]): row for row in selection_freeze if row["search"] == "expanded_grid"}
    for fold in nested_development_folds(training):
        frozen = selected[fold.fold]
        model = EpisodeSeparatedHazardModel(
            features, max_horizon=max_horizon, competing=True,
            hyperparameters=frozen["configuration"],
        ).fit(fold.outer_fit)
        matrices = {}
        for role, frame in (("past_calibration", fold.outer_calibration), ("outer_development_validation", fold.outer_validation)):
            step = model._step_probabilities(frame, max_horizon)
            total = step[:, :, 1:].sum(axis=2)
            matrices[role] = total
            for cap in HAZARD_CAPS:
                floor = 0.0 if cap == 1.0 else 1e-8
                altered = np.clip(total, floor, cap)
                survival = np.cumprod(1 - altered, axis=1)
                before = np.column_stack((np.ones(len(frame)), survival[:, :-1]))
                # This reproduces the uncalibrated implementation's use of raw
                # cause hazards with capped total survival; a discrepancy is a
                # diagnostic, not silently repaired in the frozen estimator.
                unadjusted_cif_total = (before * total).sum(axis=1)
                failure = 1 - survival[:, -1]
                hits.append({
                    "fold": fold.fold, "partition": role, "cap": cap, "floor": floor,
                    "episodes": len(frame), "hazard_cells": total.size,
                    "raw_hazard_cells_above_cap": int((total > cap).sum()),
                    "raw_hazard_cells_below_floor": int((total < floor).sum()),
                    "episodes_with_hazard_altered": int(np.any(total != altered, axis=1).sum()),
                    "maximum_raw_total_hazard": float(total.max()),
                    "minimum_raw_total_hazard": float(total.min()),
                    "maximum_hazard_adjustment": float(np.max(np.abs(total - altered))),
                    "maximum_unadjusted_cif_mass_discrepancy": float(np.max(np.abs(unadjusted_cif_total - failure))),
                    "evidence_role": "development_only_cap_and_unadjusted_cif_diagnostic",
                })
        reference_raw = None
        reference_calibrated = None
        for cap in HAZARD_CAPS:
            floor = 0.0 if cap == 1.0 else 1e-8
            raw_cal = np.cumprod(1 - np.clip(matrices["past_calibration"], floor, cap), axis=1)
            raw_val = np.cumprod(1 - np.clip(matrices["outer_development_validation"], floor, cap), axis=1)
            calibrated, fitted = calibrated_prediction(
                raw_cal, fold.outer_calibration, raw_val, horizons,
                regularization_c=float(frozen["calibration_c"]),
            )
            if reference_raw is None:
                reference_raw, reference_calibrated = raw_val.copy(), calibrated.copy()
            rows.append({
                "fold": fold.fold, "cap": cap, "floor": floor,
                "cap_description": "probability_preserving_no_artificial_floor_or_upper_cap" if cap == 1.0 else "total_hazard_model_safeguard",
                "frozen_calibration_c": float(frozen["calibration_c"]),
                "fitted_calibration_horizons": fitted,
                "raw_integrated_brier": integrated_brier(fold.outer_validation, raw_val, horizons),
                "calibrated_integrated_brier": integrated_brier(fold.outer_validation, calibrated, horizons),
                "maximum_raw_probability_change_from_cap_095": float(np.max(np.abs(raw_val - reference_raw))),
                "mean_raw_probability_change_from_cap_095": float(np.mean(np.abs(raw_val - reference_raw))),
                "maximum_calibrated_probability_change_from_cap_095": float(np.max(np.abs(calibrated - reference_calibrated))),
                "cap_selected_using_validation": False,
                "evidence_role": "outer_development_cap_sensitivity_not_retuning",
            })
    return pd.DataFrame(rows), pd.DataFrame(hits)


def parameter_coverage_addendum() -> pd.DataFrame:
    return pd.DataFrame([{
        "parameter": "model.discrete_total_hazard_clip",
        "primary_value_or_rule": "[1e-8,0.95] in frozen empirical model",
        "category": "model_safeguard_potentially_substantive_not_physical_coefficient",
        "rationale_or_selection_rule": "Restricts extreme person-period hazard probabilities; not a learned physical risk. Alternatives are audited, not selected using validation/test outcomes.",
        "executed_sensitivity": "caps 0.95,0.99,1-1e-8; probability-preserving [0,1] with no artificial floor or upper cap",
        "evidence_role": "two_nested_development_folds_only",
        "retained_output": "hazard_cap_sensitivity.csv;hazard_cap_hits.csv",
        "primary_model_modified": False,
    }])


def monitor_logic_sensitivity() -> pd.DataFrame:
    """Vary declared e-process risk budgets on controlled loss sequences.

    No sequence is a measured campaign, so time-to-alarm is only a software
    logic response, not field detection power or a false-alarm rate estimate.
    """

    from itertools import product

    sequences = {
        "zero_loss": np.zeros(500),
        "constant_0.05": np.full(500, 0.05),
        "constant_0.25": np.full(500, 0.25),
        "controlled_abrupt_shift": np.r_[np.full(100, 0.05), np.full(400, 0.50)],
    }
    rows = []
    for null_mean, alpha, fraction in product((0.05, 0.10, 0.20), (0.005, 0.01, 0.05), (0.5, 1.0, 2.0)):
        for name, sequence in sequences.items():
            monitor = CalibrationEProcess(null_mean, alpha=alpha, betting_fraction=fraction)
            first_alarm = None
            for step, loss in enumerate(sequence, start=1):
                monitor.update(float(loss))
                if monitor.alarm and first_alarm is None:
                    first_alarm = step
            rows.append({
                "sequence": name, "null_mean": null_mean,
                "alpha": alpha, "betting_fraction": fraction,
                "observations": len(sequence), "first_alarm_observation": first_alarm,
                "alarm": first_alarm is not None,
                "evidence_role": "controlled_monitor_logic_not_field_false_alarm_or_power",
            })
    return pd.DataFrame(rows)


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
