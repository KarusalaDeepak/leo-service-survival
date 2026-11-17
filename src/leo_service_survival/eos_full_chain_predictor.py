"""Development-fitted probabilities for the synthetic EO full-chain experiment.

This is deliberately not the measured WetLinks survival model.  Its target is
single-action technical completion under the declared stochastic EO overlay,
conditional on forecast feasibility.  All risk/profile inputs are simulator
assumptions known before selection, not empirically identified physical risks.

Training, regularization selection, probability calibration and final
probability diagnostics use four separately generated partitions.  Validation
labels never select a model, calibration map, gate or scheduling policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


FEATURE_NAMES: tuple[str, ...] = (
    "log_forecast_capacity_ratio",
    "log_profile_mean_capacity_ratio",
    "profile_rate_log_sd",
    "profile_outage_probability",
    "forecast_handover_probability",
    "log1p_forecast_jitter_ms",
)
TARGET = "technical_completion"


def predictor_design(frame: pd.DataFrame) -> np.ndarray:
    """Use forecast-time fields only; no realized future quantity is read."""

    required = {
        "forecast_capacity_ratio", "profile_mean_rate_mbps", "required_bytes",
        "planned_downlink_duration_s", "profile_rate_log_sd",
        "profile_outage_probability", "forecast_handover_probability",
        "forecast_jitter_ms",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"synthetic full-chain predictor missing inputs: {sorted(missing)}")
    numeric = frame[list(required)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("synthetic full-chain predictor inputs must be finite")
    if (numeric["required_bytes"] <= 0).any():
        raise ValueError("required bytes must be positive")
    mean_ratio = (
        numeric["profile_mean_rate_mbps"].to_numpy(float) * 1_000_000 / 8.0
        * numeric["planned_downlink_duration_s"].to_numpy(float)
        / numeric["required_bytes"].to_numpy(float)
    )
    return np.column_stack([
        np.log(np.clip(numeric["forecast_capacity_ratio"].to_numpy(float), 1e-12, None)),
        np.log(np.clip(mean_ratio, 1e-12, None)),
        numeric["profile_rate_log_sd"].to_numpy(float),
        numeric["profile_outage_probability"].to_numpy(float),
        numeric["forecast_handover_probability"].to_numpy(float),
        np.log1p(np.clip(numeric["forecast_jitter_ms"].to_numpy(float), 0.0, None)),
    ])


@dataclass(frozen=True)
class FullChainCompletionPredictor:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    regularization_c: float
    calibration_intercept: float
    calibration_slope: float
    training_rows: int
    tuning_rows: int
    calibration_rows: int

    def raw_predict(self, frame: pd.DataFrame) -> np.ndarray:
        design = predictor_design(frame)
        standardized = (design - np.asarray(self.means)) / np.asarray(self.scales)
        return expit(self.intercept + standardized @ np.asarray(self.coefficients))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = np.clip(self.raw_predict(frame), 1e-12, 1.0 - 1e-12)
        probability = expit(
            self.calibration_intercept
            + self.calibration_slope * np.log(raw / (1.0 - raw))
        )
        if "path_available" in frame:
            probability = np.where(frame["path_available"].to_numpy(bool), probability, 0.0)
        return probability

    @property
    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FullChainPredictorBundle:
    model: FullChainCompletionPredictor
    tables: Mapping[str, pd.DataFrame]
    manifest: Mapping[str, Any]


def _assert_disjoint_partitions(partitions: Mapping[str, pd.DataFrame]) -> None:
    if set(partitions) != {"training", "tuning", "calibration", "validation"}:
        raise ValueError("exactly training/tuning/calibration/validation partitions are required")
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for role, frame in partitions.items():
        if len(frame) < 100 or frame[TARGET].nunique() != 2:
            raise ValueError(f"{role} needs at least 100 rows and both completion classes")
        if not frame["partition"].eq(role).all():
            raise ValueError(f"incorrect partition label in {role}")
        identifiers = set(frame["reference_row_id"].astype(str))
        seeds = set(frame["reference_seed"].astype(int))
        if len(identifiers) != len(frame) or identifiers & seen_ids or seeds & seen_seeds:
            raise ValueError("predictor partitions must have disjoint unique row IDs and seeds")
        seen_ids.update(identifiers)
        seen_seeds.update(seeds)


def _fit_calibration(raw: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Fit, not hand-set, an order-preserving probability map."""

    values = np.clip(raw, 1e-12, 1.0 - 1e-12)
    x = np.log(values / (1.0 - values))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = beta[0] + beta[1] * x
        probability = expit(linear)
        loss = np.mean(np.logaddexp(0.0, linear) - target * linear)
        residual = probability - target
        return float(loss), np.asarray([residual.mean(), np.mean(residual * x)])

    result = minimize(
        lambda beta: objective(beta)[0], np.asarray([0.0, 1.0]),
        jac=lambda beta: objective(beta)[1], method="L-BFGS-B",
        bounds=((None, None), (0.0, None)),
    )
    if not result.success:
        raise RuntimeError(f"full-chain probability calibration failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def fit_full_chain_predictor(
    partitions: Mapping[str, pd.DataFrame],
    *,
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> FullChainPredictorBundle:
    """Fit from development data and report untouched validation diagnostics.

    The regularization grid is a declared numerical search, not a set of
    physical constants.  Its full tuning scores and boundary-selection flag are
    retained.  Validation is diagnostic-only, including if calibration worsens.
    """

    _assert_disjoint_partitions(partitions)
    if not c_grid or any(not np.isfinite(value) or value <= 0 for value in c_grid):
        raise ValueError("regularization grid must contain positive finite values")
    train, tune = partitions["training"], partitions["tuning"]
    calibration, validation = partitions["calibration"], partitions["validation"]
    design = predictor_design(train)
    means = design.mean(axis=0)
    scales = design.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (design - means) / scales
    tune_design = (predictor_design(tune) - means) / scales
    target = train[TARGET].to_numpy(int)
    tuning_target = tune[TARGET].to_numpy(int)
    candidates: dict[float, LogisticRegression] = {}
    selection_rows: list[dict[str, Any]] = []
    for value in sorted(set(float(item) for item in c_grid)):
        estimator = LogisticRegression(C=value, solver="lbfgs", max_iter=2_000).fit(
            standardized, target
        )
        probability = estimator.predict_proba(tune_design)[:, 1]
        candidates[value] = estimator
        selection_rows.append({
            "regularization_c": value,
            "training_rows": len(train), "tuning_rows": len(tune),
            "tuning_brier": float(np.mean((probability - tuning_target) ** 2)),
        })
    selection = pd.DataFrame(selection_rows).sort_values(
        ["tuning_brier", "regularization_c"], ascending=[True, True]
    ).reset_index(drop=True)
    selected_c = float(selection.iloc[0]["regularization_c"])
    selection["selected"] = selection["regularization_c"].eq(selected_c)
    estimator = candidates[selected_c]
    raw_calibration = estimator.predict_proba(
        (predictor_design(calibration) - means) / scales
    )[:, 1]
    intercept, slope = _fit_calibration(raw_calibration, calibration[TARGET].to_numpy(float))
    model = FullChainCompletionPredictor(
        feature_names=FEATURE_NAMES,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]), regularization_c=selected_c,
        calibration_intercept=intercept, calibration_slope=slope,
        training_rows=len(train), tuning_rows=len(tune), calibration_rows=len(calibration),
    )
    target_validation = validation[TARGET].to_numpy(float)
    raw_validation = model.raw_predict(validation)
    calibrated_validation = model.predict(validation)
    probability_rows = validation.copy()
    probability_rows["raw_probability"] = raw_validation
    probability_rows["calibrated_probability"] = calibrated_validation
    diagnostics: list[dict[str, Any]] = []
    calibration_bins: list[dict[str, Any]] = []
    for label, probability in (("raw", raw_validation), ("calibrated", calibrated_validation)):
        clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
        bins = np.minimum((probability * 10).astype(int), 9)
        ece = 0.0
        for bin_index in range(10):
            mask = bins == bin_index
            if not mask.any():
                continue
            gap = float(probability[mask].mean() - target_validation[mask].mean())
            ece += float(mask.mean()) * abs(gap)
            calibration_bins.append({
                "prediction_type": label, "bin_index": bin_index,
                "rows": int(mask.sum()), "mean_probability": float(probability[mask].mean()),
                "observed_completion": float(target_validation[mask].mean()),
                "calibration_gap": gap,
            })
        diagnostics.append({
            "prediction_type": label, "validation_rows": len(validation),
            "validation_prevalence": float(target_validation.mean()),
            "brier": float(np.mean((probability - target_validation) ** 2)),
            "log_loss": float(-np.mean(
                target_validation * np.log(clipped)
                + (1.0 - target_validation) * np.log(1.0 - clipped)
            )),
            "roc_auc": float(roc_auc_score(target_validation, probability)),
            "ece_10_equal_width_bins": ece,
            "validation_used_for_selection": False,
        })
    coefficients = pd.DataFrame([
        {"feature": feature, "standardized_coefficient": coefficient,
         "training_mean": mean, "training_scale": scale}
        for feature, coefficient, mean, scale in zip(
            FEATURE_NAMES, model.coefficients, model.means, model.scales
        )
    ])
    reference = pd.concat(partitions.values(), ignore_index=True)
    partition_manifest = {}
    for role, frame in partitions.items():
        record_bytes = frame.to_json(orient="records", double_precision=15).encode()
        partition_manifest[role] = {
            "rows": len(frame), "seeds": sorted(frame["reference_seed"].unique().tolist()),
            "positive_endpoints": int(frame[TARGET].sum()),
            "records_sha256": sha256(record_bytes).hexdigest(),
        }
    manifest = {
        "model_type": "development_fitted_logistic_with_fitted_monotone_logit_calibration",
        "model_sha256": model.sha256, "model": asdict(model),
        "target": "single_action_technical_completion_conditional_on_forecast_feasibility",
        "target_does_not_include": "multi-task resource competition or field service survival",
        "partitions": partition_manifest,
        "training_tuning_calibration_validation_seeds_disjoint": True,
        "validation_role": "report_only_no_model_calibration_or_policy_selection",
        "regularization_c_grid": sorted(candidates),
        "selected_c_on_grid_boundary": selected_c in (min(candidates), max(candidates)),
        "evidence_tier": "synthetic_simulation",
        "claim_boundary": (
            "Coefficients are fitted to generated technical outcomes, not measured physical hazards. "
            "Simulator risks, profile distributions, handover degradation and workload remain assumptions."
        ),
    }
    tables = {
        "predictor_reference": reference,
        "predictor_regularization_selection": selection,
        "predictor_coefficients": coefficients,
        "predictor_probability_diagnostics": pd.DataFrame(diagnostics),
        "predictor_calibration_bins": pd.DataFrame(calibration_bins),
        "predictor_probability_validation": probability_rows,
    }
    return FullChainPredictorBundle(model=model, tables=tables, manifest=manifest)


def audit_full_chain_regularization_grid(
    reference: pd.DataFrame,
    *,
    frozen_regularization_c: float,
    extended_c_grid: tuple[float, ...] = (
        0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,
    ),
) -> pd.DataFrame:
    """Audit search-range sensitivity using training/tuning rows only.

    This does not revise the frozen model or inspect probability-validation,
    scheduling-evaluation, or policy-outcome rows.  It is useful when the
    originally selected regularization lies on the primary grid boundary.
    A better extended candidate is reported, not silently promoted.
    """

    train = reference.loc[reference["partition"].eq("training")].copy()
    tuning = reference.loc[reference["partition"].eq("tuning")].copy()
    if len(train) < 100 or len(tuning) < 100:
        raise ValueError("range audit needs training and tuning partitions")
    if set(train["reference_seed"]) & set(tuning["reference_seed"]):
        raise ValueError("range audit training/tuning seeds overlap")
    if frozen_regularization_c not in extended_c_grid:
        raise ValueError("extended audit grid must include the frozen regularization")
    design = predictor_design(train)
    means = design.mean(axis=0)
    scales = np.where(design.std(axis=0) > 1e-12, design.std(axis=0), 1.0)
    tuning_design = (predictor_design(tuning) - means) / scales
    target = train[TARGET].to_numpy(int)
    tuning_target = tuning[TARGET].to_numpy(int)
    records = []
    for value in sorted(set(extended_c_grid)):
        estimator = LogisticRegression(C=value, solver="lbfgs", max_iter=2_000).fit(
            (design - means) / scales, target,
        )
        probability = estimator.predict_proba(tuning_design)[:, 1]
        records.append({
            "regularization_c": value,
            "training_rows": len(train), "tuning_rows": len(tuning),
            "tuning_brier": float(np.mean((probability - tuning_target) ** 2)),
            "frozen_model_setting": value == frozen_regularization_c,
            "validation_or_evaluation_outcomes_used": False,
            "evidence_tier": "synthetic_simulation",
        })
    table = pd.DataFrame(records).sort_values(["tuning_brier", "regularization_c"]).reset_index(drop=True)
    best_c = float(table.iloc[0]["regularization_c"])
    table["best_within_extended_grid"] = table["regularization_c"].eq(best_c)
    return table
