"""Censoring-aware survival baselines and the calibrated competing-risk model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier

from .metrics import censoring_weighted_brier_score, ipcw_survival_targets
from .parameters import (
    BASELINE_SEARCH_GRIDS,
    CALIBRATION_C_GRID,
    FOCAL_HAZARD_SEARCH_GRID,
)


class FeatureEncoder:
    def __init__(self, columns: Iterable[str]):
        self.columns = tuple(columns)
        self.median_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "FeatureEncoder":
        values = frame.loc[:, self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        self.median_ = np.nanmedian(values, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = np.where(np.isfinite(values), values, self.median_)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = filled.std(axis=0)
        self.scale_ = np.where(self.scale_ > 1e-8, self.scale_, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.median_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("encoder is not fitted")
        values = frame.loc[:, self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        filled = np.where(np.isfinite(values), values, self.median_)
        return (filled - self.mean_) / self.scale_


class SurvivalModel:
    name = "base"

    def fit(self, frame: pd.DataFrame) -> "SurvivalModel":
        raise NotImplementedError

    def predict_survival(self, frame: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
        raise NotImplementedError


class KaplanMeierModel(SurvivalModel):
    name = "kaplan_meier"

    def fit(self, frame: pd.DataFrame) -> "KaplanMeierModel":
        durations = frame["duration_bins"].to_numpy(int)
        events = frame["event"].to_numpy(int)
        at_risk = len(frame)
        survival = 1.0
        self.timeline_: list[int] = []
        self.survival_: list[float] = []
        for time in np.sort(np.unique(durations)):
            deaths = int(((durations == time) & (events == 1)).sum())
            censored = int(((durations == time) & (events == 0)).sum())
            removed = int((durations == time).sum())
            # A censor boundary at integer d occurs before the unknown hazard
            # outcome in bin d, so tied censors leave the risk set first.
            event_risk = at_risk - censored
            if event_risk > 0 and deaths:
                survival *= 1.0 - deaths / event_risk
            self.timeline_.append(int(time))
            self.survival_.append(float(survival))
            at_risk -= removed
        return self

    def predict_survival(self, frame: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
        values = []
        timeline = np.asarray(self.timeline_, dtype=int)
        survival = np.asarray(self.survival_, dtype=float)
        for horizon in horizons:
            eligible = np.flatnonzero(timeline <= int(horizon))
            values.append(float(survival[eligible[-1]]) if len(eligible) else 1.0)
        return np.tile(np.asarray(values, dtype=float), (len(frame), 1))


class ExponentialModel(SurvivalModel):
    name = "constant_exponential"

    def fit(self, frame: pd.DataFrame) -> "ExponentialModel":
        events = float(frame["event"].sum())
        duration = frame["duration_bins"].to_numpy(float)
        observed = frame["event"].to_numpy(int)
        # A censor boundary d says bins 1..d-1 were observed; bin d is
        # unknown.  Events at d remain observed through the event bin.
        exposure = float(np.maximum(duration - np.where(observed == 0, 1.0, 0.0), 0.0).sum())
        self.hazard_ = max(events / max(exposure, 1.0), 1e-8)
        return self

    def predict_survival(self, frame: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
        values = np.exp(-self.hazard_ * np.asarray(list(horizons), dtype=float))
        return np.tile(values, (len(frame), 1))


class WeibullAFTModel(SurvivalModel):
    name = "weibull_aft"

    def __init__(self, feature_columns: Iterable[str], l2: float = 1e-3):
        self.encoder = FeatureEncoder(feature_columns)
        self.l2 = float(l2)

    def fit(self, frame: pd.DataFrame) -> "WeibullAFTModel":
        self.encoder.fit(frame)
        x = self.encoder.transform(frame)
        x = np.column_stack([np.ones(len(x)), x])
        event = frame["event"].to_numpy(float)
        # Match the discrete boundary convention used by every other model:
        # a censor at d contributes survival only through d-1.
        time = frame["duration_bins"].to_numpy(float) - np.where(event == 0, 1.0, 0.0)
        time = np.clip(time, 1e-3, None)

        def objective(params: np.ndarray) -> float:
            beta = params[:-1]
            shape = np.exp(np.clip(params[-1], -3.0, 3.0))
            log_scale = np.clip(x @ beta, -8.0, 8.0)
            z = np.exp(np.clip(shape * (np.log(time) - log_scale), -30.0, 30.0))
            event_ll = np.log(shape) - shape * log_scale + (shape - 1.0) * np.log(time) - z
            censor_ll = -z
            ll = event * event_ll + (1.0 - event) * censor_ll
            return float(-ll.mean() + self.l2 * np.sum(beta[1:] ** 2))

        initial = np.zeros(x.shape[1] + 1)
        initial[0] = np.log(max(np.median(time), 1.0))
        result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": 400})
        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(f"Weibull fit failed: {result.message}")
        self.beta_ = result.x[:-1]
        self.shape_ = float(np.exp(np.clip(result.x[-1], -3.0, 3.0)))
        return self

    def predict_survival(self, frame: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
        x = self.encoder.transform(frame)
        x = np.column_stack([np.ones(len(x)), x])
        scale = np.exp(np.clip(x @ self.beta_, -8.0, 8.0))
        horizons = np.asarray(list(horizons), dtype=float)
        values = np.exp(-((horizons[None, :] / scale[:, None]) ** self.shape_))
        return np.clip(values, 0.0, 1.0)


class CoxPHModel(SurvivalModel):
    name = "cox_ph"

    def __init__(self, feature_columns: Iterable[str], l2: float = 5e-3):
        self.encoder = FeatureEncoder(feature_columns)
        self.l2 = float(l2)

    def fit(self, frame: pd.DataFrame) -> "CoxPHModel":
        self.encoder.fit(frame)
        x = self.encoder.transform(frame)
        time = frame["duration_bins"].to_numpy(float)
        event = frame["event"].to_numpy(int)
        event_times = np.sort(np.unique(time[event == 1]))

        def objective(beta: np.ndarray) -> float:
            score = np.clip(x @ beta, -20.0, 20.0)
            pll = 0.0
            for t in event_times:
                deaths = (time == t) & (event == 1)
                # Censoring tied at t occurs before the discrete event at t.
                risk = (time > t) | ((time == t) & (event == 1))
                pll += score[deaths].sum() - deaths.sum() * logsumexp(score[risk])
            return float(-pll / max(event.sum(), 1) + self.l2 * np.sum(beta**2))

        result = minimize(objective, np.zeros(x.shape[1]), method="L-BFGS-B", options={"maxiter": 300})
        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(f"Cox fit failed: {result.message}")
        self.beta_ = result.x
        score = np.exp(np.clip(x @ self.beta_, -20.0, 20.0))
        increments = []
        cumulative = 0.0
        for t in event_times:
            deaths = int(((time == t) & (event == 1)).sum())
            risk = (time > t) | ((time == t) & (event == 1))
            risk_sum = float(score[risk].sum())
            cumulative += deaths / max(risk_sum, 1e-12)
            increments.append(cumulative)
        self.baseline_times_ = event_times
        self.baseline_cumulative_ = np.asarray(increments, dtype=float)
        return self

    def predict_survival(self, frame: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
        x = self.encoder.transform(frame)
        relative_risk = np.exp(np.clip(x @ self.beta_, -20.0, 20.0))
        baseline = []
        for horizon in horizons:
            eligible = np.flatnonzero(self.baseline_times_ <= int(horizon))
            baseline.append(float(self.baseline_cumulative_[eligible[-1]]) if len(eligible) else 0.0)
        return np.exp(-relative_risk[:, None] * np.asarray(baseline)[None, :])


def _expand_person_period(
    frame: pd.DataFrame,
    encoder: FeatureEncoder,
    max_horizon: int,
    *,
    competing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    base = encoder.transform(frame)
    causes = sorted(cause for cause in frame.loc[frame["event"].eq(1), "cause"].dropna().unique())
    cause_to_class = {cause: index + 1 for index, cause in enumerate(causes)}
    rows: list[np.ndarray] = []
    labels: list[int] = []
    episode_weights: list[float] = []
    for idx, (_, episode) in enumerate(frame.iterrows()):
        observed_duration = int(episode["duration_bins"])
        is_observed_event = int(episode["event"]) == 1
        # An event at d contributes the hazard row d.  Right-censoring at d
        # contributes only rows strictly before d.  Administrative truncation
        # beyond H is encoded at H+1 and therefore contributes all H rows.
        stop = observed_duration if is_observed_event else observed_duration - 1
        stop = min(stop, max_horizon)
        if stop < 1:
            continue
        for step in range(1, stop + 1):
            time_features = np.asarray([step / max_horizon, np.log1p(step)], dtype=float)
            interaction = np.asarray([time_features[0] * base[idx, 0]], dtype=float)
            rows.append(np.concatenate([base[idx], time_features, interaction]))
            is_event = is_observed_event and observed_duration == step
            if not is_event:
                labels.append(0)
            elif competing:
                labels.append(cause_to_class.get(str(episode["cause"]), 0))
            else:
                labels.append(1)
            # One row represents one conditional hazard contribution.  Equal
            # row weights preserve the person-period likelihood;
            # length-normalising an episode would change its semantics.
            episode_weights.append(1.0)
    return (
        np.vstack(rows),
        np.asarray(labels, dtype=int),
        np.asarray(episode_weights, dtype=float),
        cause_to_class,
    )


def _bounded_cause_hazards(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the same bounded hazard mass for survival and cause incidence.

    The pre-existing positive-hazard safeguards remain 1e-8 and 0.95.  When
    either activates, cause hazards are scaled proportionally to that total;
    using their unadjusted mass would break ``sum CIF = 1 - survival``.
    An exactly zero learned event mass stays zero rather than inventing a
    failure with no event cause.  This affects survival only for exact-zero
    classifier probabilities, where the previous implementation inserted the
    artificial 1e-8 floor.
    """

    causes = np.asarray(probabilities[:, :, 1:], dtype=float)
    raw_total = causes.sum(axis=2)
    total = np.where(raw_total > 0.0, np.clip(raw_total, 1e-8, 0.95), 0.0)
    multiplier = np.divide(total, raw_total, out=np.zeros_like(total), where=raw_total > 0.0)
    return total, causes * multiplier[:, :, None]


class DiscreteHazardModel(SurvivalModel):
    def __init__(
        self,
        feature_columns: Iterable[str],
        *,
        max_horizon: int = 12,
        nonlinear: bool = True,
        competing: bool = False,
        random_state: int = 17,
        hyperparameters: Mapping[str, float | int] | None = None,
        linear_c: float = 0.5,
    ):
        self.encoder = FeatureEncoder(feature_columns)
        self.max_horizon = int(max_horizon)
        self.nonlinear = bool(nonlinear)
        self.competing = bool(competing)
        self.random_state = int(random_state)
        self.linear_c = float(linear_c)
        if self.linear_c <= 0.0:
            raise ValueError("linear_c must be positive")
        default_parameters = dict(FOCAL_HAZARD_SEARCH_GRID[1])
        supplied = default_parameters if hyperparameters is None else dict(hyperparameters)
        if set(supplied) != set(default_parameters):
            raise ValueError(
                "hazard hyperparameters must contain exactly "
                f"{sorted(default_parameters)}"
            )
        self.hyperparameters = supplied
        if competing:
            self.name = "competing_risk_boosted"
        elif nonlinear:
            self.name = "boosted_discrete_hazard"
        else:
            self.name = "linear_discrete_hazard"

    def fit(self, frame: pd.DataFrame) -> "DiscreteHazardModel":
        self.encoder.fit(frame)
        x, y, weights, cause_to_class = _expand_person_period(
            frame, self.encoder, self.max_horizon, competing=self.competing
        )
        self.cause_to_class_ = cause_to_class
        # Do not class-balance the likelihood: doing so improves classification
        # recall at the expense of calibrated hazard probabilities.  The
        # separate calibration split corrects only ordinary sampling drift.
        if self.nonlinear:
            self.model_ = HistGradientBoostingClassifier(
                learning_rate=float(self.hyperparameters["learning_rate"]),
                max_iter=int(self.hyperparameters["max_iter"]),
                max_leaf_nodes=int(self.hyperparameters["max_leaf_nodes"]),
                min_samples_leaf=int(self.hyperparameters["min_samples_leaf"]),
                l2_regularization=float(self.hyperparameters["l2_regularization"]),
                random_state=self.random_state,
            )
        else:
            self.model_ = LogisticRegression(
                max_iter=500,
                C=self.linear_c,
                multi_class="auto",
                random_state=self.random_state,
            )
        self.model_.fit(x, y, sample_weight=weights)
        self.classes_ = np.asarray(self.model_.classes_, dtype=int)
        return self

    def _step_probabilities(self, frame: pd.DataFrame, max_horizon: int) -> np.ndarray:
        base = self.encoder.transform(frame)
        outputs = []
        for step in range(1, max_horizon + 1):
            time_features = np.column_stack(
                [
                    np.full(len(frame), step / self.max_horizon),
                    np.full(len(frame), np.log1p(step)),
                ]
            )
            interaction = (time_features[:, :1] * base[:, :1])
            x = np.column_stack([base, time_features, interaction])
            raw = self.model_.predict_proba(x)
            probabilities = np.zeros((len(frame), int(max(self.classes_.max(), 1)) + 1))
            probabilities[:, self.classes_] = raw
            outputs.append(probabilities)
        return np.stack(outputs, axis=1)

    def predict_survival(self, frame: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
        horizons = np.asarray(list(horizons), dtype=int)
        max_horizon = int(max(horizons))
        probabilities = self._step_probabilities(frame, max_horizon)
        total_hazard, _cause_hazards = _bounded_cause_hazards(probabilities)
        survival_steps = np.cumprod(1.0 - total_hazard, axis=1)
        return survival_steps[:, horizons - 1]

    def predict_cumulative_incidence(self, frame: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
        probabilities = self._step_probabilities(frame, int(horizon))
        total_hazard, cause_hazards = _bounded_cause_hazards(probabilities)
        survival_before = np.column_stack(
            [np.ones(len(frame)), np.cumprod(1.0 - total_hazard[:, :-1], axis=1)]
        )
        result: dict[str, np.ndarray] = {}
        for cause, class_index in self.cause_to_class_.items():
            if class_index >= probabilities.shape[2]:
                continue
            result[cause] = (survival_before * cause_hazards[:, :, class_index - 1]).sum(axis=1)
        return result


class AdditiveDiscreteHazardModel(SurvivalModel):
    """Linear-probability discrete hazard baseline.

    This is an additive-hazard comparison, not a claim to implement the full
    continuous-time Aalen estimator.
    """

    name = "additive_discrete_hazard"

    def __init__(self, feature_columns: Iterable[str], *, max_horizon: int = 12, alpha: float = 5.0):
        self.encoder = FeatureEncoder(feature_columns)
        self.max_horizon = int(max_horizon)
        self.alpha = float(alpha)

    def fit(self, frame: pd.DataFrame) -> "AdditiveDiscreteHazardModel":
        self.encoder.fit(frame)
        x, y, _, _ = _expand_person_period(
            frame, self.encoder, self.max_horizon, competing=False
        )
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(x, y.astype(float))
        return self

    def predict_survival(self, frame: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
        horizons = np.asarray(list(horizons), dtype=int)
        base = self.encoder.transform(frame)
        hazards = []
        for step in range(1, int(horizons.max()) + 1):
            time_features = np.column_stack(
                [
                    np.full(len(frame), step / self.max_horizon),
                    np.full(len(frame), np.log1p(step)),
                ]
            )
            interaction = time_features[:, :1] * base[:, :1]
            design = np.column_stack([base, time_features, interaction])
            hazards.append(np.clip(self.model_.predict(design), 1e-8, 0.95))
        survival = np.cumprod(1.0 - np.column_stack(hazards), axis=1)
        return survival[:, horizons - 1]


class RandomForestDiscreteHazardModel(SurvivalModel):
    """Random-forest person-period hazard baseline."""

    name = "random_forest_discrete_hazard"

    def __init__(
        self,
        feature_columns: Iterable[str],
        *,
        max_horizon: int = 12,
        random_state: int = 17,
        hyperparameters: Mapping[str, int] | None = None,
    ):
        self.encoder = FeatureEncoder(feature_columns)
        self.max_horizon = int(max_horizon)
        self.random_state = int(random_state)
        self.hyperparameters = dict(
            BASELINE_SEARCH_GRIDS["random_forest"][1]
            if hyperparameters is None
            else hyperparameters
        )

    def fit(self, frame: pd.DataFrame) -> "RandomForestDiscreteHazardModel":
        self.encoder.fit(frame)
        x, y, _, _ = _expand_person_period(
            frame, self.encoder, self.max_horizon, competing=False
        )
        self.model_ = RandomForestClassifier(
            n_estimators=int(self.hyperparameters["n_estimators"]),
            max_depth=int(self.hyperparameters["max_depth"]),
            min_samples_leaf=int(self.hyperparameters["min_samples_leaf"]),
            max_features="sqrt",
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model_.fit(x, y)
        return self

    def predict_survival(self, frame: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
        horizons = np.asarray(list(horizons), dtype=int)
        base = self.encoder.transform(frame)
        hazards = []
        for step in range(1, int(horizons.max()) + 1):
            time_features = np.column_stack(
                [
                    np.full(len(frame), step / self.max_horizon),
                    np.full(len(frame), np.log1p(step)),
                ]
            )
            interaction = time_features[:, :1] * base[:, :1]
            design = np.column_stack([base, time_features, interaction])
            raw = self.model_.predict_proba(design)
            event_column = int(np.flatnonzero(self.model_.classes_ == 1)[0])
            hazards.append(np.clip(raw[:, event_column], 1e-8, 0.95))
        survival = np.cumprod(1.0 - np.column_stack(hazards), axis=1)
        return survival[:, horizons - 1]


class XGBoostDiscreteHazardModel(SurvivalModel):
    """XGBoost person-period survival-hazard baseline.

    This is intentionally named for its actual discrete-time construction; it
    is not presented as XGBoost's continuous-time AFT objective or as a
    cause-specific competing-risk model.
    """

    name = "xgboost_discrete_hazard"

    def __init__(
        self,
        feature_columns: Iterable[str],
        *,
        max_horizon: int = 12,
        random_state: int = 17,
        hyperparameters: Mapping[str, float | int] | None = None,
    ):
        self.encoder = FeatureEncoder(feature_columns)
        self.max_horizon = int(max_horizon)
        self.random_state = int(random_state)
        self.hyperparameters = dict(
            BASELINE_SEARCH_GRIDS["xgboost"][1]
            if hyperparameters is None
            else hyperparameters
        )

    def fit(self, frame: pd.DataFrame) -> "XGBoostDiscreteHazardModel":
        self.encoder.fit(frame)
        x, y, weights, _ = _expand_person_period(
            frame,
            self.encoder,
            self.max_horizon,
            competing=False,
        )
        self.model_ = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=int(self.hyperparameters["n_estimators"]),
            learning_rate=float(self.hyperparameters["learning_rate"]),
            max_depth=int(self.hyperparameters["max_depth"]),
            min_child_weight=float(self.hyperparameters["min_child_weight"]),
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.05,
            tree_method="hist",
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.model_.fit(x, y, sample_weight=weights, verbose=False)
        return self

    def predict_survival(self, frame: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
        horizons = np.asarray(list(horizons), dtype=int)
        base = self.encoder.transform(frame)
        hazards = []
        for step in range(1, int(horizons.max()) + 1):
            time_features = np.column_stack(
                [
                    np.full(len(frame), step / self.max_horizon),
                    np.full(len(frame), np.log1p(step)),
                ]
            )
            interaction = time_features[:, :1] * base[:, :1]
            design = np.column_stack([base, time_features, interaction])
            hazards.append(
                np.clip(self.model_.predict_proba(design)[:, 1], 1e-8, 0.95)
            )
        survival = np.cumprod(1.0 - np.column_stack(hazards), axis=1)
        return survival[:, horizons - 1]


class CalibratedSurvivalModel(SurvivalModel):
    name = "calibrated_competing_risk"

    def __init__(
        self,
        base_model: DiscreteHazardModel,
        calibration_horizons: Iterable[int],
        *,
        name: str = "calibrated_competing_risk",
        calibration_c: float = 0.1,
    ):
        self.base_model = base_model
        self.calibration_horizons = tuple(sorted(set(int(x) for x in calibration_horizons)))
        self.calibrators_: dict[int, LogisticRegression | None] = {}
        self.name = name
        self.calibration_c = float(calibration_c)

    def fit(self, frame: pd.DataFrame) -> "CalibratedSurvivalModel":
        self.base_model.fit(frame)
        return self

    def calibrate(self, frame: pd.DataFrame) -> "CalibratedSurvivalModel":
        raw = self.base_model.predict_survival(frame, self.calibration_horizons)
        for index, horizon in enumerate(self.calibration_horizons):
            known, target, weights = ipcw_survival_targets(frame, horizon)
            if known.sum() < 25 or np.unique(target[known]).size < 2:
                self.calibrators_[horizon] = None
                continue
            calibrator = LogisticRegression(
                C=self.calibration_c,
                max_iter=1000,
                random_state=17,
            )
            clipped = np.clip(raw[known, index], 1e-6, 1.0 - 1e-6)
            logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            calibrator.fit(
                logit,
                target[known],
                sample_weight=weights[known],
            )
            self.calibrators_[horizon] = calibrator
        return self

    def predict_survival(self, frame: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
        horizons = tuple(int(x) for x in horizons)
        if not horizons:
            raise ValueError("at least one prediction horizon is required")
        if min(horizons) < 1:
            raise ValueError("prediction horizons must be positive integers")

        # Calibrators are fitted separately by horizon, so their unconstrained
        # outputs need not form a survival curve.  Project on the same complete
        # discrete grid on every call and select the requested columns only
        # afterwards.  Otherwise S(h) could depend on whether earlier horizons
        # happened to be requested by the caller.
        full_horizons = tuple(range(1, max(horizons) + 1))
        raw = self.base_model.predict_survival(frame, full_horizons)
        calibrated = raw.copy()
        for index, horizon in enumerate(full_horizons):
            calibrator = self.calibrators_.get(horizon)
            if calibrator is not None:
                clipped = np.clip(raw[:, index], 1e-6, 1.0 - 1e-6)
                logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
                calibrated[:, index] = calibrator.predict_proba(logit)[:, 1]
        monotone = np.minimum.accumulate(calibrated, axis=1)
        requested_columns = np.asarray(horizons, dtype=int) - 1
        return np.clip(monotone[:, requested_columns], 0.0, 1.0)

    def predict_cumulative_incidence(self, frame: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("prediction horizon must be a positive integer")

        # Allocate each calibrated total-incidence increment according to the
        # base model's cause-hazard proportions at that same step.  Scaling each
        # endpoint CIF independently can make a cause-specific cumulative
        # incidence decrease with time.  Increment allocation preserves both
        # monotonicity and the competing-risk mass identity
        # sum_k F_k(h) = 1 - S(h).
        probabilities = self.base_model._step_probabilities(frame, horizon)
        causes = [
            (cause, class_index)
            for cause, class_index in self.base_model.cause_to_class_.items()
            if class_index < probabilities.shape[2]
        ]
        if not causes:
            return {}
        cause_hazards = np.stack(
            [probabilities[:, :, class_index] for _, class_index in causes],
            axis=2,
        )
        raw_total_hazard = cause_hazards.sum(axis=2, keepdims=True)
        shares = np.divide(
            cause_hazards,
            raw_total_hazard,
            out=np.zeros_like(cause_hazards),
            where=raw_total_hazard > 0.0,
        )
        zero_total = raw_total_hazard[:, :, 0] <= 0.0
        if zero_total.any():
            shares[zero_total] = 1.0 / len(causes)

        calibrated_survival = self.predict_survival(frame, range(1, horizon + 1))
        survival_before = np.column_stack(
            [np.ones(len(frame), dtype=float), calibrated_survival[:, :-1]]
        )
        total_incidence_increment = np.maximum(
            survival_before - calibrated_survival,
            0.0,
        )
        cause_incidence = (
            total_incidence_increment[:, :, None] * shares
        ).sum(axis=1)
        return {
            cause: cause_incidence[:, index]
            for index, (cause, _class_index) in enumerate(causes)
        }


@dataclass
class ModelSuite:
    models: dict[str, SurvivalModel]
    proposed_name: str


def build_model_suite(
    feature_columns: Iterable[str],
    *,
    max_horizon: int,
    random_state: int = 17,
    focal_hyperparameters: Mapping[str, float | int] | None = None,
    calibration_c: float = 0.1,
    baseline_hyperparameters: Mapping[str, Any] | None = None,
) -> ModelSuite:
    features = tuple(feature_columns)
    baseline = dict(baseline_hyperparameters or {})
    competing = DiscreteHazardModel(
        features,
        max_horizon=max_horizon,
        nonlinear=True,
        competing=True,
        random_state=random_state,
        hyperparameters=focal_hyperparameters,
    )
    proposed = CalibratedSurvivalModel(
        competing,
        range(1, max_horizon + 1),
        name="calibrated_competing_risk",
        calibration_c=calibration_c,
    )
    calibrated_single = CalibratedSurvivalModel(
        DiscreteHazardModel(
            features,
            max_horizon=max_horizon,
            nonlinear=True,
            competing=False,
            random_state=random_state,
        ),
        range(1, max_horizon + 1),
        name="calibrated_single_risk",
    )
    models: dict[str, SurvivalModel] = {
        "constant_exponential": ExponentialModel(),
        "kaplan_meier": KaplanMeierModel(),
        "weibull_aft": WeibullAFTModel(
            features, l2=float(baseline.get("weibull_aft_l2", 1e-3))
        ),
        "cox_ph": CoxPHModel(
            features, l2=float(baseline.get("cox_ph_l2", 5e-3))
        ),
        "additive_discrete_hazard": AdditiveDiscreteHazardModel(
            features,
            max_horizon=max_horizon,
            alpha=float(baseline.get("additive_alpha", 5.0)),
        ),
        "linear_discrete_hazard": DiscreteHazardModel(
            features,
            max_horizon=max_horizon,
            nonlinear=False,
            competing=False,
            random_state=random_state,
            linear_c=float(baseline.get("linear_hazard_c", 0.5)),
        ),
        "boosted_discrete_hazard": DiscreteHazardModel(
            features,
            max_horizon=max_horizon,
            nonlinear=True,
            competing=False,
            random_state=random_state,
            hyperparameters=baseline.get("boosted_hazard"),
        ),
        "random_forest_discrete_hazard": RandomForestDiscreteHazardModel(
            features,
            max_horizon=max_horizon,
            random_state=random_state,
            hyperparameters=baseline.get("random_forest"),
        ),
        "xgboost_discrete_hazard": XGBoostDiscreteHazardModel(
            features,
            max_horizon=max_horizon,
            random_state=random_state,
            hyperparameters=baseline.get("xgboost"),
        ),
        "calibrated_single_risk": calibrated_single,
        "competing_risk_uncalibrated": competing,
        "calibrated_competing_risk": proposed,
    }
    return ModelSuite(models=models, proposed_name="calibrated_competing_risk")


def select_focal_model_configuration(
    fit_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    max_horizon: int,
    random_state: int = 17,
    candidate_configs: tuple[Mapping[str, float | int], ...] = FOCAL_HAZARD_SEARCH_GRID,
    calibration_c_grid: tuple[float, ...] = CALIBRATION_C_GRID,
) -> tuple[dict[str, float | int], float, pd.DataFrame]:
    """Select focal model settings using development partitions only.

    The caller must keep all three frames earlier than the locked test period.
    Every configuration receives the same fit, calibration and validation rows.
    The smallest mean IPCW Brier score wins; ties prefer fewer leaves, stronger
    regularization and the smaller calibration ``C``.
    """

    horizons = tuple(range(1, int(max_horizon) + 1))
    rows: list[dict[str, float | int | bool]] = []
    for config_index, candidate in enumerate(candidate_configs):
        for calibration_c in calibration_c_grid:
            model = CalibratedSurvivalModel(
                DiscreteHazardModel(
                    feature_columns,
                    max_horizon=max_horizon,
                    nonlinear=True,
                    competing=True,
                    random_state=random_state,
                    hyperparameters=candidate,
                ),
                horizons,
                calibration_c=float(calibration_c),
            )
            model.fit(fit_frame)
            model.calibrate(calibration_frame)
            prediction = model.predict_survival(validation_frame, horizons)
            scores = [
                censoring_weighted_brier_score(
                    validation_frame,
                    prediction[:, index],
                    horizon,
                )
                for index, horizon in enumerate(horizons)
            ]
            rows.append(
                {
                    "configuration_index": config_index,
                    **dict(candidate),
                    "calibration_c": float(calibration_c),
                    "development_integrated_brier": float(np.mean(scores)),
                    "fit_rows": int(len(fit_frame)),
                    "calibration_rows": int(len(calibration_frame)),
                    "validation_rows": int(len(validation_frame)),
                }
            )
    selection = pd.DataFrame(rows).sort_values(
        [
            "development_integrated_brier",
            "max_leaf_nodes",
            "l2_regularization",
            "calibration_c",
        ],
        ascending=[True, True, False, True],
    )
    winner = selection.iloc[0]
    selected_index = int(winner["configuration_index"])
    selected_c = float(winner["calibration_c"])
    selection["selected"] = (
        selection["configuration_index"].eq(selected_index)
        & selection["calibration_c"].eq(selected_c)
    )
    return dict(candidate_configs[selected_index]), selected_c, selection.reset_index(drop=True)


def select_baseline_model_configurations(
    fit_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    max_horizon: int,
    random_state: int = 17,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Apply equal development-only selection rules to trainable baselines."""

    horizons = tuple(range(1, int(max_horizon) + 1))
    features = tuple(feature_columns)

    def score(model: SurvivalModel) -> float:
        model.fit(fit_frame)
        prediction = model.predict_survival(validation_frame, horizons)
        return float(
            np.mean(
                [
                    censoring_weighted_brier_score(
                        validation_frame, prediction[:, index], horizon
                    )
                    for index, horizon in enumerate(horizons)
                ]
            )
        )

    candidate_builders: dict[str, list[tuple[Any, SurvivalModel]]] = {
        "weibull_aft_l2": [
            (value, WeibullAFTModel(features, l2=float(value)))
            for value in BASELINE_SEARCH_GRIDS["weibull_aft_l2"]
        ],
        "cox_ph_l2": [
            (value, CoxPHModel(features, l2=float(value)))
            for value in BASELINE_SEARCH_GRIDS["cox_ph_l2"]
        ],
        "additive_alpha": [
            (
                value,
                AdditiveDiscreteHazardModel(
                    features, max_horizon=max_horizon, alpha=float(value)
                ),
            )
            for value in BASELINE_SEARCH_GRIDS["additive_alpha"]
        ],
        "linear_hazard_c": [
            (
                value,
                DiscreteHazardModel(
                    features,
                    max_horizon=max_horizon,
                    nonlinear=False,
                    competing=False,
                    random_state=random_state,
                    linear_c=float(value),
                ),
            )
            for value in BASELINE_SEARCH_GRIDS["linear_hazard_c"]
        ],
        "boosted_hazard": [
            (
                dict(value),
                DiscreteHazardModel(
                    features,
                    max_horizon=max_horizon,
                    nonlinear=True,
                    competing=False,
                    random_state=random_state,
                    hyperparameters=value,
                ),
            )
            for value in FOCAL_HAZARD_SEARCH_GRID
        ],
        "random_forest": [
            (
                dict(value),
                RandomForestDiscreteHazardModel(
                    features,
                    max_horizon=max_horizon,
                    random_state=random_state,
                    hyperparameters=value,
                ),
            )
            for value in BASELINE_SEARCH_GRIDS["random_forest"]
        ],
        "xgboost": [
            (
                dict(value),
                XGBoostDiscreteHazardModel(
                    features,
                    max_horizon=max_horizon,
                    random_state=random_state,
                    hyperparameters=value,
                ),
            )
            for value in BASELINE_SEARCH_GRIDS["xgboost"]
        ],
    }
    selected: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for family, candidates in candidate_builders.items():
        family_rows: list[dict[str, Any]] = []
        for index, (configuration, model) in enumerate(candidates):
            family_rows.append(
                {
                    "family": family,
                    "configuration_index": index,
                    "configuration_json": json.dumps(
                        configuration, sort_keys=True
                    ),
                    "development_integrated_brier": score(model),
                    "fit_rows": int(len(fit_frame)),
                    "validation_rows": int(len(validation_frame)),
                }
            )
        family_table = pd.DataFrame(family_rows).sort_values(
            ["development_integrated_brier", "configuration_index"]
        )
        winner_index = int(family_table.iloc[0]["configuration_index"])
        selected[family] = candidates[winner_index][0]
        family_table["selected"] = family_table["configuration_index"].eq(
            winner_index
        )
        rows.extend(family_table.to_dict(orient="records"))
    return selected, pd.DataFrame(rows)
