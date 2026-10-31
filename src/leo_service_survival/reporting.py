"""Cluster-aware reporting helpers for the exploratory manuscript.

These routines deliberately separate uncertainty produced by a controlled
scenario generator from uncertainty across empirical calendar blocks.  They
do not turn the supplied traces into independent deployment campaigns.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import brentq
from scipy.stats import nct
from scipy.stats import t as student_t

from .metrics import ipcw_survival_targets


def _sign_flip_p_value(values: np.ndarray) -> float:
    """Exhaustive two-sided sign-flip sensitivity for paired block effects."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    observed = abs(float(values.mean()))
    randomized = [
        abs(float(np.mean(values * np.asarray(signs, dtype=float))))
        for signs in product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(randomized) >= observed - 1e-15))


def scheduler_two_level_inference(
    detail: pd.DataFrame,
    *,
    reference: str = "survival_aware",
    metrics: Iterable[str] = ("completion_rate_all", "priority_reward", "fairness"),
    repetitions: int = 10_000,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired scheduler effects with two explicitly different uncertainty layers.

    The conditional layer resamples generated scenarios within each source
    calendar block.  The empirical layer first averages repeated scenarios in
    each calendar block and then treats the few block effects as the evidence
    units.  Neither layer represents independent operator deployments.
    """

    required = {"policy", "scenario", "source_block", *metrics}
    missing = required.difference(detail.columns)
    if missing:
        raise ValueError(f"scheduler detail is missing columns: {sorted(missing)}")
    policies = sorted(set(detail["policy"].astype(str)) - {reference})
    rng = np.random.default_rng(seed)
    effect_rows: list[dict] = []
    inference_rows: list[dict] = []

    for metric in metrics:
        pivot = detail.pivot_table(
            index=["source_block", "scenario"],
            columns="policy",
            values=metric,
            aggfunc="mean",
        )
        if reference not in pivot:
            raise ValueError(f"reference policy {reference!r} is absent")
        for baseline in policies:
            if baseline not in pivot:
                continue
            pair = pivot[[reference, baseline]].dropna().reset_index()
            pair["difference"] = pair[reference] - pair[baseline]
            for row in pair.itertuples(index=False):
                effect_rows.append(
                    {
                        "source_block": str(row.source_block),
                        "scenario": int(row.scenario),
                        "reference": reference,
                        "baseline": baseline,
                        "metric": metric,
                        "reference_value": float(getattr(row, reference)),
                        "baseline_value": float(getattr(row, baseline)),
                        "reference_minus_baseline": float(row.difference),
                    }
                )

            blocks = pair["source_block"].astype(str).unique()
            by_block = {
                block: pair.loc[
                    pair["source_block"].astype(str).eq(block), "difference"
                ].to_numpy(float)
                for block in blocks
            }
            scenario_draws = np.empty(int(repetitions), dtype=float)
            for iteration in range(int(repetitions)):
                sampled = [
                    rng.choice(values, size=len(values), replace=True)
                    for values in by_block.values()
                ]
                scenario_draws[iteration] = float(np.concatenate(sampled).mean())
            point = float(pair["difference"].mean())
            inference_rows.append(
                {
                    "reference": reference,
                    "baseline": baseline,
                    "metric": metric,
                    "uncertainty_layer": "conditional_generator",
                    "mean_difference": point,
                    "ci_low": float(np.percentile(scenario_draws, 2.5)),
                    "ci_high": float(np.percentile(scenario_draws, 97.5)),
                    "p_sign_flip": np.nan,
                    "units": int(len(pair)),
                    "empirical_blocks": int(len(blocks)),
                    "bootstrap_repetitions": int(repetitions),
                    "minimum_attainable_two_sided_p": np.nan,
                    "scope": (
                        "paired generated-scenario uncertainty conditional on the "
                        "held-out traces and scenario generator"
                    ),
                }
            )

            block_effects = pair.groupby("source_block", sort=True)["difference"].mean()
            values = block_effects.to_numpy(float)
            mean = float(values.mean())
            if len(values) > 1:
                half_width = float(
                    student_t.ppf(0.975, len(values) - 1)
                    * values.std(ddof=1)
                    / np.sqrt(len(values))
                )
                low, high = mean - half_width, mean + half_width
            else:
                low = high = np.nan
            inference_rows.append(
                {
                    "reference": reference,
                    "baseline": baseline,
                    "metric": metric,
                    "uncertainty_layer": "empirical_trace_block",
                    "mean_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "p_sign_flip": _sign_flip_p_value(values),
                    "units": int(len(values)),
                    "empirical_blocks": int(len(values)),
                    "bootstrap_repetitions": 0,
                    "minimum_attainable_two_sided_p": float(2.0 ** (1 - len(values))),
                    "scope": (
                        "approximate t interval and exhaustive sign-flip sensitivity "
                        "across held-out trace blocks; conditional on the supplied blocks"
                    ),
                }
            )

    return pd.DataFrame(effect_rows), pd.DataFrame(inference_rows)


def competing_risk_event_adequacy(
    partitions: Mapping[str, pd.DataFrame],
    *,
    design_columns: int,
    causes: Iterable[str] = ("loss", "latency", "joint"),
) -> pd.DataFrame:
    """Describe cause support without iid binomial intervals or post-hoc power."""

    rows: list[dict] = []
    for partition, frame in partitions.items():
        block_column = (
            "analysis_block"
            if "analysis_block" in frame
            else ("calendar_block" if "calendar_block" in frame else "block_id")
        )
        for cause in causes:
            selected = frame["event"].eq(1) & frame["cause"].eq(cause)
            events = int(selected.sum())
            if cause == "joint" and partition in {"calibration", "test"} and events == 0:
                status = "not_estimable"
            elif cause == "latency" and partition == "test" and events < 10:
                status = "insufficient_external_events"
            elif events < design_columns:
                status = "sparse_relative_to_design"
            else:
                status = "limited_clustered_support"
            rows.append(
                {
                    "partition": partition,
                    "cause": cause,
                    "episodes": int(len(frame)),
                    "events": events,
                    "event_fraction_descriptive": (
                        float(events / len(frame)) if len(frame) else np.nan
                    ),
                    "sites_with_event": int(frame.loc[selected, "site"].nunique()),
                    "blocks_with_event": int(frame.loc[selected, block_column].nunique()),
                    "total_blocks": int(frame[block_column].nunique()),
                    "expanded_design_columns": int(design_columns),
                    "events_per_design_column_heuristic": float(events / design_columns),
                    "adequacy_status": status,
                    "interpretation": (
                        "descriptive clustered support only; no iid confidence interval "
                        "and no post-hoc observed power"
                    ),
                }
            )
    return pd.DataFrame(rows)


def aalen_johansen_cif(frame: pd.DataFrame, cause: str, horizon: int) -> float:
    """Discrete Aalen--Johansen cumulative incidence with censor-before-event ties."""

    duration = frame["duration_bins"].to_numpy(int)
    event = frame["event"].to_numpy(int)
    labels = frame["cause"].astype(str).to_numpy()
    survival = 1.0
    incidence = 0.0
    for time in range(1, int(horizon) + 1):
        # A censor at the boundary is unknown at that boundary and is removed
        # before same-time service events enter the risk set.
        at_risk = (duration > time) | ((duration == time) & (event == 1))
        risk_count = int(at_risk.sum())
        if risk_count == 0:
            break
        all_events = int(((duration == time) & (event == 1)).sum())
        cause_events = int(
            ((duration == time) & (event == 1) & (labels == cause)).sum()
        )
        incidence += survival * cause_events / risk_count
        survival *= 1.0 - all_events / risk_count
    return float(incidence)


def competing_risk_cif_validation(
    frame: pd.DataFrame,
    predicted_cif: Mapping[int, Mapping[str, np.ndarray]],
    *,
    repetitions: int = 3_125,
    seed: int = 17,
    group_column: str = "calendar_block",
) -> pd.DataFrame:
    """Observed-versus-predicted cause incidence with whole-block bands."""

    labels = frame[group_column].astype(str).to_numpy()
    groups = pd.unique(labels)
    indices = {group: np.flatnonzero(labels == group) for group in groups}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for horizon, cause_predictions in sorted(predicted_cif.items()):
        for cause, prediction in sorted(cause_predictions.items()):
            prediction = np.asarray(prediction, dtype=float)
            observed = aalen_johansen_cif(frame, cause, int(horizon))
            predicted = float(np.mean(prediction))
            observed_draws: list[float] = []
            predicted_draws: list[float] = []
            for _ in range(int(repetitions)):
                selected = rng.choice(groups, size=len(groups), replace=True)
                sample_index = np.concatenate([indices[group] for group in selected])
                sample = frame.iloc[sample_index].reset_index(drop=True)
                observed_draws.append(aalen_johansen_cif(sample, cause, int(horizon)))
                predicted_draws.append(float(np.mean(prediction[sample_index])))
            event_mask = (
                frame["event"].eq(1)
                & frame["cause"].eq(cause)
                & frame["duration_bins"].le(int(horizon))
            )
            event_count = int(event_mask.sum())
            status = (
                "not_estimable"
                if event_count == 0
                else ("insufficient_external_events" if event_count < 10 else "descriptive")
            )
            gap_draws = np.asarray(predicted_draws) - np.asarray(observed_draws)
            rows.append(
                {
                    "horizon_bins": int(horizon),
                    "cause": cause,
                    "observed_cif_aalen_johansen": observed,
                    "predicted_mean_cif": predicted,
                    "predicted_minus_observed": predicted - observed,
                    "gap_ci_low": float(np.percentile(gap_draws, 2.5)),
                    "gap_ci_high": float(np.percentile(gap_draws, 97.5)),
                    "cause_events": event_count,
                    "blocks_with_event": int(
                        frame.loc[event_mask, group_column].nunique()
                    ),
                    "heldout_blocks": int(len(groups)),
                    "bootstrap_repetitions": int(repetitions),
                    "adequacy_status": status,
                    "scope": "descriptive whole-calendar-block uncertainty",
                }
            )
    return pd.DataFrame(rows)


def _weighted_logistic_calibration(
    probability: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1.0 - 1e-5)
    outcome = np.asarray(outcome, dtype=float)
    weight = np.asarray(weight, dtype=float)
    selected = np.isfinite(probability) & np.isfinite(outcome) & (weight > 0)
    probability, outcome, weight = probability[selected], outcome[selected], weight[selected]
    if len(probability) < 5 or len(np.unique(outcome)) < 2:
        return float("nan"), float("nan")
    design = np.column_stack([np.ones(len(probability)), logit(probability)])
    coefficients = np.asarray([0.0, 1.0])
    for _ in range(100):
        fitted = expit(design @ coefficients)
        variance_weight = weight * np.maximum(fitted * (1.0 - fitted), 1e-8)
        information = design.T @ (variance_weight[:, None] * design)
        score = design.T @ (weight * (outcome - fitted))
        try:
            step = np.linalg.solve(information + np.eye(2) * 1e-10, score)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        coefficients += step
        if np.max(np.abs(step)) < 1e-8:
            break
        if np.max(np.abs(coefficients)) > 100:
            return float("nan"), float("nan")
    return float(coefficients[0]), float(coefficients[1])


def calibration_intercept_slope(
    frame: pd.DataFrame,
    survival: np.ndarray,
    horizons: Iterable[int],
    *,
    repetitions: int = 3_125,
    seed: int = 17,
    group_column: str = "calendar_block",
) -> pd.DataFrame:
    """IPCW calibration intercept/slope with descriptive block intervals."""

    horizons = tuple(int(value) for value in horizons)
    survival = np.asarray(survival, dtype=float)
    if survival.ndim != 2 or survival.shape != (len(frame), len(horizons)):
        raise ValueError(
            "survival columns must align one-for-one with the supplied horizons"
        )
    labels = frame[group_column].astype(str).to_numpy()
    groups = pd.unique(labels)
    indices = {group: np.flatnonzero(labels == group) for group in groups}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for column, horizon in enumerate(horizons):
        known, outcome, weights = ipcw_survival_targets(frame, horizon)
        intercept, slope = _weighted_logistic_calibration(
            survival[known, column], outcome[known], weights[known]
        )
        draws: list[tuple[float, float]] = []
        for _ in range(int(repetitions)):
            selected = rng.choice(groups, size=len(groups), replace=True)
            sample_index = np.concatenate([indices[group] for group in selected])
            sample = frame.iloc[sample_index].reset_index(drop=True)
            sample_known, sample_outcome, sample_weights = ipcw_survival_targets(
                sample, horizon
            )
            draw = _weighted_logistic_calibration(
                survival[sample_index, column][sample_known],
                sample_outcome[sample_known],
                sample_weights[sample_known],
            )
            if np.all(np.isfinite(draw)):
                draws.append(draw)
        values = np.asarray(draws, dtype=float)
        for parameter, index, ideal in (("intercept", 0, 0.0), ("slope", 1, 1.0)):
            point_estimate = intercept if index == 0 else slope
            point_estimable = bool(np.isfinite(point_estimate))
            rows.append(
                {
                    "horizon_bins": horizon,
                    "parameter": parameter,
                    "estimate": point_estimate,
                    "ideal_value": ideal,
                    "ci_low": float(np.percentile(values[:, index], 2.5))
                    if len(values) and point_estimable
                    else np.nan,
                    "ci_high": float(np.percentile(values[:, index], 97.5))
                    if len(values) and point_estimable
                    else np.nan,
                    "successful_block_resamples": int(len(values)),
                    "failed_block_resamples": int(repetitions - len(values)),
                    "heldout_blocks": int(len(groups)),
                    "estimation_status": (
                        "estimable"
                        if point_estimable
                        else "not_estimable_on_full_test_due_to_separation_or_instability"
                    ),
                    "scope": (
                        "IPCW pseudo-likelihood diagnostic with descriptive whole-block "
                        "interval; not an ordinary likelihood-ratio test"
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_proper_score_block_inference(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    horizons: Iterable[int],
    *,
    reference: str,
    baselines: Iterable[str] | None = None,
    group_column: str = "calendar_block",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired IPCW Brier/log-score effects using one global censoring estimate.

    Global test-set IPCW weights are held fixed when forming block effects so
    displayed blocks decompose the same pooled proper-score contrast.  The
    equal-block estimand, approximate t interval, and exhaustive sign-flip
    sensitivity are reported separately from the episode-weighted pooled
    estimand.
    """

    horizons = tuple(int(value) for value in horizons)
    if reference not in predictions:
        raise KeyError(f"reference model {reference!r} is absent")
    selected_baselines = tuple(
        name for name in (baselines or predictions.keys()) if name != reference
    )
    labels = frame[group_column].astype(str).to_numpy()
    groups = pd.unique(labels)
    weights_by_horizon = []
    outcomes_by_horizon = []
    known_by_horizon = []
    for horizon in horizons:
        known, outcome, weights = ipcw_survival_targets(frame, horizon)
        known_by_horizon.append(known)
        outcomes_by_horizon.append(outcome)
        weights_by_horizon.append(weights)
    weights_matrix = np.column_stack(weights_by_horizon)
    outcomes_matrix = np.column_stack(outcomes_by_horizon)
    known_matrix = np.column_stack(known_by_horizon)

    block_rows: list[dict] = []
    summary_rows: list[dict] = []
    for baseline in selected_baselines:
        if baseline not in predictions:
            raise KeyError(f"baseline model {baseline!r} is absent")
        reference_probability = np.clip(
            np.asarray(predictions[reference], dtype=float), 1e-6, 1.0 - 1e-6
        )
        baseline_probability = np.clip(
            np.asarray(predictions[baseline], dtype=float), 1e-6, 1.0 - 1e-6
        )
        if reference_probability.shape != outcomes_matrix.shape:
            raise ValueError("prediction shape does not match episodes and horizons")
        brier_difference = weights_matrix * (
            np.square(outcomes_matrix - reference_probability)
            - np.square(outcomes_matrix - baseline_probability)
        )
        log_reference = -outcomes_matrix * np.log(reference_probability) - (
            1.0 - outcomes_matrix
        ) * np.log(1.0 - reference_probability)
        log_baseline = -outcomes_matrix * np.log(baseline_probability) - (
            1.0 - outcomes_matrix
        ) * np.log(1.0 - baseline_probability)
        log_difference = weights_matrix * (log_reference - log_baseline)

        for metric, loss_difference in (
            ("integrated_brier", brier_difference),
            ("binary_cross_entropy", log_difference),
        ):
            block_effects: list[float] = []
            for group in groups:
                selected = labels == group
                if metric == "integrated_brier":
                    effect = float(loss_difference[selected].sum() / (selected.sum() * len(horizons)))
                else:
                    denominator = float(weights_matrix[selected].sum())
                    effect = (
                        float(loss_difference[selected].sum() / denominator)
                        if denominator > 0
                        else np.nan
                    )
                flat_weights = weights_matrix[selected].ravel()
                positive = flat_weights > 0
                effective_n = (
                    float(flat_weights[positive].sum() ** 2 / np.square(flat_weights[positive]).sum())
                    if positive.any()
                    else 0.0
                )
                block_effects.append(effect)
                block_rows.append(
                    {
                        group_column: str(group),
                        "reference": reference,
                        "baseline": baseline,
                        "metric": metric,
                        "episodes": int(selected.sum()),
                        "events": int(frame.loc[selected, "event"].sum()),
                        "known_horizon_outcomes": int(known_matrix[selected].sum()),
                        "ipcw_effective_sample_size_episode_horizons": effective_n,
                        "reference_minus_baseline": effect,
                        "weight_source": "one fixed full-test censoring Kaplan-Meier estimate",
                    }
                )
            effects = np.asarray(block_effects, dtype=float)
            effects = effects[np.isfinite(effects)]
            if metric == "integrated_brier":
                pooled = float(brier_difference.sum() / (len(frame) * len(horizons)))
            else:
                pooled = float(log_difference.sum() / weights_matrix.sum())
            equal_block = float(effects.mean()) if len(effects) else np.nan
            if len(effects) > 1:
                half_width = float(
                    student_t.ppf(0.975, len(effects) - 1)
                    * effects.std(ddof=1)
                    / np.sqrt(len(effects))
                )
                low, high = equal_block - half_width, equal_block + half_width
                standard_deviation = float(effects.std(ddof=1))
            else:
                low = high = standard_deviation = np.nan
            summary_rows.append(
                {
                    "reference": reference,
                    "baseline": baseline,
                    "metric": metric,
                    "pooled_episode_weighted_difference": pooled,
                    "equal_block_mean_difference": equal_block,
                    "t_interval_low": low,
                    "t_interval_high": high,
                    "block_effect_standard_deviation": standard_deviation,
                    "p_exhaustive_sign_flip": _sign_flip_p_value(effects),
                    "blocks": int(len(effects)),
                    "minimum_attainable_two_sided_p": float(2.0 ** (1 - len(effects))),
                    "sign_flip_assumption": (
                        "sharp null with exchangeable paired block-effect signs; "
                        "not equality of population mean performance"
                    ),
                    "scope": (
                        "few-block exploratory inference within two supplied acquisitions; "
                        "not independent-campaign confirmation"
                    ),
                }
            )
    return pd.DataFrame(block_rows), pd.DataFrame(summary_rows)


def proper_score_power_planning(
    paired_summary: pd.DataFrame,
    *,
    baseline: str = "cox_ph",
    metric: str = "integrated_brier",
    target_power: Iterable[float] = (0.8, 0.9),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Illustrative future-block precision planning from a pilot block SD.

    This is prospective planning conditional on a noisy pilot standard
    deviation.  It is not retrospective or observed power and assumes future
    block effects are approximately independent and Gaussian.
    """

    selected = paired_summary.loc[
        paired_summary["baseline"].eq(baseline)
        & paired_summary["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError("paired_summary must contain one requested baseline/metric row")
    row = selected.iloc[0]
    scale = float(row["block_effect_standard_deviation"])
    current_effect = abs(float(row["equal_block_mean_difference"]))
    rows: list[dict] = []
    for blocks in (5, 8, 12, 20, 30, 50):
        degrees = blocks - 1
        critical = float(student_t.ppf(1.0 - alpha / 2.0, degrees))

        def power_for(effect: float) -> float:
            noncentrality = effect * np.sqrt(blocks) / scale
            return float(
                nct.cdf(-critical, degrees, noncentrality)
                + 1.0
                - nct.cdf(critical, degrees, noncentrality)
            )

        for desired_power in target_power:
            if not np.isfinite(scale) or scale <= 0:
                detectable = np.nan
            else:
                upper = max(scale, current_effect, 1e-6)
                while power_for(upper) < desired_power and upper < scale * 1e4:
                    upper *= 2.0
                detectable = float(
                    brentq(lambda effect: power_for(effect) - desired_power, 0.0, upper)
                )
            rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "future_independent_blocks": blocks,
                    "target_power": float(desired_power),
                    "two_sided_alpha": float(alpha),
                    "pilot_block_effect_sd": scale,
                    "minimum_detectable_absolute_difference": detectable,
                    "current_absolute_equal_block_effect_for_scale_only": current_effect,
                    "interpretation": (
                        "prospective Gaussian block-effect planning conditional on pilot SD; "
                        "not observed power and not a guarantee"
                    ),
                }
            )
    return pd.DataFrame(rows)
