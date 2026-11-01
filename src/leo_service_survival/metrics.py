"""Prediction, calibration, uncertainty, and paired-comparison metrics."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product

import numpy as np
import pandas as pd


PREDICTION_METRICS = (
    "integrated_brier",
    "mean_calibration_error",
    "binary_cross_entropy",
    "restricted_mean_time_mae_bins",
    "c_index",
)

# Only these two entries in ``PREDICTION_METRICS`` are proper probability
# scores with an additive paired-loss interpretation in this implementation.
# Calibration error, restricted-time error, and concordance remain useful
# diagnostics, but a block-label permutation p-value for them would invite an
# inferential interpretation that their nonlinear estimands do not support.
PROPER_SCORE_METRICS = (
    "integrated_brier",
    "binary_cross_entropy",
)


def known_survival_outcome(frame: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return evaluable rows and the discrete survival indicator at ``horizon``.

    ``duration_bins`` is the event or censoring boundary.  An event at bin
    ``h`` means the service did not survive beyond ``h``.  A row censored at or
    before ``h`` is unknown at that horizon.  Administrative truncation after
    the analysis horizon is encoded at ``h + 1`` by the runner.
    """

    duration = frame["duration_bins"].to_numpy(int)
    event = frame["event"].to_numpy(int)
    known = (duration > horizon) | ((event == 1) & (duration <= horizon))
    survived = (duration > horizon).astype(float)
    return known, survived


def _censoring_survival(frame: pd.DataFrame, query: np.ndarray, *, left_limit: bool) -> np.ndarray:
    """Discrete Kaplan--Meier estimate of post-boundary ``P(C > t)``.

    Event rows are censored observations for the censoring distribution and
    administrative-censoring rows are its events.  ``left_limit=True`` returns
    the pre-boundary value ``P(C >= t)`` when the query equals a boundary;
    otherwise the post-boundary value ``P(C > t)`` is returned.
    """

    duration = frame["duration_bins"].to_numpy(int)
    censor_event = 1 - frame["event"].to_numpy(int)
    unique_times = np.sort(np.unique(duration))
    at_risk = len(frame)
    survival = 1.0
    after: dict[int, float] = {}
    before: dict[int, float] = {}
    for time in unique_times:
        before[int(time)] = survival
        censored = int(((duration == time) & (censor_event == 1)).sum())
        removed = int((duration == time).sum())
        if at_risk > 0 and censored:
            survival *= 1.0 - censored / at_risk
        after[int(time)] = survival
        at_risk -= removed
    values = []
    for item in np.asarray(query, dtype=int):
        eligible = unique_times[unique_times <= item]
        if not len(eligible):
            values.append(1.0)
            continue
        time = int(eligible[-1])
        values.append(before[time] if left_limit and time == item else after[time])
    return np.clip(np.asarray(values, dtype=float), 1e-3, 1.0)


def ipcw_brier_score(frame: pd.DataFrame, probability: np.ndarray, horizon: int) -> float:
    """Graf-style inverse-probability-of-censoring weighted Brier score."""

    duration = frame["duration_bins"].to_numpy(int)
    event = frame["event"].to_numpy(int)
    probability = np.asarray(probability, dtype=float)
    failed = (event == 1) & (duration <= horizon)
    survived = duration > horizon
    _, _, weights = ipcw_survival_targets(frame, horizon)
    target = survived.astype(float)
    return float(np.mean(weights * np.square(target - probability)))


def ipcw_survival_targets(
    frame: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return evaluability, binary survival targets and IPCW weights.

    ``duration_bins`` is the observed follow-up boundary ``U=min(T, C)`` and
    ``event`` is the observed-event indicator.  Under the discrete convention,
    a censoring boundary at ``u`` means that the outcome in bin ``u`` is
    unknown, whereas an event at ``u`` is observed.  Thus a record is evaluable
    at ``horizon`` only when an event is observed by the horizon or the observed
    boundary lies strictly after it.  The post-boundary Kaplan--Meier value used
    below estimates the corresponding ``P(C > u)`` denominator.
    """

    duration = frame["duration_bins"].to_numpy(int)
    event = frame["event"].to_numpy(int)
    failed = (event == 1) & (duration <= horizon)
    survived = duration > horizon
    known = failed | survived
    target = survived.astype(float)
    weights = np.zeros(len(frame), dtype=float)
    if failed.any():
        weights[failed] = 1.0 / _censoring_survival(
            frame, duration[failed], left_limit=False
        )
    if survived.any():
        g_horizon = float(
            _censoring_survival(frame, np.asarray([horizon]), left_limit=False)[0]
        )
        weights[survived] = 1.0 / g_horizon
    return known, target, weights


def ipcw_restricted_duration_targets(
    frame: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return observed restricted-duration references and their IPCW weights.

    For an event observed at ``U <= H``, the reference is ``U``.  For a record
    known to survive beyond ``H``, the reference is ``H``.  A record censored at
    or before ``H`` is not evaluable and receives a zero weight; its returned
    reference is ``NaN`` so downstream code cannot accidentally treat the
    censoring boundary as a realized event time.
    """

    known, _target, weights = ipcw_survival_targets(frame, int(horizon))
    observed_time = frame["duration_bins"].to_numpy(float)
    reference = np.full(len(frame), np.nan, dtype=float)
    reference[known] = np.minimum(observed_time[known], float(horizon))
    return known, reference, weights


def stratified_ipcw_survival_targets(
    frame: pd.DataFrame,
    horizon: int,
    *,
    stratum_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return IPCW targets with the censoring distribution fitted per stratum.

    This is a sensitivity estimator, not an automatic improvement over the
    marginal estimator.  A stratum must contain at least one event or
    administrative-censoring boundary and one row; otherwise the function
    fails rather than silently borrowing a different stratum's distribution.
    """

    if stratum_column not in frame:
        raise KeyError(stratum_column)
    known = np.zeros(len(frame), dtype=bool)
    target = np.zeros(len(frame), dtype=float)
    weights = np.zeros(len(frame), dtype=float)
    labels = frame[stratum_column].astype(str).to_numpy()
    for label in pd.unique(labels):
        index = np.flatnonzero(labels == label)
        if not len(index):
            continue
        local_known, local_target, local_weights = ipcw_survival_targets(
            frame.iloc[index].reset_index(drop=True), horizon
        )
        known[index] = local_known
        target[index] = local_target
        weights[index] = local_weights
    return known, target, weights


def censoring_weighted_brier_score(
    frame: pd.DataFrame,
    probability: np.ndarray,
    horizon: int,
    *,
    stratum_column: str | None = None,
) -> float:
    """Graf-style Brier score under marginal or stratified censoring weights."""

    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 1 or len(probability) != len(frame):
        raise ValueError("probability must be one-dimensional and match frame length")
    if stratum_column is None:
        _known, target, weights = ipcw_survival_targets(frame, horizon)
    else:
        _known, target, weights = stratified_ipcw_survival_targets(
            frame, horizon, stratum_column=stratum_column
        )
    return float(np.mean(weights * np.square(target - probability)))


def censoring_support_table(
    frame: pd.DataFrame,
    horizons: Iterable[int],
    *,
    stratum_column: str | None = None,
) -> pd.DataFrame:
    """Describe endpoint support and censoring at each requested horizon."""

    if stratum_column is not None and stratum_column not in frame:
        raise KeyError(stratum_column)
    groups = [("all", frame)]
    if stratum_column is not None:
        groups = list(frame.groupby(stratum_column, sort=True, dropna=False))
    rows: list[dict] = []
    for label, group in groups:
        group = group.reset_index(drop=True)
        duration = group["duration_bins"].to_numpy(int)
        event = group["event"].to_numpy(int)
        for horizon in (int(value) for value in horizons):
            known, _target, weights = ipcw_survival_targets(group, horizon)
            positive_weights = weights[weights > 0.0]
            effective_n = (
                float(positive_weights.sum() ** 2 / np.square(positive_weights).sum())
                if len(positive_weights) and np.square(positive_weights).sum() > 0.0
                else 0.0
            )
            rows.append(
                {
                    "stratum_column": stratum_column or "marginal",
                    "stratum": str(label),
                    "horizon_bins": horizon,
                    "episodes": int(len(group)),
                    "events_by_horizon": int(((event == 1) & (duration <= horizon)).sum()),
                    "known_survivors_through_horizon": int((duration > horizon).sum()),
                    "censored_unknown_by_horizon": int(
                        ((event == 0) & (duration <= horizon)).sum()
                    ),
                    "evaluable_episodes": int(known.sum()),
                    "evaluable_fraction": float(known.mean()) if len(group) else np.nan,
                    "censoring_survival_at_horizon": float(
                        _censoring_survival(
                            group, np.asarray([horizon]), left_limit=False
                        )[0]
                    ),
                    "ipcw_effective_sample_size": effective_n,
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(
    probability: np.ndarray,
    outcome: np.ndarray,
    bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> float:
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if len(probability) == 0:
        return float("nan")
    weights = (
        np.ones(len(probability), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            selected = (probability >= edges[index]) & (probability < edges[index + 1])
        if selected.any():
            bin_weight = weights[selected]
            bin_total = float(bin_weight.sum())
            if bin_total <= 0:
                continue
            predicted = float(np.average(probability[selected], weights=bin_weight))
            observed = float(np.average(outcome[selected], weights=bin_weight))
            total += (bin_total / weight_sum) * abs(predicted - observed)
    return float(total)


def concordance_index(frame: pd.DataFrame, risk: np.ndarray, max_rows: int = 4000) -> float:
    duration = frame["duration_bins"].to_numpy(float)
    event = frame["event"].to_numpy(int)
    risk = np.asarray(risk, dtype=float)
    if len(frame) > max_rows:
        rng = np.random.default_rng(17)
        order = np.sort(rng.choice(len(frame), size=max_rows, replace=False))
        duration, event, risk = duration[order], event[order], risk[order]
    concordant = 0.0
    comparable = 0
    for i in np.flatnonzero(event == 1):
        eligible = duration > duration[i]
        if not eligible.any():
            continue
        comparable += int(eligible.sum())
        concordant += float((risk[i] > risk[eligible]).sum())
        concordant += 0.5 * float((risk[i] == risk[eligible]).sum())
    return float(concordant / comparable) if comparable else float("nan")


def evaluate_survival_predictions(
    frame: pd.DataFrame,
    survival: np.ndarray,
    horizons: Iterable[int],
) -> dict[str, float]:
    horizons = tuple(int(value) for value in horizons)
    survival = np.asarray(survival, dtype=float)
    briers: list[float] = []
    eces: list[float] = []
    nll_numerator = 0.0
    nll_denominator = 0.0
    for column, horizon in enumerate(horizons):
        known, outcome, weights = ipcw_survival_targets(frame, horizon)
        prediction = np.clip(survival[:, column], 1e-6, 1.0 - 1e-6)
        briers.append(ipcw_brier_score(frame, prediction, horizon))
        if known.any():
            eces.append(
                expected_calibration_error(
                    prediction[known],
                    outcome[known],
                    sample_weight=weights[known],
                )
            )
            losses = -outcome[known] * np.log(prediction[known]) - (
                1.0 - outcome[known]
            ) * np.log(1.0 - prediction[known])
            nll_numerator += float(np.sum(weights[known] * losses))
            nll_denominator += float(weights[known].sum())
    max_horizon = max(horizons)
    full_horizons = np.arange(1, max_horizon + 1)
    if tuple(horizons) == tuple(full_horizons):
        restricted_mean = 1.0 + survival[:, :-1].sum(axis=1)
    else:
        restricted_mean = np.trapz(
            np.column_stack([np.ones(len(frame)), survival]),
            x=np.asarray([0, *horizons], dtype=float),
            axis=1,
        )
    # ``restricted_mean`` is the model's individual expected restricted
    # duration, 1 + sum_{h=1}^{H-1} S(h), under the positive-integer event-time
    # convention.  The comparison target is the observed restricted reference
    # (event edge U when delta=1 and U<=H, or H when U>H), not a latent event
    # time or a second estimate of a population RMST.  IPCW makes the mean
    # absolute error identifiable under the stated independent-censoring
    # assumption; rows censored at or before H receive zero weight.
    rmst_known, observed_rmst, rmst_weights = ipcw_restricted_duration_targets(
        frame, max_horizon
    )
    evaluable_weights = rmst_weights[rmst_known]
    if rmst_known.any() and evaluable_weights.sum() > 0:
        # Mask before subtraction: censored-at-or-before-H references are NaN,
        # and zero IPCW weight does not make arithmetic with NaN well defined.
        evaluable_error = np.abs(
            restricted_mean[rmst_known] - observed_rmst[rmst_known]
        )
        rmst_mae = float(np.average(evaluable_error, weights=evaluable_weights))
    else:
        rmst_mae = float("nan")
    return {
        "integrated_brier": float(np.mean(briers)) if briers else float("nan"),
        "mean_calibration_error": float(np.mean(eces)) if eces else float("nan"),
        "binary_cross_entropy": float(nll_numerator / nll_denominator)
        if nll_denominator > 0
        else float("nan"),
        "restricted_mean_time_mae_bins": rmst_mae,
        "c_index": concordance_index(frame, 1.0 - survival[:, -1]),
        "n_test_episodes": int(len(frame)),
        "event_fraction": float(frame["event"].mean()),
    }


def campaign_bootstrap_difference(
    frame: pd.DataFrame,
    first_survival: np.ndarray,
    second_survival: np.ndarray,
    horizons: Iterable[int],
    *,
    repetitions: int = 500,
    seed: int = 17,
    group_column: str | None = None,
) -> dict[str, float]:
    """Bootstrap the first-minus-second integrated Brier difference by block.

    WetLinks calendar weeks are dependence blocks, not independent campaigns.
    The returned metadata makes that distinction explicit.
    """

    horizons = tuple(int(value) for value in horizons)
    group_column = group_column or (
        "calendar_block"
        if "calendar_block" in frame
        else ("block_id" if "block_id" in frame else "campaign_id")
    )
    campaigns = frame[group_column].astype(str).unique()
    if len(campaigns) < 2:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "groups": len(campaigns),
            "group_column": group_column,
        }
    group_indices = {
        group: np.flatnonzero(frame[group_column].astype(str).to_numpy() == group)
        for group in campaigns
    }
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        selected = rng.choice(campaigns, size=len(campaigns), replace=True)
        indices = np.concatenate([group_indices[group] for group in selected])
        sample = frame.iloc[indices].reset_index(drop=True)
        first = float(
            np.mean(
                [
                    ipcw_brier_score(sample, first_survival[indices, column], int(horizon))
                    for column, horizon in enumerate(horizons)
                ]
            )
        )
        second = float(
            np.mean(
                [
                    ipcw_brier_score(sample, second_survival[indices, column], int(horizon))
                    for column, horizon in enumerate(horizons)
                ]
            )
        )
        differences.append(first - second)
    values = np.asarray(differences, dtype=float)
    return {
        "mean": float(np.nanmean(values)),
        "ci_low": float(np.nanpercentile(values, 2.5)),
        "ci_high": float(np.nanpercentile(values, 97.5)),
        "groups": int(len(campaigns)),
        "group_column": group_column,
    }


def _block_membership(
    frame: pd.DataFrame,
    group_column: str | None = None,
) -> tuple[str, np.ndarray, dict[str, np.ndarray]]:
    group_column = group_column or (
        "calendar_block"
        if "calendar_block" in frame
        else ("block_id" if "block_id" in frame else "campaign_id")
    )
    labels = frame[group_column].astype(str).to_numpy()
    groups = pd.unique(labels)
    indices = {group: np.flatnonzero(labels == group) for group in groups}
    return group_column, np.asarray(groups, dtype=str), indices


def _paired_block_swap_p_value(
    frame: pd.DataFrame,
    first_survival: np.ndarray,
    second_survival: np.ndarray,
    horizons: tuple[int, ...],
    metric: str,
    groups: np.ndarray,
    group_indices: dict[str, np.ndarray],
) -> float:
    """Enumerate a paired block-label randomization p-value for a proper score.

    With five held-out weeks there are only 32 possible block-label swaps, so
    the smallest attainable two-sided p-value is 0.0625.  This deliberately
    exposes the limited inferential resolution instead of manufacturing
    precision from episode-level pseudo-replication.  Exactness requires the
    sharp-null assumption that the two model labels are exchangeable within
    every block; it is not an exact test of equality of mean scores alone.
    """

    if metric not in PROPER_SCORE_METRICS:
        raise ValueError(
            "paired block-label randomization is restricted to proper scores: "
            f"{PROPER_SCORE_METRICS}; received {metric!r}"
        )

    first_survival = np.asarray(first_survival, dtype=float)
    second_survival = np.asarray(second_survival, dtype=float)
    first_point = evaluate_survival_predictions(frame, first_survival, horizons)[metric]
    second_point = evaluate_survival_predictions(frame, second_survival, horizons)[metric]
    observed = float(first_point - second_point)
    if not np.isfinite(observed):
        return float("nan")
    randomized: list[float] = []
    for swap_pattern in product((False, True), repeat=len(groups)):
        first = first_survival.copy()
        second = second_survival.copy()
        for swap, group in zip(swap_pattern, groups):
            if not swap:
                continue
            index = group_indices[str(group)]
            first[index], second[index] = second_survival[index], first_survival[index]
        first_value = evaluate_survival_predictions(frame, first, horizons)[metric]
        second_value = evaluate_survival_predictions(frame, second, horizons)[metric]
        randomized.append(float(first_value - second_value))
    values = np.asarray(randomized, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    return float(np.mean(np.abs(values) >= abs(observed) - 1e-15))


def block_bootstrap_prediction_inference(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    horizons: Iterable[int],
    *,
    reference: str,
    repetitions: int = 500,
    seed: int = 17,
    group_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conditional block uncertainty for model metrics and paired differences.

    Models are not refitted inside the resamples.  The intervals therefore
    quantify variation across the supplied held-out dependence blocks,
    conditional on the fitted models and acquisitions; they are not external-
    campaign confidence intervals.
    """

    horizons = tuple(int(value) for value in horizons)
    group_column, groups, group_indices = _block_membership(frame, group_column)
    if reference not in predictions:
        raise KeyError(f"reference model {reference!r} is absent")
    point = {
        name: evaluate_survival_predictions(frame, survival, horizons)
        for name, survival in predictions.items()
    }
    draws = {
        name: {metric: [] for metric in PREDICTION_METRICS}
        for name in predictions
    }
    rng = np.random.default_rng(seed)
    for _ in range(int(repetitions)):
        selected = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([group_indices[str(group)] for group in selected])
        sample = frame.iloc[index].reset_index(drop=True)
        for name, survival in predictions.items():
            scores = evaluate_survival_predictions(sample, survival[index], horizons)
            for metric in PREDICTION_METRICS:
                draws[name][metric].append(float(scores[metric]))

    interval_rows: list[dict] = []
    for name in predictions:
        for metric in PREDICTION_METRICS:
            values = np.asarray(draws[name][metric], dtype=float)
            values = values[np.isfinite(values)]
            interval_rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "estimate": float(point[name][metric]),
                    "ci_low": float(np.percentile(values, 2.5)) if len(values) else np.nan,
                    "ci_high": float(np.percentile(values, 97.5)) if len(values) else np.nan,
                    "resampling_unit": group_column,
                    "groups": int(len(groups)),
                    "bootstrap_repetitions": int(repetitions),
                    "inference_scope": (
                        "conditional held-out-block variation; fitted models fixed; "
                        "not external-campaign inference"
                    ),
                }
            )

    comparison_rows: list[dict] = []
    for baseline in predictions:
        if baseline == reference:
            continue
        for metric in PROPER_SCORE_METRICS:
            reference_draws = np.asarray(draws[reference][metric], dtype=float)
            baseline_draws = np.asarray(draws[baseline][metric], dtype=float)
            difference_draws = reference_draws - baseline_draws
            difference_draws = difference_draws[np.isfinite(difference_draws)]
            comparison_rows.append(
                {
                    "reference": reference,
                    "baseline": baseline,
                    "metric": metric,
                    "reference_estimate": float(point[reference][metric]),
                    "baseline_estimate": float(point[baseline][metric]),
                    "difference": float(point[reference][metric] - point[baseline][metric]),
                    "ci_low": (
                        float(np.percentile(difference_draws, 2.5))
                        if len(difference_draws)
                        else np.nan
                    ),
                    "ci_high": (
                        float(np.percentile(difference_draws, 97.5))
                        if len(difference_draws)
                        else np.nan
                    ),
                    "p_exact_block_swap": _paired_block_swap_p_value(
                        frame,
                        predictions[reference],
                        predictions[baseline],
                        horizons,
                        metric,
                        groups,
                        group_indices,
                    ),
                    "resampling_unit": group_column,
                    "groups": int(len(groups)),
                    "bootstrap_repetitions": int(repetitions),
                    "block_label_swap_assumption": (
                        "sharp null with within-block model-label exchangeability; "
                        "not a test of equality of mean scores alone"
                    ),
                    "minimum_two_sided_p": float(2.0 ** (1 - len(groups))),
                    "inference_scope": (
                        "conditional paired proper-score comparison across held-out blocks; "
                        "fitted models fixed; not external-campaign inference"
                    ),
                }
            )
    return pd.DataFrame(interval_rows), pd.DataFrame(comparison_rows)


def calibration_curve_block_intervals(
    frame: pd.DataFrame,
    probability: np.ndarray,
    horizon: int,
    *,
    calibration_probability: np.ndarray,
    bins: int = 6,
    repetitions: int = 500,
    seed: int = 17,
    group_column: str | None = None,
) -> pd.DataFrame:
    """IPCW reliability points using bins frozen on a calibration partition.

    ``calibration_probability`` must contain predictions from data disjoint
    from ``frame``.  Only its marginal prediction distribution is used to
    define the quantile cut points.  Those cut points remain fixed throughout
    held-out block resampling, avoiding test-adaptive bin definitions.  Tied
    cut points may leave bins empty, but the result always contains exactly
    ``bins`` rows so the requested estimand is explicit.
    """

    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    calibration_probability = np.clip(
        np.asarray(calibration_probability, dtype=float), 0.0, 1.0
    )
    if probability.ndim != 1 or len(probability) != len(frame):
        raise ValueError("probability must be one-dimensional and match frame length")
    if not np.isfinite(probability).all():
        raise ValueError("probability must contain only finite values")
    if calibration_probability.ndim != 1 or not len(calibration_probability):
        raise ValueError("calibration_probability must be a non-empty one-dimensional array")
    if not np.isfinite(calibration_probability).all():
        raise ValueError("calibration_probability must contain only finite values")
    if int(bins) != bins or int(bins) < 1:
        raise ValueError("bins must be a positive integer")
    if int(repetitions) < 1:
        raise ValueError("repetitions must be positive")
    bins = int(bins)
    group_column, groups, group_indices = _block_membership(frame, group_column)
    inner_edges = np.quantile(
        calibration_probability,
        np.linspace(0.0, 1.0, bins + 1)[1:-1],
    )

    def summarize(sample: pd.DataFrame, sample_probability: np.ndarray) -> dict[int, tuple]:
        known, target, weights = ipcw_survival_targets(sample, int(horizon))
        assignments = np.searchsorted(inner_edges, sample_probability, side="right")
        result: dict[int, tuple] = {}
        block_labels = sample[group_column].astype(str).to_numpy()
        for bin_index in range(bins):
            selected = (assignments == bin_index) & known & (weights > 0)
            total = assignments == bin_index
            if not selected.any():
                result[bin_index] = (
                    np.nan,
                    np.nan,
                    np.nan,
                    int(total.sum()),
                    0,
                    0.0,
                    0.0,
                    0,
                )
                continue
            selected_weights = weights[selected]
            weight_sum = float(selected_weights.sum())
            weight_square_sum = float(np.square(selected_weights).sum())
            effective_sample_size = (
                weight_sum**2 / weight_square_sum if weight_square_sum > 0 else 0.0
            )
            predicted = float(
                np.average(sample_probability[selected], weights=selected_weights)
            )
            observed = float(np.average(target[selected], weights=selected_weights))
            result[bin_index] = (
                predicted,
                observed,
                observed - predicted,
                int(total.sum()),
                int(selected.sum()),
                weight_sum,
                float(effective_sample_size),
                int(pd.unique(block_labels[selected]).size),
            )
        return result

    point = summarize(frame, probability)
    observed_draws = {index: [] for index in range(bins)}
    predicted_draws = {index: [] for index in range(bins)}
    gap_draws = {index: [] for index in range(bins)}
    rng = np.random.default_rng(seed)
    for _ in range(int(repetitions)):
        selected_groups = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([group_indices[str(group)] for group in selected_groups])
        sample = frame.iloc[index].reset_index(drop=True)
        summary = summarize(sample, probability[index])
        for bin_index, (predicted, observed, gap, *_counts) in summary.items():
            predicted_draws[bin_index].append(predicted)
            observed_draws[bin_index].append(observed)
            gap_draws[bin_index].append(gap)

    rows: list[dict] = []
    for bin_index in range(bins):
        (
            predicted,
            observed,
            gap,
            total,
            evaluable,
            weight_sum,
            effective_sample_size,
            represented_groups,
        ) = point[bin_index]
        predicted_values = np.asarray(predicted_draws[bin_index], dtype=float)
        observed_values = np.asarray(observed_draws[bin_index], dtype=float)
        gap_values = np.asarray(gap_draws[bin_index], dtype=float)
        predicted_values = predicted_values[np.isfinite(predicted_values)]
        observed_values = observed_values[np.isfinite(observed_values)]
        gap_values = gap_values[np.isfinite(gap_values)]
        lower_edge = 0.0 if bin_index == 0 else float(inner_edges[bin_index - 1])
        upper_edge = 1.0 if bin_index == bins - 1 else float(inner_edges[bin_index])
        rows.append(
            {
                "horizon_bins": int(horizon),
                "bin": int(bin_index + 1),
                "bin_lower": lower_edge,
                "bin_upper": upper_edge,
                "predicted_survival": predicted,
                "predicted_ci_low": (
                    float(np.percentile(predicted_values, 2.5)) if len(predicted_values) else np.nan
                ),
                "predicted_ci_high": (
                    float(np.percentile(predicted_values, 97.5)) if len(predicted_values) else np.nan
                ),
                "observed_survival_ipcw": observed,
                "observed_ci_low": (
                    float(np.percentile(observed_values, 2.5)) if len(observed_values) else np.nan
                ),
                "observed_ci_high": (
                    float(np.percentile(observed_values, 97.5)) if len(observed_values) else np.nan
                ),
                "calibration_gap_observed_minus_predicted": gap,
                "gap_ci_low": (
                    float(np.percentile(gap_values, 2.5)) if len(gap_values) else np.nan
                ),
                "gap_ci_high": (
                    float(np.percentile(gap_values, 97.5)) if len(gap_values) else np.nan
                ),
                "episodes": int(total),
                "evaluable_episodes": int(evaluable),
                "ipcw_weight_sum": float(weight_sum),
                "ipcw_effective_sample_size": float(effective_sample_size),
                "groups_represented": int(represented_groups),
                "resampling_unit": group_column,
                "groups": int(len(groups)),
                "bootstrap_repetitions": int(repetitions),
                "binning_source": "external calibration-partition predictions",
                "inference_scope": (
                    "conditional reliability-curve uncertainty within held-out blocks; "
                    "bin edges frozen before held-out evaluation"
                ),
            }
        )
    return pd.DataFrame(rows)


def circular_moving_block_paired_inference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    block_lengths: Iterable[int] = (30, 60, 120),
    repetitions: int = 10_000,
    seed: int = 17,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Paired mean-difference intervals for one dependent time series.

    The two policy outcomes are differenced first, so every bootstrap draw uses
    identical time indices for both policies.  Circular moving blocks preserve
    local serial dependence and avoid unequal terminal fragments.  The result
    describes uncertainty within the supplied series under approximate
    stationarity and weak dependence; it is not evidence across independent
    campaigns and intentionally provides no pseudo-replicated p-value.
    """

    first = np.asarray(values_a, dtype=float)
    second = np.asarray(values_b, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("values_a and values_b must be paired one-dimensional arrays")
    if len(first) < 2:
        raise ValueError("at least two paired time points are required")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("paired time-series values must be finite")
    if int(repetitions) < 1:
        raise ValueError("repetitions must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    lengths = tuple(dict.fromkeys(int(value) for value in block_lengths))
    if not lengths:
        raise ValueError("at least one block length is required")
    if any(value < 1 or value > len(first) for value in lengths):
        raise ValueError("every block length must be between one and the series length")

    difference = first - second
    estimate = float(difference.mean())
    tail = (1.0 - float(confidence)) / 2.0
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for block_length in lengths:
        blocks_per_draw = int(np.ceil(len(difference) / block_length))
        draws = np.empty(int(repetitions), dtype=float)
        batch_size = min(512, int(repetitions))
        offsets = np.arange(block_length, dtype=int)
        for start in range(0, int(repetitions), batch_size):
            stop = min(start + batch_size, int(repetitions))
            starts = rng.integers(
                0,
                len(difference),
                size=(stop - start, blocks_per_draw),
            )
            indices = (starts[:, :, None] + offsets[None, None, :]) % len(difference)
            samples = difference[indices].reshape(stop - start, -1)[:, : len(difference)]
            draws[start:stop] = samples.mean(axis=1)
        low, high = np.quantile(draws, [tail, 1.0 - tail])
        rows.append(
            {
                "observations": int(len(difference)),
                "block_length": int(block_length),
                "mean_difference": estimate,
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_standard_error": float(draws.std(ddof=1))
                if len(draws) > 1
                else 0.0,
                "confidence": float(confidence),
                "bootstrap_repetitions": int(repetitions),
                "method": "paired circular moving-block bootstrap; percentile interval",
                "assumption": "approximately stationary weakly dependent time series",
                "inference_scope": (
                    "within-series dependence uncertainty; not independent-campaign inference"
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_block_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    blocks: np.ndarray,
    *,
    repetitions: int = 1000,
    seed: int = 17,
) -> tuple[float, float, float]:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    blocks = np.asarray(blocks)
    unique = np.unique(blocks)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        selected = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([np.flatnonzero(blocks == block) for block in selected])
        deltas.append(float(np.nanmean(values_a[index] - values_b[index])))
    delta = float(np.nanmean(values_a - values_b))
    low, high = np.nanpercentile(deltas, [2.5, 97.5])
    return delta, float(low), float(high)
