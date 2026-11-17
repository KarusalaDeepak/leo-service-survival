"""Controlled fusion of Paper 1 geometry with empirical QoS survival.

The functions in this module never infer RF outcomes from a TLE.  Geometry is
used only as a deterministic upper bound on action duration.  A separately
fitted measured-QoS model supplies the conditional survival curve.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .models import SurvivalModel


def restricted_eust_seconds(
    survival: np.ndarray,
    geometric_duration_seconds: float,
    *,
    bin_seconds: int,
) -> float:
    """Trapezoidal integral of measured-QoS survival up to geometry end."""

    survival = np.asarray(survival, dtype=float)
    grid = np.arange(len(survival) + 1, dtype=float) * float(bin_seconds)
    values = np.concatenate([[1.0], np.clip(survival, 0.0, 1.0)])
    end = max(0.0, min(float(geometric_duration_seconds), float(grid[-1])))
    inside = grid < end
    x = grid[inside].tolist()
    y = values[inside].tolist()
    x.append(end)
    y.append(float(np.interp(end, grid, values)))
    return float(np.trapz(np.asarray(y), np.asarray(x))) if len(x) > 1 else 0.0


def build_geometry_qos_fusion(
    windows: pd.DataFrame,
    measured_states: pd.DataFrame,
    model: SurvivalModel,
    *,
    horizons: Iterable[int] = range(1, 13),
    bin_seconds: int = 300,
    task_seconds: int = 300,
    seed: int = 17,
) -> pd.DataFrame:
    """Pair simulated visibility windows with held-out measured QoS states.

    Pairing is deliberately random and therefore forms a controlled interface
    demonstration, not a claim that the WetLinks dish used any listed TLE
    satellite.  The output keeps both evidence sources in separate columns.
    """

    horizons = tuple(int(value) for value in horizons)
    rng = np.random.default_rng(seed)
    # Every satellite in one simultaneous choice set receives the same
    # measured environment state.  Giving each candidate a different random
    # future trace would let the policy choose an uncontrollable scenario.
    choice_sets = windows["start_time_utc"].astype(str)
    state_by_choice = {
        choice: int(rng.integers(0, len(measured_states)))
        for choice in choice_sets.unique()
    }
    state_indices = np.asarray([state_by_choice[choice] for choice in choice_sets], dtype=int)
    states = measured_states.iloc[state_indices].reset_index(drop=True)
    predictions = model.predict_survival(states, horizons)
    task_bins = int(np.ceil(task_seconds / bin_seconds))
    if task_bins not in horizons:
        raise ValueError("task duration is outside the predicted horizon grid")
    task_column = horizons.index(task_bins)
    rows = []
    for position, (_, window) in enumerate(windows.reset_index(drop=True).iterrows()):
        duration_seconds = float(window["duration_s"])
        state = states.iloc[position]
        state_duration = int(state["duration_bins"])
        state_event = int(state["event"])
        measured_known = state_event == 1 or state_duration > task_bins
        measured_completion = int(state_duration > task_bins) if measured_known else 0
        geometry_allows = duration_seconds >= task_seconds
        completion_probability = float(predictions[position, task_column]) if geometry_allows else 0.0
        rows.append(
            {
                "satellite_id": window["satellite_id"],
                "start_time_utc": window["start_time_utc"],
                "geometry_duration_s": duration_seconds,
                "paper1_engineering_eust_s": float(window["expected_service_s"]),
                "paper1_engineering_utility": float(window["utility"]),
                "paper1_modeled_drop_probability": float(window["drop_probability"]),
                "paired_measured_episode": state["episode_id"],
                "measured_site": state["site"],
                "fused_empirical_eust_s": restricted_eust_seconds(
                    predictions[position],
                    duration_seconds,
                    bin_seconds=bin_seconds,
                ),
                "task_completion_probability": completion_probability,
                "geometry_allows_task": int(geometry_allows),
                "measured_outcome_known": int(measured_known),
                "controlled_task_completion": int(geometry_allows and measured_known and measured_completion),
            }
        )
    return pd.DataFrame(rows)
