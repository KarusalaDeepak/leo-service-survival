"""Decision-level replay and trace-driven multi-user scheduling experiments."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .models import SurvivalModel
from .parameters import PRIMARY_ENDPOINT


def _select_reactive(group: pd.DataFrame, latency_limit_ms: float) -> int | None:
    available = group[group["observed_replies"].fillna(0).gt(0)]
    if available.empty:
        return None
    compliant = available[available["latency_mean_ms"].le(latency_limit_ms)]
    candidate = compliant if not compliant.empty else available
    return int(candidate["latency_mean_ms"].idxmin())


def run_commect_replay(
    test: pd.DataFrame,
    model: SurvivalModel,
    *,
    horizon_seconds: int = 5,
    latency_limit_ms: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay matched policies on concurrent COMMECT candidates.

    The evidence-gated row deliberately equals the reactive policy because the
    trace contains one acquisition campaign.  The proposed survival policy is
    reported as an ungated candidate, not as deployment-authorized control.
    """

    rows: list[dict] = []
    for epoch, group in test.groupby("decision_epoch", sort=True):
        group = group.copy()
        reactive_index = _select_reactive(group, latency_limit_ms)
        if reactive_index is None:
            continue
        available = group[group["observed_replies"].fillna(0).gt(0)]
        compliant = available[available["latency_mean_ms"].le(latency_limit_ms)]
        candidate = compliant if not compliant.empty else available
        survival = model.predict_survival(candidate, [horizon_seconds])[:, 0]
        candidate = candidate.assign(predicted_completion_probability=survival)
        proposed_index = int(
            candidate.sort_values(
                ["predicted_completion_probability", "latency_mean_ms"],
                ascending=[False, True],
            ).index[0]
        )
        leo = available[available["is_leo"].eq(1)]
        cellular = available[available["is_leo"].eq(0)]
        selections: dict[str, int | None] = {
            "reactive": reactive_index,
            "evidence_gated_survival": reactive_index,
            "survival_candidate": proposed_index,
            "leo_only": int(leo["latency_mean_ms"].idxmin()) if not leo.empty else None,
            "best_5g_only": int(cellular["latency_mean_ms"].idxmin()) if not cellular.empty else None,
        }
        known = group[group["future_five_known"].eq(1)]
        oracle_index = None
        if not known.empty:
            successful = known[known["future_success_h5"].eq(1)]
            oracle_pool = successful if not successful.empty else known
            oracle_index = int(oracle_pool["future_mean_latency_h5"].idxmin())
        selections["retrospective_oracle"] = oracle_index
        for policy, selected_index in selections.items():
            if selected_index is None:
                rows.append(
                    {
                        "decision_epoch": int(epoch),
                        "policy": policy,
                        "known_outcome": 0,
                        "success": 0.0,
                        "future_mean_latency_h5": np.nan,
                        "chosen_path": "no_action",
                    }
                )
                continue
            selected = group.loc[selected_index]
            rows.append(
                {
                    "decision_epoch": int(epoch),
                    "policy": policy,
                    "known_outcome": int(selected["future_five_known"]),
                    "success": float(selected["future_success_h5"]) if selected["future_five_known"] else np.nan,
                    "future_mean_latency_h5": selected["future_mean_latency_h5"],
                    "chosen_path": selected["relative_path"],
                }
            )
    decisions = pd.DataFrame(rows).sort_values(["policy", "decision_epoch"]).reset_index(drop=True)
    decisions["switch"] = decisions.groupby("policy")["chosen_path"].transform(lambda values: values.ne(values.shift()).astype(int))
    summaries = []
    minimum_epoch = int(test["decision_epoch"].min())
    maximum_epoch = int(test["decision_epoch"].max())
    # The scheduled denominator is the complete one-second grid, including
    # epochs at which no eligible action was available.
    total_epochs = maximum_epoch - minimum_epoch + 1
    for policy, group in decisions.groupby("policy", sort=True):
        known = group["known_outcome"].eq(1)
        successes = group.loc[known, "success"].sum()
        summaries.append(
            {
                "policy": policy,
                "scheduled_epochs": int(total_epochs),
                "action_epochs": int(group["decision_epoch"].nunique()),
                "action_coverage": float(group["decision_epoch"].nunique() / total_epochs),
                "evaluable_epochs": int(known.sum()),
                "complete_case_success": float(group.loc[known, "success"].mean()) if known.any() else np.nan,
                "conservative_scheduled_success": float(successes / total_epochs) if total_epochs else np.nan,
                "mean_future_latency_h5_ms": float(group.loc[known, "future_mean_latency_h5"].mean()) if known.any() else np.nan,
                "switch_rate": float(group["switch"].iloc[1:].mean()) if len(group) > 1 else 0.0,
            }
        )
    return decisions, pd.DataFrame(summaries)


@dataclass
class SchedulingScenario:
    tasks: pd.DataFrame
    candidates: pd.DataFrame


def make_trace_driven_scenario(
    trace: pd.DataFrame,
    survival_predictions: np.ndarray,
    prediction_horizons: np.ndarray,
    *,
    task_count: int,
    candidate_count: int,
    horizon_slots: int,
    satellite_count: int,
    ground_station_count: int,
    seed: int,
) -> SchedulingScenario:
    """Create a controlled contact-allocation benchmark.

    Measured WetLinks episodes supply service outcomes and model probabilities;
    releases, deadlines, users and mutually exclusive satellite/ground-station
    resources are generated from a fixed stochastic design.  This is a
    trace-driven stress test, not an operator EO deployment.
    """

    rng = np.random.default_rng(seed)
    eligible = trace.copy().reset_index(drop=True)
    survival_predictions = np.asarray(survival_predictions, dtype=float)
    prediction_horizons = np.asarray(prediction_horizons, dtype=int)
    tasks = []
    candidates = []
    durations = np.asarray([1, 3, 6, 12], dtype=int)
    for task_id in range(task_count):
        requirement = int(rng.choice(durations, p=[0.35, 0.30, 0.25, 0.10]))
        priority = int(rng.integers(1, 11))
        release = int(rng.integers(0, max(1, horizon_slots - requirement - 3)))
        slack = int(rng.integers(requirement + 1, requirement + 12))
        deadline = min(horizon_slots, release + slack)
        if deadline - release < requirement:
            deadline = min(horizon_slots, release + requirement)
        user_id = int(rng.integers(0, 5))
        tasks.append(
            {
                "task_id": task_id,
                "user_id": user_id,
                "required_bins": requirement,
                "priority": priority,
                "release_slot": release,
                "deadline_slot": deadline,
            }
        )
        sample = eligible.iloc[rng.integers(0, len(eligible), size=candidate_count)]
        horizon_column = int(np.flatnonzero(prediction_horizons == requirement)[0])
        for rank, (source_index, row) in enumerate(sample.iterrows()):
            observed_duration = int(row["duration_bins"])
            observed_event = int(row["event"])
            known = observed_event == 1 or observed_duration > requirement
            completes = int(observed_duration > requirement) if known else 0
            latency = float(row.get("latency_mean_ms", np.nan))
            loss = float(row.get("packet_loss_pct", np.nan))
            latency_headroom = (
                (PRIMARY_ENDPOINT.latency_limit_ms - latency)
                / PRIMARY_ENDPOINT.latency_limit_ms
                if np.isfinite(latency)
                else -1.0
            )
            loss_headroom = (
                (PRIMARY_ENDPOINT.loss_limit_pct - loss)
                / PRIMARY_ENDPOINT.loss_limit_pct
                if np.isfinite(loss)
                else -1.0
            )
            # A deliberately coefficient-free baseline: rank by the worse of
            # the two normalized current-QoS margins.  It uses no future label
            # and no hand-chosen RTT-versus-loss weight.
            heuristic_reliability = float(min(latency_headroom, loss_headroom))
            latest_start = max(release, deadline - requirement)
            start_slot = int(rng.integers(release, latest_start + 1))
            end_slot = start_slot + requirement
            candidates.append(
                {
                    "task_id": task_id,
                    "candidate_rank": rank,
                    "required_bins": requirement,
                    "observed_duration_bins": observed_duration,
                    "source_episode_id": str(row.get("episode_id", source_index)),
                    "source_block": str(row.get("calendar_block", row.get("block_id", "unknown"))),
                    "known_outcome": int(known),
                    "completes": int(completes),
                    "predicted_survival": float(survival_predictions[source_index, horizon_column]),
                    # Uses only current-bin QoS; realized future duration is
                    # retained solely for outcome scoring and never for ranking.
                    "heuristic_reliability": heuristic_reliability,
                    "start_slot": start_slot,
                    "end_slot": end_slot,
                    "satellite_id": int(rng.integers(0, satellite_count)),
                    "ground_station_id": int(rng.integers(0, ground_station_count)),
                }
            )
    task_frame = pd.DataFrame(tasks)
    task_frame.attrs.update(
        {
            "horizon_slots": int(horizon_slots),
            "satellite_count": int(satellite_count),
            "ground_station_count": int(ground_station_count),
        }
    )
    candidate_frame = pd.DataFrame(candidates)
    pressures = []
    for _, candidate in candidate_frame.iterrows():
        overlaps = (
            (candidate_frame["start_slot"] < candidate["end_slot"])
            & (candidate_frame["end_slot"] > candidate["start_slot"])
            & (
                candidate_frame["satellite_id"].eq(candidate["satellite_id"])
                | candidate_frame["ground_station_id"].eq(candidate["ground_station_id"])
            )
        )
        pressures.append(max(0, int(overlaps.sum()) - 1))
    maximum = max(max(pressures, default=0), 1)
    candidate_frame["congestion"] = np.asarray(pressures, dtype=float) / maximum
    return SchedulingScenario(task_frame, candidate_frame)


def _contact_is_free(
    candidate: pd.Series,
    occupied_satellite: set[tuple[int, int]],
    occupied_ground: set[tuple[int, int]],
) -> bool:
    slots = range(int(candidate["start_slot"]), int(candidate["end_slot"]))
    satellite = int(candidate["satellite_id"])
    ground = int(candidate["ground_station_id"])
    return all(
        (satellite, slot) not in occupied_satellite and (ground, slot) not in occupied_ground
        for slot in slots
    )


def _schedule_policy(
    scenario: SchedulingScenario,
    policy: str,
    *,
    survival_congestion_weight: float = 0.25,
) -> dict:
    start = perf_counter()
    tasks = scenario.tasks.copy()
    candidates = scenario.candidates.copy()
    if policy == "edf":
        tasks = tasks.sort_values(["deadline_slot", "priority"], ascending=[True, False])
    elif policy == "highest_priority":
        tasks = tasks.sort_values(["priority", "deadline_slot"], ascending=[False, True])
    elif policy == "racs_like":
        # Preserve the common EDF task order.  The comparator differs only in
        # its coefficient-free current-QoS candidate ranking below.
        tasks = tasks.sort_values(["deadline_slot", "priority"], ascending=[True, False])
    elif policy == "survival_aware":
        # Use the same EDF task order as the strongest non-oracle baseline so
        # that the comparison isolates candidate-contact selection rather than
        # granting either method a different task set or resource action space.
        tasks = tasks.sort_values(["deadline_slot", "priority"], ascending=[True, False])
    else:
        raise ValueError(policy)
    accepted = []
    occupied_satellite: set[tuple[int, int]] = set()
    occupied_ground: set[tuple[int, int]] = set()
    for _, task in tasks.iterrows():
        pool = candidates[candidates["task_id"].eq(task["task_id"])].copy()
        feasible = pool[
            pool["start_slot"].ge(task["release_slot"])
            & pool["end_slot"].le(task["deadline_slot"])
        ]
        feasible = feasible[
            feasible.apply(
                _contact_is_free,
                axis=1,
                occupied_satellite=occupied_satellite,
                occupied_ground=occupied_ground,
            )
        ]
        if feasible.empty:
            continue
        if policy == "survival_aware":
            feasible = feasible.assign(
                _selection_score=(
                feasible["predicted_survival"]
                - float(survival_congestion_weight) * feasible["congestion"]
                )
            )
            selected = feasible.sort_values(
                ["_selection_score", "end_slot", "candidate_rank"],
                ascending=[False, True, True],
            ).iloc[0]
        elif policy == "racs_like":
            selected = feasible.sort_values(
                ["heuristic_reliability", "end_slot", "candidate_rank"],
                ascending=[False, True, True],
            ).iloc[0]
        else:
            selected = feasible.sort_values(["end_slot", "congestion"]).iloc[0]
        for slot in range(int(selected["start_slot"]), int(selected["end_slot"])):
            occupied_satellite.add((int(selected["satellite_id"]), slot))
            occupied_ground.add((int(selected["ground_station_id"]), slot))
        accepted.append({**task.to_dict(), **selected.to_dict()})
    runtime_ms = (perf_counter() - start) * 1000.0
    selected = pd.DataFrame(accepted)
    if selected.empty:
        return {
            "policy": policy,
            "accepted": 0,
            "completed": 0,
            "completion_rate_all": 0.0,
            "deadline_miss_rate": 1.0,
            "priority_reward": 0.0,
            "congestion": 0.0,
            "fairness": 0.0,
            "runtime_ms": runtime_ms,
        }
    completed = selected["known_outcome"].eq(1) & selected["completes"].eq(1)
    all_tasks = tasks[["task_id", "user_id", "priority"]].copy()
    completed_ids = set(selected.loc[completed, "task_id"].astype(int))
    all_tasks["completed"] = all_tasks["task_id"].astype(int).isin(completed_ids).astype(float)
    completion_by_user = all_tasks.groupby("user_id")["completed"].mean()
    values = completion_by_user.to_numpy(float)
    fairness = float(values.sum() ** 2 / (len(values) * np.square(values).sum())) if np.square(values).sum() else 0.0
    return {
        "policy": policy,
        "accepted": int(len(selected)),
        "completed": int(completed.sum()),
        "completion_rate_all": float(completed.sum() / len(tasks)),
        "completion_rate_evaluable": float(completed.sum() / selected["known_outcome"].sum())
        if selected["known_outcome"].sum()
        else np.nan,
        "deadline_miss_rate": float(1.0 - completed.sum() / len(tasks)),
        "priority_reward": float(selected.loc[completed, "priority"].sum() / tasks["priority"].sum()),
        "congestion": float(selected["congestion"].mean()),
        "fairness": fairness,
        "occupied_satellite_slots": int(len(occupied_satellite)),
        "occupied_ground_slots": int(len(occupied_ground)),
        "runtime_ms": runtime_ms,
    }


def run_trace_driven_scheduling(
    trace: pd.DataFrame,
    model: SurvivalModel,
    *,
    scenarios: int = 100,
    task_count: int = 80,
    candidate_count: int = 4,
    horizon_slots: int = 48,
    satellite_count: int = 4,
    ground_station_count: int = 2,
    survival_congestion_weight: float = 0.25,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizons = np.asarray([1, 3, 6, 12], dtype=int)
    trace = trace.reset_index(drop=True)
    predictions = model.predict_survival(trace, horizons)
    block_column = "calendar_block" if "calendar_block" in trace else None
    source_blocks = (
        sorted(trace[block_column].astype(str).unique()) if block_column else ["all"]
    )
    rows = []
    for scenario_index in range(scenarios):
        source_block = source_blocks[scenario_index % len(source_blocks)]
        if block_column:
            source_indices = np.flatnonzero(
                trace[block_column].astype(str).to_numpy() == source_block
            )
            scenario_trace = trace.iloc[source_indices].reset_index(drop=True)
            scenario_predictions = predictions[source_indices]
        else:
            scenario_trace = trace
            scenario_predictions = predictions
        scenario = make_trace_driven_scenario(
            scenario_trace,
            scenario_predictions,
            horizons,
            task_count=task_count,
            candidate_count=candidate_count,
            horizon_slots=horizon_slots,
            satellite_count=satellite_count,
            ground_station_count=ground_station_count,
            seed=seed + scenario_index,
        )
        for policy in ("edf", "highest_priority", "racs_like", "survival_aware"):
            result = _schedule_policy(
                scenario,
                policy,
                survival_congestion_weight=survival_congestion_weight,
            )
            result["scenario"] = scenario_index
            result["source_block"] = source_block
            rows.append(result)
    detail = pd.DataFrame(rows)
    summary = detail.groupby("policy", as_index=False).agg(
        completion_rate_all=("completion_rate_all", "mean"),
        completion_std=("completion_rate_all", "std"),
        deadline_miss_rate=("deadline_miss_rate", "mean"),
        completion_rate_evaluable=("completion_rate_evaluable", "mean"),
        priority_reward=("priority_reward", "mean"),
        congestion=("congestion", "mean"),
        fairness=("fairness", "mean"),
        runtime_ms=("runtime_ms", "mean"),
    )
    return detail, summary
