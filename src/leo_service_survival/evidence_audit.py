"""Focused audits for claims that are easy to conflate across experiments.

The functions in this module do not rerun a model or simulator.  They verify
row-level relationships in frozen exports and attach the denominators and claim
boundaries needed to interpret those exports.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


EOS_BASELINE = "edf_reactive"
EOS_INTEGRATED = "integrated_risk_gated"


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a source artifact."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _as_bool(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
        "nan": False,
        "none": False,
        "": False,
    }
    invalid = sorted(set(normalized).difference(mapping))
    if invalid:
        raise ValueError(f"{label} contains non-Boolean values: {invalid}")
    return normalized.map(mapping).astype(bool)


def _state_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts = frame[column].fillna("<missing>").astype(str).value_counts()
    return {str(key): int(value) for key, value in sorted(counts.items())}


def audit_eos_candidate_fallback(
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    """Audit candidate OOD screening separately from decision fallback.

    EOS v2 evaluates the frozen domain gate on one candidate row at a time, so
    ``gate_ood_fraction`` is binary.  The exported candidate table is duplicated
    across the paired policy arms; counts are therefore reported both as exported
    rows and as unique common-action records.
    """

    candidate_required = {
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
    }
    decision_required = {
        "campaign_id",
        "decision_id",
        "simulation_seed",
        "logical_task_id",
        "policy_arm",
        "selected_path_id",
        "gate_state",
        "gate_reason",
        "shift_alarm_state",
    }
    _require_columns(candidates, candidate_required, "EOS candidate table")
    _require_columns(decisions, decision_required, "EOS decision table")

    candidate = candidates.loc[:, sorted(candidate_required)].copy()
    action = decisions.loc[:, sorted(decision_required)].copy()
    candidate["resource_feasible"] = _as_bool(
        candidate["resource_feasible"], "resource_feasible"
    )
    candidate["gate_release"] = _as_bool(candidate["gate_release"], "gate_release")
    action["shift_alarm_state"] = _as_bool(
        action["shift_alarm_state"], "shift_alarm_state"
    )
    candidate["gate_ood_fraction"] = pd.to_numeric(
        candidate["gate_ood_fraction"], errors="raise"
    )
    candidate["gate_feature_coverage"] = pd.to_numeric(
        candidate["gate_feature_coverage"], errors="raise"
    )
    candidate["candidate_ood"] = candidate["gate_ood_fraction"].gt(0.0)

    lookup = action[["campaign_id", "decision_id", "policy_arm"]].drop_duplicates()
    if lookup.duplicated(["campaign_id", "decision_id"]).any():
        raise ValueError("EOS decision identifiers do not map to one policy arm")
    candidate = candidate.merge(
        lookup,
        on=["campaign_id", "decision_id"],
        how="left",
        validate="many_to_one",
    )
    if candidate["policy_arm"].isna().any():
        raise ValueError("some EOS candidate rows have no decision-action record")

    # In EOS v2 all non-domain criteria are configured to pass and each gate call
    # contains exactly one row.  Thus release is resource feasibility AND an
    # in-domain candidate.  This is a run-specific audit, not a generic gate rule.
    expected_release = candidate["resource_feasible"] & ~candidate["candidate_ood"]
    release_violations = int(candidate["gate_release"].ne(expected_release).sum())
    reason_text = candidate["gate_reasons"].fillna("").astype(str)
    expected_ood_reason = reason_text.str.split("|").apply(
        lambda values: "out_of_domain_fraction_exceeded" in values
    )
    reason_violations = int(expected_ood_reason.ne(candidate["candidate_ood"]).sum())

    common_keys = ["simulation_seed", "logical_task_id", "candidate_path_id"]
    common = candidate.drop_duplicates(common_keys)
    paired = candidate.groupby(common_keys, sort=False).agg(
        exported_copies=("policy_arm", "size"),
        policy_arms=("policy_arm", "nunique"),
        feasible_values=("resource_feasible", "nunique"),
        release_values=("gate_release", "nunique"),
        ood_values=("gate_ood_fraction", "nunique"),
    )
    paired_violations = int(
        (
            paired["exported_copies"].ne(2)
            | paired["policy_arms"].ne(2)
            | paired["feasible_values"].ne(1)
            | paired["release_values"].ne(1)
            | paired["ood_values"].ne(1)
        ).sum()
    )

    by_arm: dict[str, dict[str, int]] = {}
    for arm, group in candidate.groupby("policy_arm", sort=True):
        by_arm[str(arm)] = {
            "candidate_rows": int(len(group)),
            "ood_candidate_rows": int(group["candidate_ood"].sum()),
            "resource_feasible_candidate_rows": int(group["resource_feasible"].sum()),
            "resource_feasible_ood_candidate_rows": int(
                (group["resource_feasible"] & group["candidate_ood"]).sum()
            ),
            "released_candidate_rows": int(group["gate_release"].sum()),
        }

    integrated_candidates = candidate[candidate["policy_arm"].eq(EOS_INTEGRATED)].copy()
    integrated_decisions = action[action["policy_arm"].eq(EOS_INTEGRATED)].copy()
    if integrated_candidates.empty or integrated_decisions.empty:
        raise ValueError("integrated EOS policy rows are absent")
    choice_keys = ["simulation_seed", "logical_task_id"]
    choice = integrated_candidates.groupby(choice_keys, as_index=False).agg(
        any_ood_candidate=("candidate_ood", "any"),
        any_resource_feasible_candidate=("resource_feasible", "any"),
        any_released_candidate=("gate_release", "any"),
    )
    feasible_ood = (
        integrated_candidates.assign(
            feasible_ood=lambda frame: frame["resource_feasible"] & frame["candidate_ood"]
        )
        .groupby(choice_keys, as_index=False)["feasible_ood"]
        .any()
    )
    choice = choice.merge(feasible_ood, on=choice_keys, validate="one_to_one")
    decision_choice = integrated_decisions.merge(
        choice,
        on=choice_keys,
        how="left",
        validate="one_to_one",
    )
    if decision_choice["any_ood_candidate"].isna().any():
        raise ValueError("some integrated decisions have no candidate choice set")

    selected_decisions = integrated_decisions[
        integrated_decisions["selected_path_id"].notna()
    ].copy()
    selected = selected_decisions.merge(
        integrated_candidates,
        left_on=["campaign_id", "decision_id", "selected_path_id"],
        right_on=["campaign_id", "decision_id", "candidate_path_id"],
        how="left",
        validate="one_to_one",
        suffixes=("_decision", "_candidate"),
        indicator=True,
    )
    missing_selected = int(selected["_merge"].ne("both").sum())
    selected_ood = selected["gate_ood_fraction"].fillna(1.0).gt(0.0)
    selected_alarm_mismatch = int(
        selected["shift_alarm_state"].ne(selected_ood).sum()
    )

    no_action = integrated_decisions[integrated_decisions["gate_state"].eq("no_action")]
    ood_state_table = pd.crosstab(
        decision_choice["any_ood_candidate"], decision_choice["gate_state"]
    )
    ood_state_counts = {
        str(bool(ood)): {str(state): int(count) for state, count in row.items()}
        for ood, row in ood_state_table.iterrows()
    }

    fallback_count = int(integrated_decisions["gate_state"].eq("fallback").sum())
    any_ood_sets = int(choice["any_ood_candidate"].sum())
    return {
        "protocol": "eos-full-chain-sim-v2",
        "evidence_tier": "synthetic_simulation",
        "candidate_release_rule": (
            "For this locked v2 run: gate_release = resource_feasible AND "
            "(one-row gate_ood_fraction == 0). Other EvidenceGate checks were "
            "configured to pass for this simulation-only release path."
        ),
        "exported_candidate_rows": int(len(candidate)),
        "exported_ood_candidate_rows": int(candidate["candidate_ood"].sum()),
        "exported_ood_fraction": float(candidate["candidate_ood"].mean()),
        "unique_common_action_records": int(len(common)),
        "unique_common_action_ood_records": int(common["candidate_ood"].sum()),
        "paired_arm_duplication_verified": paired_violations == 0,
        "paired_arm_duplication_violations": paired_violations,
        "by_policy_arm": by_arm,
        "release_rule_compliant_rows": int(len(candidate) - release_violations),
        "release_rule_total_rows": int(len(candidate)),
        "release_rule_violations": release_violations,
        "ood_reason_compliant_rows": int(len(candidate) - reason_violations),
        "ood_reason_violations": reason_violations,
        "gate_feature_coverage_min": float(candidate["gate_feature_coverage"].min()),
        "gate_ood_fraction_values": sorted(
            float(value) for value in candidate["gate_ood_fraction"].unique()
        ),
        "integrated_choice_sets": int(len(choice)),
        "choice_sets_with_any_ood_candidate": any_ood_sets,
        "choice_sets_with_any_resource_feasible_ood_candidate": int(
            choice["feasible_ood"].sum()
        ),
        "choice_sets_with_any_resource_feasible_candidate": int(
            choice["any_resource_feasible_candidate"].sum()
        ),
        "choice_sets_with_any_released_candidate": int(
            choice["any_released_candidate"].sum()
        ),
        "choice_sets_with_resource_feasible_but_no_release": int(
            (
                choice["any_resource_feasible_candidate"]
                & ~choice["any_released_candidate"]
            ).sum()
        ),
        "integrated_decision_gate_states": _state_counts(
            integrated_decisions, "gate_state"
        ),
        "integrated_no_action_reasons": _state_counts(no_action, "gate_reason"),
        "selected_integrated_actions": int(len(selected)),
        "selected_action_join_missing": missing_selected,
        "selected_released_actions": int(selected["gate_release"].fillna(False).sum()),
        "selected_ood_actions": int(selected_ood.sum()),
        "selected_shift_alarm_actions": int(
            selected["shift_alarm_state"].fillna(False).sum()
        ),
        "selected_shift_alarm_vs_selected_ood_mismatches": selected_alarm_mismatch,
        "integrated_fallback_decisions": fallback_count,
        "choice_set_ood_by_decision_gate_state": ood_state_counts,
        "candidate_ood_implies_decision_fallback": False,
        "why_candidate_ood_does_not_imply_fallback": (
            f"{any_ood_sets} integrated choice sets contained an OOD candidate, but "
            f"the decision-level fallback state occurred {fallback_count} times. "
            "OOD is candidate-specific; another released candidate can be chosen."
        ),
        "claim_boundary": (
            "This verifies screening and candidate/action linkage in a synthetic "
            "run. It does not establish a field fallback rate or fallback safety."
        ),
    }


def audit_eos_effect(
    ledger: pd.DataFrame,
    paired_seed_results: pd.DataFrame,
    recorded_inference: Mapping[str, Any],
    *,
    bootstrap_seed: int = 20261201,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """Verify the EOS endpoint, denominator, pairing and seed-cluster interval."""

    required = {
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
    }
    _require_columns(ledger, required, "EOS task ledger")
    seed_required = {
        "simulation_seed",
        "paired_tasks",
        "paired_difference",
    }
    _require_columns(paired_seed_results, seed_required, "EOS paired-seed table")

    frame = ledger.copy()
    bool_columns = (
        "action_executed",
        "no_action",
        "install_failed",
        "late_install",
        "geometry_allows_task",
        "observation_acquired",
        "product_valid",
        "application_complete",
        "met_deadline",
    )
    for column in bool_columns:
        frame[column] = _as_bool(frame[column], column)
    expected_endpoint = (
        frame["action_executed"]
        & ~frame["no_action"]
        & ~frame["install_failed"]
        & ~frame["late_install"]
        & frame["geometry_allows_task"]
        & frame["observation_acquired"]
        & frame["product_valid"]
        & pd.to_numeric(frame["bytes_delivered"], errors="coerce")
        .fillna(0)
        .ge(pd.to_numeric(frame["required_bytes"], errors="coerce").fillna(0))
        & frame["application_complete"]
        & frame["met_deadline"]
    ).astype(int)
    recorded_endpoint = pd.to_numeric(
        frame["conservative_on_time_completion"], errors="raise"
    ).astype(int)
    endpoint_violations = int(expected_endpoint.ne(recorded_endpoint).sum())

    policy_rows: dict[str, dict[str, Any]] = {}
    for policy, group in frame.groupby("policy_arm", sort=True):
        successes = int(
            pd.to_numeric(
                group["conservative_on_time_completion"], errors="raise"
            ).sum()
        )
        policy_rows[str(policy)] = {
            "denominator_tasks": int(len(group)),
            "successful_tasks": successes,
            "completion_rate": float(successes / len(group)),
            "no_action_tasks": int(group["no_action"].sum()),
        }

    pivot = frame.pivot(
        index=["simulation_seed", "logical_task_id"],
        columns="policy_arm",
        values="conservative_on_time_completion",
    ).reset_index()
    if {EOS_BASELINE, EOS_INTEGRATED}.difference(pivot.columns):
        raise ValueError("EOS task pairs do not contain both policy arms")
    paired_difference = (
        pd.to_numeric(pivot[EOS_INTEGRATED], errors="raise")
        - pd.to_numeric(pivot[EOS_BASELINE], errors="raise")
    )

    by_seed = paired_seed_results.sort_values("simulation_seed").reset_index(drop=True)
    differences = pd.to_numeric(by_seed["paired_difference"], errors="raise").to_numpy(float)
    weights = pd.to_numeric(by_seed["paired_tasks"], errors="raise").to_numpy(float)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_resamples, dtype=float)
    for iteration in range(bootstrap_resamples):
        selected = rng.integers(0, len(by_seed), size=len(by_seed))
        bootstrap[iteration] = float(
            np.average(differences[selected], weights=weights[selected])
        )
    reconstructed_ci = [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]
    recorded_ci = [
        float(value) for value in recorded_inference["cluster_bootstrap_95_ci"]
    ]

    return {
        "protocol": "eos-full-chain-sim-v2",
        "evidence_tier": "synthetic_simulation",
        "primary_endpoint": "conservative_on_time_completion",
        "endpoint_conjunction": [
            "action_executed",
            "not no_action",
            "not install_failed",
            "not late_install",
            "geometry_allows_task",
            "observation_acquired",
            "product_valid",
            "bytes_delivered >= required_bytes",
            "application_complete",
            "met_deadline",
        ],
        "endpoint_rule_compliant_rows": int(len(frame) - endpoint_violations),
        "endpoint_rule_total_rows": int(len(frame)),
        "endpoint_rule_violations": endpoint_violations,
        "policy_results": policy_rows,
        "paired_task_denominator": int(len(pivot)),
        "paired_improved_tasks": int(paired_difference.gt(0).sum()),
        "paired_regressed_tasks": int(paired_difference.lt(0).sum()),
        "paired_tied_tasks": int(paired_difference.eq(0).sum()),
        "paired_completion_difference": float(paired_difference.mean()),
        "paired_completion_difference_percentage_points": float(
            100.0 * paired_difference.mean()
        ),
        "bootstrap_cluster": "simulation_seed",
        "bootstrap_clusters": int(len(by_seed)),
        "tasks_per_cluster": {
            str(int(row.simulation_seed)): int(row.paired_tasks)
            for row in by_seed.itertuples(index=False)
        },
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_resamples": int(bootstrap_resamples),
        "bootstrap_construction": (
            "Resample the 10 simulation-seed labels with replacement; for every "
            "sampled label carry all paired task differences in that seed, take "
            "their mean, repeat 2,000 times, and use np.quantile at 0.025 and 0.975."
        ),
        "reconstructed_cluster_bootstrap_95_ci": reconstructed_ci,
        "recorded_cluster_bootstrap_95_ci": recorded_ci,
        "interval_reconstruction_matches_recorded": bool(
            np.allclose(reconstructed_ci, recorded_ci, atol=1e-12, rtol=0.0)
        ),
        "comparator": {
            "baseline": EOS_BASELINE,
            "baseline_rule": (
                "Order tasks by earliest deadline, then higher priority; among "
                "resource-feasible actions choose earliest window, then highest "
                "current SNR, subject to satellite/gateway conflicts."
            ),
            "integrated": EOS_INTEGRATED,
            "integrated_rule": (
                "Keep deadline primary, use feasible-opportunity scarcity and "
                "priority as tie-breaks, and rank released actions by predicted "
                "completion probability, subject to the same resource conflicts."
            ),
        },
        "claim_boundary": (
            "The interval resamples stochastic seeds that reuse one EOS-Bench "
            "scenario. It is not an independent-site, operator, or campaign interval."
        ),
    }


def audit_commect_paper4(
    policy_summary: pd.DataFrame,
    time_series_inference: pd.DataFrame,
    split_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the exact protocol to the Paper 4 63.27% versus 57.31% result."""

    required = {
        "policy",
        "scheduled_epochs",
        "action_epochs",
        "evaluable_epochs",
        "complete_case_success",
        "conservative_scheduled_success",
    }
    _require_columns(policy_summary, required, "Paper 4 COMMECT policy summary")
    indexed = policy_summary.set_index("policy")
    for policy in ("survival_candidate", "reactive", "evidence_gated_survival"):
        if policy not in indexed.index:
            raise ValueError(f"Paper 4 COMMECT summary lacks {policy!r}")
    candidate = indexed.loc["survival_candidate"]
    reactive = indexed.loc["reactive"]
    gated = indexed.loc["evidence_gated_survival"]
    denominator = int(candidate["scheduled_epochs"])
    if int(reactive["scheduled_epochs"]) != denominator:
        raise ValueError("Paper 4 COMMECT policies use different scheduled denominators")
    candidate_successes = int(
        round(float(candidate["conservative_scheduled_success"]) * denominator)
    )
    reactive_successes = int(
        round(float(reactive["conservative_scheduled_success"]) * denominator)
    )
    infer = time_series_inference[
        time_series_inference["reference"].eq("survival_candidate")
        & time_series_inference["baseline"].eq("reactive")
        & pd.to_numeric(time_series_inference["block_length"], errors="coerce").eq(60)
    ]
    if len(infer) != 1:
        raise ValueError("expected one 60-second Paper 4 COMMECT comparison")
    inference_row = infer.iloc[0]

    return {
        "experiment_id": "P4-COMMECT-5S-CS",
        "paper_role": "Paper 4 journal-extension experiment",
        "dataset_scope": "one COMMECT Denmark acquisition; offline concurrent shadow replay",
        "split": {
            "method": (
                "common-wall-clock chronological 60% train / 20% calibration / "
                "20% test with a 30-second purge at each boundary"
            ),
            **{str(key): int(value) for key, value in split_audit.items()},
        },
        "decision_cadence_seconds": 1,
        "selection_horizon_seconds": 5,
        "survival_model_max_horizon_seconds": 10,
        "target": (
            "selected path has a known exact sequence for the next five one-second "
            "bins and every future bin mean RTT is <= 60 ms"
        ),
        "eligibility_rule": (
            "Keep replying paths; if any have current mean RTT <= 60 ms, retain "
            "only those, otherwise retain all replying paths."
        ),
        "candidate_policy": (
            "Choose maximum predicted five-second survival probability; break ties "
            "with lower current mean RTT."
        ),
        "baseline_policy": "Choose minimum current mean RTT from the same eligible set.",
        "denominator_type": "complete scheduled one-second test grid",
        "scheduled_test_seconds": denominator,
        "action_seconds": int(candidate["action_epochs"]),
        "no_action_seconds": int(denominator - candidate["action_epochs"]),
        "candidate_evaluable_seconds": int(candidate["evaluable_epochs"]),
        "reactive_evaluable_seconds": int(reactive["evaluable_epochs"]),
        "unknown_or_no_action_treatment": "zero success on the scheduled denominator",
        "candidate_successes": candidate_successes,
        "candidate_success_rate": float(candidate["conservative_scheduled_success"]),
        "reactive_successes": reactive_successes,
        "reactive_success_rate": float(reactive["conservative_scheduled_success"]),
        "absolute_difference": float((candidate_successes - reactive_successes) / denominator),
        "absolute_difference_percentage_points": float(
            100.0 * (candidate_successes - reactive_successes) / denominator
        ),
        "moving_block_60s_95_ci": [
            float(inference_row["ci_low"]),
            float(inference_row["ci_high"]),
        ],
        "gated_matches_reactive": bool(
            float(gated["conservative_scheduled_success"])
            == float(reactive["conservative_scheduled_success"])
        ),
        "authorization": (
            "The survival candidate is ungated. With one acquisition, the study-level "
            "admission rule retains reactive control."
        ),
        "claim_boundary": (
            "Within-series moving-block uncertainty from one shadow-replay campaign; "
            "not an installed policy trial or independent-campaign replication."
        ),
    }


def audit_commect_paper2(
    policy_summary: pd.DataFrame,
    validation_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the exact protocol to Paper 2's fixed 1-second comparison."""

    required = {"policy_name", "decision_count", "success_rate_under_60ms"}
    _require_columns(policy_summary, required, "Paper 2 COMMECT policy summary")
    indexed = policy_summary.set_index("policy_name")
    candidate_name = "qos_filter_then_ensemble_selector"
    baseline_name = "reactive_greedy"
    for policy in (candidate_name, baseline_name, "validation_gated_qos_selector"):
        if policy not in indexed.index:
            raise ValueError(f"Paper 2 COMMECT summary lacks {policy!r}")
    candidate = indexed.loc[candidate_name]
    baseline = indexed.loc[baseline_name]
    denominator = int(candidate["decision_count"])
    if int(baseline["decision_count"]) != denominator:
        raise ValueError("Paper 2 COMMECT policies use different complete-case denominators")
    candidate_successes = int(round(float(candidate["success_rate_under_60ms"]) * denominator))
    baseline_successes = int(round(float(baseline["success_rate_under_60ms"]) * denominator))
    split = validation_metadata["split_boundary_audit"]
    scheduled = int(validation_metadata["scheduled_test_epoch_count"])

    return {
        "experiment_id": "P2-COMMECT-1S-CC-FTR",
        "paper_role": "Paper 2 fixed one-second alignment/denominator audit",
        "dataset_scope": "one COMMECT Denmark acquisition; offline concurrent shadow replay",
        "split": {
            "method": (
                "fixed wall-clock split frozen before completeness filtering: "
                "55% train / 15% calibration / 15% policy selection / 15% test"
            ),
            "ratios": split["ratios"],
            "test_first_epoch": int(split["partitions"]["test"]["first_bin_epoch"]),
            "test_last_epoch": int(split["partitions"]["test"]["last_bin_epoch"]),
            "targets_crossing_boundaries_excluded": True,
        },
        "decision_cadence_seconds": 1,
        "prediction_horizon_seconds": 1,
        "target": "selected path's exact next-interval mean RTT is <= 60 ms",
        "complete_case_rule": (
            "Retain epochs with the candidate observations and exact next-interval "
            "outcomes required for matched comparison."
        ),
        "candidate_policy": (
            "Ungated filter-then-rank ensemble: rank by the frozen ensemble score "
            "within currently <=60-ms paths, or all available paths if none comply."
        ),
        "baseline_policy": "Choose minimum current mean RTT among available paths.",
        "denominator_type": "evaluable complete-case test decisions",
        "scheduled_test_epochs": scheduled,
        "complete_case_decisions": denominator,
        "unknown_or_excluded_scheduled_epochs": int(scheduled - denominator),
        "candidate_successes": candidate_successes,
        "candidate_success_rate": float(candidate["success_rate_under_60ms"]),
        "baseline_successes": baseline_successes,
        "baseline_success_rate": float(baseline["success_rate_under_60ms"]),
        "absolute_difference": float((candidate_successes - baseline_successes) / denominator),
        "absolute_difference_percentage_points": float(
            100.0 * (candidate_successes - baseline_successes) / denominator
        ),
        "candidate_scheduled_lower_bound": float(candidate_successes / scheduled),
        "baseline_scheduled_lower_bound": float(baseline_successes / scheduled),
        "authorization": (
            "The filter-then-rank ensemble was an unscreened ablation outside the "
            "declared gate candidate family; it was not authorized."
        ),
        "claim_boundary": (
            "This is a separate fixed 1-second complete-case protocol. It is not a "
            "replication of P4-COMMECT-5S-CS and not a live steering result."
        ),
    }


def describe_gate_mechanisms(
    immediate_summary: pd.DataFrame,
    delayed_summary: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return non-overlapping definitions for the repository's three safeguards."""

    if len(immediate_summary) != 1 or len(delayed_summary) != 1:
        raise ValueError("gate stress summaries must each contain exactly one row")
    immediate = immediate_summary.iloc[0]
    delayed = delayed_summary.iloc[0]
    return [
        {
            "mechanism_id": "input_ood_guard",
            "unit": "candidate feature vector (or non-overlapping feature batch in stress audit)",
            "timing": "before a candidate is released for learned ranking",
            "rule": (
                "Development-fitted median imputation and standardization, Ledoit-Wolf "
                "Mahalanobis distance, and a frozen 0.99 development quantile."
            ),
            "eos_v2_effect": (
                "A one-row OOD candidate is not released; another in-domain candidate "
                "may still be selected, so an OOD row is not a fallback decision."
            ),
            "stress_audit": {
                "evidence_tier": str(immediate["evidence_tier"]),
                "stable_campaigns": int(immediate["stable_campaigns"]),
                "false_alarm_campaigns": int(immediate["false_alarm_campaigns"]),
                "shift_campaigns": int(immediate["shift_campaigns"]),
                "detected_shift_campaigns": int(immediate["detected_shift_campaigns"]),
                "median_detection_delay_rows": float(
                    immediate["median_detection_delay_rows"]
                ),
            },
            "not_this": "not the delayed loss monitor and not whole-policy admission",
        },
        {
            "mechanism_id": "delayed_stream_monitor",
            "unit": "matured bounded calibration-loss stream, with a fresh monitor per campaign",
            "timing": "after outcomes/labels mature",
            "rule": (
                "E-process with increments exp(lambda*(loss-null_mean)-lambda^2/8); "
                "alarm when E >= 1/alpha. The locked stress uses null_mean=0.10, "
                "lambda=1, and alpha=0.01."
            ),
            "stress_audit": {
                "evidence_tier": str(delayed["evidence_tier"]),
                "stable_campaigns": int(delayed["stable_campaigns"]),
                "false_alarm_campaigns": int(delayed["false_alarm_campaigns"]),
                "shift_campaigns": int(delayed["shift_campaigns"]),
                "detected_shift_campaigns": int(delayed["detected_shift_campaigns"]),
                "median_detection_delay_outcomes": float(
                    delayed["median_detection_delay_outcomes"]
                ),
            },
            "not_this": (
                "cannot screen the current unlabeled row and does not itself authorize a policy"
            ),
        },
        {
            "mechanism_id": "offline_campaign_admission_gate",
            "unit": "entire candidate policy across locked campaign-level evidence",
            "timing": "offline, before deployment/test release",
            "rule": (
                "Check schema/clock validity, independent-campaign support, calibration "
                "and safety bounds, and control-loop timing against frozen thresholds; "
                "failure retains the named baseline."
            ),
            "commect_paper4_status": (
                "One acquisition is insufficient, so evidence_gated_survival is "
                "explicitly the reactive choice at every action epoch."
            ),
            "prospective_validation_contract": (
                "assess_research_completion additionally requires a frozen gate, enough "
                "independent field campaigns, false-alarm upper bound, detection lower "
                "bound, and fallback-safety lower bound."
            ),
            "not_this": (
                "not inferred from one OOD candidate and not validated by the synthetic "
                "delayed-monitor operating curve"
            ),
        },
    ]
