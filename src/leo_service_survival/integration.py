"""Validated relational contract for prospective end-to-end LEO studies.

The existing exploratory experiments intentionally keep unrelated evidence sources
separate.  This module defines the stricter contract required before a result may be
described as a continuous geometry-to-application experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


EVIDENCE_TIERS = (
    "field_live_randomized",
    "field_shadow_concurrent",
    "field_observational",
    "hardware_in_loop",
    "measured_trace_replay",
    "synthetic_simulation",
    "geometry_only",
)

FIELD_EVIDENCE_TIERS = (
    "field_live_randomized",
    "field_shadow_concurrent",
    "field_observational",
)

COMPLETION_CRITERIA = (
    "continuous_field_chain",
    "independent_multisite_multioperator_validation",
    "matched_satellite_beam_gateway_handover_rf",
    "operational_eo_workload_validation",
    "validated_data_driven_gate",
    "replicated_primary_endpoint_benefit",
)


REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "campaign_manifest": frozenset(
        {
            "campaign_id",
            "independent_cluster_id",
            "site_id",
            "collection_date",
            "randomized_policy_arm",
            "protocol_version",
            "model_sha256",
            "config_sha256",
            "evidence_tier",
        }
    ),
    "candidate_path_observation": frozenset(
        {
            "campaign_id",
            "decision_id",
            "candidate_path_id",
            "interface_type",
            "operator_id",
            "path_available",
            "telemetry_sample_ns",
            "telemetry_cutoff_ns",
            "observation_age_ms",
            "satellite_id",
            "beam_id",
            "gateway_id",
            "handover_state",
            "rf_snr_db",
            "geometry_allows_task",
            "predicted_completion_probability",
            "missingness_reason",
        }
    ),
    "task": frozenset(
        {
            "campaign_id",
            "task_id",
            "decision_id",
            "release_time_ns",
            "required_duration_s",
            "required_bytes",
            "priority",
            "deadline_ns",
            "task_type",
        }
    ),
    "decision_action": frozenset(
        {
            "campaign_id",
            "task_id",
            "decision_id",
            "policy_arm",
            "selected_path_id",
            "baseline_path_id",
            "gate_state",
            "gate_reason",
            "shift_alarm_state",
            "decision_start_ns",
            "decision_end_ns",
            "install_ack_ns",
            "installed_path_id",
            "install_failed",
            "late_install",
            "rollback",
        }
    ),
    "eo_resource_state": frozenset(
        {
            "campaign_id",
            "task_id",
            "decision_id",
            "satellite_id",
            "ground_station_id",
            "contact_id",
            "observation_feasible",
            "storage_available_bytes",
            "downlink_capacity_bytes",
            "predicted_congestion",
        }
    ),
    "outcome": frozenset(
        {
            "campaign_id",
            "task_id",
            "decision_id",
            "action_executed",
            "observation_acquired",
            "product_valid",
            "bytes_delivered",
            "application_complete",
            "completion_time_ns",
            "met_deadline",
            "no_action",
            "failure_stage",
        }
    ),
}


CRITICAL_NONNULL: dict[str, frozenset[str]] = {
    "campaign_manifest": frozenset(
        {
            "campaign_id",
            "independent_cluster_id",
            "site_id",
            "randomized_policy_arm",
            "protocol_version",
            "model_sha256",
            "config_sha256",
            "evidence_tier",
        }
    ),
    "candidate_path_observation": frozenset(
        {
            "campaign_id",
            "decision_id",
            "candidate_path_id",
            "path_available",
            "telemetry_sample_ns",
            "telemetry_cutoff_ns",
            "geometry_allows_task",
            "predicted_completion_probability",
        }
    ),
    "task": frozenset(
        {
            "campaign_id",
            "task_id",
            "decision_id",
            "release_time_ns",
            "deadline_ns",
        }
    ),
    "decision_action": frozenset(
        {
            "campaign_id",
            "task_id",
            "decision_id",
            "policy_arm",
            "gate_state",
            "decision_start_ns",
            "decision_end_ns",
            "install_failed",
            "late_install",
            "rollback",
        }
    ),
    "eo_resource_state": frozenset(
        {
            "campaign_id",
            "task_id",
            "decision_id",
            "observation_feasible",
        }
    ),
    "outcome": frozenset(
        {
            "campaign_id",
            "task_id",
            "decision_id",
            "action_executed",
            "observation_acquired",
            "product_valid",
            "bytes_delivered",
            "application_complete",
            "met_deadline",
            "no_action",
        }
    ),
}


@dataclass(frozen=True)
class FullChainAudit:
    """Machine-readable validation result for a full-chain evidence bundle."""

    checks: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool(len(self.checks) and self.checks["passed"].all())

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        failed = self.checks.loc[~self.checks["passed"], ["check", "details"]]
        details = "; ".join(
            f"{row.check}: {row.details}" for row in failed.itertuples(index=False)
        )
        raise ValueError(f"full-chain evidence audit failed: {details}")


@dataclass(frozen=True)
class ResearchCompletionAudit:
    """Pass/fail ledger for the six thesis-completion evidence requirements.

    This audit is intentionally stricter than :class:`FullChainAudit`.  The
    latter checks whether tables form a coherent engineering chain.  This class
    additionally distinguishes field evidence from trace replay and simulation,
    and requires replication and inferential support before research is labelled
    complete.
    """

    checks: pd.DataFrame

    @property
    def completed(self) -> bool:
        return bool(len(self.checks) and self.checks["passed"].all())

    @property
    def completed_count(self) -> int:
        return int(self.checks["passed"].sum()) if len(self.checks) else 0

    def raise_for_incomplete(self) -> None:
        if self.completed:
            return
        failed = self.checks.loc[~self.checks["passed"], ["criterion", "details"]]
        details = "; ".join(
            f"{row.criterion}: {row.details}" for row in failed.itertuples(index=False)
        )
        raise ValueError(f"research completion audit failed: {details}")


def _check_row(check: str, passed: bool, details: str, failed_rows: int = 0) -> dict:
    return {
        "check": check,
        "passed": bool(passed),
        "failed_rows": int(failed_rows),
        "details": str(details),
    }


def validate_full_chain_bundle(tables: Mapping[str, pd.DataFrame]) -> FullChainAudit:
    """Validate schema, keys, clocks, installed actions, and evidence tiers.

    A successful audit means the supplied tables are internally capable of
    representing a continuous experiment.  It does not by itself establish that a
    field campaign is independent, randomized, sufficiently powered, or beneficial.
    """

    checks: list[dict] = []
    missing_tables = sorted(set(REQUIRED_COLUMNS).difference(tables))
    checks.append(
        _check_row(
            "required_tables",
            not missing_tables,
            "all required tables present" if not missing_tables else f"missing {missing_tables}",
        )
    )
    if missing_tables:
        return FullChainAudit(pd.DataFrame(checks))

    for name, required in REQUIRED_COLUMNS.items():
        frame = tables[name]
        missing = sorted(required.difference(frame.columns))
        checks.append(
            _check_row(
                f"{name}.required_columns",
                not missing,
                "complete" if not missing else f"missing {missing}",
            )
        )
        if missing:
            continue
        critical = sorted(CRITICAL_NONNULL[name])
        null_rows = int(frame[critical].isna().any(axis=1).sum())
        checks.append(
            _check_row(
                f"{name}.critical_nonnull",
                null_rows == 0,
                "complete" if null_rows == 0 else "critical identifiers or outcomes are null",
                null_rows,
            )
        )

    manifest = tables["campaign_manifest"]
    if REQUIRED_COLUMNS["campaign_manifest"].issubset(manifest.columns):
        duplicates = int(manifest["campaign_id"].duplicated().sum())
        invalid_tiers = int((~manifest["evidence_tier"].isin(EVIDENCE_TIERS)).sum())
        checks.append(
            _check_row(
                "campaign_manifest.unique_campaign",
                duplicates == 0,
                "unique" if duplicates == 0 else "duplicate campaign identifiers",
                duplicates,
            )
        )
        checks.append(
            _check_row(
                "campaign_manifest.evidence_tier",
                invalid_tiers == 0,
                "valid" if invalid_tiers == 0 else "unknown evidence tier",
                invalid_tiers,
            )
        )

    unique_keys = {
        "task": ["task_id"],
        "decision_action": ["decision_id"],
        "outcome": ["task_id"],
        "candidate_path_observation": ["decision_id", "candidate_path_id"],
        "eo_resource_state": ["task_id", "contact_id"],
    }
    for name, columns in unique_keys.items():
        frame = tables[name]
        if set(columns).issubset(frame.columns):
            duplicate_rows = int(frame.duplicated(columns).sum())
            checks.append(
                _check_row(
                    f"{name}.unique_key",
                    duplicate_rows == 0,
                    "unique" if duplicate_rows == 0 else f"duplicate key {columns}",
                    duplicate_rows,
                )
            )

    task = tables["task"]
    action = tables["decision_action"]
    outcome = tables["outcome"]
    candidate = tables["candidate_path_observation"]
    resource = tables["eo_resource_state"]
    campaigns = set(manifest.get("campaign_id", pd.Series(dtype=str)).dropna())

    for name, frame in tables.items():
        if name == "campaign_manifest" or "campaign_id" not in frame:
            continue
        orphan = int((~frame["campaign_id"].isin(campaigns)).sum())
        checks.append(
            _check_row(
                f"{name}.campaign_foreign_key",
                orphan == 0,
                "valid" if orphan == 0 else "campaign absent from manifest",
                orphan,
            )
        )

    if {"task_id", "decision_id"}.issubset(task.columns):
        task_ids = set(task["task_id"].dropna())
        decision_ids = set(task["decision_id"].dropna())
        for name, frame in (("decision_action", action), ("outcome", outcome), ("eo_resource_state", resource)):
            orphan_tasks = int((~frame["task_id"].isin(task_ids)).sum())
            orphan_decisions = int((~frame["decision_id"].isin(decision_ids)).sum())
            checks.append(
                _check_row(
                    f"{name}.task_foreign_key",
                    orphan_tasks == 0,
                    "valid" if orphan_tasks == 0 else "task absent from task table",
                    orphan_tasks,
                )
            )
            checks.append(
                _check_row(
                    f"{name}.decision_foreign_key",
                    orphan_decisions == 0,
                    "valid" if orphan_decisions == 0 else "decision absent from task table",
                    orphan_decisions,
                )
            )

        candidate_orphans = int((~candidate["decision_id"].isin(decision_ids)).sum())
        checks.append(
            _check_row(
                "candidate_path_observation.decision_foreign_key",
                candidate_orphans == 0,
                "valid" if candidate_orphans == 0 else "decision absent from task table",
                candidate_orphans,
            )
        )

    if {"decision_id", "selected_path_id", "installed_path_id", "install_failed"}.issubset(action.columns):
        candidate_keys = set(
            zip(candidate["decision_id"].astype(str), candidate["candidate_path_id"].astype(str))
        )
        selected_present = action["selected_path_id"].notna()
        selected_keys = list(
            zip(
                action.loc[selected_present, "decision_id"].astype(str),
                action.loc[selected_present, "selected_path_id"].astype(str),
            )
        )
        missing_selected = sum(key not in candidate_keys for key in selected_keys)
        checks.append(
            _check_row(
                "decision_action.selected_candidate_exists",
                missing_selected == 0,
                "valid" if missing_selected == 0 else "selected path absent from candidate table",
                missing_selected,
            )
        )
        installed_mismatch = (
            selected_present
            & ~action["install_failed"].astype(bool)
            & action["installed_path_id"].astype(str).ne(action["selected_path_id"].astype(str))
        )
        mismatch_rows = int(installed_mismatch.sum())
        checks.append(
            _check_row(
                "decision_action.installed_action_verified",
                mismatch_rows == 0,
                "valid" if mismatch_rows == 0 else "installed path differs without install failure",
                mismatch_rows,
            )
        )

    if {"telemetry_sample_ns", "telemetry_cutoff_ns", "observation_age_ms"}.issubset(candidate.columns):
        age_ns = candidate["telemetry_cutoff_ns"] - candidate["telemetry_sample_ns"]
        invalid_clock = (
            age_ns.lt(0)
            | candidate["observation_age_ms"].lt(0)
            | ~np.isclose(
                age_ns.to_numpy(float) / 1_000_000.0,
                candidate["observation_age_ms"].to_numpy(float),
                atol=1.0,
                equal_nan=False,
            )
        )
        invalid_rows = int(invalid_clock.sum())
        checks.append(
            _check_row(
                "candidate_path_observation.clock_consistency",
                invalid_rows == 0,
                "valid" if invalid_rows == 0 else "negative or inconsistent observation age",
                invalid_rows,
            )
        )

    if {"decision_start_ns", "decision_end_ns", "install_ack_ns"}.issubset(action.columns):
        invalid_decision = action["decision_end_ns"].lt(action["decision_start_ns"])
        acknowledged = action["install_ack_ns"].notna()
        invalid_ack = acknowledged & action["install_ack_ns"].lt(action["decision_end_ns"])
        invalid_rows = int((invalid_decision | invalid_ack).sum())
        checks.append(
            _check_row(
                "decision_action.clock_order",
                invalid_rows == 0,
                "valid" if invalid_rows == 0 else "decision/install timestamps are out of order",
                invalid_rows,
            )
        )

    return FullChainAudit(pd.DataFrame(checks))


def build_full_chain_ledger(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one conservative row per task after a successful evidence audit."""

    audit = validate_full_chain_bundle(tables)
    audit.raise_for_errors()

    task = tables["task"].copy()
    action = tables["decision_action"].copy()
    outcome = tables["outcome"].copy()
    resource = tables["eo_resource_state"].copy()
    manifest = tables["campaign_manifest"].copy()
    candidate = tables["candidate_path_observation"].copy()

    selected = action[["decision_id", "selected_path_id"]].dropna().copy()
    selected["decision_id"] = selected["decision_id"].astype(str)
    selected["selected_path_id"] = selected["selected_path_id"].astype(str)
    candidate["decision_id"] = candidate["decision_id"].astype(str)
    candidate["candidate_path_id"] = candidate["candidate_path_id"].astype(str)
    selected_candidate = selected.merge(
        candidate,
        left_on=["decision_id", "selected_path_id"],
        right_on=["decision_id", "candidate_path_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_candidate"),
    )
    selected_columns = [
        "decision_id",
        "candidate_path_id",
        "operator_id",
        "satellite_id",
        "beam_id",
        "gateway_id",
        "handover_state",
        "rf_snr_db",
        "geometry_allows_task",
        "predicted_completion_probability",
    ]
    selected_candidate = selected_candidate[selected_columns]

    resources = resource.groupby(["task_id", "decision_id"], as_index=False).agg(
        feasible_contacts=("observation_feasible", "sum"),
        available_downlink_bytes=("downlink_capacity_bytes", "sum"),
        maximum_predicted_congestion=("predicted_congestion", "max"),
    )

    ledger = (
        task.merge(action, on=["campaign_id", "task_id", "decision_id"], validate="one_to_one")
        .merge(outcome, on=["campaign_id", "task_id", "decision_id"], validate="one_to_one")
        .merge(resources, on=["task_id", "decision_id"], validate="one_to_one")
        .merge(selected_candidate, on="decision_id", how="left", validate="one_to_one")
        .merge(
            manifest[
                [
                    "campaign_id",
                    "independent_cluster_id",
                    "site_id",
                    "randomized_policy_arm",
                    "evidence_tier",
                    "model_sha256",
                    "config_sha256",
                ]
            ],
            on="campaign_id",
            validate="many_to_one",
        )
    )

    declared_bytes = pd.to_numeric(ledger["required_bytes"], errors="coerce").fillna(0)
    delivered_bytes = pd.to_numeric(ledger["bytes_delivered"], errors="coerce").fillna(0)
    selected_geometry = ledger["geometry_allows_task"].fillna(False).astype(bool)
    ledger["conservative_on_time_completion"] = (
        ledger["action_executed"].astype(bool)
        & ~ledger["no_action"].astype(bool)
        & ~ledger["install_failed"].astype(bool)
        & ~ledger["late_install"].astype(bool)
        & selected_geometry
        & ledger["observation_acquired"].astype(bool)
        & ledger["product_valid"].astype(bool)
        & delivered_bytes.ge(declared_bytes)
        & ledger["application_complete"].astype(bool)
        & ledger["met_deadline"].astype(bool)
    ).astype(int)
    ledger["control_loop_ms"] = (
        pd.to_numeric(ledger["install_ack_ns"], errors="coerce")
        - pd.to_numeric(ledger["decision_start_ns"], errors="coerce")
    ) / 1_000_000.0
    ledger["prediction_error"] = (
        ledger["predicted_completion_probability"]
        - ledger["conservative_on_time_completion"]
    )
    return ledger.sort_values(["campaign_id", "task_id"]).reset_index(drop=True)


def summarize_full_chain_completion(ledger: pd.DataFrame) -> pd.DataFrame:
    """Report policy effects without pretending campaigns are iid time bins."""

    required = {
        "independent_cluster_id",
        "policy_arm",
        "evidence_tier",
        "conservative_on_time_completion",
        "control_loop_ms",
    }
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError(f"ledger is missing columns: {sorted(missing)}")
    return (
        ledger.groupby(["evidence_tier", "policy_arm"], as_index=False)
        .agg(
            tasks=("conservative_on_time_completion", "size"),
            independent_clusters=("independent_cluster_id", "nunique"),
            completion_rate=("conservative_on_time_completion", "mean"),
            control_loop_p95_ms=("control_loop_ms", lambda value: float(np.nanpercentile(value, 95))),
        )
        .sort_values(["evidence_tier", "policy_arm"])
        .reset_index(drop=True)
    )


def assess_research_completion(
    tables: Mapping[str, pd.DataFrame],
    *,
    gate_validation: pd.DataFrame | None = None,
    paired_policy_effects: pd.DataFrame | None = None,
    minimum_sites: int = 4,
    minimum_operator_contexts: int = 2,
    minimum_independent_clusters: int = 30,
) -> ResearchCompletionAudit:
    """Evaluate the six requested completion criteria without tier conflation.

    Parameters are protocol thresholds and must be fixed before a confirmatory
    test is inspected.  Simulation, hardware-in-the-loop, and trace replay can
    exercise the same software path, but only prospective field rows are allowed
    to satisfy the empirical criteria below.

    ``gate_validation`` must contain one locked summary row with the columns
    ``gate_frozen_before_test``, ``independent_validation_campaigns``,
    ``false_alarm_rate_upper``, ``maximum_false_alarm_rate``,
    ``detection_rate_lower``, ``minimum_detection_rate``,
    ``fallback_safety_effect_lower``, and ``minimum_fallback_safety_effect``.

    ``paired_policy_effects`` must identify a prespecified primary comparison
    using ``prespecified_primary``, ``evidence_tier``, ``independent_clusters``,
    ``effect_ci_lower``, and ``minimum_practical_gain``.  A universal-win
    requirement is deliberately not used; the primary field endpoint must clear
    its frozen practical threshold.
    """

    checks: list[dict] = []

    def add(criterion: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "criterion": criterion,
                "passed": bool(passed),
                "details": str(details),
            }
        )

    full_audit = validate_full_chain_bundle(tables)
    manifest = tables.get("campaign_manifest", pd.DataFrame())
    task = tables.get("task", pd.DataFrame())
    candidates = tables.get("candidate_path_observation", pd.DataFrame())
    outcomes = tables.get("outcome", pd.DataFrame())

    field_campaigns: set[str] = set()
    if {"campaign_id", "evidence_tier"}.issubset(manifest.columns):
        field_campaigns = set(
            manifest.loc[
                manifest["evidence_tier"].isin(FIELD_EVIDENCE_TIERS), "campaign_id"
            ].astype(str)
        )
    live_campaigns: set[str] = set()
    if {"campaign_id", "evidence_tier"}.issubset(manifest.columns):
        live_campaigns = set(
            manifest.loc[
                manifest["evidence_tier"].eq("field_live_randomized"), "campaign_id"
            ].astype(str)
        )

    field_manifest = (
        manifest[manifest["campaign_id"].astype(str).isin(field_campaigns)]
        if "campaign_id" in manifest
        else pd.DataFrame()
    )

    field_tasks = (
        task[task["campaign_id"].astype(str).isin(field_campaigns)]
        if "campaign_id" in task
        else pd.DataFrame()
    )
    field_outcomes = (
        outcomes[outcomes["campaign_id"].astype(str).isin(field_campaigns)]
        if "campaign_id" in outcomes
        else pd.DataFrame()
    )
    has_observed_completion = bool(
        len(field_outcomes)
        and {"application_complete", "met_deadline"}.issubset(field_outcomes.columns)
        and field_outcomes[["application_complete", "met_deadline"]].notna().all(axis=None)
    )
    continuous_pass = bool(
        full_audit.passed
        and field_campaigns
        and len(field_tasks)
        and has_observed_completion
    )
    add(
        "continuous_field_chain",
        continuous_pass,
        (
            f"schema_passed={full_audit.passed}; field_campaigns={len(field_campaigns)}; "
            f"field_tasks={len(field_tasks)}; observed_completion={has_observed_completion}"
        ),
    )

    site_count = (
        int(field_manifest["site_id"].nunique()) if "site_id" in field_manifest else 0
    )
    cluster_count = (
        int(field_manifest["independent_cluster_id"].nunique())
        if "independent_cluster_id" in field_manifest
        else 0
    )
    operator_columns = [
        column
        for column in ("operator_context_id", "leo_provider_id", "terminal_context_id")
        if column in field_manifest
    ]
    if operator_columns:
        contexts = field_manifest[operator_columns].astype(str).agg("|".join, axis=1)
        context_count = int(contexts.nunique())
    else:
        context_count = 0
    independent_pass = bool(
        len(field_campaigns)
        and site_count >= minimum_sites
        and context_count >= minimum_operator_contexts
        and cluster_count >= minimum_independent_clusters
    )
    add(
        "independent_multisite_multioperator_validation",
        independent_pass,
        (
            f"field_sites={site_count}/{minimum_sites}; operator_or_terminal_contexts="
            f"{context_count}/{minimum_operator_contexts}; independent_clusters="
            f"{cluster_count}/{minimum_independent_clusters}"
        ),
    )

    field_candidates = (
        candidates[candidates["campaign_id"].astype(str).isin(field_campaigns)]
        if "campaign_id" in candidates
        else pd.DataFrame()
    )
    leo_candidates = (
        field_candidates[
            field_candidates["interface_type"].astype(str).str.lower().isin(
                {"leo", "satellite", "ntn", "starlink", "oneweb"}
            )
        ]
        if "interface_type" in field_candidates
        else pd.DataFrame()
    )
    matched_columns = (
        "satellite_id",
        "beam_id",
        "gateway_id",
        "handover_state",
        "rf_snr_db",
    )
    matched_available = set(matched_columns).issubset(leo_candidates.columns)
    matched_rows = bool(
        len(leo_candidates)
        and matched_available
        and leo_candidates[list(matched_columns)].notna().all(axis=None)
    )
    add(
        "matched_satellite_beam_gateway_handover_rf",
        matched_rows,
        (
            f"field_leo_candidates={len(leo_candidates)}; required_fields_present="
            f"{matched_available}; all_matched_nonnull={matched_rows}"
        ),
    )

    operational_tasks = pd.DataFrame()
    if {"campaign_id", "workload_origin"}.issubset(task.columns):
        operational_tasks = task[
            task["campaign_id"].astype(str).isin(live_campaigns)
            & task["workload_origin"].astype(str).eq("operational")
        ]
    operational_ids = set(operational_tasks.get("task_id", pd.Series(dtype=str)).astype(str))
    operational_outcomes = (
        outcomes[outcomes["task_id"].astype(str).isin(operational_ids)]
        if "task_id" in outcomes
        else pd.DataFrame()
    )
    operational_pass = bool(
        live_campaigns
        and len(operational_tasks)
        and len(operational_outcomes) == len(operational_tasks)
        and {"application_complete", "met_deadline"}.issubset(
            operational_outcomes.columns
        )
        and operational_outcomes[["application_complete", "met_deadline"]]
        .notna()
        .all(axis=None)
    )
    add(
        "operational_eo_workload_validation",
        operational_pass,
        (
            f"live_randomized_campaigns={len(live_campaigns)}; operational_tasks="
            f"{len(operational_tasks)}; linked_outcomes={len(operational_outcomes)}"
        ),
    )

    gate_pass = False
    gate_details = "no locked prospective gate-validation summary supplied"
    gate_required = {
        "gate_frozen_before_test",
        "independent_validation_campaigns",
        "false_alarm_rate_upper",
        "maximum_false_alarm_rate",
        "detection_rate_lower",
        "minimum_detection_rate",
        "fallback_safety_effect_lower",
        "minimum_fallback_safety_effect",
    }
    if gate_validation is not None and len(gate_validation) == 1:
        missing = gate_required.difference(gate_validation.columns)
        if not missing:
            row = gate_validation.iloc[0]
            gate_pass = bool(
                row["gate_frozen_before_test"]
                and int(row["independent_validation_campaigns"])
                >= minimum_operator_contexts
                and float(row["false_alarm_rate_upper"])
                <= float(row["maximum_false_alarm_rate"])
                and float(row["detection_rate_lower"])
                >= float(row["minimum_detection_rate"])
                and float(row["fallback_safety_effect_lower"])
                >= float(row["minimum_fallback_safety_effect"])
            )
            gate_details = (
                f"frozen={bool(row['gate_frozen_before_test'])}; validation_campaigns="
                f"{int(row['independent_validation_campaigns'])}; false_alarm_upper="
                f"{float(row['false_alarm_rate_upper']):.4f}; detection_lower="
                f"{float(row['detection_rate_lower']):.4f}; fallback_safety_lower="
                f"{float(row['fallback_safety_effect_lower']):.4f}"
            )
        else:
            gate_details = f"gate-validation summary missing {sorted(missing)}"
    add("validated_data_driven_gate", gate_pass, gate_details)

    benefit_pass = False
    benefit_details = "no prespecified paired field-effect summary supplied"
    benefit_required = {
        "prespecified_primary",
        "evidence_tier",
        "independent_clusters",
        "effect_ci_lower",
        "minimum_practical_gain",
    }
    if paired_policy_effects is not None:
        missing = benefit_required.difference(paired_policy_effects.columns)
        if not missing:
            eligible = paired_policy_effects[
                paired_policy_effects["prespecified_primary"].astype(bool)
                & paired_policy_effects["evidence_tier"].eq("field_live_randomized")
            ]
            if len(eligible) == 1:
                row = eligible.iloc[0]
                benefit_pass = bool(
                    int(row["independent_clusters"]) >= minimum_independent_clusters
                    and float(row["effect_ci_lower"])
                    > float(row["minimum_practical_gain"])
                )
                benefit_details = (
                    f"independent_clusters={int(row['independent_clusters'])}/"
                    f"{minimum_independent_clusters}; effect_ci_lower="
                    f"{float(row['effect_ci_lower']):.6f}; required_gain="
                    f"{float(row['minimum_practical_gain']):.6f}"
                )
            else:
                benefit_details = (
                    "expected exactly one prespecified primary live-field comparison; "
                    f"found {len(eligible)}"
                )
        else:
            benefit_details = f"paired-effect summary missing {sorted(missing)}"
    add("replicated_primary_endpoint_benefit", benefit_pass, benefit_details)

    frame = pd.DataFrame(checks)
    if tuple(frame["criterion"]) != COMPLETION_CRITERIA:
        raise AssertionError("completion criteria were emitted out of protocol order")
    return ResearchCompletionAudit(frame)
