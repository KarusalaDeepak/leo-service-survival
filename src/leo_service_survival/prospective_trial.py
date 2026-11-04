"""Prospective closed-loop trial controls for Objective 4.

This module supplies the execution safeguards that cannot be inferred from an
offline trace: acquisition-role separation, immutable action/outcome records,
verified route installation, first-packet observation, and automatic fallback.
It deliberately does *not* claim that a field trial has occurred.  Hardware
drivers and field observations must be supplied by an operator deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd


PROTOCOL_ROLES = ("pilot", "development", "gate_evidence", "final_test")
CAMPAIGN_STATUSES = ("planned", "collecting", "complete", "locked", "excluded")
RECORD_TYPES = ("action", "outcome")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable collection of validation checks."""

    checks: pd.DataFrame

    @property
    def passed(self) -> bool:
        return bool(len(self.checks) and self.checks["passed"].all())

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        failures = self.checks.loc[~self.checks["passed"], ["check", "details"]]
        detail = "; ".join(
            f"{row.check}: {row.details}" for row in failures.itertuples(index=False)
        )
        raise ValueError(f"prospective-trial validation failed: {detail}")


def _report(rows: list[tuple[str, bool, str]]) -> ValidationReport:
    return ValidationReport(
        pd.DataFrame(rows, columns=("check", "passed", "details"))
    )


def _is_sha256(value: object) -> bool:
    return bool(_SHA256_PATTERN.fullmatch(str(value)))


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return the canonical representation used by the ledger hash chain."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_campaign_register(
    frame: pd.DataFrame,
    *,
    require_locked: bool = False,
    minimum_gate_campaigns: int = 5,
    minimum_gate_families: int = 3,
) -> ValidationReport:
    """Audit acquisition roles without treating repeated days as independent.

    The same acquisition family may contain multiple campaigns, but it may not
    cross development, gate-evidence, and final-test roles.  A shared acquisition
    fingerprint may not be relabelled as two families.  The numerical floors are
    eligibility checks only; power and precision may require more campaigns.
    """

    required = {
        "study_id",
        "campaign_id",
        "acquisition_family_id",
        "role",
        "acquisition_pipeline_id",
        "site_id",
        "operator_id",
        "terminal_id",
        "route_id",
        "server_id",
        "collection_start_utc",
        "collection_end_utc",
        "protocol_version",
        "model_sha256",
        "config_sha256",
        "clock_sync_method",
        "maximum_clock_error_ms",
        "status",
        "evidence_tier",
    }
    rows: list[tuple[str, bool, str]] = []
    missing = sorted(required.difference(frame.columns))
    rows.append(
        (
            "register.required_columns",
            not missing,
            "complete" if not missing else f"missing {missing}",
        )
    )
    if missing:
        return _report(rows)

    rows.append(
        (
            "register.nonempty",
            bool(len(frame)),
            f"rows={len(frame)}",
        )
    )
    duplicate_campaigns = int(frame["campaign_id"].duplicated().sum())
    rows.append(
        (
            "register.unique_campaign_id",
            duplicate_campaigns == 0,
            f"duplicate_rows={duplicate_campaigns}",
        )
    )

    invalid_roles = sorted(set(frame["role"].dropna()) - set(PROTOCOL_ROLES))
    invalid_status = sorted(set(frame["status"].dropna()) - set(CAMPAIGN_STATUSES))
    rows.append(
        (
            "register.valid_roles",
            not invalid_roles,
            "valid" if not invalid_roles else f"invalid={invalid_roles}",
        )
    )
    rows.append(
        (
            "register.valid_statuses",
            not invalid_status,
            "valid" if not invalid_status else f"invalid={invalid_status}",
        )
    )

    non_pilot = frame.loc[frame["role"].isin({"development", "gate_evidence", "final_test"})]
    roles_per_family = non_pilot.groupby("acquisition_family_id")["role"].nunique()
    role_leakage = int(roles_per_family.gt(1).sum())
    rows.append(
        (
            "register.family_role_separation",
            role_leakage == 0,
            f"families_crossing_roles={role_leakage}",
        )
    )

    fingerprint_columns = [
        "acquisition_pipeline_id",
        "site_id",
        "operator_id",
        "terminal_id",
        "route_id",
        "server_id",
    ]
    fingerprint_families = non_pilot.groupby(fingerprint_columns, dropna=False)[
        "acquisition_family_id"
    ].nunique()
    relabelled = int(fingerprint_families.gt(1).sum())
    rows.append(
        (
            "register.no_family_relabelling",
            relabelled == 0,
            f"shared_fingerprints_with_multiple_family_ids={relabelled}",
        )
    )

    starts = pd.to_datetime(frame["collection_start_utc"], utc=True, errors="coerce")
    ends = pd.to_datetime(frame["collection_end_utc"], utc=True, errors="coerce")
    date_required = frame["status"].isin({"collecting", "complete", "locked"})
    invalid_dates = date_required & (starts.isna() | ends.isna() | ends.lt(starts))
    rows.append(
        (
            "register.collection_interval",
            not bool(invalid_dates.any()),
            f"invalid_rows={int(invalid_dates.sum())}",
        )
    )

    clock_error = pd.to_numeric(frame["maximum_clock_error_ms"], errors="coerce")
    invalid_clock = date_required & (clock_error.isna() | clock_error.lt(0))
    rows.append(
        (
            "register.clock_audit",
            not bool(invalid_clock.any()),
            f"invalid_rows={int(invalid_clock.sum())}",
        )
    )

    locked_mask = frame["status"].eq("locked")
    bad_hash = locked_mask & (
        ~frame["model_sha256"].map(_is_sha256)
        | ~frame["config_sha256"].map(_is_sha256)
    )
    rows.append(
        (
            "register.locked_artifact_hashes",
            not bool(bad_hash.any()),
            f"invalid_locked_rows={int(bad_hash.sum())}",
        )
    )

    gate = frame.loc[
        frame["role"].eq("gate_evidence") & ~frame["status"].eq("excluded")
    ]
    gate_campaigns = int(gate["campaign_id"].nunique())
    gate_families = int(gate["acquisition_family_id"].nunique())
    rows.append(
        (
            "register.gate_eligibility_floor",
            gate_campaigns >= minimum_gate_campaigns
            and gate_families >= minimum_gate_families,
            (
                f"gate_campaigns={gate_campaigns}/{minimum_gate_campaigns}; "
                f"gate_families={gate_families}/{minimum_gate_families}; "
                "power/precision may require more"
            ),
        )
    )

    if require_locked:
        eligible = frame.loc[~frame["status"].eq("excluded")]
        unlocked = int((~eligible["status"].eq("locked")).sum())
        role_families = eligible.groupby("role")["acquisition_family_id"].nunique()
        required_roles = {"development", "gate_evidence", "final_test"}
        missing_roles = sorted(required_roles.difference(role_families.index))
        rows.append(
            (
                "register.all_included_campaigns_locked",
                unlocked == 0,
                f"unlocked_rows={unlocked}",
            )
        )
        rows.append(
            (
                "register.required_role_families",
                not missing_roles,
                "complete" if not missing_roles else f"missing={missing_roles}",
            )
        )

    return _report(rows)


def validate_trial_configuration(config: Mapping[str, Any]) -> ValidationReport:
    """Validate the frozen gate/canary/rollback contract."""

    rows: list[tuple[str, bool, str]] = []
    required_sections = {
        "schema_version",
        "protocol_status",
        "field_trial_completed",
        "primary_endpoint",
        "gate",
        "canary",
        "rollback",
        "instrumentation",
        "analysis",
    }
    missing = sorted(required_sections.difference(config))
    rows.append(
        (
            "config.required_sections",
            not missing,
            "complete" if not missing else f"missing={missing}",
        )
    )
    if missing:
        return _report(rows)

    status = str(config["protocol_status"])
    rows.append(
        (
            "config.status",
            status in {"DRAFT_NOT_REGISTERED", "FROZEN_REGISTERED"},
            status,
        )
    )
    rows.append(
        (
            "config.no_false_completion_claim",
            config["field_trial_completed"] is False,
            f"field_trial_completed={config['field_trial_completed']!r}",
        )
    )

    endpoint = config["primary_endpoint"]
    endpoint_ok = (
        endpoint.get("name") == "conservative_on_time_task_completion"
        and endpoint.get("denominator") == "all_scheduled_tasks_intention_to_treat"
        and endpoint.get("no_action_is_failure") is True
        and endpoint.get("missing_completion_evidence_is_failure") is True
    )
    rows.append(("config.primary_endpoint", endpoint_ok, str(endpoint)))

    gate = config["gate"]
    try:
        minimum_campaigns = int(gate["minimum_independent_gate_campaigns"])
        minimum_families = int(gate["minimum_independent_gate_families"])
        coverage = float(gate["minimum_feature_coverage"])
        ood = float(gate["maximum_ood_fraction"])
        noninferiority = float(gate["minimum_candidate_minus_baseline_effect_lower"])
        ci_width = float(gate["maximum_effect_ci_half_width"])
        gate_ok = (
            minimum_campaigns >= 2
            and minimum_families >= 2
            and 0 < coverage <= 1
            and 0 <= ood < 1
            and -1 < noninferiority <= 0
            and 0 < ci_width < 1
        )
    except (KeyError, TypeError, ValueError):
        gate_ok = False
    rows.append(("config.gate_thresholds", gate_ok, str(gate)))

    canary = config["canary"]
    try:
        allocation = float(canary["initial_candidate_allocation_fraction"])
        canary_ok = (
            0 < allocation <= 0.25
            and canary.get("candidate_requires_gate_authorization") is True
            and canary.get("candidate_starts_in_shadow_mode") is True
        )
    except (KeyError, TypeError, ValueError):
        canary_ok = False
    rows.append(("config.canary_guardrails", canary_ok, str(canary)))

    rollback = config["rollback"]
    required_triggers = {
        "emergency_stop",
        "install_not_acknowledged",
        "installed_path_mismatch",
        "first_packet_timeout",
        "control_loop_budget_exceeded",
        "active_shift_alarm",
        "endpoint_safety_boundary_crossed",
        "clock_or_logging_failure",
    }
    supplied_triggers = set(rollback.get("required_triggers", ()))
    rollback_ok = (
        required_triggers.issubset(supplied_triggers)
        and rollback.get("fallback_path") == "reactive_baseline"
    )
    rows.append(
        (
            "config.rollback_triggers",
            rollback_ok,
            f"missing={sorted(required_triggers - supplied_triggers)}",
        )
    )

    instrumentation = config["instrumentation"]
    required_instrumentation = {
        "proposed_path",
        "installed_path",
        "installation_acknowledgement",
        "first_post_installation_packet",
        "monotonic_stage_timestamps",
        "wall_clock_timestamps",
        "route_table_snapshot_before_and_after",
        "timestamped_loss_and_timeout_records",
    }
    enabled = {key for key, value in instrumentation.items() if value is True}
    rows.append(
        (
            "config.instrumentation_contract",
            required_instrumentation.issubset(enabled),
            f"missing={sorted(required_instrumentation - enabled)}",
        )
    )

    analysis = config["analysis"]
    analysis_ok = (
        analysis.get("independent_unit") == "acquisition_family_or_randomized_block"
        and analysis.get("primary_estimand")
        == "intention_to_treat_risk_difference_candidate_minus_baseline"
        and analysis.get("multiplicity_control") == "Holm_familywise"
    )
    rows.append(("config.analysis_contract", analysis_ok, str(analysis)))

    return _report(rows)


def load_ledger_schema(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if "fields" not in schema or "required_common" not in schema:
        raise ValueError("ledger schema is missing fields or required_common")
    return schema


def _type_matches(value: object, declared: str) -> bool:
    if declared == "string":
        return isinstance(value, str)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "string_array":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    raise ValueError(f"unsupported schema type: {declared}")


def validate_ledger_record(
    record: Mapping[str, Any], schema: Mapping[str, Any]
) -> ValidationReport:
    """Validate one action or outcome event before it enters the ledger."""

    rows: list[tuple[str, bool, str]] = []
    record_type = record.get("record_type")
    valid_type = record_type in RECORD_TYPES
    rows.append(("ledger.record_type", valid_type, str(record_type)))
    if not valid_type:
        return _report(rows)

    fields = schema["fields"]
    required = set(schema["required_common"]) | set(
        schema["required_by_record_type"][record_type]
    )
    missing = sorted(field for field in required if field not in record)
    null_required = sorted(
        field for field in required if field in record and record[field] is None
    )
    rows.append(
        (
            "ledger.required_fields",
            not missing and not null_required,
            f"missing={missing}; null={null_required}",
        )
    )

    unknown = sorted(set(record).difference(fields).difference({"previous_record_sha256", "record_sha256"}))
    rows.append(
        (
            "ledger.known_fields",
            not unknown,
            "valid" if not unknown else f"unknown={unknown}",
        )
    )

    type_errors: list[str] = []
    enum_errors: list[str] = []
    format_errors: list[str] = []
    for field, value in record.items():
        if field not in fields or value is None:
            continue
        definition = fields[field]
        if not _type_matches(value, definition["type"]):
            type_errors.append(field)
            continue
        if "enum" in definition and value not in definition["enum"]:
            enum_errors.append(field)
        if definition.get("format") == "sha256" and not _is_sha256(value):
            format_errors.append(field)
    rows.append(
        (
            "ledger.field_types",
            not type_errors,
            "valid" if not type_errors else f"invalid={type_errors}",
        )
    )
    rows.append(
        (
            "ledger.field_enums",
            not enum_errors,
            "valid" if not enum_errors else f"invalid={enum_errors}",
        )
    )
    rows.append(
        (
            "ledger.field_formats",
            not format_errors,
            "valid" if not format_errors else f"invalid={format_errors}",
        )
    )

    if record_type == "action":
        installed = bool(record.get("install_verified"))
        path_consistent = (
            not installed
            or record.get("installed_path_id") == record.get("intended_path_id")
        )
        packet_consistent = (
            not record.get("first_packet_observed")
            or record.get("first_packet_path_id") == record.get("intended_path_id")
        )
        rollback_consistent = (
            not record.get("rollback_triggered")
            or (
                record.get("rollback_verified") is True
                and record.get("rollback_acknowledged") is True
                and record.get("rollback_install_verified") is True
                and record.get("rollback_first_packet_observed") is True
                and record.get("rollback_first_packet_path_id")
                == record.get("baseline_path_id")
                and record.get("final_fib_path_id") == record.get("baseline_path_id")
            )
        )
        timing_fields = (
            "collect_ms",
            "feature_ms",
            "decision_ms",
            "disseminate_ms",
            "install_ms",
            "switch_ms",
            "total_loop_ms",
        )
        extra_timing_fields = (
            "rollback_install_ms",
            "rollback_switch_ms",
            "rollback_total_ms",
            "action_total_ms",
        )
        timing = [record.get(field) for field in timing_fields + extra_timing_fields]
        timing_ok = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in timing
        )
        if timing_ok:
            component_sum = sum(float(record[field]) for field in timing_fields[:-1])
            timing_ok = abs(component_sum - float(record["total_loop_ms"])) <= 1.0
            rollback_component_sum = float(record.get("rollback_install_ms", -1)) + float(
                record.get("rollback_switch_ms", -1)
            )
            timing_ok = timing_ok and abs(
                rollback_component_sum - float(record.get("rollback_total_ms", -1))
            ) <= 1.0
            timing_ok = timing_ok and abs(
                float(record["total_loop_ms"])
                + float(record.get("rollback_total_ms", -1))
                - float(record.get("action_total_ms", -1))
            ) <= 1.0
        rows.append(
            (
                "ledger.installed_path_consistency",
                path_consistent,
                "valid" if path_consistent else "verified path differs from intended path",
            )
        )
        rows.append(
            (
                "ledger.first_packet_path_consistency",
                packet_consistent,
                "valid" if packet_consistent else "first packet differs from intended path",
            )
        )
        rows.append(
            (
                "ledger.rollback_consistency",
                rollback_consistent,
                "valid" if rollback_consistent else "rollback did not verify the baseline path",
            )
        )
        rows.append(
            (
                "ledger.timing_decomposition",
                timing_ok,
                "valid" if timing_ok else "stage times are negative, nonfinite, or do not sum",
            )
        )

    if record_type == "outcome":
        required_bytes = record.get("required_bytes")
        delivered_bytes = record.get("delivered_bytes")
        deadline_wall_ns = record.get("deadline_wall_ns")
        completion_wall_ns = record.get("completion_wall_ns")
        derived_met_deadline = bool(
            record.get("observation_valid")
            and record.get("application_complete")
            and delivered_bytes is not None
            and required_bytes is not None
            and delivered_bytes >= required_bytes
            and isinstance(deadline_wall_ns, int)
            and not isinstance(deadline_wall_ns, bool)
            and isinstance(completion_wall_ns, int)
            and not isinstance(completion_wall_ns, bool)
            and completion_wall_ns <= deadline_wall_ns
        )
        deadline_consistent = record.get("met_deadline") is derived_met_deadline
        definition_consistent = bool(record.get("conservative_on_time_completion")) == bool(
            derived_met_deadline
            and not record.get("no_action")
            and not record.get("controller_failure")
            and not record.get("late_install")
            and delivered_bytes is not None
            and required_bytes is not None
            and delivered_bytes >= required_bytes
        )
        evidence_status = record.get("outcome_evidence_status")
        evidence_origin = record.get("outcome_evidence_origin")
        evidence_tier = record.get("evidence_tier")
        evidence_binding_consistent = bool(
            (
                (
                    evidence_tier == "controlled_fake_driver_test"
                    and evidence_status == "FAKE_DRIVER_TEST"
                    and evidence_origin == "fake_driver_test"
                    and record.get("application_acknowledgement_authenticated") is False
                    and record.get("application_acknowledgement_key_id") == ""
                )
                or (
                    evidence_tier != "controlled_fake_driver_test"
                    and evidence_status == "LIVE_APPLICATION_ACKNOWLEDGEMENT"
                    and evidence_origin == "executed_live"
                    and record.get("application_acknowledgement_authenticated") is True
                    and bool(
                        re.fullmatch(
                            r"[0-9a-f]{16}",
                            str(record.get("application_acknowledgement_key_id", "")),
                        )
                    )
                )
            )
            and bool(record.get("application_acknowledgement_id"))
            and bool(record.get("application_acknowledgement_source"))
        )
        acknowledgement_time_consistent = bool(
            isinstance(record.get("application_acknowledgement_wall_ns"), int)
            and not isinstance(record.get("application_acknowledgement_wall_ns"), bool)
            and isinstance(completion_wall_ns, int)
            and not isinstance(completion_wall_ns, bool)
            and record["application_acknowledgement_wall_ns"] >= completion_wall_ns
        )
        rows.append(
            (
                "ledger.deadline_label_derived",
                deadline_consistent,
                "valid" if deadline_consistent else "met_deadline contradicts timestamps, bytes, completion, or validity",
            )
        )
        rows.append(
            (
                "ledger.conservative_completion_definition",
                definition_consistent,
                "valid" if definition_consistent else "completion label contradicts component outcomes",
            )
        )
        rows.append(
            (
                "ledger.outcome_evidence_binding",
                evidence_binding_consistent,
                "valid" if evidence_binding_consistent else "acknowledgement status, origin, authentication, or evidence tier contradicts",
            )
        )
        rows.append(
            (
                "ledger.outcome_acknowledgement_time",
                acknowledgement_time_consistent,
                "valid" if acknowledgement_time_consistent else "acknowledgement precedes completion",
            )
        )

    return _report(rows)


def append_hash_chained_record(
    path: str | Path,
    record: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Append a validated, tamper-evident JSON Lines record.

    The append is flushed and fsynced before return.  The function refuses to
    extend an invalid existing chain.  It does not provide distributed consensus;
    one trial logger process must own a ledger file.
    """

    validation = validate_ledger_record(record, schema)
    validation.raise_for_errors()
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    previous = "0" * 64
    if ledger_path.exists() and ledger_path.stat().st_size:
        chain = audit_hash_chain(ledger_path, schema)
        chain.raise_for_errors()
        with ledger_path.open("rb") as handle:
            last = list(handle)[-1]
        previous = json.loads(last)["record_sha256"]

    payload = dict(record)
    payload["previous_record_sha256"] = previous
    payload["record_sha256"] = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    descriptor = os.open(ledger_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def audit_hash_chain(path: str | Path, schema: Mapping[str, Any]) -> ValidationReport:
    """Verify schema, ordering, linkage, and content hashes for a JSONL ledger."""

    ledger_path = Path(path)
    rows: list[tuple[str, bool, str]] = []
    if not ledger_path.exists():
        return _report([("ledger.file_exists", False, str(ledger_path))])
    previous = "0" * 64
    errors: list[str] = []
    count = 0
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            count += 1
            try:
                payload = json.loads(line)
                claimed = payload.pop("record_sha256")
                linked = payload.get("previous_record_sha256")
                expected = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
                core = {
                    key: value
                    for key, value in payload.items()
                    if key != "previous_record_sha256"
                }
                validation = validate_ledger_record(core, schema)
                if not validation.passed:
                    errors.append(f"line {line_number}: schema validation failed")
                if linked != previous:
                    errors.append(f"line {line_number}: previous hash mismatch")
                if claimed != expected:
                    errors.append(f"line {line_number}: record hash mismatch")
                previous = claimed
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"line {line_number}: {exc}")
    rows.append(("ledger.nonempty", count > 0, f"records={count}"))
    rows.append(
        (
            "ledger.hash_chain",
            not errors,
            "valid" if not errors else "; ".join(errors[:10]),
        )
    )
    return _report(rows)


def audit_decision_lifecycles(
    path: str | Path,
    schema: Mapping[str, Any],
    *,
    require_mature_outcomes: bool = False,
) -> ValidationReport:
    """Audit action/outcome joins and the conservative endpoint definition."""

    chain = audit_hash_chain(path, schema)
    rows: list[tuple[str, bool, str]] = [
        ("lifecycle.hash_chain", chain.passed, "valid" if chain.passed else "invalid")
    ]
    if not chain.passed:
        return _report(rows)

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            payload.pop("record_sha256", None)
            payload.pop("previous_record_sha256", None)
            records.append(payload)

    frame = pd.DataFrame(records)
    actions = frame.loc[frame["record_type"].eq("action")].copy()
    outcomes = frame.loc[frame["record_type"].eq("outcome")].copy()
    duplicate_actions = int(actions["decision_id"].duplicated(keep=False).sum())
    duplicate_outcomes = int(outcomes["decision_id"].duplicated(keep=False).sum())
    rows.append(
        (
            "lifecycle.unique_action",
            duplicate_actions == 0 and bool(len(actions)),
            f"actions={len(actions)}; duplicate_rows={duplicate_actions}",
        )
    )
    rows.append(
        (
            "lifecycle.at_most_one_outcome",
            duplicate_outcomes == 0,
            f"outcomes={len(outcomes)}; duplicate_rows={duplicate_outcomes}",
        )
    )

    action_ids = set(actions.get("decision_id", pd.Series(dtype=str)))
    outcome_ids = set(outcomes.get("decision_id", pd.Series(dtype=str)))
    orphan_outcomes = sorted(outcome_ids - action_ids)
    missing_outcomes = sorted(action_ids - outcome_ids)
    rows.append(
        (
            "lifecycle.no_orphan_outcome",
            not orphan_outcomes,
            "valid" if not orphan_outcomes else f"orphan={orphan_outcomes[:10]}",
        )
    )
    rows.append(
        (
            "lifecycle.mature_outcomes",
            not require_mature_outcomes or not missing_outcomes,
            (
                "complete"
                if not missing_outcomes
                else f"pending={len(missing_outcomes)}"
            ),
        )
    )

    linked = actions.merge(
        outcomes,
        on="decision_id",
        how="inner",
        suffixes=("_action", "_outcome"),
        validate="one_to_one" if duplicate_actions == 0 and duplicate_outcomes == 0 else "many_to_many",
    )
    identity_fields = (
        "study_id",
        "trial_id",
        "protocol_version",
        "campaign_id",
        "acquisition_family_id",
        "session_id",
        "task_id",
        "model_sha256",
        "config_sha256",
        "evidence_tier",
    )
    identity_mismatch = 0
    for field in identity_fields:
        if f"{field}_action" in linked and f"{field}_outcome" in linked:
            identity_mismatch += int(
                linked[f"{field}_action"].ne(linked[f"{field}_outcome"]).sum()
            )
    rows.append(
        (
            "lifecycle.identity_join",
            identity_mismatch == 0,
            f"mismatches={identity_mismatch}",
        )
    )

    if len(linked):
        outcome_after_action = linked["recorded_wall_ns_outcome"].ge(
            linked["recorded_wall_ns_action"]
        )
        bytes_complete = linked["delivered_bytes_outcome"].ge(
            linked["required_bytes_outcome"]
        )
        expected = (
            linked["install_verified_action"].astype(bool)
            & linked["first_packet_observed_action"].astype(bool)
            & ~linked["rollback_triggered_action"].astype(bool)
            & linked["observation_valid_outcome"].astype(bool)
            & linked["application_complete_outcome"].astype(bool)
            & linked["met_deadline_outcome"].astype(bool)
            & ~linked["no_action_outcome"].astype(bool)
            & ~linked["controller_failure_outcome"].astype(bool)
            & ~linked["late_install_outcome"].astype(bool)
            & bytes_complete
        )
        endpoint_mismatch = int(
            expected.ne(
                linked["conservative_on_time_completion_outcome"].astype(bool)
            ).sum()
        )
        unauthorized = (
            linked["installed_path_id_action"].eq(linked["proposed_path_id_action"])
            & linked["proposed_path_id_action"].ne(linked["baseline_path_id_action"])
            & (
                linked["gate_state_action"].ne("authorized")
                | linked["guard_state_action"].ne("pass")
                | linked["monitor_state_action"].ne("pass")
                | linked["canary_state_action"].ne("candidate")
            )
        )
        rollback_unverified = linked["rollback_triggered_action"].astype(bool) & (
            ~linked["rollback_verified_action"].astype(bool)
            | linked["final_fib_path_id_action"].ne(
                linked["baseline_path_id_action"]
            )
        )
        rows.append(
            (
                "lifecycle.outcome_after_action",
                bool(outcome_after_action.all()),
                f"invalid_rows={int((~outcome_after_action).sum())}",
            )
        )
        rows.append(
            (
                "lifecycle.primary_endpoint",
                endpoint_mismatch == 0,
                f"mismatches={endpoint_mismatch}",
            )
        )
        rows.append(
            (
                "lifecycle.no_unauthorized_candidate_installation",
                not bool(unauthorized.any()),
                f"unauthorized_rows={int(unauthorized.sum())}",
            )
        )
        rows.append(
            (
                "lifecycle.rollback_verified",
                not bool(rollback_unverified.any()),
                f"unverified_rows={int(rollback_unverified.sum())}",
            )
        )
    else:
        rows.extend(
            (
                ("lifecycle.outcome_after_action", not require_mature_outcomes, "no mature outcomes"),
                ("lifecycle.primary_endpoint", not require_mature_outcomes, "no mature outcomes"),
                ("lifecycle.no_unauthorized_candidate_installation", True, "no mature outcomes"),
                ("lifecycle.rollback_verified", True, "no mature outcomes"),
            )
        )
    return _report(rows)


@dataclass(frozen=True)
class ClockSample:
    wall_ns: int
    monotonic_ns: int


class Clock(Protocol):
    def sample(self) -> ClockSample: ...


class SystemClock:
    def sample(self) -> ClockSample:
        return ClockSample(time.time_ns(), time.monotonic_ns())


@dataclass(frozen=True)
class RouteSnapshot:
    path_id: str | None
    raw: str


@dataclass(frozen=True)
class InstallAcknowledgement:
    acknowledged: bool
    raw: str


@dataclass(frozen=True)
class FirstPacketObservation:
    observed: bool
    path_id: str | None
    raw: str


class RouteDriver(Protocol):
    def snapshot(self, destination: str) -> RouteSnapshot: ...

    def install(self, path_id: str, destination: str) -> InstallAcknowledgement: ...

    def observe_first_packet(
        self, path_id: str, destination: str, timeout_ms: float
    ) -> FirstPacketObservation: ...


class CommandRouteDriver:
    """Operator-configured no-shell route/FIB scaffold.

    Commands are argument arrays, never shell strings.  Mutation is disabled by
    default and must be explicitly enabled by the deployment operator.  The
    snapshot command must emit the resolved path identifier as its first line.
    The first-packet command must exit zero only after observing a packet on the
    requested path.  This adapter is intentionally generic because Linux policy
    routing, MP-QUIC, MPTCP, and anchored tunnels use different control planes.
    """

    def __init__(
        self,
        *,
        snapshot_command: Sequence[str],
        install_commands: Mapping[str, Sequence[str]],
        first_packet_commands: Mapping[str, Sequence[str]],
        allow_live_mutation: bool = False,
        command_timeout_s: float = 5.0,
    ) -> None:
        self.snapshot_command = tuple(snapshot_command)
        self.install_commands = {
            key: tuple(value) for key, value in install_commands.items()
        }
        self.first_packet_commands = {
            key: tuple(value) for key, value in first_packet_commands.items()
        }
        self.allow_live_mutation = bool(allow_live_mutation)
        self.command_timeout_s = float(command_timeout_s)

    @staticmethod
    def _expand(command: Sequence[str], *, destination: str, path_id: str) -> list[str]:
        return [
            token.replace("{destination}", destination).replace("{path_id}", path_id)
            for token in command
        ]

    def _run(self, command: Sequence[str], timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=self.command_timeout_s if timeout_s is None else timeout_s,
        )

    def snapshot(self, destination: str) -> RouteSnapshot:
        command = self._expand(
            self.snapshot_command, destination=destination, path_id=""
        )
        result = self._run(command)
        raw = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            return RouteSnapshot(None, raw)
        first_line = result.stdout.strip().splitlines()
        return RouteSnapshot(first_line[0].strip() if first_line else None, raw)

    def install(self, path_id: str, destination: str) -> InstallAcknowledgement:
        if not self.allow_live_mutation:
            return InstallAcknowledgement(False, "live route mutation disabled")
        if path_id not in self.install_commands:
            return InstallAcknowledgement(False, f"unknown path_id={path_id}")
        command = self._expand(
            self.install_commands[path_id], destination=destination, path_id=path_id
        )
        result = self._run(command)
        raw = (result.stdout + result.stderr).strip()
        return InstallAcknowledgement(result.returncode == 0, raw)

    def observe_first_packet(
        self, path_id: str, destination: str, timeout_ms: float
    ) -> FirstPacketObservation:
        if path_id not in self.first_packet_commands:
            return FirstPacketObservation(False, None, f"unknown path_id={path_id}")
        command = self._expand(
            self.first_packet_commands[path_id],
            destination=destination,
            path_id=path_id,
        )
        try:
            result = self._run(command, timeout_s=max(timeout_ms / 1000.0, 0.001))
        except subprocess.TimeoutExpired:
            return FirstPacketObservation(False, None, "first-packet timeout")
        raw = (result.stdout + result.stderr).strip()
        observed_lines = result.stdout.strip().splitlines()
        observed_path = observed_lines[0].strip() if observed_lines else None
        return FirstPacketObservation(
            result.returncode == 0 and observed_path == path_id,
            observed_path if result.returncode == 0 else None,
            raw,
        )


@dataclass(frozen=True)
class DecisionContext:
    destination: str
    baseline_path_id: str
    proposed_path_id: str
    gate_state: str
    gate_reason: str
    guard_state: str
    monitor_state: str
    canary_state: str
    collect_ms: float
    feature_ms: float
    decision_ms: float
    disseminate_ms: float


@dataclass(frozen=True)
class ClosedLoopResult:
    intended_path_id: str
    installed_path_id: str | None
    candidate_authorized: bool
    install_acknowledged: bool
    install_verified: bool
    first_packet_observed: bool
    first_packet_path_id: str | None
    install_ack_evidence_sha256: str
    first_packet_evidence_sha256: str
    fallback_triggered: bool
    fallback_reason: str
    rollback_triggered: bool
    rollback_reason: str
    rollback_acknowledged: bool
    rollback_install_verified: bool
    rollback_first_packet_observed: bool
    rollback_first_packet_path_id: str | None
    rollback_ack_evidence_sha256: str
    rollback_first_packet_evidence_sha256: str
    rollback_verified: bool
    pre_fib_path_id: str | None
    post_fib_path_id: str | None
    final_fib_path_id: str | None
    install_ms: float
    switch_ms: float
    total_loop_ms: float
    rollback_install_ms: float
    rollback_switch_ms: float
    rollback_total_ms: float
    action_total_ms: float
    action_start: ClockSample
    action_end: ClockSample


class ClosedLoopController:
    """Fail-closed action executor with FIB and first-packet verification."""

    def __init__(
        self,
        driver: RouteDriver,
        *,
        clock: Clock | None = None,
        maximum_total_loop_ms: float = 100.0,
        first_packet_timeout_ms: float = 500.0,
    ) -> None:
        if maximum_total_loop_ms <= 0 or first_packet_timeout_ms <= 0:
            raise ValueError("timing thresholds must be positive")
        self.driver = driver
        self.clock = SystemClock() if clock is None else clock
        self.maximum_total_loop_ms = float(maximum_total_loop_ms)
        self.first_packet_timeout_ms = float(first_packet_timeout_ms)

    def execute(self, context: DecisionContext) -> ClosedLoopResult:
        stage_values = (
            context.collect_ms,
            context.feature_ms,
            context.decision_ms,
            context.disseminate_ms,
        )
        if any(not math.isfinite(value) or value < 0 for value in stage_values):
            raise ValueError("pre-install stage durations must be finite and nonnegative")

        start = self.clock.sample()
        pre = self.driver.snapshot(context.destination)
        candidate_authorized = (
            context.gate_state == "authorized"
            and context.guard_state == "pass"
            and context.monitor_state == "pass"
            and context.canary_state == "candidate"
        )
        intended = (
            context.proposed_path_id
            if candidate_authorized
            else context.baseline_path_id
        )
        fallback = not candidate_authorized
        fallback_reason = "" if candidate_authorized else (
            "candidate_not_authorized:" + ",".join(
                state
                for state in (
                    context.gate_reason,
                    context.guard_state,
                    context.monitor_state,
                    context.canary_state,
                )
                if state
            )
        )

        # Installation time includes the pre-install route observation and the
        # independent post-install route verification.  This avoids hiding FIB
        # inspection overhead outside the declared loop decomposition.
        install_start = start
        if pre.path_id == intended:
            acknowledgement = InstallAcknowledgement(True, "path already installed")
        else:
            acknowledgement = self.driver.install(intended, context.destination)
        post = self.driver.snapshot(context.destination)
        install_end = self.clock.sample()
        install_ms = max(
            0.0, (install_end.monotonic_ns - install_start.monotonic_ns) / 1_000_000.0
        )
        verified = bool(
            acknowledgement.acknowledged and post.path_id == intended
        )

        packet_start = self.clock.sample()
        packet = (
            self.driver.observe_first_packet(
                intended, context.destination, self.first_packet_timeout_ms
            )
            if verified
            else FirstPacketObservation(False, None, "installation not verified")
        )
        packet_end = self.clock.sample()
        switch_ms = max(
            0.0, (packet_end.monotonic_ns - packet_start.monotonic_ns) / 1_000_000.0
        )
        total_loop_ms = sum(stage_values) + install_ms + switch_ms

        rollback_reasons: list[str] = []
        if not acknowledgement.acknowledged:
            rollback_reasons.append("install_not_acknowledged")
        if acknowledgement.acknowledged and not verified:
            rollback_reasons.append("installed_path_mismatch")
        if verified and not (packet.observed and packet.path_id == intended):
            rollback_reasons.append("first_packet_timeout")
        if total_loop_ms > self.maximum_total_loop_ms:
            rollback_reasons.append("control_loop_budget_exceeded")

        rollback = bool(rollback_reasons)
        rollback_acknowledged = False
        rollback_install_verified = False
        rollback_first_packet_observed = False
        rollback_first_packet_path_id: str | None = None
        rollback_ack_raw = ""
        rollback_packet_raw = ""
        rollback_install_ms = 0.0
        rollback_switch_ms = 0.0
        if rollback:
            fallback = True
            fallback_reason = fallback_reason or "runtime_safety_failure"
            rollback_install_start = self.clock.sample()
            current = self.driver.snapshot(context.destination)
            if current.path_id == context.baseline_path_id:
                rollback_ack = InstallAcknowledgement(
                    True, "baseline already installed at rollback"
                )
            else:
                rollback_ack = self.driver.install(
                    context.baseline_path_id, context.destination
                )
            final = self.driver.snapshot(context.destination)
            rollback_install_end = self.clock.sample()
            rollback_install_ms = max(
                0.0,
                (
                    rollback_install_end.monotonic_ns
                    - rollback_install_start.monotonic_ns
                )
                / 1_000_000.0,
            )
            rollback_acknowledged = rollback_ack.acknowledged
            rollback_ack_raw = rollback_ack.raw
            rollback_install_verified = bool(
                rollback_acknowledged
                and final.path_id == context.baseline_path_id
            )
            rollback_packet_start = self.clock.sample()
            rollback_packet = (
                self.driver.observe_first_packet(
                    context.baseline_path_id,
                    context.destination,
                    self.first_packet_timeout_ms,
                )
                if rollback_install_verified
                else FirstPacketObservation(
                    False, None, "rollback installation not verified"
                )
            )
            rollback_packet_end = self.clock.sample()
            rollback_switch_ms = max(
                0.0,
                (
                    rollback_packet_end.monotonic_ns
                    - rollback_packet_start.monotonic_ns
                )
                / 1_000_000.0,
            )
            rollback_first_packet_observed = bool(
                rollback_packet.observed
                and rollback_packet.path_id == context.baseline_path_id
            )
            rollback_first_packet_path_id = rollback_packet.path_id
            rollback_packet_raw = rollback_packet.raw
        else:
            final = post
        end = self.clock.sample()
        rollback_total_ms = rollback_install_ms + rollback_switch_ms
        action_total_ms = total_loop_ms + rollback_total_ms
        rollback_verified = bool(
            not rollback
            or (
                rollback_acknowledged
                and rollback_install_verified
                and rollback_first_packet_observed
                and final.path_id == context.baseline_path_id
            )
        )

        return ClosedLoopResult(
            intended_path_id=intended,
            installed_path_id=post.path_id if verified else None,
            candidate_authorized=candidate_authorized,
            install_acknowledged=acknowledgement.acknowledged,
            install_verified=verified,
            first_packet_observed=bool(
                packet.observed and packet.path_id == intended
            ),
            first_packet_path_id=packet.path_id,
            install_ack_evidence_sha256=sha256(
                acknowledgement.raw.encode("utf-8")
            ).hexdigest(),
            first_packet_evidence_sha256=sha256(
                packet.raw.encode("utf-8")
            ).hexdigest(),
            fallback_triggered=fallback,
            fallback_reason=fallback_reason,
            rollback_triggered=rollback,
            rollback_reason=",".join(rollback_reasons),
            rollback_acknowledged=rollback_acknowledged,
            rollback_install_verified=rollback_install_verified,
            rollback_first_packet_observed=rollback_first_packet_observed,
            rollback_first_packet_path_id=rollback_first_packet_path_id,
            rollback_ack_evidence_sha256=sha256(
                rollback_ack_raw.encode("utf-8")
            ).hexdigest(),
            rollback_first_packet_evidence_sha256=sha256(
                rollback_packet_raw.encode("utf-8")
            ).hexdigest(),
            rollback_verified=rollback_verified,
            pre_fib_path_id=pre.path_id,
            post_fib_path_id=post.path_id,
            final_fib_path_id=final.path_id,
            install_ms=install_ms,
            switch_ms=switch_ms,
            total_loop_ms=total_loop_ms,
            rollback_install_ms=rollback_install_ms,
            rollback_switch_ms=rollback_switch_ms,
            rollback_total_ms=rollback_total_ms,
            action_total_ms=action_total_ms,
            action_start=start,
            action_end=end,
        )


def build_reproducibility_manifest(
    root: str | Path,
    relative_paths: Sequence[str],
    *,
    protocol_status: str,
) -> dict[str, Any]:
    """Build a hash manifest without asserting that trial data exist."""

    base = Path(root).resolve()
    files: list[dict[str, Any]] = []
    for relative in sorted(set(relative_paths)):
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"manifest path escapes root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "manifest_version": "1.0.0",
        "protocol_status": protocol_status,
        "field_trial_completed": False,
        "generated_wall_ns": time.time_ns(),
        "hash_algorithm": "sha256",
        "files": files,
        "claim_boundary": (
            "This manifest verifies local trial-package artifacts only; it does not "
            "assert field acquisition, gate authorization, canary execution, or benefit."
        ),
    }


def verify_reproducibility_manifest(
    root: str | Path, manifest: Mapping[str, Any]
) -> ValidationReport:
    base = Path(root).resolve()
    rows: list[tuple[str, bool, str]] = []
    rows.append(
        (
            "manifest.no_false_completion_claim",
            manifest.get("field_trial_completed") is False,
            str(manifest.get("field_trial_completed")),
        )
    )
    mismatches: list[str] = []
    for item in manifest.get("files", []):
        path = (base / item["path"]).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            mismatches.append(f"{item['path']}: outside root")
            continue
        if not path.is_file():
            mismatches.append(f"{item['path']}: missing")
        elif sha256_file(path) != item.get("sha256"):
            mismatches.append(f"{item['path']}: hash mismatch")
    rows.append(
        (
            "manifest.file_hashes",
            not mismatches and bool(manifest.get("files")),
            "valid" if not mismatches else "; ".join(mismatches),
        )
    )
    return _report(rows)
