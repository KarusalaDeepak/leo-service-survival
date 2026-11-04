"""Fail-closed preparation controls for the prospective field extension to Objective 4.

The functions in this module make the first three external-execution steps
testable without pretending that hardware or field observations exist.  A
structurally valid template remains ``LIVE_REQUIRED``/``NOT_RUN``.  A
deterministic fake-driver run is labelled ``FAKE_DRIVER_TEST`` and is never
accepted as live evidence.

The module covers three boundaries:

* testbed and route-driver qualification;
* immutable, role-separated campaign-register locking; and
* synchronized, loss-preserving telemetry ingestion.

No function turns a template or fake record into a live result.  Live
qualification requires explicit ``live_hardware`` provenance, non-placeholder
identifiers, immutable evidence hashes, and passing read-only qualification
probes collected on the named deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import pandas as pd


NOT_RUN = "NOT_RUN"
LIVE_REQUIRED = "LIVE_REQUIRED"
FAKE_DRIVER_TEST = "FAKE_DRIVER_TEST"
LIVE_QUALIFIED = "LIVE_QUALIFIED"
ALLOWED_EVIDENCE_SOURCES = {LIVE_REQUIRED, FAKE_DRIVER_TEST, "live_hardware"}
ROLE_SET = {"pilot", "development", "gate_evidence", "final_test"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_RE = re.compile(
    r"(?:^|[-_:/])(TBD|TODO|REPLACE|PLACEHOLDER|EXAMPLE)(?:$|[-_:/])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PreparationReport:
    """A validation report with an explicit evidence boundary."""

    checks: pd.DataFrame
    evidence_status: str
    live_eligible: bool = False

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
        raise ValueError(f"field-preparation validation failed: {details}")


@dataclass(frozen=True)
class ProbeResult:
    """One non-mutating testbed qualification probe."""

    probe_id: str
    passed: bool
    observed: str
    evidence_sha256: str
    command_sha256: str = ""
    evidence_origin: str = "imported_evidence"


class ProbeRunner(Protocol):
    """Interface used for live read-only probes and deterministic fake tests."""

    def run(self, probe_id: str, command: Sequence[str]) -> ProbeResult: ...


def _report(
    rows: list[tuple[str, bool, str]],
    *,
    evidence_status: str,
    live_eligible: bool = False,
) -> PreparationReport:
    return PreparationReport(
        pd.DataFrame(rows, columns=("check", "passed", "details")),
        evidence_status=evidence_status,
        live_eligible=live_eligible,
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def probe_observation_sha256(observed: str) -> str:
    """Return the reproducible content hash used by qualification probes."""

    return _sha256_bytes(str(observed).encode("utf-8"))


def probe_command_sha256(command: Sequence[str]) -> str:
    """Bind a probe result to the exact configured no-shell argv vector."""

    return _sha256_bytes(
        json.dumps(list(command), sort_keys=False, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value)))


def _is_placeholder(value: object) -> bool:
    text = str(value).strip()
    return not text or bool(_PLACEHOLDER_RE.search(text))


def _valid_command(command: object, *, allow_fake: bool) -> bool:
    if not isinstance(command, list) or not command:
        return False
    if not all(isinstance(part, str) and part for part in command):
        return False
    executable = Path(command[0])
    if not executable.is_absolute() or _is_placeholder(str(executable)):
        return False
    if not allow_fake and executable.name in {"true", "false", "printf", "echo"}:
        return False
    forbidden = {"sh", "bash", "zsh", "fish", "osascript"}
    return executable.name not in forbidden


def _valid_destination(value: object) -> bool:
    text = str(value).strip()
    if _is_placeholder(text):
        return False
    try:
        ipaddress.ip_network(text, strict=False)
        return True
    except ValueError:
        labels = text.rstrip(".").split(".")
        return (
            len(labels) >= 2
            and all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) for label in labels)
        )


def validate_route_driver_configuration(
    config: Mapping[str, Any],
    *,
    allow_fake_driver: bool = False,
) -> PreparationReport:
    """Validate a real route driver without executing a route mutation."""

    rows: list[tuple[str, bool, str]] = []
    evidence_source = str(config.get("evidence_source", LIVE_REQUIRED))
    status = str(config.get("status", NOT_RUN))
    source_ok = evidence_source in ALLOWED_EVIDENCE_SOURCES
    rows.append(("route.evidence_source", source_ok, evidence_source))

    fake = evidence_source == FAKE_DRIVER_TEST
    rows.append(
        (
            "route.fake_driver_explicit_opt_in",
            not fake or allow_fake_driver,
            f"fake={fake}; allow_fake_driver={allow_fake_driver}",
        )
    )
    allowed_status = {"FROZEN_DEPLOYMENT_CONFIG"}
    if allow_fake_driver:
        allowed_status.add(FAKE_DRIVER_TEST)
    rows.append(("route.status", status in allowed_status, status))
    rows.append(
        (
            "route.destination",
            _valid_destination(config.get("destination")),
            str(config.get("destination")),
        )
    )

    baseline = str(config.get("baseline_path_id", ""))
    candidates = config.get("candidate_path_ids", [])
    path_ids_ok = (
        bool(baseline)
        and not _is_placeholder(baseline)
        and isinstance(candidates, list)
        and len(candidates) >= 2
        and all(isinstance(item, str) and not _is_placeholder(item) for item in candidates)
        and baseline not in candidates
        and len(set(candidates)) == len(candidates)
    )
    rows.append(("route.path_ids", path_ids_ok, f"baseline={baseline}; candidates={candidates}"))

    expected_paths = {baseline, *candidates} if path_ids_ok else set()
    snapshot = config.get("snapshot_command")
    installs = config.get("install_commands", {})
    packets = config.get("first_packet_commands", {})
    commands: list[object] = [snapshot]
    if isinstance(installs, Mapping):
        commands.extend(installs.values())
    if isinstance(packets, Mapping):
        commands.extend(packets.values())
    command_ok = bool(commands) and all(
        _valid_command(item, allow_fake=fake and allow_fake_driver) for item in commands
    )
    rows.append(("route.command_safety", command_ok, f"commands={len(commands)}"))
    rows.append(
        (
            "route.complete_path_commands",
            isinstance(installs, Mapping)
            and isinstance(packets, Mapping)
            and set(installs) == expected_paths
            and set(packets) == expected_paths,
            f"install={sorted(installs) if isinstance(installs, Mapping) else 'invalid'}; "
            f"packet={sorted(packets) if isinstance(packets, Mapping) else 'invalid'}",
        )
    )
    contract = config.get("contract", {})
    contract_ok = (
        isinstance(contract, Mapping)
        and contract.get("commands_use_shell") is False
        and contract.get("live_mutation_default") is False
        and contract.get("snapshot_stdout_first_line")
        and contract.get("install_exit_zero")
        and contract.get("first_packet_exit_zero")
        and contract.get("first_packet_stdout_first_line")
    )
    rows.append(("route.fail_closed_contract", bool(contract_ok), str(contract)))

    try:
        maximum_total_loop_ms = float(config.get("maximum_total_loop_ms"))
        first_packet_timeout_ms = float(config.get("first_packet_timeout_ms"))
        timing_ok = (
            math.isfinite(maximum_total_loop_ms)
            and maximum_total_loop_ms > 0
            and math.isfinite(first_packet_timeout_ms)
            and first_packet_timeout_ms > 0
        )
    except (TypeError, ValueError):
        timing_ok = False
    rows.append(
        (
            "route.timing_budgets",
            timing_ok,
            (
                f"maximum_total_loop_ms={config.get('maximum_total_loop_ms')}; "
                f"first_packet_timeout_ms={config.get('first_packet_timeout_ms')}"
            ),
        )
    )

    drill = config.get("isolated_rollback_drill", {})
    required_drill_commands = {
        "inject_fault_command",
        "restore_baseline_command",
        "verify_baseline_command",
        "observe_first_packet_command",
    }
    drill_commands_ok = isinstance(drill, Mapping) and all(
        _valid_command(
            drill.get(name),
            allow_fake=fake and allow_fake_driver,
        )
        for name in required_drill_commands
    )
    drill_contract_ok = bool(
        isinstance(drill, Mapping)
        and drill.get("isolation_attested") is True
        and not _is_placeholder(drill.get("namespace_id"))
        and drill_commands_ok
    )
    rows.append(
        (
            "route.isolated_rollback_drill",
            drill_contract_ok,
            (
                f"namespace_id={drill.get('namespace_id') if isinstance(drill, Mapping) else None}; "
                f"commands={sorted(required_drill_commands) if drill_commands_ok else 'invalid'}"
            ),
        )
    )

    live_eligible = (
        evidence_source == "live_hardware"
        and status == "FROZEN_DEPLOYMENT_CONFIG"
        and all(item[1] for item in rows)
    )
    evidence_status = LIVE_QUALIFIED if live_eligible else (FAKE_DRIVER_TEST if fake else LIVE_REQUIRED)
    return _report(rows, evidence_status=evidence_status, live_eligible=live_eligible)


def validate_testbed_qualification(
    config: Mapping[str, Any],
    *,
    probe_results: Sequence[ProbeResult] = (),
    allow_fake_driver: bool = False,
) -> PreparationReport:
    """Audit the hardware, permissions, clock, and continuity evidence contract."""

    rows: list[tuple[str, bool, str]] = []
    source = str(config.get("evidence_source", LIVE_REQUIRED))
    fake = source == FAKE_DRIVER_TEST
    rows.append(("testbed.evidence_source", source in ALLOWED_EVIDENCE_SOURCES, source))
    rows.append(
        (
            "testbed.fake_driver_explicit_opt_in",
            not fake or allow_fake_driver,
            f"fake={fake}; allow_fake_driver={allow_fake_driver}",
        )
    )
    expected_status = config.get("qualification_status")
    status_ok = expected_status in {NOT_RUN, LIVE_REQUIRED}
    if fake and allow_fake_driver:
        status_ok = expected_status == FAKE_DRIVER_TEST
    if source == "live_hardware":
        status_ok = expected_status in {LIVE_REQUIRED, LIVE_QUALIFIED}
    rows.append(("testbed.qualification_status", status_ok, str(expected_status)))
    rows.append(
        (
            "testbed.no_false_trial_completion",
            config.get("field_trial_completed") is False,
            f"field_trial_completed={config.get('field_trial_completed')!r}",
        )
    )

    required_components = {
        "cellular_5g_terminal",
        "leo_terminal",
        "dual_access_gateway",
        "remote_anchor",
        "clock_sync",
        "session_preserving_path",
    }
    components = config.get("components", {})
    missing_components = sorted(required_components.difference(components)) if isinstance(components, Mapping) else sorted(required_components)
    rows.append(
        (
            "testbed.required_components",
            not missing_components,
            "complete" if not missing_components else f"missing={missing_components}",
        )
    )

    identifiers: list[str] = []
    evidence_hashes: list[str] = []
    component_states: list[str] = []
    if isinstance(components, Mapping):
        for name in sorted(required_components.intersection(components)):
            item = components[name]
            if not isinstance(item, Mapping):
                identifiers.append("")
                evidence_hashes.append("")
                component_states.append("INVALID")
                continue
            identifiers.append(str(item.get("asset_or_service_id", "")))
            evidence_hashes.append(str(item.get("evidence_sha256", "")))
            component_states.append(str(item.get("status", "")))

    if source == "live_hardware":
        identifier_ok = bool(identifiers) and all(not _is_placeholder(item) for item in identifiers)
        hashes_ok = bool(evidence_hashes) and all(_is_sha256(item) for item in evidence_hashes)
        states_ok = bool(component_states) and all(item == "QUALIFIED" for item in component_states)
    elif fake:
        identifier_ok = bool(identifiers) and all(item.startswith("FAKE-") for item in identifiers)
        hashes_ok = bool(evidence_hashes) and all(_is_sha256(item) for item in evidence_hashes)
        states_ok = bool(component_states) and all(item == FAKE_DRIVER_TEST for item in component_states)
    else:
        identifier_ok = True
        hashes_ok = True
        states_ok = all(item in {NOT_RUN, LIVE_REQUIRED, ""} for item in component_states)
    rows.append(("testbed.component_identifiers", identifier_ok, str(identifiers)))
    rows.append(("testbed.component_evidence_hashes", hashes_ok, f"count={len(evidence_hashes)}"))
    rows.append(("testbed.component_states", states_ok, str(component_states)))

    cellular = components.get("cellular_5g_terminal", {}) if isinstance(components, Mapping) else {}
    leo = components.get("leo_terminal", {}) if isinstance(components, Mapping) else {}
    gateway = components.get("dual_access_gateway", {}) if isinstance(components, Mapping) else {}
    anchor = components.get("remote_anchor", {}) if isinstance(components, Mapping) else {}
    component_capabilities_ok = all(
        (
            isinstance(cellular, Mapping),
            not _is_placeholder(cellular.get("interface")),
            cellular.get("service_active") is True,
            cellular.get("telemetry_export_available") is True,
            isinstance(leo, Mapping),
            not _is_placeholder(leo.get("interface")),
            leo.get("service_active") is True,
            leo.get("telemetry_export_available") is True,
            isinstance(gateway, Mapping),
            not _is_placeholder(gateway.get("cellular_interface")),
            not _is_placeholder(gateway.get("leo_interface")),
            gateway.get("cellular_interface") != gateway.get("leo_interface"),
            gateway.get("independent_fib_observer") is True,
            gateway.get("emergency_baseline_restore") is True,
            isinstance(anchor, Mapping),
            anchor.get("controlled_endpoint") is True,
            anchor.get("application_acknowledgement") is True,
        )
    )
    if source == LIVE_REQUIRED:
        component_capabilities_ok = True
    rows.append(
        (
            "testbed.component_capabilities",
            component_capabilities_ok,
            "qualified" if component_capabilities_ok else "missing active service, telemetry, dual-interface safety, or anchor acknowledgement",
        )
    )

    permissions = config.get("permissions", {})
    required_permissions = {"institutional", "operator", "network_change", "ethics_or_exemption"}
    missing_permissions = sorted(required_permissions.difference(permissions)) if isinstance(permissions, Mapping) else sorted(required_permissions)
    permission_ok = not missing_permissions
    if permission_ok and source == "live_hardware":
        permission_ok = all(
            isinstance(permissions[key], Mapping)
            and permissions[key].get("status") == "APPROVED"
            and not _is_placeholder(permissions[key].get("approval_id"))
            and _is_sha256(permissions[key].get("evidence_sha256"))
            for key in required_permissions
        )
    elif permission_ok and fake:
        permission_ok = all(
            isinstance(permissions[key], Mapping)
            and permissions[key].get("status") == FAKE_DRIVER_TEST
            for key in required_permissions
        )
    rows.append(
        (
            "testbed.permissions",
            permission_ok,
            "complete" if permission_ok else f"missing_or_unapproved={missing_permissions}",
        )
    )

    clock = components.get("clock_sync", {}) if isinstance(components, Mapping) else {}
    try:
        clock_ok = (
            float(clock.get("maximum_error_ms")) >= 0
            and float(clock.get("maximum_error_ms")) <= float(config.get("maximum_clock_error_ms"))
            and str(clock.get("method")) in {"PTP", "GNSS_PPS", "NTP_AUDITED", "FAKE_DETERMINISTIC"}
        )
    except (TypeError, ValueError):
        clock_ok = False
    rows.append(("testbed.clock_budget", clock_ok, str(clock)))

    continuity = components.get("session_preserving_path", {}) if isinstance(components, Mapping) else {}
    continuity_ok = (
        isinstance(continuity, Mapping)
        and continuity.get("mechanism") in {"anchored_tunnel", "MP-QUIC", "MPTCP", "FAKE_DETERMINISTIC"}
        and continuity.get("preserves_session_identity") is True
    )
    rows.append(("testbed.session_continuity", continuity_ok, str(continuity)))

    probes = config.get("read_only_probes", {})
    configured_probe_ids = set(probes) if isinstance(probes, Mapping) else set()
    required_probes = {
        "cellular_link",
        "leo_link",
        "dual_interface",
        "remote_anchor",
        "clock_error",
        "session_continuity",
        "permission_bundle",
    }
    probe_config_ok = required_probes.issubset(configured_probe_ids) and all(
        _valid_command(probes[key], allow_fake=fake and allow_fake_driver)
        for key in required_probes.intersection(configured_probe_ids)
    )
    rows.append(
        (
            "testbed.read_only_probe_contract",
            probe_config_ok,
            f"missing={sorted(required_probes - configured_probe_ids)}",
        )
    )

    supplied = {result.probe_id: result for result in probe_results}
    duplicate_results = len(supplied) != len(probe_results)
    evidence_hashes_valid = all(
        result.evidence_sha256 == probe_observation_sha256(result.observed)
        for result in supplied.values()
    )
    command_hashes_valid = all(
        result.probe_id in probes
        and result.command_sha256 == probe_command_sha256(probes[result.probe_id])
        for result in supplied.values()
    )
    live_origin_valid = all(
        result.evidence_origin == "executed_live" for result in supplied.values()
    )
    if source in {"live_hardware", FAKE_DRIVER_TEST}:
        probes_ok = (
            not duplicate_results
            and required_probes == set(supplied)
            and all(result.passed and _is_sha256(result.evidence_sha256) for result in supplied.values())
            and evidence_hashes_valid
            and command_hashes_valid
            and (source != "live_hardware" or live_origin_valid)
        )
    else:
        probes_ok = not probe_results
    rows.append(
        (
            "testbed.probe_evidence",
            probes_ok,
            (
                f"supplied={sorted(supplied)}; duplicates={duplicate_results}; "
                f"observation_hashes={evidence_hashes_valid}; "
                f"command_hashes={command_hashes_valid}; "
                f"live_execution_origin={live_origin_valid}"
            ),
        )
    )

    all_checks = all(row[1] for row in rows)
    live_eligible = source == "live_hardware" and all_checks
    evidence_status = LIVE_QUALIFIED if live_eligible else (FAKE_DRIVER_TEST if fake else LIVE_REQUIRED)
    return _report(rows, evidence_status=evidence_status, live_eligible=live_eligible)


def run_read_only_qualification_probes(
    config: Mapping[str, Any],
    runner: ProbeRunner,
) -> tuple[ProbeResult, ...]:
    """Execute only explicitly declared read-only qualification probes."""

    probes = config.get("read_only_probes", {})
    if not isinstance(probes, Mapping):
        raise ValueError("read_only_probes must be a mapping")
    results: list[ProbeResult] = []
    for probe_id in sorted(probes):
        command = probes[probe_id]
        if not _valid_command(
            command,
            allow_fake=config.get("evidence_source") == FAKE_DRIVER_TEST,
        ):
            raise ValueError(f"unsafe or invalid read-only command for {probe_id}")
        results.append(runner.run(probe_id, tuple(command)))
    return tuple(results)


class SubprocessReadOnlyProbeRunner:
    """Explicit live probe runner; commands are argv arrays and never use a shell."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, probe_id: str, command: Sequence[str]) -> ProbeResult:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            shell=False,
        )
        observed = json.dumps(
            {
                "probe_id": probe_id,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            sort_keys=True,
        )
        return ProbeResult(
            probe_id=probe_id,
            passed=completed.returncode == 0,
            observed=observed,
            evidence_sha256=probe_observation_sha256(observed),
            command_sha256=probe_command_sha256(command),
            evidence_origin="executed_live",
        )


LOCK_REQUIRED_COLUMNS = {
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
    "testbed_qualification_sha256",
    "raw_capture_sha256",
    "gate_opportunity_sha256",
    "telemetry_manifest_sha256",
    "outcome_manifest_sha256",
    "clock_sync_method",
    "maximum_clock_error_ms",
    "status",
    "evidence_tier",
    "role_access_group",
}


def validate_lockable_campaign_registry(
    frame: pd.DataFrame,
    *,
    allow_fake: bool = False,
    minimum_gate_campaigns: int = 5,
    minimum_gate_families: int = 3,
) -> PreparationReport:
    """Validate a register immediately before creating an immutable lock."""

    rows: list[tuple[str, bool, str]] = []
    missing = sorted(LOCK_REQUIRED_COLUMNS.difference(frame.columns))
    rows.append(("registry.required_columns", not missing, "complete" if not missing else f"missing={missing}"))
    if missing:
        return _report(rows, evidence_status=LIVE_REQUIRED)

    rows.append(("registry.nonempty", bool(len(frame)), f"rows={len(frame)}"))
    rows.append(
        (
            "registry.unique_campaign_id",
            not frame["campaign_id"].duplicated().any(),
            f"duplicates={int(frame['campaign_id'].duplicated().sum())}",
        )
    )
    roles = set(frame["role"])
    rows.append(("registry.valid_roles", roles.issubset(ROLE_SET), str(sorted(roles))))
    required_roles = {"development", "gate_evidence", "final_test"}
    rows.append(("registry.required_roles", required_roles.issubset(roles), f"missing={sorted(required_roles - roles)}"))

    gate = frame.loc[frame["role"].eq("gate_evidence")]
    gate_campaigns = int(gate["campaign_id"].nunique())
    gate_families = int(gate["acquisition_family_id"].nunique())
    rows.append(
        (
            "registry.frozen_gate_eligibility_floor",
            gate_campaigns >= minimum_gate_campaigns
            and gate_families >= minimum_gate_families,
            f"campaigns={gate_campaigns}/{minimum_gate_campaigns}; families={gate_families}/{minimum_gate_families}",
        )
    )

    role_counts = frame.groupby("acquisition_family_id")["role"].nunique()
    rows.append(
        (
            "registry.family_role_separation",
            not role_counts.gt(1).any(),
            f"cross_role_families={int(role_counts.gt(1).sum())}",
        )
    )
    fingerprint = [
        "acquisition_pipeline_id",
        "site_id",
        "operator_id",
        "terminal_id",
        "route_id",
        "server_id",
    ]
    fingerprint_roles = frame.groupby(fingerprint, dropna=False)["role"].nunique()
    rows.append(
        (
            "registry.fingerprint_role_separation",
            not fingerprint_roles.gt(1).any(),
            f"cross_role_fingerprints={int(fingerprint_roles.gt(1).sum())}",
        )
    )

    hash_columns = [
        "model_sha256",
        "config_sha256",
        "testbed_qualification_sha256",
        "raw_capture_sha256",
        "telemetry_manifest_sha256",
        "outcome_manifest_sha256",
    ]
    invalid_hashes = sum(int(not _is_sha256(value)) for column in hash_columns for value in frame[column])
    rows.append(("registry.immutable_hashes", invalid_hashes == 0, f"invalid={invalid_hashes}"))
    gate_rows = frame["role"].eq("gate_evidence")
    invalid_gate_opportunity_hashes = int(
        sum(not _is_sha256(value) for value in frame.loc[gate_rows, "gate_opportunity_sha256"])
    )
    non_gate_values = frame.loc[~gate_rows, "gate_opportunity_sha256"].astype(str)
    invalid_non_gate_opportunity_hashes = int(
        sum(bool(value.strip()) and not _is_sha256(value) for value in non_gate_values)
    )
    rows.append(
        (
            "registry.gate_opportunity_hashes",
            invalid_gate_opportunity_hashes == 0
            and invalid_non_gate_opportunity_hashes == 0,
            (
                f"invalid_gate={invalid_gate_opportunity_hashes}; "
                f"invalid_non_gate={invalid_non_gate_opportunity_hashes}"
            ),
        )
    )
    duplicated_raw = frame["raw_capture_sha256"].duplicated(keep=False)
    rows.append(
        (
            "registry.no_raw_capture_reuse",
            not duplicated_raw.any(),
            f"duplicate_rows={int(duplicated_raw.sum())}",
        )
    )
    gate_opportunity_hashes = frame.loc[gate_rows, "gate_opportunity_sha256"]
    duplicated_gate_opportunities = gate_opportunity_hashes.duplicated(keep=False)
    rows.append(
        (
            "registry.no_gate_opportunity_reuse",
            not duplicated_gate_opportunities.any(),
            f"duplicate_rows={int(duplicated_gate_opportunities.sum())}",
        )
    )

    all_text = frame.astype(str)
    placeholders = sum(
        int(_is_placeholder(value))
        for column in LOCK_REQUIRED_COLUMNS
        - {"maximum_clock_error_ms", "gate_opportunity_sha256"}
        for value in all_text[column]
    )
    if allow_fake:
        placeholders = 0
    rows.append(("registry.no_placeholders", placeholders == 0, f"placeholder_values={placeholders}"))

    starts = pd.to_datetime(frame["collection_start_utc"], utc=True, errors="coerce")
    ends = pd.to_datetime(frame["collection_end_utc"], utc=True, errors="coerce")
    valid_intervals = starts.notna() & ends.notna() & ends.ge(starts)
    rows.append(("registry.valid_intervals", bool(valid_intervals.all()), f"invalid={int((~valid_intervals).sum())}"))
    clock = pd.to_numeric(frame["maximum_clock_error_ms"], errors="coerce")
    rows.append(("registry.clock_error_recorded", bool(clock.notna().all() and clock.ge(0).all()), f"invalid={int((clock.isna() | clock.lt(0)).sum())}"))

    states = set(frame["status"])
    rows.append(("registry.lockable_status", states.issubset({"complete", "locked"}), str(sorted(states))))
    access_group_roles = frame.groupby("role_access_group")["role"].nunique()
    rows.append(
        (
            "registry.role_access_separation",
            not access_group_roles.gt(1).any(),
            f"shared_access_groups={int(access_group_roles.gt(1).sum())}",
        )
    )

    overlap_count = 0
    indexed = frame.assign(lock_start=starts, lock_end=ends).sort_values("lock_start")
    for _, group in indexed.groupby(["terminal_id", "server_id"], dropna=False):
        records = list(group.to_dict(orient="records"))
        for position, left in enumerate(records):
            for right in records[position + 1 :]:
                if (
                    left["role"] != right["role"]
                    and left["lock_end"] >= right["lock_start"]
                    and right["lock_end"] >= left["lock_start"]
                ):
                    overlap_count += 1
    rows.append(("registry.no_cross_role_time_overlap", overlap_count == 0, f"overlaps={overlap_count}"))

    evidence_status = FAKE_DRIVER_TEST if allow_fake else LIVE_REQUIRED
    live_eligible = not allow_fake and all(row[1] for row in rows)
    if live_eligible:
        evidence_status = LIVE_QUALIFIED
    return _report(rows, evidence_status=evidence_status, live_eligible=live_eligible)


def build_campaign_registry_lock(
    frame: pd.DataFrame,
    *,
    protocol_sha256: str,
    qualification_sha256: str,
    lock_timestamp_utc: str,
    allow_fake: bool = False,
) -> dict[str, Any]:
    """Build a deterministic immutable register snapshot after leakage checks."""

    report = validate_lockable_campaign_registry(frame, allow_fake=allow_fake)
    report.raise_for_errors()
    if not _is_sha256(protocol_sha256) or not _is_sha256(qualification_sha256):
        raise ValueError("protocol and qualification hashes must be SHA-256 values")
    parsed_lock = pd.to_datetime(lock_timestamp_utc, utc=True, errors="coerce")
    if pd.isna(parsed_lock):
        raise ValueError("lock_timestamp_utc must be an ISO-8601 timestamp")

    columns = sorted(LOCK_REQUIRED_COLUMNS)
    rows: list[dict[str, Any]] = []
    for record in frame.sort_values("campaign_id")[columns].to_dict(orient="records"):
        normalized = {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in record.items()
        }
        normalized["status"] = "locked"
        normalized["row_sha256"] = _sha256_bytes(_canonical_json(normalized).encode("utf-8"))
        rows.append(normalized)
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "evidence_status": FAKE_DRIVER_TEST if allow_fake else LIVE_QUALIFIED,
        "live_field_evidence": not allow_fake,
        "lock_timestamp_utc": pd.Timestamp(parsed_lock).isoformat(),
        "protocol_sha256": protocol_sha256,
        "qualification_sha256": qualification_sha256,
        "campaigns": rows,
    }
    payload["registry_sha256"] = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    return payload


def verify_campaign_registry_lock(lock: Mapping[str, Any]) -> PreparationReport:
    """Verify every row hash, the whole-register hash, and role separation."""

    rows: list[tuple[str, bool, str]] = []
    evidence_status = str(lock.get("evidence_status", LIVE_REQUIRED))
    campaigns = lock.get("campaigns", [])
    rows.append(("lock.campaigns", isinstance(campaigns, list) and bool(campaigns), f"rows={len(campaigns) if isinstance(campaigns, list) else 0}"))
    row_errors = 0
    if isinstance(campaigns, list):
        for campaign in campaigns:
            if not isinstance(campaign, Mapping):
                row_errors += 1
                continue
            core = dict(campaign)
            claimed = core.pop("row_sha256", None)
            expected = _sha256_bytes(_canonical_json(core).encode("utf-8"))
            if claimed != expected:
                row_errors += 1
    rows.append(("lock.row_hashes", row_errors == 0, f"errors={row_errors}"))

    payload = dict(lock)
    claimed_registry = payload.pop("registry_sha256", None)
    expected_registry = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    rows.append(("lock.registry_hash", claimed_registry == expected_registry, f"claimed={claimed_registry}"))
    rows.append(("lock.protocol_hash", _is_sha256(lock.get("protocol_sha256")), str(lock.get("protocol_sha256"))))
    rows.append(("lock.qualification_hash", _is_sha256(lock.get("qualification_sha256")), str(lock.get("qualification_sha256"))))

    if isinstance(campaigns, list) and campaigns:
        frame = pd.DataFrame(campaigns).drop(columns=["row_sha256"], errors="ignore")
        nested = validate_lockable_campaign_registry(
            frame,
            allow_fake=evidence_status == FAKE_DRIVER_TEST,
        )
        rows.append(("lock.leakage_checks", nested.passed, "valid" if nested.passed else "failed"))
    else:
        rows.append(("lock.leakage_checks", False, "no campaigns"))

    live_eligible = evidence_status == LIVE_QUALIFIED and all(row[1] for row in rows)
    return _report(rows, evidence_status=evidence_status, live_eligible=live_eligible)


TELEMETRY_RECORD_TYPES = {
    "network_probe",
    "rf_beam_handover",
    "route_observation",
    "application_eo_outcome",
}


def validate_telemetry_record(record: Mapping[str, Any]) -> PreparationReport:
    """Validate one synchronized telemetry event without discarding failures."""

    rows: list[tuple[str, bool, str]] = []
    record_type = record.get("record_type")
    rows.append(("telemetry.record_type", record_type in TELEMETRY_RECORD_TYPES, str(record_type)))
    common = {
        "schema_version",
        "record_type",
        "evidence_status",
        "wall_time_ns",
        "monotonic_time_ns",
        "clock_epoch_id",
        "maximum_clock_error_ms",
        "clock_evidence_sha256",
        "study_id",
        "campaign_id",
        "acquisition_family_id",
        "session_id",
        "source_id",
        "sequence_number",
    }
    missing = sorted(key for key in common if key not in record)
    rows.append(("telemetry.common_fields", not missing, "complete" if not missing else f"missing={missing}"))
    if missing or record_type not in TELEMETRY_RECORD_TYPES:
        return _report(rows, evidence_status=str(record.get("evidence_status", LIVE_REQUIRED)))

    evidence_status_value = str(record.get("evidence_status"))
    rows.append(
        (
            "telemetry.evidence_status",
            evidence_status_value in ALLOWED_EVIDENCE_SOURCES,
            evidence_status_value,
        )
    )

    numeric_ok = all(
        isinstance(record.get(key), int) and not isinstance(record.get(key), bool) and record[key] >= 0
        for key in ("wall_time_ns", "monotonic_time_ns", "sequence_number")
    )
    try:
        clock_ok = math.isfinite(float(record["maximum_clock_error_ms"])) and float(record["maximum_clock_error_ms"]) >= 0
    except (TypeError, ValueError):
        clock_ok = False
    rows.append(("telemetry.timestamps", numeric_ok and clock_ok, f"clock_error_ms={record.get('maximum_clock_error_ms')}"))
    rows.append(("telemetry.clock_evidence", _is_sha256(record.get("clock_evidence_sha256")), str(record.get("clock_evidence_sha256"))))
    identifiers_ok = all(not _is_placeholder(record.get(key)) for key in ("study_id", "campaign_id", "acquisition_family_id", "session_id", "source_id", "clock_epoch_id"))
    rows.append(("telemetry.identifiers", identifiers_ok, "valid" if identifiers_ok else "placeholder or empty identifier"))

    if record_type == "network_probe":
        required = {"probe_id", "path_id", "disposition", "sent_wall_time_ns", "received_wall_time_ns", "rtt_ms", "failure_reason"}
        absent = sorted(required.difference(record))
        rows.append(("telemetry.probe_fields", not absent, "complete" if not absent else f"missing={absent}"))
        disposition = record.get("disposition")
        disposition_ok = disposition in {"reply", "loss", "timeout"}
        sent_ok = (
            isinstance(record.get("sent_wall_time_ns"), int)
            and not isinstance(record.get("sent_wall_time_ns"), bool)
            and record["sent_wall_time_ns"] >= 0
        )
        if disposition == "reply":
            values_ok = (
                isinstance(record.get("received_wall_time_ns"), int)
                and record["received_wall_time_ns"] >= record.get("sent_wall_time_ns", -1)
                and isinstance(record.get("rtt_ms"), (int, float))
                and not isinstance(record.get("rtt_ms"), bool)
                and float(record["rtt_ms"]) >= 0
                and record.get("failure_reason") == ""
            )
        elif disposition in {"loss", "timeout"}:
            values_ok = (
                record.get("received_wall_time_ns") is None
                and record.get("rtt_ms") is None
                and isinstance(record.get("failure_reason"), str)
                and bool(record.get("failure_reason"))
            )
        else:
            values_ok = False
        rows.append(("telemetry.probe_disposition", disposition_ok and sent_ok and values_ok, str(disposition)))
    elif record_type == "rf_beam_handover":
        required = {"path_id", "access_type", "rf_metrics", "beam_id", "satellite_id", "gateway_id", "handover_event"}
        absent = sorted(required.difference(record))
        values_ok = (
            not absent
            and record.get("access_type") in {"cellular_5g", "leo"}
            and isinstance(record.get("rf_metrics"), Mapping)
            and record.get("handover_event") in {"none", "start", "complete", "failure"}
        )
        rows.append(("telemetry.rf_beam_handover_fields", values_ok, "complete" if values_ok else f"missing={absent}"))
    elif record_type == "route_observation":
        required = {"decision_id", "task_id", "stage", "path_id", "destination", "acknowledged", "evidence_sha256"}
        absent = sorted(required.difference(record))
        values_ok = (
            not absent
            and record.get("stage") in {"proposed", "install_request", "install_ack", "pre_fib", "post_fib", "first_packet", "final_fib", "rollback"}
            and isinstance(record.get("acknowledged"), bool)
            and _is_sha256(record.get("evidence_sha256"))
            and _valid_destination(record.get("destination"))
        )
        rows.append(("telemetry.route_fields", values_ok, "complete" if values_ok else f"missing={absent}"))
    elif record_type == "application_eo_outcome":
        required = {"decision_id", "task_id", "workload_type", "required_bytes", "delivered_bytes", "deadline_wall_time_ns", "completion_wall_time_ns", "application_complete", "met_deadline", "no_action", "failure_reason"}
        absent = sorted(required.difference(record))
        values_ok = (
            not absent
            and record.get("workload_type") in {"application", "earth_observation"}
            and all(isinstance(record.get(key), int) and not isinstance(record.get(key), bool) and record[key] >= 0 for key in ("required_bytes", "delivered_bytes", "deadline_wall_time_ns"))
            and (record.get("completion_wall_time_ns") is None or isinstance(record.get("completion_wall_time_ns"), int))
            and all(isinstance(record.get(key), bool) for key in ("application_complete", "met_deadline", "no_action"))
        )
        if values_ok:
            completed = bool(record["application_complete"])
            on_time = bool(record["met_deadline"])
            no_action = bool(record["no_action"])
            completion = record["completion_wall_time_ns"]
            if completed:
                values_ok = (
                    isinstance(completion, int)
                    and record["delivered_bytes"] >= record["required_bytes"]
                    and not no_action
                )
            if on_time:
                values_ok = (
                    values_ok
                    and completed
                    and isinstance(completion, int)
                    and completion <= record["deadline_wall_time_ns"]
                )
            if no_action:
                values_ok = values_ok and not completed and not on_time
        rows.append(("telemetry.outcome_fields", values_ok, "complete" if values_ok else f"missing={absent}"))

    evidence_status = str(record.get("evidence_status", LIVE_REQUIRED))
    return _report(rows, evidence_status=evidence_status, live_eligible=evidence_status == "live_hardware" and all(row[1] for row in rows))


def append_telemetry_record(
    path: str | Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Append a validated synchronized event to a SHA-256 hash chain."""

    report = validate_telemetry_record(record)
    report.raise_for_errors()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = "0" * 64
    if target.exists() and target.stat().st_size:
        audit = audit_telemetry_stream(
            target,
            require_bundle_complete=False,
            require_sequence_continuity=False,
        )
        audit.raise_for_errors()
        with target.open("rb") as handle:
            last = list(handle)[-1]
        previous = json.loads(last)["record_sha256"]
    payload = dict(record)
    payload["previous_record_sha256"] = previous
    payload["record_sha256"] = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def audit_telemetry_stream(
    path: str | Path,
    *,
    expected_probe_ids: Iterable[str] | None = None,
    campaign_lock: Mapping[str, Any] | None = None,
    require_complete_action_outcome: bool = False,
    require_bundle_complete: bool = True,
    require_sequence_continuity: bool = True,
) -> PreparationReport:
    """Audit hashes, clocks, probe completeness, roles, actions, and outcomes."""

    target = Path(path)
    rows: list[tuple[str, bool, str]] = []
    if not target.exists():
        return _report([("telemetry.file_exists", False, str(target))], evidence_status=LIVE_REQUIRED)
    records: list[dict[str, Any]] = []
    previous = "0" * 64
    errors: list[str] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
                claimed = payload.pop("record_sha256")
                linked = payload.pop("previous_record_sha256")
                expected = _sha256_bytes(_canonical_json({**payload, "previous_record_sha256": linked}).encode("utf-8"))
                validation = validate_telemetry_record(payload)
                if not validation.passed:
                    errors.append(f"line {line_number}: schema")
                if linked != previous:
                    errors.append(f"line {line_number}: previous hash")
                if claimed != expected:
                    errors.append(f"line {line_number}: record hash")
                previous = claimed
                records.append(payload)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"line {line_number}: {exc}")
    rows.append(("telemetry.nonempty", bool(records), f"records={len(records)}"))
    rows.append(("telemetry.hash_chain", not errors, "valid" if not errors else "; ".join(errors[:10])))
    if not records:
        return _report(rows, evidence_status=LIVE_REQUIRED)

    frame = pd.DataFrame(records)
    duplicate_sequence = int(frame.duplicated(["source_id", "sequence_number"]).sum())
    rows.append(("telemetry.unique_source_sequence", duplicate_sequence == 0, f"duplicates={duplicate_sequence}"))
    ordered = frame.sort_values(["source_id", "sequence_number"])
    monotonic_error = 0
    sequence_gap_sources = 0
    for _, group in ordered.groupby("source_id"):
        if not group["monotonic_time_ns"].is_monotonic_increasing:
            monotonic_error += 1
        sequence = group["sequence_number"].astype(int).tolist()
        if any(right - left != 1 for left, right in zip(sequence, sequence[1:])):
            sequence_gap_sources += 1
    rows.append(("telemetry.monotonic_per_source", monotonic_error == 0, f"invalid_sources={monotonic_error}"))
    rows.append(
        (
            "telemetry.contiguous_source_sequence",
            not require_sequence_continuity or sequence_gap_sources == 0,
            f"sources_with_gaps={sequence_gap_sources}",
        )
    )

    clock_epochs = int(frame["clock_epoch_id"].nunique())
    clock_evidence = int(frame["clock_evidence_sha256"].nunique())
    rows.append(
        (
            "telemetry.synchronized_clock_epoch",
            clock_epochs == 1 and clock_evidence == 1,
            f"clock_epochs={clock_epochs}; clock_evidence_hashes={clock_evidence}",
        )
    )

    probes = frame.loc[frame["record_type"].eq("network_probe")].copy()
    expected_set = set(expected_probe_ids or ())
    observed_set = set(probes.get("probe_id", pd.Series(dtype=str)))
    duplicates = int(probes.get("probe_id", pd.Series(dtype=str)).duplicated().sum())
    probe_complete = not expected_set or (expected_set == observed_set and duplicates == 0)
    rows.append(
        (
            "telemetry.loss_preserving_probe_schedule",
            probe_complete,
            f"missing={sorted(expected_set - observed_set)}; extra={sorted(observed_set - expected_set) if expected_set else []}; duplicates={duplicates}",
        )
    )

    required_types = TELEMETRY_RECORD_TYPES if require_bundle_complete else set()
    present_types = set(frame["record_type"])
    rows.append(("telemetry.required_record_types", required_types.issubset(present_types), f"missing={sorted(required_types - present_types)}"))

    routes = frame.loc[frame["record_type"].eq("route_observation")].copy()
    outcomes = frame.loc[frame["record_type"].eq("application_eo_outcome")].copy()
    route_stage_errors = 0
    route_order_errors = 0
    route_path_errors = 0
    missing_outcomes = 0
    if len(routes):
        required_stages = {"proposed", "install_ack", "post_fib", "first_packet", "final_fib"}
        for decision_id, group in routes.groupby("decision_id"):
            if not required_stages.issubset(set(group["stage"])):
                route_stage_errors += 1
            else:
                first = group.sort_values("wall_time_ns").drop_duplicates("stage")
                time_by_stage = first.set_index("stage")["wall_time_ns"].to_dict()
                stage_order = ("proposed", "install_ack", "post_fib", "first_packet", "final_fib")
                if any(time_by_stage[left] > time_by_stage[right] for left, right in zip(stage_order, stage_order[1:])):
                    route_order_errors += 1
                path_by_stage = first.set_index("stage")["path_id"].to_dict()
                installed_path = path_by_stage["post_fib"]
                if any(
                    path_by_stage[stage] != installed_path
                    for stage in ("proposed", "install_ack", "first_packet")
                ):
                    route_path_errors += 1
            if decision_id not in set(outcomes.get("decision_id", pd.Series(dtype=str))):
                missing_outcomes += 1
    rows.append(
        (
            "telemetry.route_lifecycle",
            not require_bundle_complete or route_stage_errors == 0,
            f"incomplete_decisions={route_stage_errors}",
        )
    )
    rows.append(("telemetry.route_stage_order", not require_bundle_complete or route_order_errors == 0, f"misordered_decisions={route_order_errors}"))
    rows.append(("telemetry.route_path_reconciliation", not require_bundle_complete or route_path_errors == 0, f"mismatched_decisions={route_path_errors}"))
    rows.append(("telemetry.action_outcome_link", not require_complete_action_outcome or missing_outcomes == 0, f"missing_outcomes={missing_outcomes}"))

    duplicate_outcomes = int(outcomes.get("decision_id", pd.Series(dtype=str)).duplicated().sum())
    task_mismatch = 0
    if len(routes) and len(outcomes):
        route_tasks = routes.groupby("decision_id")["task_id"].first()
        for row in outcomes.itertuples(index=False):
            if row.decision_id in route_tasks and route_tasks[row.decision_id] != row.task_id:
                task_mismatch += 1
    rows.append(("telemetry.unique_outcome_per_decision", duplicate_outcomes == 0, f"duplicates={duplicate_outcomes}"))
    rows.append(("telemetry.route_outcome_task_identity", task_mismatch == 0, f"mismatches={task_mismatch}"))

    identity_errors = 0
    if campaign_lock is not None:
        lock_report = verify_campaign_registry_lock(campaign_lock)
        if not lock_report.passed:
            identity_errors += 1
        allowed = {
            (row["campaign_id"], row["acquisition_family_id"])
            for row in campaign_lock.get("campaigns", [])
        }
        identity_errors += sum(
            (record["campaign_id"], record["acquisition_family_id"]) not in allowed
            for record in records
        )
    rows.append(("telemetry.campaign_lock_identity", identity_errors == 0, f"errors={identity_errors}"))

    statuses = {str(record.get("evidence_status", LIVE_REQUIRED)) for record in records}
    mixed = len(statuses) != 1
    rows.append(("telemetry.single_evidence_status", not mixed, str(sorted(statuses))))
    evidence_status = next(iter(statuses)) if len(statuses) == 1 else LIVE_REQUIRED
    live_eligible = evidence_status == "live_hardware" and all(row[1] for row in rows)
    return _report(rows, evidence_status=evidence_status, live_eligible=live_eligible)


def ingest_telemetry_bundle(
    records: Iterable[Mapping[str, Any]],
    destination: str | Path,
    *,
    expected_probe_ids: Iterable[str],
    campaign_lock: Mapping[str, Any] | None = None,
    require_complete_action_outcome: bool = True,
) -> PreparationReport:
    """Normalize a bundle into a durable, complete, hash-chained JSONL stream."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite telemetry stream: {target}")
    for record in records:
        append_telemetry_record(target, record)
    report = audit_telemetry_stream(
        target,
        expected_probe_ids=expected_probe_ids,
        campaign_lock=campaign_lock,
        require_complete_action_outcome=require_complete_action_outcome,
    )
    report.raise_for_errors()
    return report
