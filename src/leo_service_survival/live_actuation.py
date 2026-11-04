"""Fail-closed bindings for Objective 4 live route mutation.

This module is intentionally stricter than the offline/dry-run utilities.  A
route-changing process must prove that it is using an authenticated live
qualification, the exact frozen route/testbed/schema files, a role-correct
campaign lock, and the exact hash-chained telemetry capture named by that lock.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .field_preparation import (
    LIVE_QUALIFIED,
    ProbeResult,
    audit_telemetry_stream,
    validate_route_driver_configuration,
    validate_testbed_qualification,
    verify_campaign_registry_lock,
)
from .gate_freeze import CONFIRMATORY_LIVE, TrialFreezeManifest, verify_trial_freeze_manifest
from .prospective_release import verify_signed_artifact
from .prospective_trial import sha256_file


LIVE_QUALIFICATION_PURPOSE = "O4_LIVE_TESTBED_QUALIFICATION_V1"
PHASE_ROLE = {
    "rejection_verification": "gate_evidence",
    "canary": "gate_evidence",
    "final_test": "final_test",
}


@dataclass(frozen=True)
class LiveActionBindings:
    trial_freeze_manifest_sha256: str
    qualification_artifact_sha256: str
    qualification_file_sha256: str
    testbed_config_sha256: str
    route_config_sha256: str
    ledger_schema_sha256: str
    campaign_lock_sha256: str
    raw_telemetry_sha256: str
    campaign_role: str


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def verify_live_qualification(
    *,
    qualification_path: str | Path,
    testbed_config_path: str | Path,
    route_config_path: str | Path,
    ledger_schema_path: str | Path,
    signing_key: bytes,
) -> dict[str, Any]:
    """Verify an authenticated qualification and all bound frozen files."""

    qualification = _load_json(qualification_path)
    verify_signed_artifact(
        qualification,
        signing_key,
        purpose=LIVE_QUALIFICATION_PURPOSE,
    ).raise_for_errors()
    required_state = {
        "artifact_type": "live_testbed_qualification",
        "qualification_status": LIVE_QUALIFIED,
        "evidence_source": "live_hardware",
        "probe_evidence_origin": "AUTHENTICATED_LIVE_EXECUTION",
        "field_trial_status": "NOT_RUN",
        "passed": True,
        "testbed_live_eligible": True,
        "route_live_eligible": True,
    }
    mismatches = {
        key: (qualification.get(key), expected)
        for key, expected in required_state.items()
        if qualification.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"qualification is not authenticated LIVE_QUALIFIED evidence: {mismatches}"
        )
    exact_hashes = {
        "testbed_config_sha256": sha256_file(testbed_config_path),
        "route_config_sha256": sha256_file(route_config_path),
        "ledger_schema_sha256": sha256_file(ledger_schema_path),
    }
    changed = {
        key: (qualification.get(key), expected)
        for key, expected in exact_hashes.items()
        if qualification.get(key) != expected
    }
    if changed:
        raise RuntimeError(f"qualified frozen file changed: {changed}")

    testbed = _load_json(testbed_config_path)
    route = _load_json(route_config_path)
    probes = tuple(
        ProbeResult(
            probe_id=item["probe_id"],
            passed=item["passed"],
            observed=item["observed"],
            evidence_sha256=item["evidence_sha256"],
            command_sha256=item.get("command_sha256", ""),
            evidence_origin=item.get("evidence_origin", "imported_evidence"),
        )
        for item in qualification.get("probe_results", ())
    )
    validate_testbed_qualification(testbed, probe_results=probes).raise_for_errors()
    validate_route_driver_configuration(route).raise_for_errors()
    return qualification


def verify_live_action_bindings(
    *,
    qualification_path: str | Path,
    testbed_config_path: str | Path,
    route_config_path: str | Path,
    trial_config_path: str | Path,
    ledger_schema_path: str | Path,
    campaign_lock_path: str | Path,
    trial_freeze_manifest_path: str | Path,
    raw_telemetry_path: str | Path,
    signing_key: bytes,
    campaign_id: str,
    acquisition_family_id: str,
    session_id: str,
    release_phase: str,
    model_sha256: str,
    final_test_unlock: Mapping[str, Any] | None = None,
) -> LiveActionBindings:
    """Verify every immutable binding required before live route mutation."""

    if release_phase not in PHASE_ROLE:
        raise RuntimeError(f"unsupported live release phase: {release_phase}")
    qualification = verify_live_qualification(
        qualification_path=qualification_path,
        testbed_config_path=testbed_config_path,
        route_config_path=route_config_path,
        ledger_schema_path=ledger_schema_path,
        signing_key=signing_key,
    )
    manifest_value = _load_json(trial_freeze_manifest_path)
    manifest = TrialFreezeManifest.from_dict(manifest_value)
    verify_trial_freeze_manifest(
        manifest,
        signing_key=signing_key,
        expected_mode=CONFIRMATORY_LIVE,
    )
    exact_manifest_files = {
        "testbed_qualification": sha256_file(qualification_path),
        "route_config": sha256_file(route_config_path),
        "campaign_registry_lock": sha256_file(campaign_lock_path),
        "ledger_schema": sha256_file(ledger_schema_path),
        "trial_config": sha256_file(trial_config_path),
    }
    manifest_mismatches = {
        name: (manifest.artifacts.get(name, {}).get("sha256"), digest)
        for name, digest in exact_manifest_files.items()
        if manifest.artifacts.get(name, {}).get("sha256") != digest
    }
    if manifest_mismatches:
        raise RuntimeError(
            f"live action files differ from the authenticated trial freeze: {manifest_mismatches}"
        )
    if manifest.artifacts.get("model", {}).get("sha256") != model_sha256:
        raise RuntimeError("live action model differs from the authenticated trial freeze")
    lock = _load_json(campaign_lock_path)
    lock_report = verify_campaign_registry_lock(lock)
    lock_report.raise_for_errors()
    if not lock_report.live_eligible or lock.get("evidence_status") != LIVE_QUALIFIED:
        raise RuntimeError("live action requires a LIVE_QUALIFIED campaign lock")
    if lock.get("protocol_sha256") != sha256_file(trial_config_path):
        raise RuntimeError("campaign lock is bound to a different trial protocol")
    if lock.get("qualification_sha256") != sha256_file(qualification_path):
        raise RuntimeError("campaign lock is bound to a different qualification")

    matches = [
        row
        for row in lock.get("campaigns", ())
        if row.get("campaign_id") == campaign_id
        and row.get("acquisition_family_id") == acquisition_family_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "campaign and acquisition family must identify exactly one locked row"
        )
    row = matches[0]
    expected_role = PHASE_ROLE[release_phase]
    if row.get("role") != expected_role or row.get("status") != "locked":
        raise RuntimeError(
            f"release phase {release_phase} requires a locked {expected_role} campaign"
        )
    if row.get("config_sha256") != sha256_file(trial_config_path):
        raise RuntimeError("campaign row is bound to a different trial configuration")
    if row.get("model_sha256") != model_sha256:
        raise RuntimeError("campaign row is bound to a different model")
    if row.get("testbed_qualification_sha256") != sha256_file(qualification_path):
        raise RuntimeError("campaign row is bound to a different qualification")

    raw_hash = sha256_file(raw_telemetry_path)
    if row.get("raw_capture_sha256") != raw_hash:
        raise RuntimeError("raw telemetry file does not match the locked campaign hash")
    telemetry_report = audit_telemetry_stream(
        raw_telemetry_path,
        campaign_lock=lock,
        require_bundle_complete=False,
        require_complete_action_outcome=False,
    )
    telemetry_report.raise_for_errors()
    if not telemetry_report.live_eligible:
        raise RuntimeError("raw telemetry is not eligible live-hardware evidence")
    identities: set[tuple[str, str, str]] = set()
    for line in Path(raw_telemetry_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        identities.add(
            (
                str(item.get("campaign_id")),
                str(item.get("acquisition_family_id")),
                str(item.get("session_id")),
            )
        )
    expected_identity = (campaign_id, acquisition_family_id, session_id)
    if identities != {expected_identity}:
        raise RuntimeError(
            f"raw telemetry identity mismatch: expected={expected_identity}; observed={sorted(identities)}"
        )

    if release_phase == "final_test":
        if final_test_unlock is None:
            raise RuntimeError("final-test live action requires an unlock artifact")
        verify_signed_artifact(
            final_test_unlock,
            signing_key,
            purpose="O4_FINAL_TEST_UNLOCK_V1",
        ).raise_for_errors()
        if campaign_id not in set(final_test_unlock.get("final_test_campaign_ids", ())):
            raise RuntimeError("campaign is absent from the final-test unlock")
        if acquisition_family_id not in set(
            final_test_unlock.get("final_test_family_ids", ())
        ):
            raise RuntimeError("acquisition family is absent from the final-test unlock")
    elif final_test_unlock is not None:
        raise RuntimeError("final-test unlock cannot be used outside the final-test phase")

    return LiveActionBindings(
        trial_freeze_manifest_sha256=manifest.manifest_sha256,
        qualification_artifact_sha256=str(qualification["artifact_sha256"]),
        qualification_file_sha256=sha256_file(qualification_path),
        testbed_config_sha256=sha256_file(testbed_config_path),
        route_config_sha256=sha256_file(route_config_path),
        ledger_schema_sha256=sha256_file(ledger_schema_path),
        campaign_lock_sha256=sha256_file(campaign_lock_path),
        raw_telemetry_sha256=raw_hash,
        campaign_role=str(row["role"]),
    )
