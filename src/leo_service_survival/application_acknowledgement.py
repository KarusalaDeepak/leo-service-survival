"""Production emission of authenticated application outcome acknowledgements.

This module records endpoint observations; it does not decide whether a task
succeeded.  The emitted acknowledgement contains explicit byte, deadline, and
validity evidence bound to one existing action record.  The downstream ledger
appender remains responsible for deriving deadline and conservative completion
endpoints.

Live acknowledgements use the project's purpose-bound HMAC-SHA256 convention.
Files are promoted atomically with create-only semantics and read-only mode so
the emitter cannot silently replace prior evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from .outcome_evidence import (
    IDENTITY_FIELDS,
    LIVE_APPLICATION_ACKNOWLEDGEMENT,
    LIVE_EVIDENCE_ORIGIN,
    OUTCOME_ACK_ARTIFACT_TYPE,
    OUTCOME_ACK_PURPOSE,
    OUTCOME_FIELDS,
    load_verified_outcome_evidence,
)
from .prospective_release import sign_artifact, verify_signed_artifact
from .prospective_trial import audit_hash_chain, load_ledger_schema


DEFAULT_CLAIM_BOUNDARY = (
    "One authenticated endpoint acknowledgement for one bound action; "
    "the ledger derives completion and this artifact supports no broader "
    "performance or deployment-benefit claim."
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LIVE_EVIDENCE_TIERS = frozenset(
    {
        "field_live_randomized",
        "field_shadow_concurrent",
        "field_observational",
        "hardware_in_loop",
    }
)
_BODY_REQUIRED = frozenset(
    {
        "schema_version",
        "artifact_type",
        "evidence_status",
        "evidence_origin",
        "acknowledgement_id",
        "acknowledgement_source",
        "acknowledged_wall_ns",
        "claim_boundary",
        *IDENTITY_FIELDS,
        *OUTCOME_FIELDS,
    }
)
_AUTHENTICATION_FIELDS = frozenset({"algorithm", "purpose", "key_id", "signature"})


@dataclass(frozen=True)
class ApplicationAcknowledgementContract:
    """Supported constants extracted from the frozen acknowledgement schema."""

    schema_version: str
    artifact_type: str
    authentication_purpose: str


@dataclass(frozen=True)
class ApplicationOutcomeObservation:
    """Raw endpoint facts required to emit one acknowledgement."""

    required_bytes: int
    delivered_bytes: int
    deadline_wall_ns: int
    completion_wall_ns: int
    observation_valid: bool
    application_complete: bool
    no_action: bool
    controller_failure: bool
    late_install: bool
    missingness_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in OUTCOME_FIELDS
        }


def _json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_application_acknowledgement_contract(
    schema_path: str | Path,
) -> ApplicationAcknowledgementContract:
    """Load and fail closed if the schema no longer matches this emitter.

    This is deliberately stricter than permissive JSON-schema validation.  A
    schema version or authentication-purpose change requires an emitter update
    instead of silently producing an artifact under a different contract.
    """

    schema = _json_object(schema_path)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("acknowledgement schema has no properties object")
    required = schema.get("required")
    if not isinstance(required, list) or set(required) != _BODY_REQUIRED:
        raise ValueError(
            "acknowledgement schema required fields differ from the supported contract"
        )
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("acknowledgement schema must be a closed JSON object")

    schema_version = properties.get("schema_version", {}).get("const")
    artifact_type = properties.get("artifact_type", {}).get("const")
    if schema_version != "1.0.0":
        raise ValueError(f"unsupported acknowledgement schema version: {schema_version!r}")
    if artifact_type != OUTCOME_ACK_ARTIFACT_TYPE:
        raise ValueError(f"unsupported acknowledgement artifact type: {artifact_type!r}")

    status_values = set(properties.get("evidence_status", {}).get("enum", ()))
    origin_values = set(properties.get("evidence_origin", {}).get("enum", ()))
    if LIVE_APPLICATION_ACKNOWLEDGEMENT not in status_values:
        raise ValueError("schema does not permit live application acknowledgements")
    if LIVE_EVIDENCE_ORIGIN not in origin_values:
        raise ValueError("schema does not permit executed-live evidence origin")

    for field in (
        "required_bytes",
        "delivered_bytes",
        "deadline_wall_ns",
        "completion_wall_ns",
    ):
        if properties.get(field, {}).get("type") != "integer":
            raise ValueError(f"schema field {field} must remain an integer")
    for field in (
        "observation_valid",
        "application_complete",
        "no_action",
        "controller_failure",
        "late_install",
    ):
        if properties.get(field, {}).get("type") != "boolean":
            raise ValueError(f"schema field {field} must remain boolean")

    authentication = properties.get("authentication")
    if not isinstance(authentication, Mapping):
        raise ValueError("schema has no authentication contract")
    authentication_properties = authentication.get("properties")
    if not isinstance(authentication_properties, Mapping):
        raise ValueError("schema authentication has no properties object")
    if set(authentication.get("required", ())) != _AUTHENTICATION_FIELDS:
        raise ValueError("schema authentication fields differ from the supported contract")
    if authentication.get("additionalProperties") is not False:
        raise ValueError("schema authentication object must reject extra properties")
    if authentication_properties.get("algorithm", {}).get("const") != "HMAC-SHA256":
        raise ValueError("schema authentication algorithm must be HMAC-SHA256")
    purpose = authentication_properties.get("purpose", {}).get("const")
    if purpose != OUTCOME_ACK_PURPOSE:
        raise ValueError(
            "schema authentication purpose differs from the supported acknowledgement purpose"
        )

    live_requires_authentication = False
    for clause in schema.get("allOf", ()):
        if not isinstance(clause, Mapping):
            continue
        condition = clause.get("if", {}).get("properties", {}).get("evidence_status", {})
        live_required = set(clause.get("then", {}).get("required", ()))
        if condition.get("const") == LIVE_APPLICATION_ACKNOWLEDGEMENT and {
            "artifact_sha256",
            "authentication",
        }.issubset(live_required):
            live_requires_authentication = True
    if not live_requires_authentication:
        raise ValueError("schema does not require hash and authentication for live evidence")

    return ApplicationAcknowledgementContract(
        schema_version=schema_version,
        artifact_type=artifact_type,
        authentication_purpose=purpose,
    )


def load_bound_live_action(
    ledger_path: str | Path,
    ledger_schema_path: str | Path,
    decision_id: str,
) -> dict[str, Any]:
    """Audit a ledger and return the only unclosed live action for a decision."""

    ledger = Path(ledger_path)
    schema = load_ledger_schema(ledger_schema_path)
    audit_hash_chain(ledger, schema).raise_for_errors()
    records = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actions = [
        row
        for row in records
        if row.get("record_type") == "action" and row.get("decision_id") == decision_id
    ]
    outcomes = [
        row
        for row in records
        if row.get("record_type") == "outcome" and row.get("decision_id") == decision_id
    ]
    if len(actions) != 1:
        raise ValueError(f"expected exactly one linked action; found {len(actions)}")
    if outcomes:
        raise ValueError("an outcome already exists for this decision")
    action = actions[0]
    if action.get("evidence_tier") not in _LIVE_EVIDENCE_TIERS:
        raise ValueError(
            "production acknowledgement emission requires a live or hardware evidence tier"
        )
    return action


def _validate_observation(
    observation: ApplicationOutcomeObservation,
    *,
    action_recorded_wall_ns: int,
    acknowledged_wall_ns: int,
) -> None:
    for field in (
        "required_bytes",
        "delivered_bytes",
        "deadline_wall_ns",
        "completion_wall_ns",
    ):
        value = getattr(observation, field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    for field in (
        "observation_valid",
        "application_complete",
        "no_action",
        "controller_failure",
        "late_install",
    ):
        if not isinstance(getattr(observation, field), bool):
            raise ValueError(f"{field} must be boolean")
    if observation.deadline_wall_ns == 0:
        raise ValueError("deadline_wall_ns must be a real nonzero deadline")
    if observation.completion_wall_ns < action_recorded_wall_ns:
        raise ValueError("completion_wall_ns precedes the linked action record")
    if acknowledged_wall_ns < observation.completion_wall_ns:
        raise ValueError("acknowledged_wall_ns precedes completion_wall_ns")
    if observation.application_complete and (
        observation.delivered_bytes < observation.required_bytes
    ):
        raise ValueError("application_complete contradicts an incomplete byte count")
    if observation.no_action and (
        observation.application_complete or observation.delivered_bytes != 0
    ):
        raise ValueError("no_action requires zero delivered bytes and an incomplete application")
    if observation.controller_failure and observation.application_complete:
        raise ValueError("controller_failure contradicts a completed application")
    if observation.observation_valid and observation.missingness_reason:
        raise ValueError("valid observations must not carry a missingness reason")
    if not observation.observation_valid and not observation.missingness_reason:
        raise ValueError("invalid observations require a missingness reason")


def build_live_application_acknowledgement(
    *,
    action: Mapping[str, Any],
    observation: ApplicationOutcomeObservation,
    acknowledgement_id: str,
    acknowledgement_source: str,
    acknowledged_wall_ns: int,
    signing_key: bytes,
    contract: ApplicationAcknowledgementContract,
    claim_boundary: str = DEFAULT_CLAIM_BOUNDARY,
) -> dict[str, Any]:
    """Build and authenticate one acknowledgement without deriving an outcome."""

    for label, value in (
        ("acknowledgement_id", acknowledgement_id),
        ("acknowledgement_source", acknowledgement_source),
        ("claim_boundary", claim_boundary),
    ):
        if not isinstance(value, str) or not value.strip() or value.strip().upper() == "TBD":
            raise ValueError(f"{label} must be a concrete nonempty string")
    if not isinstance(acknowledged_wall_ns, int) or isinstance(acknowledged_wall_ns, bool):
        raise ValueError("acknowledged_wall_ns must be an integer")
    if acknowledged_wall_ns < 0:
        raise ValueError("acknowledged_wall_ns must be nonnegative")
    if action.get("record_type") != "action":
        raise ValueError("linked record is not an action")
    if action.get("evidence_tier") not in _LIVE_EVIDENCE_TIERS:
        raise ValueError("linked action is not eligible for production acknowledgement evidence")

    for field in IDENTITY_FIELDS:
        value = action.get(field)
        if not isinstance(value, str) or not value or value.upper() == "TBD":
            raise ValueError(f"linked action has no concrete {field}")
    for field in ("model_sha256", "config_sha256"):
        value = str(action[field])
        if not _SHA256_RE.fullmatch(value) or value == "0" * 64:
            raise ValueError(f"linked action has an invalid or placeholder {field}")
    action_recorded_wall_ns = action.get("recorded_wall_ns")
    if (
        not isinstance(action_recorded_wall_ns, int)
        or isinstance(action_recorded_wall_ns, bool)
        or action_recorded_wall_ns < 0
    ):
        raise ValueError("linked action has an invalid recorded_wall_ns")

    _validate_observation(
        observation,
        action_recorded_wall_ns=action_recorded_wall_ns,
        acknowledged_wall_ns=acknowledged_wall_ns,
    )
    body = {
        "schema_version": contract.schema_version,
        "artifact_type": contract.artifact_type,
        "evidence_status": LIVE_APPLICATION_ACKNOWLEDGEMENT,
        "evidence_origin": LIVE_EVIDENCE_ORIGIN,
        "acknowledgement_id": acknowledgement_id,
        "acknowledgement_source": acknowledgement_source,
        "acknowledged_wall_ns": acknowledged_wall_ns,
        "claim_boundary": claim_boundary,
        **{field: action[field] for field in IDENTITY_FIELDS},
        **observation.as_dict(),
    }
    if set(body) != _BODY_REQUIRED:
        raise RuntimeError("emitter body does not match the loaded schema contract")
    signed = sign_artifact(
        body,
        signing_key,
        purpose=contract.authentication_purpose,
    )
    verify_signed_artifact(
        signed,
        signing_key,
        purpose=contract.authentication_purpose,
    ).raise_for_errors()
    if set(signed) != _BODY_REQUIRED | {"artifact_sha256", "authentication"}:
        raise RuntimeError("signed acknowledgement contains unexpected fields")
    return signed


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write while emitting acknowledgement")
        view = view[written:]


def write_immutable_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    validator: Callable[[Path], None] | None = None,
) -> Path:
    """Atomically create a read-only JSON artifact without overwrite support."""

    output = Path(path)
    parent = output.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {parent}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing acknowledgement: {output}")
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = parent / f".{output.name}.tmp.{os.getpid()}.{uuid4().hex}"
    fd: int | None = None
    promoted = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        _write_all(fd, content)
        os.fsync(fd)
        os.close(fd)
        fd = None
        if validator is not None:
            validator(temporary)
        os.chmod(temporary, 0o400)
        os.link(temporary, output)
        promoted = True
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return output
    finally:
        if fd is not None:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()
        if not promoted and output.exists():
            # The emitter never removes a pre-existing output.  This branch is
            # reachable only if linking succeeded and a later durability step
            # failed; keep that evidence rather than silently destroying it.
            pass


def emit_live_application_acknowledgement(
    *,
    output_path: str | Path,
    action: Mapping[str, Any],
    observation: ApplicationOutcomeObservation,
    acknowledgement_id: str,
    acknowledgement_source: str,
    signing_key: bytes,
    contract: ApplicationAcknowledgementContract,
    acknowledged_wall_ns: int | None = None,
    claim_boundary: str = DEFAULT_CLAIM_BOUNDARY,
) -> dict[str, Any]:
    """Build, verify, and immutably emit one live endpoint acknowledgement."""

    acknowledged = time.time_ns() if acknowledged_wall_ns is None else acknowledged_wall_ns
    payload = build_live_application_acknowledgement(
        action=action,
        observation=observation,
        acknowledgement_id=acknowledgement_id,
        acknowledgement_source=acknowledgement_source,
        acknowledged_wall_ns=acknowledged,
        signing_key=signing_key,
        contract=contract,
        claim_boundary=claim_boundary,
    )

    def verify_temporary(path: Path) -> None:
        verified = load_verified_outcome_evidence(
            path,
            action=action,
            signing_key=signing_key,
        )
        if verified.acknowledgement_sha256 != payload["artifact_sha256"]:
            raise ValueError("emitted acknowledgement hash failed self-verification")

    write_immutable_json(output_path, payload, validator=verify_temporary)
    return payload
