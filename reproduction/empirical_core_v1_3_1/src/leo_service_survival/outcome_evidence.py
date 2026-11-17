"""Authenticated application-outcome evidence for prospective trials.

The action ledger is deliberately unable to accept a human-supplied success
flag.  Instead, this module validates one application acknowledgement against
the already-recorded action identity and derives the deadline endpoint from
the acknowledgement's timestamps, bytes, and validity fields.

Live acknowledgements are purpose-bound HMAC artifacts.  Deterministic
software rehearsals are accepted only when both the action and acknowledgement
are explicitly labelled ``FAKE_DRIVER_TEST``; that path can never be promoted
to field evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
from pathlib import Path
from typing import Any, Mapping

from .prospective_release import verify_signed_artifact
from .prospective_trial import canonical_json, sha256_file


LIVE_APPLICATION_ACKNOWLEDGEMENT = "LIVE_APPLICATION_ACKNOWLEDGEMENT"
FAKE_DRIVER_TEST = "FAKE_DRIVER_TEST"
LIVE_EVIDENCE_ORIGIN = "executed_live"
FAKE_EVIDENCE_ORIGIN = "fake_driver_test"
OUTCOME_ACK_PURPOSE = "O4_APPLICATION_OUTCOME_ACKNOWLEDGEMENT_V1"
OUTCOME_ACK_ARTIFACT_TYPE = "application_outcome_acknowledgement"

IDENTITY_FIELDS = (
    "study_id",
    "trial_id",
    "protocol_version",
    "campaign_id",
    "acquisition_family_id",
    "session_id",
    "task_id",
    "decision_id",
    "model_sha256",
    "config_sha256",
)

OUTCOME_FIELDS = (
    "required_bytes",
    "delivered_bytes",
    "deadline_wall_ns",
    "completion_wall_ns",
    "observation_valid",
    "application_complete",
    "no_action",
    "controller_failure",
    "late_install",
    "missingness_reason",
)


@dataclass(frozen=True)
class VerifiedOutcomeEvidence:
    """A validated acknowledgement plus immutable ledger metadata."""

    payload: dict[str, Any]
    evidence_file_sha256: str
    acknowledgement_sha256: str
    authenticated: bool
    authentication_key_id: str

    @property
    def met_deadline(self) -> bool:
        """Derive, rather than accept, the complete-valid-by-deadline label."""

        return bool(
            self.payload["observation_valid"]
            and self.payload["application_complete"]
            and self.payload["delivered_bytes"] >= self.payload["required_bytes"]
            and self.payload["completion_wall_ns"] <= self.payload["deadline_wall_ns"]
        )


def _canonical_without_authentication(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_sha256", "authentication"}
    }


def _validate_shape(payload: Mapping[str, Any]) -> list[str]:
    required = {
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
    errors: list[str] = []
    forbidden_derived = sorted(
        {"met_deadline", "conservative_on_time_completion", "failure_stage"}.intersection(
            payload
        )
    )
    if forbidden_derived:
        errors.append(
            "derived outcome fields must not be supplied by the observer: "
            f"{forbidden_derived}"
        )
    missing = sorted(required.difference(payload))
    if missing:
        return [f"missing required fields: {missing}"]
    if payload.get("schema_version") != "1.0.0":
        errors.append("schema_version must be 1.0.0")
    if payload.get("artifact_type") != OUTCOME_ACK_ARTIFACT_TYPE:
        errors.append(f"artifact_type must be {OUTCOME_ACK_ARTIFACT_TYPE}")
    for field in (*IDENTITY_FIELDS, "acknowledgement_id", "acknowledgement_source"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            errors.append(f"{field} must be a nonempty string")
    for field in (
        "acknowledged_wall_ns",
        "required_bytes",
        "delivered_bytes",
        "deadline_wall_ns",
        "completion_wall_ns",
    ):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{field} must be a nonnegative integer")
    for field in (
        "observation_valid",
        "application_complete",
        "no_action",
        "controller_failure",
        "late_install",
    ):
        if not isinstance(payload.get(field), bool):
            errors.append(f"{field} must be boolean")
    if not isinstance(payload.get("missingness_reason"), str):
        errors.append("missingness_reason must be a string")
    if not isinstance(payload.get("claim_boundary"), str) or not payload["claim_boundary"]:
        errors.append("claim_boundary must be a nonempty string")
    return errors


def _validate_semantics(payload: Mapping[str, Any], action: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in IDENTITY_FIELDS:
        if payload.get(field) != action.get(field):
            errors.append(
                f"identity mismatch for {field}: "
                f"acknowledgement={payload.get(field)!r}; action={action.get(field)!r}"
            )

    completion = payload.get("completion_wall_ns")
    acknowledged = payload.get("acknowledged_wall_ns")
    action_recorded = action.get("recorded_wall_ns")
    if isinstance(completion, int) and isinstance(action_recorded, int):
        if completion < action_recorded:
            errors.append("completion_wall_ns precedes the linked action record")
    if isinstance(acknowledged, int) and isinstance(completion, int):
        if acknowledged < completion:
            errors.append("acknowledged_wall_ns precedes completion_wall_ns")

    if payload.get("no_action") and (
        payload.get("application_complete") or payload.get("delivered_bytes", 0) > 0
    ):
        errors.append("no_action contradicts completed or delivered application data")
    if payload.get("application_complete") and (
        payload.get("delivered_bytes", -1) < payload.get("required_bytes", 0)
    ):
        errors.append("application_complete contradicts an incomplete byte count")

    action_is_fake = action.get("evidence_tier") == "controlled_fake_driver_test"
    status = payload.get("evidence_status")
    origin = payload.get("evidence_origin")
    if action_is_fake:
        if status != FAKE_DRIVER_TEST or origin != FAKE_EVIDENCE_ORIGIN:
            errors.append(
                "controlled fake-driver actions require FAKE_DRIVER_TEST/fake_driver_test outcome evidence"
            )
    elif status != LIVE_APPLICATION_ACKNOWLEDGEMENT or origin != LIVE_EVIDENCE_ORIGIN:
        errors.append(
            "field actions require LIVE_APPLICATION_ACKNOWLEDGEMENT/executed_live outcome evidence"
        )
    return errors


def load_verified_outcome_evidence(
    path: str | Path,
    *,
    action: Mapping[str, Any],
    signing_key: bytes | None = None,
) -> VerifiedOutcomeEvidence:
    """Load, authenticate, identity-bind, and hash one outcome acknowledgement.

    For a live action, ``signing_key`` is mandatory and the acknowledgement
    must carry a valid purpose-bound HMAC.  For a controlled fake action, the
    exact file is still hashed, but its status remains explicitly non-field.
    """

    evidence_path = Path(path)
    raw = evidence_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"outcome acknowledgement is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("outcome acknowledgement must be a JSON object")

    errors = _validate_shape(payload)
    if not errors:
        errors.extend(_validate_semantics(payload, action))
    if errors:
        raise ValueError("invalid outcome acknowledgement: " + "; ".join(errors))

    action_is_fake = action.get("evidence_tier") == "controlled_fake_driver_test"
    authenticated = False
    key_id = ""
    if not action_is_fake:
        if signing_key is None:
            raise ValueError("a signing key is required for live outcome evidence")
        signature = verify_signed_artifact(
            payload,
            signing_key,
            purpose=OUTCOME_ACK_PURPOSE,
        )
        signature.raise_for_errors()
        authenticated = True
        key_id = str(payload["authentication"]["key_id"])
        acknowledgement_sha256 = str(payload["artifact_sha256"])
    else:
        # A fake acknowledgement cannot become live evidence merely by carrying
        # a signature.  Its semantic body hash is retained for reproducibility.
        body = _canonical_without_authentication(payload)
        acknowledgement_sha256 = sha256(
            canonical_json(body).encode("utf-8")
        ).hexdigest()

    return VerifiedOutcomeEvidence(
        payload=payload,
        evidence_file_sha256=sha256_file(evidence_path),
        acknowledgement_sha256=acknowledgement_sha256,
        authenticated=authenticated,
        authentication_key_id=key_id,
    )


def verify_outcome_evidence_binding(
    record: Mapping[str, Any],
    evidence_path: str | Path,
    *,
    action: Mapping[str, Any],
    signing_key: bytes | None = None,
) -> bool:
    """Recompute both stored hashes and all acknowledgement bindings."""

    try:
        verified = load_verified_outcome_evidence(
            evidence_path,
            action=action,
            signing_key=signing_key,
        )
    except (OSError, ValueError):
        return False
    payload = verified.payload
    identity_matches = all(record.get(field) == payload[field] for field in IDENTITY_FIELDS)
    outcome_matches = all(record.get(field) == payload[field] for field in OUTCOME_FIELDS)
    metadata_matches = bool(
        record.get("outcome_evidence_status") == payload["evidence_status"]
        and record.get("outcome_evidence_origin") == payload["evidence_origin"]
        and record.get("application_acknowledgement_id")
        == payload["acknowledgement_id"]
        and record.get("application_acknowledgement_source")
        == payload["acknowledgement_source"]
        and record.get("application_acknowledgement_wall_ns")
        == payload["acknowledged_wall_ns"]
        and record.get("application_acknowledgement_authenticated")
        is verified.authenticated
        and record.get("application_acknowledgement_key_id")
        == verified.authentication_key_id
        and record.get("met_deadline") is verified.met_deadline
    )
    return bool(
        hmac.compare_digest(
            str(record.get("outcome_evidence_sha256", "")),
            verified.evidence_file_sha256,
        )
        and hmac.compare_digest(
            str(record.get("application_acknowledgement_sha256", "")),
            verified.acknowledgement_sha256,
        )
        and record.get("application_acknowledgement_id")
        == verified.payload["acknowledgement_id"]
        and identity_matches
        and outcome_matches
        and metadata_matches
    )


def derive_failure_stage(
    *,
    action: Mapping[str, Any],
    evidence: VerifiedOutcomeEvidence,
) -> str:
    """Choose the first conservative failure stage without a human label."""

    payload = evidence.payload
    if payload["no_action"]:
        return "installation"
    if payload["controller_failure"]:
        return "measurement"
    if not action.get("install_verified"):
        return "installation"
    if not action.get("first_packet_observed") or action.get("rollback_triggered"):
        return "switch"
    if not payload["observation_valid"]:
        return "observation"
    if (
        not payload["application_complete"]
        or payload["delivered_bytes"] < payload["required_bytes"]
    ):
        return "delivery"
    if payload["late_install"] or payload["completion_wall_ns"] > payload["deadline_wall_ns"]:
        return "deadline"
    return "none"
