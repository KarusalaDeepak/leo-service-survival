"""Fail-closed release controls for prospective field-extension execution.

The functions in this module implement the *mechanics* of prospective gate,
rejection, canary, final-test, and audit execution.  They do not manufacture
field evidence.  The repository's shipped protocol is deliberately left in
``DRAFT_NOT_REGISTERED`` state and all real execution paths require frozen,
hash-matched inputs.

The signed artifacts use HMAC-SHA256.  In deployment the signing key must be
stored outside the repository and made available only to the release operator.
HMAC provides integrity and shared-key authentication, not public
non-repudiation; the key identifier is recorded in every artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import hmac
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd

from .gate_freeze import (
    CONFIRMATORY_LIVE,
    CONTROLLED_FIXTURE,
    FAKE_DRIVER_TEST,
    TrialFreezeManifest,
    assert_gate_evidence_recomputed,
    load_verified_registry_lock,
)
from .prospective_trial import (
    ClosedLoopController,
    DecisionContext,
    ValidationReport,
    audit_decision_lifecycles,
    audit_hash_chain,
    canonical_json,
    load_ledger_schema,
    sha256_file,
    validate_campaign_register,
)


GATE_DECISIONS = ("AUTHORIZED", "REJECTED", "INSUFFICIENT_EVIDENCE")
ZERO_SHA256 = "0" * 64


def _report(rows: list[tuple[str, bool, str]]) -> ValidationReport:
    return ValidationReport(pd.DataFrame(rows, columns=("check", "passed", "details")))


def _canonical_without_authentication(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_sha256", "authentication"}
    }


def sign_artifact(
    payload: Mapping[str, Any],
    key: bytes,
    *,
    purpose: str,
) -> dict[str, Any]:
    """Hash and authenticate a release artifact with a purpose-bound HMAC."""

    if len(key) < 32:
        raise ValueError("release signing key must contain at least 32 bytes")
    if not purpose or "\n" in purpose:
        raise ValueError("artifact purpose must be a nonempty single line")
    body = _canonical_without_authentication(payload)
    digest = sha256(canonical_json(body).encode("utf-8")).hexdigest()
    message = f"{purpose}\n{digest}".encode("utf-8")
    signature = hmac.new(key, message, sha256).hexdigest()
    return {
        **body,
        "artifact_sha256": digest,
        "authentication": {
            "algorithm": "HMAC-SHA256",
            "purpose": purpose,
            "key_id": sha256(key).hexdigest()[:16],
            "signature": signature,
        },
    }


def verify_signed_artifact(
    payload: Mapping[str, Any],
    key: bytes,
    *,
    purpose: str,
) -> ValidationReport:
    """Verify the content hash, purpose binding, key id, and HMAC."""

    rows: list[tuple[str, bool, str]] = []
    authentication = payload.get("authentication")
    if not isinstance(authentication, Mapping):
        return _report([("signature.authentication_present", False, "missing")])
    body = _canonical_without_authentication(payload)
    expected_digest = sha256(canonical_json(body).encode("utf-8")).hexdigest()
    claimed_digest = str(payload.get("artifact_sha256", ""))
    rows.append(
        (
            "signature.content_hash",
            hmac.compare_digest(claimed_digest, expected_digest),
            f"claimed={claimed_digest}; expected={expected_digest}",
        )
    )
    expected_key_id = sha256(key).hexdigest()[:16]
    rows.append(
        (
            "signature.contract",
            authentication.get("algorithm") == "HMAC-SHA256"
            and authentication.get("purpose") == purpose
            and authentication.get("key_id") == expected_key_id,
            (
                f"algorithm={authentication.get('algorithm')}; "
                f"purpose={authentication.get('purpose')}; "
                f"key_id={authentication.get('key_id')}"
            ),
        )
    )
    expected_signature = hmac.new(
        key,
        f"{purpose}\n{expected_digest}".encode("utf-8"),
        sha256,
    ).hexdigest()
    supplied = str(authentication.get("signature", ""))
    rows.append(
        (
            "signature.hmac",
            hmac.compare_digest(supplied, expected_signature),
            "valid" if hmac.compare_digest(supplied, expected_signature) else "invalid",
        )
    )
    return _report(rows)


def load_signing_key(path: str | Path) -> bytes:
    """Load a binary/UTF-8 operator key and reject short or empty key files."""

    key = Path(path).read_bytes().strip()
    if len(key) < 32:
        raise ValueError("release signing key file must contain at least 32 bytes")
    return key


def resolve_itt_fallback_status(
    *,
    gate_decision: str,
    assigned_arm: str,
    controller_fallback_triggered: bool,
    controller_fallback_reason: str,
) -> tuple[bool, str]:
    """Separate a signed baseline assignment from fail-closed fallback.

    The ITT arm is preserved independently of the installed path.  A baseline
    block under an authorized randomized plan is an intentional control action,
    whereas a rejected gate or a candidate-arm safety intervention is fallback.
    """

    if assigned_arm not in {"reactive_baseline", "gated_candidate"}:
        raise ValueError(f"unsupported assigned arm: {assigned_arm}")
    if gate_decision not in GATE_DECISIONS:
        raise ValueError(f"unsupported gate decision: {gate_decision}")
    if gate_decision == "AUTHORIZED" and assigned_arm == "reactive_baseline":
        return False, ""
    return bool(controller_fallback_triggered), str(controller_fallback_reason)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_locked_gate_evidence(
    evidence: Mapping[str, Any],
    register: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    expected_evidence_status: str = "LOCKED_GATE_EVIDENCE",
) -> ValidationReport:
    """Validate one locked, role-separated evidence package for gate use."""

    rows: list[tuple[str, bool, str]] = []
    required = {
        "schema_version",
        "evidence_status",
        "protocol_id",
        "config_sha256",
        "model_sha256",
        "campaigns",
        "pooled_metrics",
    }
    missing = sorted(required.difference(evidence))
    rows.append(
        (
            "evidence.required_fields",
            not missing,
            "complete" if not missing else f"missing={missing}",
        )
    )
    if missing:
        return _report(rows)

    rows.append(
        (
            "evidence.frozen_status",
            evidence.get("evidence_status") == expected_evidence_status,
            str(evidence.get("evidence_status")),
        )
    )
    rows.append(
        (
            "evidence.protocol_binding",
            evidence.get("protocol_id") == config.get("protocol_id")
            and evidence.get("config_sha256") == config_sha256,
            (
                f"evidence_protocol={evidence.get('protocol_id')}; "
                f"config_protocol={config.get('protocol_id')}; "
                f"evidence_config_sha256={evidence.get('config_sha256')}"
            ),
        )
    )

    campaigns = evidence.get("campaigns")
    campaign_rows_valid = isinstance(campaigns, list) and bool(campaigns)
    required_campaign_fields = {
        "campaign_id",
        "acquisition_family_id",
        "paired_opportunity_groups",
        "raw_evidence_sha256",
    }
    if campaign_rows_valid:
        campaign_rows_valid = all(
            isinstance(item, Mapping)
            and required_campaign_fields.issubset(item)
            and isinstance(item.get("paired_opportunity_groups"), int)
            and item["paired_opportunity_groups"] >= 0
            and isinstance(item.get("raw_evidence_sha256"), str)
            and len(item["raw_evidence_sha256"]) == 64
            for item in campaigns
        )
    rows.append(
        (
            "evidence.campaign_records",
            campaign_rows_valid,
            f"records={len(campaigns) if isinstance(campaigns, list) else 0}",
        )
    )

    if campaign_rows_valid:
        supplied_pairs = {
            (str(item["campaign_id"]), str(item["acquisition_family_id"]))
            for item in campaigns
        }
        unique_supplied = len(supplied_pairs) == len(campaigns)
        registered_gate = register.loc[
            register["role"].eq("gate_evidence")
            & register["status"].eq("locked")
            & ~register["status"].eq("excluded"),
            ["campaign_id", "acquisition_family_id"],
        ]
        registered_pairs = set(map(tuple, registered_gate.astype(str).itertuples(index=False, name=None)))
        exact_match = supplied_pairs == registered_pairs
        rows.append(
            (
                "evidence.unique_campaigns",
                unique_supplied,
                f"unique={len(supplied_pairs)}; rows={len(campaigns)}",
            )
        )
        rows.append(
            (
                "evidence.gate_register_match",
                exact_match,
                (
                    f"supplied={len(supplied_pairs)}; registered={len(registered_pairs)}; "
                    f"missing={sorted(registered_pairs - supplied_pairs)}; "
                    f"extra={sorted(supplied_pairs - registered_pairs)}"
                ),
            )
        )

    gate_register = register.loc[register["role"].eq("gate_evidence")]
    model_hashes = set(gate_register["model_sha256"].astype(str))
    config_hashes = set(gate_register["config_sha256"].astype(str))
    rows.append(
        (
            "evidence.frozen_artifact_match",
            model_hashes == {str(evidence.get("model_sha256"))}
            and config_hashes == {config_sha256},
            f"register_model_hashes={sorted(model_hashes)}; register_config_hashes={sorted(config_hashes)}",
        )
    )

    metrics = evidence.get("pooled_metrics")
    metric_names = {
        "paired_opportunity_groups",
        "feature_coverage",
        "ood_fraction",
        "calibration_gap_upper",
        "candidate_minus_baseline_effect_lower",
        "effect_ci_half_width",
        "control_loop_p95_ms",
    }
    metric_ok = isinstance(metrics, Mapping) and metric_names.issubset(metrics)
    if metric_ok:
        metric_ok = all(_finite_number(metrics[name]) for name in metric_names)
    rows.append(
        (
            "evidence.pooled_metrics",
            metric_ok,
            "valid" if metric_ok else f"required={sorted(metric_names)}",
        )
    )
    if metric_ok:
        total_groups = sum(int(item["paired_opportunity_groups"]) for item in campaigns)
        rows.append(
            (
                "evidence.opportunity_group_reconciliation",
                int(metrics["paired_opportunity_groups"]) == total_groups,
                f"pooled={metrics['paired_opportunity_groups']}; campaign_sum={total_groups}",
            )
        )
        range_ok = (
            0 <= float(metrics["feature_coverage"]) <= 1
            and 0 <= float(metrics["ood_fraction"]) <= 1
            and float(metrics["calibration_gap_upper"]) >= 0
            and -1 <= float(metrics["candidate_minus_baseline_effect_lower"]) <= 1
            and 0 <= float(metrics["effect_ci_half_width"]) <= 1
            and float(metrics["control_loop_p95_ms"]) >= 0
        )
        rows.append(("evidence.metric_ranges", range_ok, str(dict(metrics))))
    return _report(rows)


def _gate_condition(
    name: str,
    observed: float,
    threshold: float,
    direction: str,
) -> dict[str, Any]:
    if direction == "at_least":
        passed = observed >= threshold
    elif direction == "at_most":
        passed = observed <= threshold
    else:
        raise ValueError(f"unsupported gate direction: {direction}")
    return {
        "condition": name,
        "observed": observed,
        "threshold": threshold,
        "direction": direction,
        "passed": bool(passed),
    }


def run_frozen_evidence_gate(
    *,
    config_path: str | Path,
    register_path: str | Path,
    evidence_path: str | Path,
    signing_key: bytes,
    created_wall_ns: int | None = None,
    execution_mode: str = CONFIRMATORY_LIVE,
    freeze_manifest_path: str | Path | None = None,
    opportunity_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the frozen gate and return a signed authorization/rejection report.

    Malformed, mutable, or role-leaking packages raise an error rather than
    producing an authorization.  Adequate evidence that fails any frozen
    performance condition produces ``REJECTED``.  A valid package below the
    frozen campaign/family/opportunity floors produces
    ``INSUFFICIENT_EVIDENCE``.
    """

    config_path = Path(config_path)
    register_path = Path(register_path)
    evidence_path = Path(evidence_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if execution_mode not in {CONFIRMATORY_LIVE, CONTROLLED_FIXTURE}:
        raise ValueError(f"unsupported evidence-gate execution mode: {execution_mode}")
    expected_evidence_status = (
        "LOCKED_GATE_EVIDENCE"
        if execution_mode == CONFIRMATORY_LIVE
        else FAKE_DRIVER_TEST
    )
    manifest: TrialFreezeManifest | None = None
    if execution_mode == CONFIRMATORY_LIVE:
        if freeze_manifest_path is None or opportunity_manifest_path is None:
            raise ValueError(
                "confirmatory_live requires --trial-freeze-manifest and "
                "--opportunity-manifest so metrics can be recomputed"
            )
    if freeze_manifest_path is not None or opportunity_manifest_path is not None:
        if freeze_manifest_path is None or opportunity_manifest_path is None:
            raise ValueError("freeze manifest and opportunity manifest must be supplied together")
        manifest = TrialFreezeManifest.from_dict(
            json.loads(Path(freeze_manifest_path).read_text(encoding="utf-8"))
        )
        lock, register = load_verified_registry_lock(
            manifest,
            signing_key=signing_key,
            expected_mode=execution_mode,
        )
        if sha256_file(config_path) != manifest.artifacts["trial_config"]["sha256"]:
            raise ValueError("gate config is not the trial config bound by the freeze manifest")
        if sha256_file(register_path) != manifest.artifacts["campaign_registry_lock"]["sha256"]:
            raise ValueError("gate register is not the registry lock bound by the freeze manifest")
        assert_gate_evidence_recomputed(
            evidence,
            freeze_manifest=manifest,
            opportunity_manifest_path=opportunity_manifest_path,
            signing_key=signing_key,
            execution_mode=execution_mode,
        )
    else:
        # Legacy CSV input is retained only for explicitly-labelled software
        # fixtures.  It is impossible to reach the live claim boundary through
        # this branch.
        if execution_mode != CONTROLLED_FIXTURE:
            raise ValueError("unverified CSV registers are forbidden in confirmatory_live")
        if evidence.get("evidence_status") != FAKE_DRIVER_TEST:
            raise ValueError("controlled_fixture evidence must be labelled FAKE_DRIVER_TEST")
        register = pd.read_csv(register_path, keep_default_na=False)
    if config.get("protocol_status") != "FROZEN_REGISTERED":
        raise ValueError("evidence gate requires protocol_status=FROZEN_REGISTERED")
    if config.get("field_trial_completed") is not False:
        raise ValueError("gate input must not assert a completed field trial")
    gate = config.get("gate")
    if not isinstance(gate, Mapping) or gate.get("all_conditions_must_pass") is not True:
        raise ValueError("frozen gate must require all conditions to pass")

    register_report = validate_campaign_register(
        register,
        require_locked=True,
        minimum_gate_campaigns=int(gate["minimum_independent_gate_campaigns"]),
        minimum_gate_families=int(gate["minimum_independent_gate_families"]),
    )
    hard_failures = register_report.checks.loc[
        ~register_report.checks["passed"]
        & ~register_report.checks["check"].eq("register.gate_eligibility_floor")
    ]
    if len(hard_failures):
        detail = hard_failures[["check", "details"]].to_dict(orient="records")
        raise ValueError(f"locked acquisition register is invalid: {detail}")

    config_digest = sha256_file(config_path)
    evidence_report = validate_locked_gate_evidence(
        evidence,
        register,
        config,
        config_sha256=config_digest,
        expected_evidence_status=expected_evidence_status,
    )
    evidence_report.raise_for_errors()

    gate_register = register.loc[register["role"].eq("gate_evidence")]
    campaign_count = int(gate_register["campaign_id"].nunique())
    family_count = int(gate_register["acquisition_family_id"].nunique())
    metrics = evidence["pooled_metrics"]
    eligibility = [
        _gate_condition(
            "independent_gate_campaigns",
            campaign_count,
            int(gate["minimum_independent_gate_campaigns"]),
            "at_least",
        ),
        _gate_condition(
            "independent_gate_families",
            family_count,
            int(gate["minimum_independent_gate_families"]),
            "at_least",
        ),
        _gate_condition(
            "paired_opportunity_groups",
            int(metrics["paired_opportunity_groups"]),
            int(gate["minimum_paired_opportunity_groups"]),
            "at_least",
        ),
    ]
    performance = [
        _gate_condition(
            "feature_coverage",
            float(metrics["feature_coverage"]),
            float(gate["minimum_feature_coverage"]),
            "at_least",
        ),
        _gate_condition(
            "ood_fraction",
            float(metrics["ood_fraction"]),
            float(gate["maximum_ood_fraction"]),
            "at_most",
        ),
        _gate_condition(
            "calibration_gap_upper",
            float(metrics["calibration_gap_upper"]),
            float(gate["maximum_calibration_gap_upper"]),
            "at_most",
        ),
        _gate_condition(
            "candidate_minus_baseline_effect_lower",
            float(metrics["candidate_minus_baseline_effect_lower"]),
            float(gate["minimum_candidate_minus_baseline_effect_lower"]),
            "at_least",
        ),
        _gate_condition(
            "effect_ci_half_width",
            float(metrics["effect_ci_half_width"]),
            float(gate["maximum_effect_ci_half_width"]),
            "at_most",
        ),
        _gate_condition(
            "control_loop_p95_ms",
            float(metrics["control_loop_p95_ms"]),
            float(gate["maximum_control_loop_p95_ms"]),
            "at_most",
        ),
    ]
    if not all(item["passed"] for item in eligibility):
        decision = "INSUFFICIENT_EVIDENCE"
    elif all(item["passed"] for item in performance):
        decision = "AUTHORIZED"
    else:
        decision = "REJECTED"

    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "frozen_evidence_gate_report",
        "created_wall_ns": time.time_ns() if created_wall_ns is None else int(created_wall_ns),
        "protocol_id": config["protocol_id"],
        "gate_profile_id": gate["profile_id"],
        "decision": decision,
        "candidate_authorized": decision == "AUTHORIZED",
        "trial_freeze_manifest_sha256": (
            manifest.manifest_sha256 if manifest is not None else ZERO_SHA256
        ),
        "frozen_artifacts": {
            "config_sha256": config_digest,
            "register_sha256": sha256_file(register_path),
            "evidence_sha256": sha256_file(evidence_path),
            "model_sha256": evidence["model_sha256"],
            "route_config_sha256": (
                manifest.artifacts["route_config"]["sha256"]
                if manifest is not None
                else ZERO_SHA256
            ),
            "ledger_schema_sha256": (
                manifest.artifacts["ledger_schema"]["sha256"]
                if manifest is not None
                else ZERO_SHA256
            ),
            "testbed_qualification_sha256": (
                manifest.artifacts["testbed_qualification"]["sha256"]
                if manifest is not None
                else ZERO_SHA256
            ),
            "campaign_registry_lock_sha256": (
                manifest.artifacts["campaign_registry_lock"]["sha256"]
                if manifest is not None
                else ZERO_SHA256
            ),
        },
        "independent_counts": {
            "gate_campaigns": campaign_count,
            "gate_families": family_count,
            "paired_opportunity_groups": int(metrics["paired_opportunity_groups"]),
        },
        "eligibility_checks": eligibility,
        "performance_checks": performance,
        "failed_conditions": [
            item["condition"] for item in eligibility + performance if not item["passed"]
        ],
        "fallback_action": gate["insufficient_or_rejected_action"],
        "execution_mode": execution_mode,
        "evidence_status": expected_evidence_status,
        "field_execution_status": (
            "GATE_RUN_ONLY"
            if execution_mode == CONFIRMATORY_LIVE
            else FAKE_DRIVER_TEST
        ),
        "claim_boundary": (
            "Authorization permits only the frozen canary. Rejection or insufficient "
            "evidence requires the reactive baseline. FAKE_DRIVER_TEST is software "
            "evidence only. This report alone does not establish deployment benefit "
            "or completion of the field trial."
        ),
    }
    return sign_artifact(payload, signing_key, purpose="O4_FROZEN_EVIDENCE_GATE_V1")


def verify_gate_report(report: Mapping[str, Any], key: bytes) -> ValidationReport:
    signature = verify_signed_artifact(
        report, key, purpose="O4_FROZEN_EVIDENCE_GATE_V1"
    )
    rows = signature.checks.to_records(index=False).tolist()
    decision = report.get("decision")
    checks = list(report.get("eligibility_checks", ())) + list(
        report.get("performance_checks", ())
    )
    expected_authorization = decision == "AUTHORIZED" and bool(checks) and all(
        item.get("passed") is True for item in checks
    )
    rows.append(("gate.valid_decision", decision in GATE_DECISIONS, str(decision)))
    rows.append(
        (
            "gate.authorization_consistency",
            report.get("candidate_authorized") is expected_authorization,
            f"candidate_authorized={report.get('candidate_authorized')}",
        )
    )
    return _report(rows)


@dataclass(frozen=True)
class RollbackDrillResult:
    isolated_namespace_id: str
    fault_injected: bool
    rollback_acknowledged: bool
    baseline_verified: bool
    first_packet_observed: bool
    details: str


class IsolatedRollbackDrillDriver(Protocol):
    def run_isolated_rollback_drill(
        self,
        *,
        isolated_namespace_id: str,
        destination: str,
        baseline_path_id: str,
    ) -> RollbackDrillResult: ...


def run_rejection_branch(
    *,
    gate_report: Mapping[str, Any],
    signing_key: bytes,
    controller: ClosedLoopController,
    context: DecisionContext,
    drill_driver: IsolatedRollbackDrillDriver | None,
    isolated_namespace_id: str | None,
    created_wall_ns: int | None = None,
) -> dict[str, Any]:
    """Verify baseline retention and an isolated rollback drill after rejection."""

    verify_gate_report(gate_report, signing_key).raise_for_errors()
    if gate_report["decision"] == "AUTHORIZED":
        raise ValueError("rejection branch cannot run for an authorized candidate")
    safe_context = replace(
        context,
        gate_state="rejected"
        if gate_report["decision"] == "REJECTED"
        else "insufficient_evidence",
        gate_reason=str(gate_report["decision"]),
        guard_state="fallback",
        monitor_state="not_mature",
        canary_state="baseline",
    )
    closed_loop = controller.execute(safe_context)
    baseline_verified = bool(
        closed_loop.intended_path_id == context.baseline_path_id
        and closed_loop.install_acknowledged
        and closed_loop.install_verified
        and closed_loop.first_packet_observed
        and closed_loop.final_fib_path_id == context.baseline_path_id
    )
    if drill_driver is None or not isolated_namespace_id:
        drill = RollbackDrillResult(
            isolated_namespace_id="NOT_RUN",
            fault_injected=False,
            rollback_acknowledged=False,
            baseline_verified=False,
            first_packet_observed=False,
            details="isolated rollback drill not executed",
        )
    else:
        drill = drill_driver.run_isolated_rollback_drill(
            isolated_namespace_id=isolated_namespace_id,
            destination=context.destination,
            baseline_path_id=context.baseline_path_id,
        )
    drill_verified = bool(
        drill.isolated_namespace_id == isolated_namespace_id
        and drill.fault_injected
        and drill.rollback_acknowledged
        and drill.baseline_verified
        and drill.first_packet_observed
    )
    status = (
        "VERIFIED"
        if baseline_verified and drill_verified
        else "PARTIAL_NOT_RUN"
        if baseline_verified and drill_driver is None
        else "FAILED"
    )
    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "rejection_branch_report",
        "created_wall_ns": time.time_ns() if created_wall_ns is None else int(created_wall_ns),
        "gate_report_sha256": gate_report["artifact_sha256"],
        "gate_decision": gate_report["decision"],
        "status": status,
        "production_baseline_verification": {
            "passed": baseline_verified,
            "intended_path_id": closed_loop.intended_path_id,
            "installed_path_id": closed_loop.installed_path_id,
            "final_fib_path_id": closed_loop.final_fib_path_id,
            "install_acknowledged": closed_loop.install_acknowledged,
            "install_verified": closed_loop.install_verified,
            "first_packet_observed": closed_loop.first_packet_observed,
            "total_loop_ms": closed_loop.total_loop_ms,
        },
        "isolated_rollback_drill": asdict(drill),
        "candidate_installed_in_production": False,
        "benefit_claim_supported": False,
        "claim_boundary": (
            "A VERIFIED rejection branch demonstrates fail-closed baseline retention "
            "and rollback readiness only; it cannot support a candidate-benefit claim."
        ),
    }
    return sign_artifact(payload, signing_key, purpose="O4_REJECTION_BRANCH_V1")


def build_fixed_canary_plan(
    *,
    gate_report: Mapping[str, Any],
    signing_key: bytes,
    block_ids: Sequence[str],
    allocation_seed: str,
    allocation_fraction: float,
    created_wall_ns: int | None = None,
) -> dict[str, Any]:
    """Create a deterministic fixed-allocation canary plan after authorization."""

    verify_gate_report(gate_report, signing_key).raise_for_errors()
    if gate_report["decision"] != "AUTHORIZED":
        raise ValueError("a canary plan requires an AUTHORIZED gate report")
    blocks = list(block_ids)
    if not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("canary block identifiers must be nonempty and unique")
    if not 0 < allocation_fraction <= 0.25:
        raise ValueError("canary allocation fraction must be in (0, 0.25]")
    if len(allocation_seed) < 16:
        raise ValueError("allocation seed must contain at least 16 characters")
    ranked = sorted(
        blocks,
        key=lambda block: sha256(f"{allocation_seed}\n{block}".encode("utf-8")).hexdigest(),
    )
    candidate_count = max(1, math.floor(len(blocks) * allocation_fraction))
    candidates = set(ranked[:candidate_count])
    assignments = [
        {
            "block_id": block,
            "assigned_arm": "gated_candidate" if block in candidates else "reactive_baseline",
        }
        for block in blocks
    ]
    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "fixed_canary_plan",
        "created_wall_ns": time.time_ns() if created_wall_ns is None else int(created_wall_ns),
        "gate_report_sha256": gate_report["artifact_sha256"],
        "frozen_artifacts": dict(gate_report["frozen_artifacts"]),
        "allocation_seed_commitment": sha256(allocation_seed.encode("utf-8")).hexdigest(),
        "allocation_fraction": float(allocation_fraction),
        "total_blocks": len(blocks),
        "candidate_blocks": candidate_count,
        "assignments": assignments,
        "retuning_permitted": False,
        "status": "PLANNED_NOT_EXECUTED",
        "claim_boundary": "This plan fixes allocation; it is not evidence that canary blocks ran.",
    }
    return sign_artifact(payload, signing_key, purpose="O4_FIXED_CANARY_PLAN_V1")


def evaluate_canary_checkpoint(
    *,
    gate_report: Mapping[str, Any],
    canary_plan: Mapping[str, Any],
    event_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    signing_key: bytes,
    created_wall_ns: int | None = None,
) -> dict[str, Any]:
    """Apply frozen stop rules and require rollback on any safety violation."""

    verify_gate_report(gate_report, signing_key).raise_for_errors()
    verify_signed_artifact(
        canary_plan, signing_key, purpose="O4_FIXED_CANARY_PLAN_V1"
    ).raise_for_errors()
    if gate_report["decision"] != "AUTHORIZED":
        raise ValueError("canary evaluation requires gate authorization")
    if canary_plan.get("gate_report_sha256") != gate_report.get("artifact_sha256"):
        raise ValueError("canary plan is not bound to the supplied gate report")
    if canary_plan.get("frozen_artifacts") != gate_report.get("frozen_artifacts"):
        raise ValueError("frozen artifacts changed between gate and canary")

    assignments = {
        item["block_id"]: item["assigned_arm"]
        for item in canary_plan.get("assignments", ())
    }
    events = list(event_records)
    required_event_fields = {
        "block_id",
        "executed_arm",
        "install_acknowledged",
        "install_verified",
        "installed_path_mismatch",
        "baseline_path_id",
        "intended_path_id",
        "first_packet_observed",
        "first_packet_path_id",
        "first_packet_evidence_sha256",
        "rollback_triggered",
        "rollback_acknowledged",
        "rollback_install_verified",
        "rollback_first_packet_observed",
        "rollback_first_packet_path_id",
        "rollback_ack_evidence_sha256",
        "rollback_first_packet_evidence_sha256",
        "rollback_verified",
        "emergency_stop",
        "unauthorized_candidate_action",
        "active_shift_alarm",
        "endpoint_safety_boundary_crossed",
        "clock_or_logging_failure",
        "total_loop_ms",
        "rollback_total_ms",
        "action_total_ms",
    }
    boolean_event_fields = required_event_fields - {
        "block_id",
        "executed_arm",
        "total_loop_ms",
        "rollback_total_ms",
        "action_total_ms",
        "baseline_path_id",
        "intended_path_id",
        "first_packet_path_id",
        "first_packet_evidence_sha256",
        "rollback_first_packet_path_id",
        "rollback_ack_evidence_sha256",
        "rollback_first_packet_evidence_sha256",
    }
    event_errors: list[str] = []
    for index, item in enumerate(events):
        missing = sorted(required_event_fields.difference(item))
        if missing:
            event_errors.append(f"event {index}: missing={missing}")
            continue
        invalid_booleans = sorted(
            field for field in boolean_event_fields if not isinstance(item[field], bool)
        )
        if invalid_booleans:
            event_errors.append(f"event {index}: non_boolean={invalid_booleans}")
        if not isinstance(item["block_id"], str) or not item["block_id"]:
            event_errors.append(f"event {index}: invalid block_id")
        if item["executed_arm"] not in {"reactive_baseline", "gated_candidate"}:
            event_errors.append(f"event {index}: invalid executed_arm")
        timing_values = (
            item["total_loop_ms"],
            item["rollback_total_ms"],
            item["action_total_ms"],
        )
        if any(not _finite_number(value) or float(value) < 0 for value in timing_values):
            event_errors.append(f"event {index}: invalid loop or rollback timing")
        elif abs(
            float(item["total_loop_ms"])
            + float(item["rollback_total_ms"])
            - float(item["action_total_ms"])
        ) > 1.0:
            event_errors.append(f"event {index}: action timing does not reconcile")
        proof_fields = (
            "first_packet_evidence_sha256",
            "rollback_ack_evidence_sha256",
            "rollback_first_packet_evidence_sha256",
        )
        if any(
            not isinstance(item[field], str)
            or len(item[field]) != 64
            or any(character not in "0123456789abcdef" for character in item[field])
            for field in proof_fields
        ):
            event_errors.append(f"event {index}: invalid observer evidence hash")
        if item["install_verified"] and not item["install_acknowledged"]:
            event_errors.append(f"event {index}: verified install lacks acknowledgement")
        if item["installed_path_mismatch"] and item["install_verified"]:
            event_errors.append(f"event {index}: path mismatch contradicts verified install")
        if item["first_packet_observed"] and not item["install_verified"]:
            event_errors.append(f"event {index}: first packet observed without verified install")
        if item["first_packet_observed"] and item["first_packet_path_id"] != item["intended_path_id"]:
            event_errors.append(f"event {index}: first packet path differs from intended path")
        if item["rollback_verified"] and not item["rollback_triggered"]:
            event_errors.append(f"event {index}: rollback verified without rollback trigger")
        if item["rollback_triggered"] and item["rollback_verified"]:
            rollback_proof = bool(
                item["rollback_acknowledged"]
                and item["rollback_install_verified"]
                and item["rollback_first_packet_observed"]
                and item["rollback_first_packet_path_id"] == item["baseline_path_id"]
            )
            if not rollback_proof:
                event_errors.append(f"event {index}: rollback verified without baseline acknowledgement and first-packet proof")
        if not item["rollback_triggered"] and any(
            (
                item["rollback_acknowledged"],
                item["rollback_install_verified"],
                item["rollback_first_packet_observed"],
                float(item["rollback_total_ms"]) != 0.0,
            )
        ):
            event_errors.append(f"event {index}: rollback evidence present without rollback trigger")
    if event_errors:
        raise ValueError("invalid canary event contract: " + "; ".join(event_errors[:10]))
    event_ids = [str(item.get("block_id")) for item in events]
    if len(events) != len(assignments) or set(event_ids) != set(assignments):
        raise ValueError("canary events must contain exactly one record for every planned block")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("duplicate canary block event")

    mismatch = 0
    install_failures = 0
    candidate_events = 0
    stop_counts = {
        "emergency_stop": 0,
        "unauthorized_candidate_action": 0,
        "install_not_acknowledged": 0,
        "installed_path_mismatch": 0,
        "active_shift_alarm": 0,
        "endpoint_safety_boundary_crossed": 0,
        "clock_or_logging_failure": 0,
        "first_packet_timeout": 0,
        "control_loop_budget_exceeded": 0,
    }
    maximum_loop_ms = float(config["rollback"]["maximum_single_decision_loop_ms"])
    rollback_required_events = 0
    rollback_verified_events = 0
    for item in events:
        block_id = str(item["block_id"])
        assigned = assignments[block_id]
        observed = item.get("executed_arm")
        mismatch += int(observed != assigned)
        if assigned == "gated_candidate":
            candidate_events += 1
            install_failures += int(
                not item.get("install_acknowledged")
                or not item.get("install_verified")
            )
            stop_counts["first_packet_timeout"] += int(
                not item.get("first_packet_observed")
            )
        per_event_rollback_trigger = bool(
            item["emergency_stop"]
            or not item["install_acknowledged"]
            or item["installed_path_mismatch"]
            or not item["first_packet_observed"]
            or item["active_shift_alarm"]
            or item["endpoint_safety_boundary_crossed"]
            or item["clock_or_logging_failure"]
            or float(item["total_loop_ms"]) > maximum_loop_ms
        )
        rollback_required_events += int(per_event_rollback_trigger)
        rollback_verified_events += int(
            per_event_rollback_trigger
            and item["rollback_triggered"]
            and item["rollback_verified"]
        )
        stop_counts["emergency_stop"] += int(item["emergency_stop"])
        stop_counts["unauthorized_candidate_action"] += int(
            item["unauthorized_candidate_action"]
        )
        stop_counts["install_not_acknowledged"] += int(not item["install_acknowledged"])
        stop_counts["installed_path_mismatch"] += int(item["installed_path_mismatch"])
        stop_counts["active_shift_alarm"] += int(item["active_shift_alarm"])
        stop_counts["endpoint_safety_boundary_crossed"] += int(
            item["endpoint_safety_boundary_crossed"]
        )
        stop_counts["clock_or_logging_failure"] += int(
            item["clock_or_logging_failure"]
        )
        loop_ms = item["total_loop_ms"]
        stop_counts["control_loop_budget_exceeded"] += int(
            not _finite_number(loop_ms) or float(loop_ms) > maximum_loop_ms
        )
    install_failure_fraction = install_failures / candidate_events if candidate_events else 1.0
    install_limit = float(config["rollback"]["maximum_install_failure_fraction"])
    triggered = [name for name, count in stop_counts.items() if count]
    if mismatch:
        triggered.append("allocation_or_execution_mismatch")
    if install_failure_fraction > install_limit:
        triggered.append("install_failure_fraction")
    rollback_complete = rollback_required_events == rollback_verified_events
    if triggered:
        status = (
            "STOPPED_ROLLBACK_VERIFIED"
            if rollback_complete
            else "STOPPED_ROLLBACK_UNVERIFIED"
        )
    else:
        status = "CANARY_CHECKPOINT_PASSED"
    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "canary_checkpoint_report",
        "created_wall_ns": time.time_ns() if created_wall_ns is None else int(created_wall_ns),
        "gate_report_sha256": gate_report["artifact_sha256"],
        "canary_plan_sha256": canary_plan["artifact_sha256"],
        "frozen_artifacts": dict(gate_report["frozen_artifacts"]),
        "status": status,
        "blocks_evaluated": len(events),
        "candidate_blocks_evaluated": candidate_events,
        "allocation_mismatches": mismatch,
        "install_failure_fraction": install_failure_fraction,
        "install_failure_limit": install_limit,
        "stop_trigger_counts": stop_counts,
        "triggered_stop_rules": sorted(set(triggered)),
        "automatic_rollback_required": bool(triggered),
        "rollback_required_events": rollback_required_events,
        "rollback_verified_events": rollback_verified_events,
        "automatic_rollback_completed": rollback_complete if triggered else True,
        "untouched_final_test_may_open": not triggered,
        "retuning_performed": False,
        "claim_boundary": (
            "A passed canary checkpoint permits a separately signed final-test unlock; "
            "it does not establish benefit. A stopped checkpoint mandates rollback."
        ),
    }
    return sign_artifact(payload, signing_key, purpose="O4_CANARY_CHECKPOINT_V1")


def unlock_untouched_final_test(
    *,
    gate_report: Mapping[str, Any],
    canary_plan: Mapping[str, Any],
    checkpoint_report: Mapping[str, Any],
    register_path: str | Path,
    signing_key: bytes,
    checkpoint_approval_id: str,
    no_retuning_attestation: bool,
    created_wall_ns: int | None = None,
) -> dict[str, Any]:
    """Issue a hash-bound unlock only after a clean, frozen canary checkpoint."""

    verify_gate_report(gate_report, signing_key).raise_for_errors()
    verify_signed_artifact(
        canary_plan, signing_key, purpose="O4_FIXED_CANARY_PLAN_V1"
    ).raise_for_errors()
    verify_signed_artifact(
        checkpoint_report, signing_key, purpose="O4_CANARY_CHECKPOINT_V1"
    ).raise_for_errors()
    if checkpoint_report.get("status") != "CANARY_CHECKPOINT_PASSED":
        raise ValueError("final test remains locked after a stopped or incomplete canary")
    if checkpoint_report.get("automatic_rollback_required") is not False:
        raise ValueError("final test remains locked while rollback is required")
    if not checkpoint_approval_id.strip():
        raise ValueError("registered checkpoint approval identifier is required")
    if no_retuning_attestation is not True:
        raise ValueError("no-retuning attestation is required")
    if not (
        canary_plan.get("gate_report_sha256") == gate_report.get("artifact_sha256")
        and checkpoint_report.get("gate_report_sha256") == gate_report.get("artifact_sha256")
        and checkpoint_report.get("canary_plan_sha256") == canary_plan.get("artifact_sha256")
    ):
        raise ValueError("gate, canary plan, and checkpoint hashes do not form one chain")
    if not (
        gate_report.get("frozen_artifacts")
        == canary_plan.get("frozen_artifacts")
        == checkpoint_report.get("frozen_artifacts")
    ):
        raise ValueError("frozen artifact hashes changed before final-test unlock")

    register_path = Path(register_path)
    register = pd.read_csv(register_path, keep_default_na=False)
    report = validate_campaign_register(register, require_locked=True)
    report.raise_for_errors()
    final_rows = register.loc[register["role"].eq("final_test")]
    final_ready = bool(len(final_rows)) and bool(final_rows["status"].eq("locked").all())
    if not final_ready:
        raise ValueError("no locked, role-separated final-test family is registered")
    frozen = gate_report["frozen_artifacts"]
    if set(final_rows["config_sha256"].astype(str)) != {frozen["config_sha256"]}:
        raise ValueError("final-test configuration hash differs from gate configuration")
    if set(final_rows["model_sha256"].astype(str)) != {frozen["model_sha256"]}:
        raise ValueError("final-test model hash differs from gate model")

    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "untouched_final_test_unlock",
        "created_wall_ns": time.time_ns() if created_wall_ns is None else int(created_wall_ns),
        "gate_report_sha256": gate_report["artifact_sha256"],
        "canary_plan_sha256": canary_plan["artifact_sha256"],
        "checkpoint_report_sha256": checkpoint_report["artifact_sha256"],
        "register_sha256_at_unlock": sha256_file(register_path),
        "final_test_campaign_ids": sorted(final_rows["campaign_id"].astype(str)),
        "final_test_family_ids": sorted(final_rows["acquisition_family_id"].astype(str).unique()),
        "checkpoint_approval_id": checkpoint_approval_id,
        "no_retuning_attestation": True,
        "frozen_artifacts": dict(frozen),
        "status": "UNTOUCHED_FINAL_TEST_UNLOCKED",
        "claim_boundary": (
            "This token authorizes collection under frozen artifacts. It is not evidence "
            "that the final test ran or that the candidate improved outcomes."
        ),
    }
    return sign_artifact(payload, signing_key, purpose="O4_FINAL_TEST_UNLOCK_V1")


def build_fixed_final_test_plan(
    *,
    gate_report: Mapping[str, Any],
    final_test_unlock: Mapping[str, Any],
    signing_key: bytes,
    analysis_spec_path: str | Path,
    block_ids: Sequence[str],
    allocation_seed: str,
    candidate_allocation_fraction: float,
    created_wall_ns: int | None = None,
) -> dict[str, Any]:
    """Create the signed final-test ITT allocation after the unlock token exists.

    The frozen analysis specification is content-bound into the allocation plan.
    This prevents an analyst from changing the endpoint, resampling unit, gain
    threshold, or minimum sample size after final-test outcomes are observed.
    """

    verify_gate_report(gate_report, signing_key).raise_for_errors()
    verify_signed_artifact(
        final_test_unlock, signing_key, purpose="O4_FINAL_TEST_UNLOCK_V1"
    ).raise_for_errors()
    if gate_report.get("decision") != "AUTHORIZED":
        raise ValueError("final-test allocation requires an authorized gate")
    if final_test_unlock.get("status") != "UNTOUCHED_FINAL_TEST_UNLOCKED":
        raise ValueError("final-test allocation requires an unlocked final test")
    if final_test_unlock.get("gate_report_sha256") != gate_report.get("artifact_sha256"):
        raise ValueError("final-test unlock is not bound to the supplied gate report")
    if final_test_unlock.get("frozen_artifacts") != gate_report.get("frozen_artifacts"):
        raise ValueError("frozen artifacts changed before final-test allocation")
    analysis_spec_path = Path(analysis_spec_path)
    analysis_spec = json.loads(analysis_spec_path.read_text(encoding="utf-8"))
    if analysis_spec.get("status") != "FROZEN_ANALYSIS_SPEC":
        raise ValueError("final-test allocation requires a frozen analysis specification")
    frozen = gate_report["frozen_artifacts"]
    if (
        analysis_spec.get("config_sha256") != frozen.get("config_sha256")
        or analysis_spec.get("model_sha256") != frozen.get("model_sha256")
    ):
        raise ValueError("analysis specification is bound to different frozen artifacts")
    blocks = list(block_ids)
    if not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("final-test block identifiers must be nonempty and unique")
    if not 0 < candidate_allocation_fraction < 1:
        raise ValueError("final-test candidate allocation fraction must be in (0, 1)")
    if len(allocation_seed) < 16:
        raise ValueError("allocation seed must contain at least 16 characters")
    ranked = sorted(
        blocks,
        key=lambda block: sha256(
            f"final-test\n{allocation_seed}\n{block}".encode("utf-8")
        ).hexdigest(),
    )
    candidate_count = min(
        len(blocks) - 1,
        max(1, round(len(blocks) * candidate_allocation_fraction)),
    )
    candidates = set(ranked[:candidate_count])
    assignments = [
        {
            "block_id": block,
            "assigned_arm": "gated_candidate" if block in candidates else "reactive_baseline",
        }
        for block in blocks
    ]
    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "fixed_final_test_plan",
        "created_wall_ns": time.time_ns() if created_wall_ns is None else int(created_wall_ns),
        "release_phase": "final_test",
        "gate_report_sha256": gate_report["artifact_sha256"],
        "final_test_unlock_sha256": final_test_unlock["artifact_sha256"],
        "analysis_spec_sha256": sha256_file(analysis_spec_path),
        "frozen_artifacts": dict(gate_report["frozen_artifacts"]),
        "allocation_seed_commitment": sha256(allocation_seed.encode("utf-8")).hexdigest(),
        "candidate_allocation_fraction": float(candidate_allocation_fraction),
        "total_blocks": len(blocks),
        "candidate_blocks": candidate_count,
        "baseline_blocks": len(blocks) - candidate_count,
        "assignments": assignments,
        "retuning_permitted": False,
        "status": "PLANNED_NOT_EXECUTED",
        "claim_boundary": (
            "This signed plan fixes final-test ITT assignment after unlock. It is not "
            "evidence that any assigned action or outcome occurred."
        ),
    }
    return sign_artifact(payload, signing_key, purpose="O4_FIXED_FINAL_TEST_PLAN_V1")


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _timing_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if _finite_number(value)]
    return {
        "n": len(finite),
        "mean_ms": statistics.fmean(finite) if finite else None,
        "p50_ms": _percentile(finite, 0.50),
        "p95_ms": _percentile(finite, 0.95),
        "max_ms": max(finite) if finite else None,
    }


def build_end_to_end_audit_report(
    *,
    ledger_path: str | Path,
    schema_path: str | Path,
    register_path: str | Path,
    gate_report: Mapping[str, Any],
    signing_key: bytes,
    checkpoint_report: Mapping[str, Any] | None = None,
    final_test_unlock: Mapping[str, Any] | None = None,
    final_test_plan: Mapping[str, Any] | None = None,
    created_wall_ns: int | None = None,
) -> dict[str, Any]:
    """Reconcile intended actions, installed state, timing, and mature outcomes."""

    verify_gate_report(gate_report, signing_key).raise_for_errors()
    if checkpoint_report is not None:
        verify_signed_artifact(
            checkpoint_report, signing_key, purpose="O4_CANARY_CHECKPOINT_V1"
        ).raise_for_errors()
    if final_test_unlock is not None:
        verify_signed_artifact(
            final_test_unlock, signing_key, purpose="O4_FINAL_TEST_UNLOCK_V1"
        ).raise_for_errors()
    if final_test_plan is not None:
        verify_signed_artifact(
            final_test_plan, signing_key, purpose="O4_FIXED_FINAL_TEST_PLAN_V1"
        ).raise_for_errors()
    schema = load_ledger_schema(schema_path)
    chain = audit_hash_chain(ledger_path, schema)
    lifecycle = audit_decision_lifecycles(
        ledger_path, schema, require_mature_outcomes=True
    )
    analysis_lifecycle = audit_decision_lifecycles(
        ledger_path, schema, require_mature_outcomes=False
    )
    register = pd.read_csv(register_path, keep_default_na=False)
    register_report = validate_campaign_register(register, require_locked=True)

    records = [
        json.loads(line)
        for line in Path(ledger_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(records)
    actions = frame.loc[frame["record_type"].eq("action")].copy()
    outcomes = frame.loc[frame["record_type"].eq("outcome")].copy()
    linked = actions.merge(
        outcomes,
        on="decision_id",
        how="left",
        suffixes=("_action", "_outcome"),
    )
    role_by_campaign = dict(
        zip(register["campaign_id"].astype(str), register["role"].astype(str))
    )
    actions["registered_role"] = actions["campaign_id"].astype(str).map(role_by_campaign)
    discrepancies = {
        "register_changed_after_unlock": int(
            final_test_unlock is not None
            and sha256_file(register_path)
            != final_test_unlock.get("register_sha256_at_unlock")
        ),
        "unregistered_campaign": int(actions["registered_role"].isna().sum()),
        "install_not_acknowledged": int((~actions["install_acknowledged"].astype(bool)).sum()),
        "intended_installed_mismatch": int(
            (
                actions["install_verified"].astype(bool)
                & actions["intended_path_id"].ne(actions["installed_path_id"])
            ).sum()
        ),
        "installed_final_fib_mismatch_without_rollback": int(
            (
                ~actions["rollback_triggered"].astype(bool)
                & actions["installed_path_id"].ne(actions["final_fib_path_id"])
            ).sum()
        ),
        "first_packet_missing": int((~actions["first_packet_observed"].astype(bool)).sum()),
        "first_packet_path_mismatch": int(
            (
                actions["first_packet_observed"].astype(bool)
                & actions["first_packet_path_id"].ne(actions["intended_path_id"])
            ).sum()
        ),
        "rollback_unverified": int(
            (
                actions["rollback_triggered"].astype(bool)
                & ~actions["rollback_verified"].astype(bool)
            ).sum()
        ),
        "missing_mature_outcome": int(linked["record_type_outcome"].isna().sum()),
    }
    action_timing = {
        name: _timing_summary(actions[name].tolist())
        for name in (
            "collect_ms",
            "feature_ms",
            "decision_ms",
            "disseminate_ms",
            "install_ms",
            "switch_ms",
            "total_loop_ms",
        )
    }
    success_count = int(outcomes["conservative_on_time_completion"].astype(bool).sum())
    outcome_count = int(len(outcomes))
    final_campaigns = set(
        register.loc[register["role"].eq("final_test"), "campaign_id"].astype(str)
    )
    final_actions = actions.loc[actions["campaign_id"].astype(str).isin(final_campaigns)]
    final_outcomes = outcomes.loc[outcomes["campaign_id"].astype(str).isin(final_campaigns)]
    gate_decision = gate_report["decision"]
    checkpoint_passed = bool(
        checkpoint_report is not None
        and checkpoint_report.get("status") == "CANARY_CHECKPOINT_PASSED"
    )
    unlock_valid = bool(
        final_test_unlock is not None
        and final_test_unlock.get("status") == "UNTOUCHED_FINAL_TEST_UNLOCKED"
        and final_test_unlock.get("gate_report_sha256") == gate_report.get("artifact_sha256")
    )
    final_plan_valid = bool(
        final_test_plan is not None
        and final_test_unlock is not None
        and final_test_plan.get("gate_report_sha256") == gate_report.get("artifact_sha256")
        and final_test_plan.get("final_test_unlock_sha256")
        == final_test_unlock.get("artifact_sha256")
        and final_test_plan.get("frozen_artifacts") == gate_report.get("frozen_artifacts")
    )
    if len(final_actions) and final_plan_valid:
        signed_assignments = {
            item["block_id"]: item["assigned_arm"]
            for item in final_test_plan.get("assignments", ())
        }
        observed_assignments = dict(
            zip(final_actions["block_id"].astype(str), final_actions["policy_arm"].astype(str))
        )
        discrepancies["final_test_assignment_mismatch"] = sum(
            signed_assignments.get(block) != arm
            for block, arm in observed_assignments.items()
        ) + sum(block not in observed_assignments for block in signed_assignments)
        discrepancies["final_test_release_hash_mismatch"] = int(
            not final_actions["gate_report_sha256"].eq(gate_report["artifact_sha256"]).all()
            or not final_actions["assignment_plan_sha256"].eq(
                final_test_plan["artifact_sha256"]
            ).all()
        )
    elif len(final_actions):
        discrepancies["final_test_assignment_mismatch"] = int(len(final_actions))
        discrepancies["final_test_release_hash_mismatch"] = int(len(final_actions))
    integrity_passed = bool(
        chain.passed
        and lifecycle.passed
        and register_report.passed
        and not any(discrepancies.values())
    )
    analysis_blocking_discrepancies = {
        key: value
        for key, value in discrepancies.items()
        if key != "missing_mature_outcome"
    }
    analysis_integrity_passed = bool(
        chain.passed
        and analysis_lifecycle.passed
        and register_report.passed
        and not any(analysis_blocking_discrepancies.values())
    )
    if gate_decision != "AUTHORIZED":
        claim_level = "FAIL_CLOSED_SAFETY_ONLY"
    elif not checkpoint_passed:
        claim_level = "AUTHORIZED_BUT_CANARY_NOT_PASSED"
    elif not unlock_valid or not final_plan_valid or len(final_actions) == 0 or len(final_outcomes) == 0:
        claim_level = "CANARY_SAFETY_ONLY"
    elif analysis_integrity_passed:
        claim_level = "FINAL_TEST_DATA_READY_FOR_PRESPECIFIED_ANALYSIS"
    else:
        claim_level = "FINAL_TEST_AUDIT_FAILED"

    last_record_hash = records[-1].get("record_sha256") if records else None
    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "end_to_end_trial_audit",
        "created_wall_ns": time.time_ns() if created_wall_ns is None else int(created_wall_ns),
        "gate_report_sha256": gate_report["artifact_sha256"],
        "checkpoint_report_sha256": (
            checkpoint_report.get("artifact_sha256") if checkpoint_report else None
        ),
        "final_test_unlock_sha256": (
            final_test_unlock.get("artifact_sha256") if final_test_unlock else None
        ),
        "final_test_plan_sha256": (
            final_test_plan.get("artifact_sha256") if final_test_plan else None
        ),
        "ledger_sha256": sha256_file(ledger_path),
        "ledger_schema_sha256": sha256_file(schema_path),
        "ledger_last_record_sha256": last_record_hash,
        "register_sha256": sha256_file(register_path),
        "hash_chain_passed": chain.passed,
        "lifecycle_audit_passed": lifecycle.passed,
        "analysis_lifecycle_audit_passed": analysis_lifecycle.passed,
        "register_audit_passed": register_report.passed,
        "integrity_passed": integrity_passed,
        "analysis_integrity_passed": analysis_integrity_passed,
        "counts": {
            "actions": int(len(actions)),
            "outcomes": outcome_count,
            "conservative_on_time_completions": success_count,
            "rollbacks": int(actions["rollback_triggered"].astype(bool).sum()),
            "fallback_actions": int(actions["fallback_triggered"].astype(bool).sum()),
            "candidate_actions": int(actions["policy_arm"].eq("gated_candidate").sum()),
            "final_test_actions": int(len(final_actions)),
            "final_test_outcomes": int(len(final_outcomes)),
        },
        "conservative_completion_fraction": (
            success_count / outcome_count if outcome_count else None
        ),
        "action_reconciliation_discrepancies": discrepancies,
        "timing": action_timing,
        "switching_interruption": action_timing["switch_ms"],
        "claim_level": claim_level,
        "benefit_claim_supported": False,
        "claim_boundary": (
            "This audit establishes trace-to-action accountability and identifies "
            "whether frozen final-test data are ready for the prespecified statistical "
            "analysis. It never converts operational integrity into a benefit claim; "
            "the registered effect analysis and uncertainty bounds remain required."
        ),
    }
    return sign_artifact(payload, signing_key, purpose="O4_END_TO_END_AUDIT_V1")


def run_prespecified_final_test_analysis(
    *,
    ledger_path: str | Path,
    schema_path: str | Path,
    register_path: str | Path,
    analysis_spec_path: str | Path,
    gate_report: Mapping[str, Any],
    final_test_unlock: Mapping[str, Any],
    final_test_plan: Mapping[str, Any],
    audit_report: Mapping[str, Any],
    signing_key: bytes,
    created_wall_ns: int | None = None,
) -> dict[str, Any]:
    """Run the frozen final-test intention-to-treat risk-difference analysis.

    Every assigned action remains in its original arm.  Missing mature outcomes,
    no-action records, controller failures, late installation, and incomplete or
    late delivery all count as failures through the conservative endpoint.  The
    interval is a percentile cluster bootstrap over the frozen cluster field.
    """

    verify_gate_report(gate_report, signing_key).raise_for_errors()
    verify_signed_artifact(
        final_test_unlock, signing_key, purpose="O4_FINAL_TEST_UNLOCK_V1"
    ).raise_for_errors()
    verify_signed_artifact(
        final_test_plan, signing_key, purpose="O4_FIXED_FINAL_TEST_PLAN_V1"
    ).raise_for_errors()
    verify_signed_artifact(
        audit_report, signing_key, purpose="O4_END_TO_END_AUDIT_V1"
    ).raise_for_errors()
    if gate_report.get("decision") != "AUTHORIZED":
        raise ValueError("final-test analysis requires an authorized gate")
    if final_test_unlock.get("status") != "UNTOUCHED_FINAL_TEST_UNLOCKED":
        raise ValueError("final-test analysis requires a valid unlock token")
    if final_test_unlock.get("gate_report_sha256") != gate_report.get("artifact_sha256"):
        raise ValueError("final-test unlock is not bound to the gate report")
    if final_test_plan.get("gate_report_sha256") != gate_report.get("artifact_sha256"):
        raise ValueError("final-test plan is not bound to the gate report")
    if final_test_plan.get("final_test_unlock_sha256") != final_test_unlock.get("artifact_sha256"):
        raise ValueError("final-test plan is not bound to the unlock token")
    if not (
        final_test_unlock.get("frozen_artifacts")
        == final_test_plan.get("frozen_artifacts")
        == gate_report.get("frozen_artifacts")
    ):
        raise ValueError("frozen artifacts changed in the final-test release chain")
    if final_test_plan.get("release_phase") != "final_test":
        raise ValueError("final-test plan has an invalid release phase")
    if final_test_plan.get("status") != "PLANNED_NOT_EXECUTED":
        raise ValueError("final-test plan has an invalid pre-execution status")
    planned_assignments = list(final_test_plan.get("assignments", ()))
    planned_blocks = [str(item.get("block_id", "")) for item in planned_assignments]
    planned_arms = [str(item.get("assigned_arm", "")) for item in planned_assignments]
    if (
        not planned_assignments
        or len(set(planned_blocks)) != len(planned_blocks)
        or any(not block for block in planned_blocks)
        or any(arm not in {"gated_candidate", "reactive_baseline"} for arm in planned_arms)
        or len(planned_assignments) != int(final_test_plan.get("total_blocks", -1))
        or planned_arms.count("gated_candidate")
        != int(final_test_plan.get("candidate_blocks", -1))
        or planned_arms.count("reactive_baseline")
        != int(final_test_plan.get("baseline_blocks", -1))
    ):
        raise ValueError("signed final-test plan has an inconsistent assignment contract")

    analysis_spec_path = Path(analysis_spec_path)
    spec = json.loads(analysis_spec_path.read_text(encoding="utf-8"))
    required_spec = {
        "schema_version",
        "status",
        "estimand",
        "endpoint",
        "candidate_arm",
        "baseline_arm",
        "cluster_field",
        "bootstrap_replicates",
        "confidence_level",
        "bootstrap_seed",
        "minimum_practical_gain",
        "missing_outcome_rule",
        "minimum_independent_clusters_per_arm",
        "config_sha256",
        "model_sha256",
    }
    missing = sorted(required_spec.difference(spec))
    if missing:
        raise ValueError(f"analysis specification missing fields: {missing}")
    if spec["status"] != "FROZEN_ANALYSIS_SPEC":
        raise ValueError("analysis specification is not frozen")
    if spec["estimand"] != "intention_to_treat_risk_difference_candidate_minus_baseline":
        raise ValueError("unsupported final-test estimand")
    if spec["endpoint"] != "conservative_on_time_task_completion":
        raise ValueError("unsupported final-test endpoint")
    if spec["missing_outcome_rule"] != "count_as_failure":
        raise ValueError("final-test analysis must count missing outcomes as failures")
    frozen = gate_report["frozen_artifacts"]
    if spec["config_sha256"] != frozen["config_sha256"] or spec["model_sha256"] != frozen["model_sha256"]:
        raise ValueError("analysis specification is bound to different frozen artifacts")
    if final_test_plan.get("analysis_spec_sha256") != sha256_file(analysis_spec_path):
        raise ValueError("analysis specification differs from the signed final-test plan")
    replicates = int(spec["bootstrap_replicates"])
    confidence = float(spec["confidence_level"])
    gain = float(spec["minimum_practical_gain"])
    minimum_clusters_per_arm = spec["minimum_independent_clusters_per_arm"]
    if replicates < 1000 or not 0.8 <= confidence < 1 or not -1 < gain < 1:
        raise ValueError("invalid frozen bootstrap, confidence, or gain parameters")
    if (
        not isinstance(minimum_clusters_per_arm, int)
        or isinstance(minimum_clusters_per_arm, bool)
        or minimum_clusters_per_arm < 10
    ):
        raise ValueError(
            "minimum_independent_clusters_per_arm must be a frozen integer of at least 10"
        )
    cluster_field = str(spec["cluster_field"])
    if cluster_field != "block_id":
        raise ValueError(
            "cluster_field must be block_id because final-test randomization occurs by block"
        )

    schema = load_ledger_schema(schema_path)
    schema_digest = sha256_file(schema_path)
    frozen_schema_digest = str(frozen.get("ledger_schema_sha256", ZERO_SHA256))
    if frozen_schema_digest != ZERO_SHA256 and schema_digest != frozen_schema_digest:
        raise ValueError("ledger schema differs from the schema frozen at the evidence gate")
    audit_hash_chain(ledger_path, schema).raise_for_errors()
    audit_decision_lifecycles(
        ledger_path, schema, require_mature_outcomes=False
    ).raise_for_errors()
    register = pd.read_csv(register_path, keep_default_na=False)
    validate_campaign_register(register, require_locked=True).raise_for_errors()
    if sha256_file(register_path) != final_test_unlock.get("register_sha256_at_unlock"):
        raise ValueError("campaign register changed after final-test unlock")
    final_campaigns = set(
        register.loc[register["role"].eq("final_test"), "campaign_id"].astype(str)
    )
    records = [
        json.loads(line)
        for line in Path(ledger_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(records)
    actions = frame.loc[
        frame["record_type"].eq("action")
        & frame["campaign_id"].astype(str).isin(final_campaigns)
    ].copy()
    if len(actions) and not actions["release_phase"].eq("final_test").all():
        raise ValueError("final-test campaigns contain non-final-test release phases")
    if len(actions) and (
        not actions["model_sha256"].eq(frozen["model_sha256"]).all()
        or not actions["config_sha256"].eq(frozen["config_sha256"]).all()
    ):
        raise ValueError("final-test actions use model or configuration outside the frozen gate")
    outcomes = frame.loc[frame["record_type"].eq("outcome"), [
        "decision_id",
        "conservative_on_time_completion",
        "no_action",
        "controller_failure",
        "late_install",
        "evidence_tier",
        "outcome_evidence_status",
        "outcome_evidence_origin",
        "application_acknowledgement_authenticated",
    ]].copy().rename(columns={
        "conservative_on_time_completion": "outcome_success",
        "no_action": "outcome_no_action",
        "controller_failure": "outcome_controller_failure",
        "late_install": "outcome_late_install",
        "evidence_tier": "outcome_evidence_tier",
        "outcome_evidence_status": "outcome_ack_status",
        "outcome_evidence_origin": "outcome_ack_origin",
        "application_acknowledgement_authenticated": "outcome_ack_authenticated",
    })
    if actions["decision_id"].duplicated().any() or outcomes["decision_id"].duplicated().any():
        raise ValueError("analysis ledger contains duplicate actions or outcomes")
    analysis = actions.merge(outcomes, on="decision_id", how="left", validate="one_to_one")
    observed_outcomes = analysis["outcome_success"].notna()
    live_evidence = bool(len(analysis)) and bool(
        actions["evidence_tier"].eq("field_live_randomized").all()
        and analysis.loc[observed_outcomes, "outcome_evidence_tier"].eq(
            "field_live_randomized"
        ).all()
        and analysis.loc[observed_outcomes, "outcome_ack_status"].eq(
            "LIVE_APPLICATION_ACKNOWLEDGEMENT"
        ).all()
        and analysis.loc[observed_outcomes, "outcome_ack_origin"].eq(
            "executed_live"
        ).all()
        and analysis.loc[observed_outcomes, "outcome_ack_authenticated"].astype(bool).all()
    )
    assignments = {
        str(item["block_id"]): str(item["assigned_arm"])
        for item in final_test_plan.get("assignments", ())
    }
    observed_blocks = analysis["block_id"].astype(str)
    if set(observed_blocks) != set(assignments):
        raise ValueError("final-test ledger does not cover exactly the signed block set")
    assignment_mismatch = sum(
        assignments[block] != arm
        for block, arm in zip(observed_blocks, analysis["policy_arm"].astype(str))
    )
    if assignment_mismatch:
        raise ValueError("final-test action arm differs from signed ITT assignment")
    if not analysis["gate_report_sha256"].eq(gate_report["artifact_sha256"]).all():
        raise ValueError("final-test actions reference a different gate report")
    if not analysis["assignment_plan_sha256"].eq(final_test_plan["artifact_sha256"]).all():
        raise ValueError("final-test actions reference a different assignment plan")
    if len(analysis) and int(final_test_plan.get("created_wall_ns", 0)) > int(
        analysis["recorded_wall_ns"].min()
    ):
        raise ValueError("final-test allocation was signed after action collection began")
    candidate_arm = str(spec["candidate_arm"])
    baseline_arm = str(spec["baseline_arm"])
    unexpected_arms = sorted(set(analysis["policy_arm"].astype(str)) - {candidate_arm, baseline_arm})
    if unexpected_arms:
        raise ValueError(f"unexpected final-test policy arms: {unexpected_arms}")
    missing_outcome = analysis["outcome_success"].isna()
    analysis["itt_success"] = (
        analysis["outcome_success"].fillna(False).astype(bool)
    )
    analysis["cluster_id"] = analysis[cluster_field].astype(str)

    arm_rows = {
        arm: analysis.loc[analysis["policy_arm"].eq(arm)].copy()
        for arm in (candidate_arm, baseline_arm)
    }
    arm_counts = {arm: int(len(rows)) for arm, rows in arm_rows.items()}
    arm_clusters = {
        arm: int(rows["cluster_id"].nunique()) for arm, rows in arm_rows.items()
    }
    sufficient = all(
        arm_counts[arm] > 0
        and arm_clusters[arm] >= minimum_clusters_per_arm
        for arm in arm_counts
    )
    risk = {
        arm: (float(rows["itt_success"].mean()) if len(rows) else None)
        for arm, rows in arm_rows.items()
    }
    observed = (
        risk[candidate_arm] - risk[baseline_arm]
        if risk[candidate_arm] is not None and risk[baseline_arm] is not None
        else None
    )

    bootstrap_values: list[float] = []
    if sufficient:
        rng = random.Random(int(spec["bootstrap_seed"]))
        grouped = {
            arm: {
                cluster: rows.loc[rows["cluster_id"].eq(cluster), "itt_success"].astype(float).tolist()
                for cluster in sorted(rows["cluster_id"].unique())
            }
            for arm, rows in arm_rows.items()
        }
        for _ in range(replicates):
            sampled_risk: dict[str, float] = {}
            for arm, clusters in grouped.items():
                cluster_ids = list(clusters)
                sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
                values = [value for cluster in sampled for value in clusters[cluster]]
                sampled_risk[arm] = statistics.fmean(values)
            bootstrap_values.append(sampled_risk[candidate_arm] - sampled_risk[baseline_arm])
    alpha = 1 - confidence
    lower = _percentile(bootstrap_values, alpha / 2) if bootstrap_values else None
    upper = _percentile(bootstrap_values, 1 - alpha / 2) if bootstrap_values else None
    supports_gain = bool(sufficient and lower is not None and lower > gain)
    ledger_digest = sha256_file(ledger_path)
    register_digest = sha256_file(register_path)
    audit_ready = bool(
        audit_report.get("analysis_integrity_passed") is True
        and audit_report.get("claim_level")
        == "FINAL_TEST_DATA_READY_FOR_PRESPECIFIED_ANALYSIS"
        and audit_report.get("gate_report_sha256") == gate_report.get("artifact_sha256")
        and audit_report.get("final_test_unlock_sha256")
        == final_test_unlock.get("artifact_sha256")
        and audit_report.get("final_test_plan_sha256")
        == final_test_plan.get("artifact_sha256")
        and audit_report.get("ledger_sha256") == ledger_digest
        and audit_report.get("ledger_schema_sha256") == schema_digest
        and audit_report.get("register_sha256") == register_digest
    )
    if not audit_ready:
        raise ValueError("signed end-to-end audit does not authorize this ledger for analysis")
    confirmatory_live = bool(
        gate_report.get("execution_mode") == CONFIRMATORY_LIVE
        and gate_report.get("evidence_status") == "LOCKED_GATE_EVIDENCE"
        and live_evidence
    )
    benefit_supported = bool(supports_gain and audit_ready and confirmatory_live)
    status = "ANALYSIS_COMPLETE" if sufficient else "ANALYSIS_INSUFFICIENT"

    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "prespecified_final_test_itt_analysis",
        "created_wall_ns": time.time_ns() if created_wall_ns is None else int(created_wall_ns),
        "gate_report_sha256": gate_report["artifact_sha256"],
        "final_test_unlock_sha256": final_test_unlock["artifact_sha256"],
        "final_test_plan_sha256": final_test_plan["artifact_sha256"],
        "analysis_spec_sha256": sha256_file(analysis_spec_path),
        "ledger_sha256": ledger_digest,
        "register_sha256": register_digest,
        "end_to_end_audit_sha256": audit_report["artifact_sha256"],
        "status": status,
        "estimand": spec["estimand"],
        "endpoint": spec["endpoint"],
        "candidate_arm": candidate_arm,
        "baseline_arm": baseline_arm,
        "cluster_field": cluster_field,
        "arm_task_counts": arm_counts,
        "arm_cluster_counts": arm_clusters,
        "minimum_independent_clusters_per_arm": minimum_clusters_per_arm,
        "missing_outcomes_counted_as_failure": int(missing_outcome.sum()),
        "no_action_counted_as_failure": int(analysis["outcome_no_action"].fillna(False).astype(bool).sum()),
        "candidate_completion_risk": risk[candidate_arm],
        "baseline_completion_risk": risk[baseline_arm],
        "risk_difference_candidate_minus_baseline": observed,
        "confidence_level": confidence,
        "cluster_bootstrap_replicates": replicates,
        "risk_difference_interval": {"lower": lower, "upper": upper},
        "minimum_practical_gain": gain,
        "gain_criterion": "lower_confidence_bound_strictly_above_minimum_practical_gain",
        "prespecified_gain_supported": supports_gain,
        "confirmatory_live_evidence": confirmatory_live,
        "retuning_performed": False,
        "benefit_claim_supported": benefit_supported,
        "claim_boundary": (
            "A positive claim is permitted only when the prespecified candidate-minus-"
            "baseline intention-to-treat interval lies wholly above the frozen practical "
            "gain, the signed audit passes, and the evidence is a confirmatory live trial. "
            "Controlled fixtures may exercise the analysis but cannot support benefit."
        ),
    }
    return sign_artifact(payload, signing_key, purpose="O4_FINAL_TEST_ITT_ANALYSIS_V1")
