"""Cryptographically bound inputs and reproducible gate-evidence derivation.

This module closes the boundary between campaign locking and the prospective
release gate.  A gate decision is never computed from hand-entered summary
metrics.  Instead, the operator freezes all trial artifacts in an authenticated
``TrialFreezeManifest`` and derives gate evidence from per-campaign opportunity
tables whose byte hashes were already recorded in the immutable campaign lock.

``confirmatory_live`` and ``controlled_fixture`` are deliberately separate
modes.  Controlled fixtures are always labelled ``FAKE_DRIVER_TEST`` and can
never be promoted to live evidence by changing a command-line description.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import hmac
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .field_preparation import (
    FAKE_DRIVER_TEST,
    LIVE_QUALIFIED,
    sha256_file,
    validate_route_driver_configuration,
    verify_campaign_registry_lock,
)


CONFIRMATORY_LIVE = "confirmatory_live"
CONTROLLED_FIXTURE = "controlled_fixture"
LOCKED_GATE_EVIDENCE = "LOCKED_GATE_EVIDENCE"
_PURPOSE = "O4_TRIAL_FREEZE_MANIFEST_V1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FREEZE_ARTIFACT_KEYS = {
    "testbed_qualification",
    "route_config",
    "campaign_registry_lock",
    "ledger_schema",
    "model",
    "trial_config",
    "analysis_plan",
    "opportunity_derivation",
}

GATE_OPPORTUNITY_COLUMNS = (
    "campaign_id",
    "acquisition_family_id",
    "opportunity_group_id",
    "cluster_id",
    "feature_present_count",
    "feature_required_count",
    "ood_flag",
    "predicted_success_probability",
    "candidate_success",
    "baseline_success",
    "control_loop_ms",
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _is_sha256(value: object) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value)))


def _require_signing_key(key: bytes) -> None:
    if len(key) < 32:
        raise ValueError("trial-freeze signing key must contain at least 32 bytes")


def _manifest_signature(key: bytes, digest: str) -> str:
    return hmac.new(key, f"{_PURPOSE}\n{digest}".encode("utf-8"), sha256).hexdigest()


@dataclass(frozen=True)
class TrialFreezeManifest:
    """Authenticated content bindings for every confirmatory gate input."""

    schema_version: str
    execution_mode: str
    evidence_status: str
    protocol_id: str
    freeze_timestamp_utc: str
    artifacts: Mapping[str, Mapping[str, str]]
    campaign_registry_content_sha256: str
    field_trial_completed: bool
    claim_boundary: str
    manifest_sha256: str
    authentication: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialFreezeManifest":
        return cls(**dict(value))

    def unsigned_body(self) -> dict[str, Any]:
        body = self.to_dict()
        body.pop("manifest_sha256", None)
        body.pop("authentication", None)
        return body


def _artifact_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"freeze artifact is not a regular file: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _validate_execution_boundary(
    *,
    execution_mode: str,
    registry_lock: Mapping[str, Any],
    qualification: Mapping[str, Any],
    route_config: Mapping[str, Any],
) -> None:
    lock_report = verify_campaign_registry_lock(registry_lock)
    lock_report.raise_for_errors()
    if execution_mode == CONFIRMATORY_LIVE:
        if not lock_report.live_eligible or lock_report.evidence_status != LIVE_QUALIFIED:
            raise ValueError("confirmatory_live requires a LIVE_QUALIFIED campaign lock")
        if not (
            qualification.get("passed") is True
            and qualification.get("testbed_live_eligible") is True
            and qualification.get("route_live_eligible") is True
            and qualification.get("testbed_evidence_status") == LIVE_QUALIFIED
            and qualification.get("route_evidence_status") == LIVE_QUALIFIED
        ):
            raise ValueError("confirmatory_live requires a passing live qualification report")
        route_report = validate_route_driver_configuration(route_config)
        if not route_report.passed or not route_report.live_eligible:
            raise ValueError("confirmatory_live requires a live-eligible route configuration")
    elif execution_mode == CONTROLLED_FIXTURE:
        if lock_report.evidence_status != FAKE_DRIVER_TEST:
            raise ValueError("controlled_fixture requires a FAKE_DRIVER_TEST campaign lock")
        if not (
            qualification.get("passed") is True
            and qualification.get("testbed_evidence_status") == FAKE_DRIVER_TEST
            and qualification.get("route_evidence_status") == FAKE_DRIVER_TEST
            and qualification.get("testbed_live_eligible") is False
            and qualification.get("route_live_eligible") is False
        ):
            raise ValueError("controlled_fixture requires an explicit fake qualification report")
        route_report = validate_route_driver_configuration(
            route_config, allow_fake_driver=True
        )
        if not route_report.passed or route_report.evidence_status != FAKE_DRIVER_TEST:
            raise ValueError("controlled_fixture requires a FAKE_DRIVER_TEST route configuration")
    else:
        raise ValueError(f"unsupported execution_mode: {execution_mode}")


def _validate_registry_artifact_bindings(
    registry_lock: Mapping[str, Any],
    artifact_records: Mapping[str, Mapping[str, str]],
    *,
    protocol_id: str,
) -> None:
    """Require every locked campaign to name the same frozen artifacts."""

    if registry_lock.get("protocol_sha256") != artifact_records["trial_config"]["sha256"]:
        raise ValueError("campaign lock protocol_sha256 does not match trial_config")
    if registry_lock.get("qualification_sha256") != artifact_records["testbed_qualification"]["sha256"]:
        raise ValueError("campaign lock qualification_sha256 does not match qualification report")
    campaigns = registry_lock.get("campaigns")
    if not isinstance(campaigns, list) or not campaigns:
        raise ValueError("campaign lock contains no campaigns")
    expected = {
        "protocol_version": protocol_id,
        "model_sha256": artifact_records["model"]["sha256"],
        "config_sha256": artifact_records["trial_config"]["sha256"],
        "testbed_qualification_sha256": artifact_records["testbed_qualification"]["sha256"],
    }
    for index, campaign in enumerate(campaigns):
        for field, value in expected.items():
            if campaign.get(field) != value:
                raise ValueError(
                    f"campaign {index} {field} is not bound to the trial freeze"
                )


def build_trial_freeze_manifest(
    *,
    execution_mode: str,
    protocol_id: str,
    freeze_timestamp_utc: str,
    artifact_paths: Mapping[str, str | Path],
    signing_key: bytes,
) -> TrialFreezeManifest:
    """Build an authenticated manifest after revalidating the execution boundary."""

    _require_signing_key(signing_key)
    missing = sorted(FREEZE_ARTIFACT_KEYS.difference(artifact_paths))
    extra = sorted(set(artifact_paths).difference(FREEZE_ARTIFACT_KEYS))
    if missing or extra:
        raise ValueError(f"freeze artifacts mismatch: missing={missing}; extra={extra}")
    timestamp = pd.to_datetime(freeze_timestamp_utc, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("freeze_timestamp_utc must be an ISO-8601 timestamp")
    records = {name: _artifact_record(path) for name, path in artifact_paths.items()}
    registry_lock = json.loads(
        Path(records["campaign_registry_lock"]["path"]).read_text(encoding="utf-8")
    )
    qualification = json.loads(
        Path(records["testbed_qualification"]["path"]).read_text(encoding="utf-8")
    )
    route_config = json.loads(
        Path(records["route_config"]["path"]).read_text(encoding="utf-8")
    )
    _validate_execution_boundary(
        execution_mode=execution_mode,
        registry_lock=registry_lock,
        qualification=qualification,
        route_config=route_config,
    )
    trial_config = json.loads(
        Path(records["trial_config"]["path"]).read_text(encoding="utf-8")
    )
    if trial_config.get("protocol_id") != protocol_id:
        raise ValueError("protocol_id does not match the frozen trial configuration")
    if trial_config.get("protocol_status") != "FROZEN_REGISTERED":
        raise ValueError("trial configuration must be FROZEN_REGISTERED before freeze")
    if trial_config.get("field_trial_completed") is not False:
        raise ValueError("freeze input must not assert a completed field trial")
    _validate_registry_artifact_bindings(
        registry_lock, records, protocol_id=protocol_id
    )

    evidence_status = (
        LOCKED_GATE_EVIDENCE
        if execution_mode == CONFIRMATORY_LIVE
        else FAKE_DRIVER_TEST
    )
    body: dict[str, Any] = {
        "schema_version": "1.0.0",
        "execution_mode": execution_mode,
        "evidence_status": evidence_status,
        "protocol_id": protocol_id,
        "freeze_timestamp_utc": pd.Timestamp(timestamp).isoformat(),
        "artifacts": records,
        "campaign_registry_content_sha256": str(registry_lock["registry_sha256"]),
        "field_trial_completed": False,
        "claim_boundary": (
            "This manifest authenticates frozen inputs. FAKE_DRIVER_TEST verifies "
            "software only; only confirmatory_live with live hardware can support a "
            "prospective field claim."
        ),
    }
    digest = sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    value = {
        **body,
        "manifest_sha256": digest,
        "authentication": {
            "algorithm": "HMAC-SHA256",
            "purpose": _PURPOSE,
            "key_id": sha256(signing_key).hexdigest()[:16],
            "signature": _manifest_signature(signing_key, digest),
        },
    }
    return TrialFreezeManifest.from_dict(value)


def verify_trial_freeze_manifest(
    manifest: TrialFreezeManifest | Mapping[str, Any],
    *,
    signing_key: bytes,
    expected_mode: str | None = None,
) -> None:
    """Recompute every file hash, registry lock, content hash, and HMAC."""

    _require_signing_key(signing_key)
    item = (
        manifest
        if isinstance(manifest, TrialFreezeManifest)
        else TrialFreezeManifest.from_dict(manifest)
    )
    body = item.unsigned_body()
    expected_digest = sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    if not hmac.compare_digest(item.manifest_sha256, expected_digest):
        raise ValueError("trial freeze manifest content hash mismatch")
    authentication = item.authentication
    expected_signature = _manifest_signature(signing_key, expected_digest)
    if not (
        authentication.get("algorithm") == "HMAC-SHA256"
        and authentication.get("purpose") == _PURPOSE
        and authentication.get("key_id") == sha256(signing_key).hexdigest()[:16]
        and hmac.compare_digest(str(authentication.get("signature", "")), expected_signature)
    ):
        raise ValueError("trial freeze manifest authentication failed")
    if expected_mode is not None and item.execution_mode != expected_mode:
        raise ValueError(
            f"freeze execution mode mismatch: {item.execution_mode} != {expected_mode}"
        )
    if set(item.artifacts) != FREEZE_ARTIFACT_KEYS:
        raise ValueError("trial freeze manifest has missing or unexpected artifacts")
    for name, record in item.artifacts.items():
        supplied_hash = record.get("sha256")
        if not _is_sha256(supplied_hash):
            raise ValueError(f"invalid frozen hash for {name}")
        path = Path(str(record.get("path", "")))
        if not path.is_file() or sha256_file(path) != supplied_hash:
            raise ValueError(f"frozen artifact changed or is missing: {name}")
    registry_lock = json.loads(
        Path(item.artifacts["campaign_registry_lock"]["path"]).read_text(encoding="utf-8")
    )
    qualification = json.loads(
        Path(item.artifacts["testbed_qualification"]["path"]).read_text(encoding="utf-8")
    )
    route_config = json.loads(
        Path(item.artifacts["route_config"]["path"]).read_text(encoding="utf-8")
    )
    _validate_execution_boundary(
        execution_mode=item.execution_mode,
        registry_lock=registry_lock,
        qualification=qualification,
        route_config=route_config,
    )
    _validate_registry_artifact_bindings(
        registry_lock, item.artifacts, protocol_id=item.protocol_id
    )
    if registry_lock.get("registry_sha256") != item.campaign_registry_content_sha256:
        raise ValueError("campaign registry content hash is not bound to the freeze")
    expected_status = (
        LOCKED_GATE_EVIDENCE
        if item.execution_mode == CONFIRMATORY_LIVE
        else FAKE_DRIVER_TEST
    )
    if item.evidence_status != expected_status:
        raise ValueError("freeze evidence status is inconsistent with execution mode")


def load_verified_registry_lock(
    manifest: TrialFreezeManifest | Mapping[str, Any],
    *,
    signing_key: bytes,
    expected_mode: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return the locked register only after full freeze-manifest verification."""

    item = (
        manifest
        if isinstance(manifest, TrialFreezeManifest)
        else TrialFreezeManifest.from_dict(manifest)
    )
    verify_trial_freeze_manifest(item, signing_key=signing_key, expected_mode=expected_mode)
    path = Path(item.artifacts["campaign_registry_lock"]["path"])
    lock = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(lock["campaigns"]).drop(columns=["row_sha256"], errors="ignore")
    return lock, frame


def validate_gate_opportunity_table(
    frame: pd.DataFrame,
    *,
    campaign_id: str,
    acquisition_family_id: str,
) -> None:
    """Validate one raw paired-opportunity table before any statistic is derived."""

    missing = sorted(set(GATE_OPPORTUNITY_COLUMNS).difference(frame.columns))
    extra = sorted(set(frame.columns).difference(GATE_OPPORTUNITY_COLUMNS))
    if missing or extra:
        raise ValueError(f"gate opportunity columns mismatch: missing={missing}; extra={extra}")
    if frame.empty:
        raise ValueError("gate opportunity table is empty")
    if set(frame["campaign_id"].astype(str)) != {campaign_id}:
        raise ValueError("opportunity table campaign_id mismatch")
    if set(frame["acquisition_family_id"].astype(str)) != {acquisition_family_id}:
        raise ValueError("opportunity table acquisition_family_id mismatch")
    if frame["opportunity_group_id"].astype(str).duplicated().any():
        raise ValueError("opportunity_group_id must be unique within a campaign")
    if frame["cluster_id"].astype(str).eq("").any():
        raise ValueError("cluster_id must be nonempty")

    present = pd.to_numeric(frame["feature_present_count"], errors="coerce")
    required = pd.to_numeric(frame["feature_required_count"], errors="coerce")
    probabilities = pd.to_numeric(frame["predicted_success_probability"], errors="coerce")
    loop = pd.to_numeric(frame["control_loop_ms"], errors="coerce")
    binary_columns = ("ood_flag", "candidate_success", "baseline_success")
    binary_ok = all(
        set(pd.to_numeric(frame[column], errors="coerce").dropna().unique()).issubset({0, 1})
        and pd.to_numeric(frame[column], errors="coerce").notna().all()
        for column in binary_columns
    )
    numeric_ok = bool(
        present.notna().all()
        and required.notna().all()
        and probabilities.notna().all()
        and loop.notna().all()
        and present.ge(0).all()
        and required.gt(0).all()
        and present.le(required).all()
        and probabilities.between(0, 1).all()
        and loop.ge(0).all()
        and np.isfinite(present).all()
        and np.isfinite(required).all()
        and np.isfinite(probabilities).all()
        and np.isfinite(loop).all()
    )
    if not binary_ok or not numeric_ok:
        raise ValueError("gate opportunity table contains invalid metric values")


def build_gate_opportunity_manifest(
    *,
    freeze_manifest: TrialFreezeManifest | Mapping[str, Any],
    table_paths: Mapping[str, str | Path],
    signing_key: bytes,
    execution_mode: str,
) -> dict[str, Any]:
    """Inventory and verify one raw table for every locked gate campaign."""

    item = (
        freeze_manifest
        if isinstance(freeze_manifest, TrialFreezeManifest)
        else TrialFreezeManifest.from_dict(freeze_manifest)
    )
    lock, register = load_verified_registry_lock(
        item, signing_key=signing_key, expected_mode=execution_mode
    )
    gate = register.loc[register["role"].eq("gate_evidence")]
    by_id = {str(row.campaign_id): row for row in gate.itertuples(index=False)}
    if set(table_paths) != set(by_id):
        raise ValueError(
            "raw opportunity paths must match every locked gate campaign exactly once"
        )
    tables: list[dict[str, str]] = []
    for campaign_id in sorted(by_id):
        row = by_id[campaign_id]
        path = Path(table_paths[campaign_id]).expanduser().resolve()
        actual_hash = sha256_file(path)
        if actual_hash != str(row.gate_opportunity_sha256):
            raise ValueError(
                f"gate opportunity table for {campaign_id} does not match the "
                "gate_opportunity_sha256 in the Step-2 lock"
            )
        frame = pd.read_csv(path, keep_default_na=False)
        validate_gate_opportunity_table(
            frame,
            campaign_id=campaign_id,
            acquisition_family_id=str(row.acquisition_family_id),
        )
        tables.append(
            {
                "campaign_id": campaign_id,
                "acquisition_family_id": str(row.acquisition_family_id),
                "path": str(path),
                "sha256": actual_hash,
            }
        )
    return {
        "schema_version": "1.0.0",
        "evidence_status": (
            "live_hardware"
            if execution_mode == CONFIRMATORY_LIVE
            else FAKE_DRIVER_TEST
        ),
        "campaign_registry_sha256": lock["registry_sha256"],
        "tables": tables,
    }


def _metric_vector(frame: pd.DataFrame) -> dict[str, float]:
    present = frame["feature_present_count"].astype(float).sum()
    required = frame["feature_required_count"].astype(float).sum()
    predicted = frame["predicted_success_probability"].astype(float)
    candidate = frame["candidate_success"].astype(float)
    baseline = frame["baseline_success"].astype(float)
    return {
        "feature_coverage": float(present / required),
        "ood_fraction": float(frame["ood_flag"].astype(float).mean()),
        "calibration_gap": float(abs(predicted.mean() - candidate.mean())),
        "candidate_minus_baseline_effect": float((candidate - baseline).mean()),
        "control_loop_p95_ms": float(np.quantile(frame["control_loop_ms"].astype(float), 0.95)),
    }


def _cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    seed: int,
    replicates: int,
    confidence_level: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if replicates < 500:
        raise ValueError("cluster bootstrap requires at least 500 replicates")
    if not 0.5 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0.5, 1)")
    clusters = sorted(frame["cluster_id"].astype(str).unique())
    if not clusters:
        raise ValueError("at least one bootstrap cluster is required")
    groups = {value: frame.loc[frame["cluster_id"].astype(str).eq(value)] for value in clusters}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "feature_coverage": [],
        "ood_fraction": [],
        "calibration_gap": [],
        "candidate_minus_baseline_effect": [],
        "control_loop_p95_ms": [],
    }
    for _ in range(replicates):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        resampled = pd.concat([groups[str(cluster)] for cluster in selected], ignore_index=True)
        metrics = _metric_vector(resampled)
        for name, value in metrics.items():
            samples[name].append(value)
    alpha = 1.0 - confidence_level
    lower = {
        name: float(np.quantile(values, alpha)) for name, values in samples.items()
    }
    upper = {
        name: float(np.quantile(values, 1.0 - alpha)) for name, values in samples.items()
    }
    two_sided_lower = float(
        np.quantile(samples["candidate_minus_baseline_effect"], alpha / 2.0)
    )
    two_sided_upper = float(
        np.quantile(samples["candidate_minus_baseline_effect"], 1.0 - alpha / 2.0)
    )
    details = {
        "effect_two_sided_lower": two_sided_lower,
        "effect_two_sided_upper": two_sided_upper,
        "effect_ci_half_width": (two_sided_upper - two_sided_lower) / 2.0,
    }
    return lower, upper, details


def _load_derivation_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "bootstrap_seed",
        "bootstrap_replicates",
        "confidence_level",
        "cluster_column",
        "calibration_metric",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"opportunity derivation config missing fields: {missing}")
    if value["cluster_column"] != "cluster_id":
        raise ValueError("only the frozen cluster_id resampling unit is supported")
    if value["calibration_metric"] != "absolute_mean_probability_error":
        raise ValueError("unsupported frozen calibration metric")
    return value


def build_locked_gate_evidence(
    *,
    freeze_manifest: TrialFreezeManifest | Mapping[str, Any],
    opportunity_manifest_path: str | Path,
    signing_key: bytes,
    execution_mode: str,
) -> dict[str, Any]:
    """Recompute conservative pooled metrics from hash-locked raw tables."""

    item = (
        freeze_manifest
        if isinstance(freeze_manifest, TrialFreezeManifest)
        else TrialFreezeManifest.from_dict(freeze_manifest)
    )
    lock, register = load_verified_registry_lock(
        item, signing_key=signing_key, expected_mode=execution_mode
    )
    opportunity_manifest_path = Path(opportunity_manifest_path)
    opportunity_manifest = json.loads(opportunity_manifest_path.read_text(encoding="utf-8"))
    expected_evidence_status = (
        "live_hardware" if execution_mode == CONFIRMATORY_LIVE else FAKE_DRIVER_TEST
    )
    if opportunity_manifest.get("evidence_status") != expected_evidence_status:
        raise ValueError("opportunity manifest evidence status does not match execution mode")
    if opportunity_manifest.get("campaign_registry_sha256") != lock.get("registry_sha256"):
        raise ValueError("opportunity manifest is not bound to the verified campaign lock")
    tables = opportunity_manifest.get("tables")
    if not isinstance(tables, list) or not tables:
        raise ValueError("opportunity manifest must list per-campaign tables")

    gate = register.loc[register["role"].eq("gate_evidence")].copy()
    gate_by_id = {str(row.campaign_id): row for row in gate.itertuples(index=False)}
    supplied_ids = [str(entry.get("campaign_id", "")) for entry in tables if isinstance(entry, Mapping)]
    if len(supplied_ids) != len(set(supplied_ids)) or set(supplied_ids) != set(gate_by_id):
        raise ValueError("opportunity tables must match every locked gate campaign exactly once")

    frames: list[pd.DataFrame] = []
    campaign_records: list[dict[str, Any]] = []
    for entry in sorted(tables, key=lambda value: str(value["campaign_id"])):
        campaign_id = str(entry["campaign_id"])
        locked = gate_by_id[campaign_id]
        family_id = str(entry.get("acquisition_family_id", ""))
        if family_id != str(locked.acquisition_family_id):
            raise ValueError(f"acquisition family mismatch for campaign {campaign_id}")
        table_path = Path(str(entry.get("path", ""))).expanduser().resolve()
        actual_hash = sha256_file(table_path)
        claimed_hash = str(entry.get("sha256", ""))
        if not _is_sha256(claimed_hash) or actual_hash != claimed_hash:
            raise ValueError(f"raw opportunity table hash mismatch for {campaign_id}")
        if actual_hash != str(locked.gate_opportunity_sha256):
            raise ValueError(
                f"gate opportunity table for {campaign_id} is not the "
                "gate_opportunity_sha256 locked in Step 2"
            )
        frame = pd.read_csv(table_path, keep_default_na=False)
        validate_gate_opportunity_table(
            frame,
            campaign_id=campaign_id,
            acquisition_family_id=family_id,
        )
        frames.append(frame)
        campaign_records.append(
            {
                "campaign_id": campaign_id,
                "acquisition_family_id": family_id,
                "paired_opportunity_groups": int(frame["opportunity_group_id"].nunique()),
                "raw_evidence_sha256": actual_hash,
                "row_count": int(len(frame)),
            }
        )
    pooled = pd.concat(frames, ignore_index=True)
    derivation_path = item.artifacts["opportunity_derivation"]["path"]
    derivation = _load_derivation_config(derivation_path)
    point = _metric_vector(pooled)
    lower, upper, interval = _cluster_bootstrap(
        pooled,
        seed=int(derivation["bootstrap_seed"]),
        replicates=int(derivation["bootstrap_replicates"]),
        confidence_level=float(derivation["confidence_level"]),
    )
    # Gate-facing metrics use conservative one-sided bounds.  Point estimates
    # are retained separately for audit and never substituted by hand.
    metrics = {
        "paired_opportunity_groups": int(
            pooled.groupby(["campaign_id", "opportunity_group_id"]).ngroups
        ),
        "feature_coverage": lower["feature_coverage"],
        "ood_fraction": upper["ood_fraction"],
        "calibration_gap_upper": upper["calibration_gap"],
        "candidate_minus_baseline_effect_lower": lower[
            "candidate_minus_baseline_effect"
        ],
        "effect_ci_half_width": interval["effect_ci_half_width"],
        "control_loop_p95_ms": upper["control_loop_p95_ms"],
    }
    evidence_status = (
        LOCKED_GATE_EVIDENCE
        if execution_mode == CONFIRMATORY_LIVE
        else FAKE_DRIVER_TEST
    )
    return {
        "schema_version": "2.0.0",
        "evidence_status": evidence_status,
        "execution_mode": execution_mode,
        "protocol_id": item.protocol_id,
        "config_sha256": item.artifacts["trial_config"]["sha256"],
        "model_sha256": item.artifacts["model"]["sha256"],
        "trial_freeze_manifest_sha256": item.manifest_sha256,
        "campaign_registry_sha256": lock["registry_sha256"],
        "opportunity_manifest_sha256": sha256_file(opportunity_manifest_path),
        "opportunity_derivation_sha256": item.artifacts["opportunity_derivation"]["sha256"],
        "campaigns": campaign_records,
        "pooled_metrics": metrics,
        "point_estimates": {
            "feature_coverage": point["feature_coverage"],
            "ood_fraction": point["ood_fraction"],
            "calibration_gap": point["calibration_gap"],
            "candidate_minus_baseline_effect": point[
                "candidate_minus_baseline_effect"
            ],
            "control_loop_p95_ms": point["control_loop_p95_ms"],
        },
        "bootstrap": {
            "method": "nonparametric_cluster_bootstrap",
            "cluster_column": "cluster_id",
            "seed": int(derivation["bootstrap_seed"]),
            "replicates": int(derivation["bootstrap_replicates"]),
            "confidence_level": float(derivation["confidence_level"]),
            "effect_two_sided_lower": interval["effect_two_sided_lower"],
            "effect_two_sided_upper": interval["effect_two_sided_upper"],
        },
        "claim_boundary": (
            "Metrics were recomputed from Step-2 hash-locked opportunity tables. "
            "FAKE_DRIVER_TEST is software evidence only and is never live field evidence."
        ),
    }


def assert_gate_evidence_recomputed(
    evidence: Mapping[str, Any],
    *,
    freeze_manifest: TrialFreezeManifest | Mapping[str, Any],
    opportunity_manifest_path: str | Path,
    signing_key: bytes,
    execution_mode: str,
) -> None:
    """Reject hand-edited metrics by exact comparison with deterministic derivation."""

    recomputed = build_locked_gate_evidence(
        freeze_manifest=freeze_manifest,
        opportunity_manifest_path=opportunity_manifest_path,
        signing_key=signing_key,
        execution_mode=execution_mode,
    )
    if _canonical_json(dict(evidence)) != _canonical_json(recomputed):
        raise ValueError("gate evidence does not equal recomputed hash-locked evidence")

