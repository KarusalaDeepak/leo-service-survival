from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from leo_service_survival.field_preparation import (
    FAKE_DRIVER_TEST,
    build_campaign_registry_lock,
    sha256_file,
)
from leo_service_survival.gate_freeze import (
    CONFIRMATORY_LIVE,
    CONTROLLED_FIXTURE,
    GATE_OPPORTUNITY_COLUMNS,
    TrialFreezeManifest,
    assert_gate_evidence_recomputed,
    build_gate_opportunity_manifest,
    build_locked_gate_evidence,
    build_trial_freeze_manifest,
    verify_trial_freeze_manifest,
)
from leo_service_survival.prospective_release import run_frozen_evidence_gate


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "prospective_trial"
KEY = b"objective-4-freeze-test-key-more-than-32-bytes"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fake_route() -> dict:
    paths = ("reactive_baseline", "cellular", "leo")
    return {
        "status": FAKE_DRIVER_TEST,
        "evidence_source": FAKE_DRIVER_TEST,
        "destination": "198.51.100.10",
        "baseline_path_id": "reactive_baseline",
        "candidate_path_ids": ["cellular", "leo"],
        "maximum_total_loop_ms": 100.0,
        "first_packet_timeout_ms": 500.0,
        "snapshot_command": ["/usr/bin/true"],
        "install_commands": {path: ["/usr/bin/true"] for path in paths},
        "first_packet_commands": {path: ["/usr/bin/true"] for path in paths},
        "isolated_rollback_drill": {
            "isolation_attested": True,
            "namespace_id": "FAKE-ISOLATED-NAMESPACE",
            "inject_fault_command": ["/usr/bin/true"],
            "restore_baseline_command": ["/usr/bin/true"],
            "verify_baseline_command": ["/usr/bin/true"],
            "observe_first_packet_command": ["/usr/bin/true"],
        },
        "contract": {
            "snapshot_stdout_first_line": "resolved path",
            "install_exit_zero": "acknowledged",
            "first_packet_exit_zero": "observed",
            "first_packet_stdout_first_line": "observed path identifier",
            "commands_use_shell": False,
            "live_mutation_default": False,
        },
    }


def _fake_qualification() -> dict:
    return {
        "field_trial_status": "NOT_RUN",
        "testbed_evidence_status": FAKE_DRIVER_TEST,
        "testbed_live_eligible": False,
        "route_evidence_status": FAKE_DRIVER_TEST,
        "route_live_eligible": False,
        "passed": True,
        "claim_boundary": "deterministic fake qualification for software tests only",
    }


def _opportunity_frame(campaign: str, family: str, offset: int) -> pd.DataFrame:
    rows = []
    for index in range(12):
        baseline = int((index + offset) % 4 != 0)
        candidate = int((index + offset) % 7 != 0)
        rows.append(
            {
                "campaign_id": campaign,
                "acquisition_family_id": family,
                "opportunity_group_id": f"{campaign}-OP-{index:03d}",
                "cluster_id": f"{campaign}-BLOCK-{index // 3:02d}",
                "feature_present_count": 10 if index != 0 else 9,
                "feature_required_count": 10,
                "ood_flag": int(index == 1),
                "predicted_success_probability": 0.84,
                "candidate_success": candidate,
                "baseline_success": baseline,
                "control_loop_ms": 35.0 + index,
            }
        )
    return pd.DataFrame(rows, columns=GATE_OPPORTUNITY_COLUMNS)


def _package(tmp_path: Path) -> tuple[TrialFreezeManifest, Path, dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    route_path = tmp_path / "route.json"
    qualification_path = tmp_path / "qualification.json"
    _write_json(route_path, _fake_route())
    _write_json(qualification_path, _fake_qualification())

    trial_config = json.loads(
        (PACKAGE / "config" / "gate_canary_rollback.json").read_text(encoding="utf-8")
    )
    trial_config["protocol_status"] = "FROZEN_REGISTERED"
    trial_path = tmp_path / "trial.json"
    _write_json(trial_path, trial_config)
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"frozen deterministic model bytes\n")
    analysis_path = tmp_path / "analysis.md"
    analysis_path.write_text("# Frozen analysis\nNo post-outcome changes.\n", encoding="utf-8")
    derivation_path = tmp_path / "derivation.json"
    _write_json(
        derivation_path,
        {
            "schema_version": "1.0.0",
            "bootstrap_seed": 20261103,
            "bootstrap_replicates": 500,
            "confidence_level": 0.95,
            "cluster_column": "cluster_id",
            "calibration_metric": "absolute_mean_probability_error",
        },
    )
    ledger_path = PACKAGE / "schema" / "decision_ledger.schema.json"

    table_paths: dict[str, Path] = {}
    definitions = [
        ("C-DEV", "F-DEV", "development"),
        ("C-GATE-1", "F-GATE-1", "gate_evidence"),
        ("C-GATE-2", "F-GATE-2", "gate_evidence"),
        ("C-GATE-3", "F-GATE-3", "gate_evidence"),
        ("C-GATE-4", "F-GATE-1", "gate_evidence"),
        ("C-GATE-5", "F-GATE-2", "gate_evidence"),
        ("C-TEST", "F-TEST", "final_test"),
    ]
    rows = []
    for index, (campaign, family, role) in enumerate(definitions):
        raw_hash = sha256(f"raw-telemetry-{campaign}".encode("utf-8")).hexdigest()
        gate_opportunity_hash = ""
        if role == "gate_evidence":
            table = _opportunity_frame(campaign, family, index)
            table_path = tmp_path / f"{campaign}.csv"
            table.to_csv(table_path, index=False)
            table_paths[campaign] = table_path
            gate_opportunity_hash = sha256_file(table_path)
        rows.append(
            {
                "study_id": "O4-FAKE",
                "campaign_id": campaign,
                "acquisition_family_id": family,
                "role": role,
                "acquisition_pipeline_id": f"PIPE-{index}",
                "site_id": f"SITE-{index}",
                "operator_id": f"OP-{index}",
                "terminal_id": f"TERM-{index}",
                "route_id": f"ROUTE-{index}",
                "server_id": f"SERVER-{index}",
                "collection_start_utc": f"2026-{index + 1:02d}-01T00:00:00Z",
                "collection_end_utc": f"2026-{index + 1:02d}-02T00:00:00Z",
                "protocol_version": trial_config["protocol_id"],
                "model_sha256": sha256_file(model_path),
                "config_sha256": sha256_file(trial_path),
                "testbed_qualification_sha256": sha256_file(qualification_path),
                "raw_capture_sha256": raw_hash,
                "gate_opportunity_sha256": gate_opportunity_hash,
                "telemetry_manifest_sha256": sha256(f"telemetry-{index}".encode()).hexdigest(),
                "outcome_manifest_sha256": sha256(f"outcome-{index}".encode()).hexdigest(),
                "clock_sync_method": "FAKE_DETERMINISTIC",
                "maximum_clock_error_ms": 0.0,
                "status": "complete",
                "evidence_tier": "hardware_in_loop",
                "role_access_group": f"ACCESS-{role}",
            }
        )
    lock = build_campaign_registry_lock(
        pd.DataFrame(rows),
        protocol_sha256=sha256_file(trial_path),
        qualification_sha256=sha256_file(qualification_path),
        lock_timestamp_utc="2026-11-03T00:00:00Z",
        allow_fake=True,
    )
    lock_path = tmp_path / "registry.lock.json"
    _write_json(lock_path, lock)

    artifacts = {
        "testbed_qualification": qualification_path,
        "route_config": route_path,
        "campaign_registry_lock": lock_path,
        "ledger_schema": ledger_path,
        "model": model_path,
        "trial_config": trial_path,
        "analysis_plan": analysis_path,
        "opportunity_derivation": derivation_path,
    }
    manifest = build_trial_freeze_manifest(
        execution_mode=CONTROLLED_FIXTURE,
        protocol_id=trial_config["protocol_id"],
        freeze_timestamp_utc="2026-11-03T01:00:00Z",
        artifact_paths=artifacts,
        signing_key=KEY,
    )
    opportunity_manifest = {
        "schema_version": "1.0.0",
        "evidence_status": FAKE_DRIVER_TEST,
        "campaign_registry_sha256": lock["registry_sha256"],
        "tables": [
            {
                "campaign_id": campaign,
                "acquisition_family_id": next(
                    family for current, family, _ in definitions if current == campaign
                ),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for campaign, path in sorted(table_paths.items())
        ],
    }
    opportunity_manifest_path = tmp_path / "opportunities.json"
    _write_json(opportunity_manifest_path, opportunity_manifest)
    return manifest, opportunity_manifest_path, artifacts


def test_freeze_manifest_binds_all_artifacts_and_rejects_tampering(tmp_path: Path) -> None:
    manifest, _, artifacts = _package(tmp_path)
    verify_trial_freeze_manifest(
        manifest, signing_key=KEY, expected_mode=CONTROLLED_FIXTURE
    )
    assert set(manifest.artifacts) == {
        "testbed_qualification",
        "route_config",
        "campaign_registry_lock",
        "ledger_schema",
        "model",
        "trial_config",
        "analysis_plan",
        "opportunity_derivation",
    }
    artifacts["analysis_plan"].write_text("tampered after freeze\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        verify_trial_freeze_manifest(
            manifest, signing_key=KEY, expected_mode=CONTROLLED_FIXTURE
        )


def test_raw_opportunity_builder_recomputes_conservative_metrics(tmp_path: Path) -> None:
    manifest, opportunities, _ = _package(tmp_path)
    evidence = build_locked_gate_evidence(
        freeze_manifest=manifest,
        opportunity_manifest_path=opportunities,
        signing_key=KEY,
        execution_mode=CONTROLLED_FIXTURE,
    )
    assert evidence["evidence_status"] == FAKE_DRIVER_TEST
    assert evidence["execution_mode"] == CONTROLLED_FIXTURE
    assert len(evidence["campaigns"]) == 5
    assert evidence["pooled_metrics"]["paired_opportunity_groups"] == 60
    assert 0 <= evidence["pooled_metrics"]["feature_coverage"] <= 1
    assert 0 <= evidence["pooled_metrics"]["ood_fraction"] <= 1
    assert evidence["bootstrap"]["replicates"] == 500
    assert_gate_evidence_recomputed(
        evidence,
        freeze_manifest=manifest,
        opportunity_manifest_path=opportunities,
        signing_key=KEY,
        execution_mode=CONTROLLED_FIXTURE,
    )


def test_opportunity_manifest_builder_uses_only_step2_locked_tables(tmp_path: Path) -> None:
    manifest, opportunities, _ = _package(tmp_path)
    existing = json.loads(opportunities.read_text(encoding="utf-8"))
    table_paths = {
        item["campaign_id"]: item["path"] for item in existing["tables"]
    }
    rebuilt = build_gate_opportunity_manifest(
        freeze_manifest=manifest,
        table_paths=table_paths,
        signing_key=KEY,
        execution_mode=CONTROLLED_FIXTURE,
    )
    assert rebuilt == existing
    table_paths.pop(next(iter(table_paths)))
    with pytest.raises(ValueError, match="every locked gate campaign"):
        build_gate_opportunity_manifest(
            freeze_manifest=manifest,
            table_paths=table_paths,
            signing_key=KEY,
            execution_mode=CONTROLLED_FIXTURE,
        )


def test_invented_gate_metric_is_rejected_by_exact_recomputation(tmp_path: Path) -> None:
    manifest, opportunities, _ = _package(tmp_path)
    evidence = build_locked_gate_evidence(
        freeze_manifest=manifest,
        opportunity_manifest_path=opportunities,
        signing_key=KEY,
        execution_mode=CONTROLLED_FIXTURE,
    )
    evidence["pooled_metrics"]["candidate_minus_baseline_effect_lower"] = 1.0
    with pytest.raises(ValueError, match="does not equal recomputed"):
        assert_gate_evidence_recomputed(
            evidence,
            freeze_manifest=manifest,
            opportunity_manifest_path=opportunities,
            signing_key=KEY,
            execution_mode=CONTROLLED_FIXTURE,
        )


def test_raw_table_tampering_and_invented_hash_are_rejected(tmp_path: Path) -> None:
    manifest, opportunities, _ = _package(tmp_path)
    value = json.loads(opportunities.read_text(encoding="utf-8"))
    table_path = Path(value["tables"][0]["path"])
    table_path.write_text(table_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_locked_gate_evidence(
            freeze_manifest=manifest,
            opportunity_manifest_path=opportunities,
            signing_key=KEY,
            execution_mode=CONTROLLED_FIXTURE,
        )

    manifest, opportunities, _ = _package(tmp_path / "invented")
    value = json.loads(opportunities.read_text(encoding="utf-8"))
    value["tables"][0]["sha256"] = "f" * 64
    _write_json(opportunities, value)
    with pytest.raises(ValueError, match="hash mismatch"):
        build_locked_gate_evidence(
            freeze_manifest=manifest,
            opportunity_manifest_path=opportunities,
            signing_key=KEY,
            execution_mode=CONTROLLED_FIXTURE,
        )


def test_fake_fixture_cannot_be_relabelled_as_confirmatory_live(tmp_path: Path) -> None:
    manifest, opportunities, _ = _package(tmp_path)
    with pytest.raises(ValueError, match="execution mode mismatch"):
        build_locked_gate_evidence(
            freeze_manifest=manifest,
            opportunity_manifest_path=opportunities,
            signing_key=KEY,
            execution_mode=CONFIRMATORY_LIVE,
        )

    forged = manifest.to_dict()
    forged["execution_mode"] = CONFIRMATORY_LIVE
    forged["evidence_status"] = "LOCKED_GATE_EVIDENCE"
    with pytest.raises(ValueError, match="content hash mismatch"):
        verify_trial_freeze_manifest(
            forged, signing_key=KEY, expected_mode=CONFIRMATORY_LIVE
        )


def test_gate_runner_recomputes_instead_of_trusting_supplied_metrics(tmp_path: Path) -> None:
    manifest, opportunities, artifacts = _package(tmp_path)
    manifest_path = tmp_path / "freeze.json"
    evidence_path = tmp_path / "evidence.json"
    _write_json(manifest_path, manifest.to_dict())
    evidence = build_locked_gate_evidence(
        freeze_manifest=manifest,
        opportunity_manifest_path=opportunities,
        signing_key=KEY,
        execution_mode=CONTROLLED_FIXTURE,
    )
    _write_json(evidence_path, evidence)
    report = run_frozen_evidence_gate(
        config_path=artifacts["trial_config"],
        register_path=artifacts["campaign_registry_lock"],
        evidence_path=evidence_path,
        signing_key=KEY,
        execution_mode=CONTROLLED_FIXTURE,
        freeze_manifest_path=manifest_path,
        opportunity_manifest_path=opportunities,
        created_wall_ns=123,
    )
    assert report["execution_mode"] == CONTROLLED_FIXTURE
    assert report["evidence_status"] == FAKE_DRIVER_TEST
    assert report["field_execution_status"] == FAKE_DRIVER_TEST

    evidence["pooled_metrics"]["feature_coverage"] = 1.0
    _write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="does not equal recomputed"):
        run_frozen_evidence_gate(
            config_path=artifacts["trial_config"],
            register_path=artifacts["campaign_registry_lock"],
            evidence_path=evidence_path,
            signing_key=KEY,
            execution_mode=CONTROLLED_FIXTURE,
            freeze_manifest_path=manifest_path,
            opportunity_manifest_path=opportunities,
        )


def test_confirmatory_gate_refuses_manual_summary_without_freeze(tmp_path: Path) -> None:
    manifest, opportunities, artifacts = _package(tmp_path)
    evidence = build_locked_gate_evidence(
        freeze_manifest=manifest,
        opportunity_manifest_path=opportunities,
        signing_key=KEY,
        execution_mode=CONTROLLED_FIXTURE,
    )
    evidence_path = tmp_path / "manual-summary.json"
    _write_json(evidence_path, evidence)
    with pytest.raises(ValueError, match="confirmatory_live requires"):
        run_frozen_evidence_gate(
            config_path=artifacts["trial_config"],
            register_path=artifacts["campaign_registry_lock"],
            evidence_path=evidence_path,
            signing_key=KEY,
        )

