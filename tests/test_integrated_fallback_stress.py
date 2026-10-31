from __future__ import annotations

from hashlib import sha256
import json

import pandas as pd
import pytest

from leo_service_survival.eos_full_chain import (
    EVIDENCE_TIER,
    POLICY_BASELINE,
    POLICY_INTEGRATED,
)
from leo_service_survival.fallback_stress import (
    FALLBACK_GATE_REASON,
    FALLBACK_POLICY_ID,
    run_synthetic_integrated_fallback,
    write_integrated_fallback_artifacts,
)


def test_unsupported_input_runs_real_reactive_fallback_with_full_trace() -> None:
    artifacts = run_synthetic_integrated_fallback()

    assert artifacts.audit.passed, artifacts.audit.checks.to_dict(orient="records")
    for frame in artifacts.tables.values():
        assert "evidence_tier" in frame
        assert frame["evidence_tier"].eq(EVIDENCE_TIER).all()

    candidates = artifacts.tables["candidate_path_observation"]
    actions = artifacts.tables["decision_action"]
    integrated = actions.loc[actions["policy_arm"].eq(POLICY_INTEGRATED)].iloc[0]
    baseline = actions.loc[actions["policy_arm"].eq(POLICY_BASELINE)].iloc[0]
    integrated_candidate = candidates.loc[
        candidates["decision_id"].eq(integrated["decision_id"])
        & candidates["candidate_path_id"].eq(integrated["selected_path_id"])
    ].iloc[0]

    assert integrated_candidate["gate_ood_fraction"] == 1.0
    assert not bool(integrated_candidate["gate_release"])
    assert "out_of_domain_fraction_exceeded" in integrated_candidate["gate_reasons"]
    assert integrated["gate_state"] == "fallback"
    assert integrated["gate_reason"] == FALLBACK_GATE_REASON
    assert integrated["selected_path_id"] == integrated["baseline_path_id"]
    assert integrated["selected_path_id"] == baseline["selected_path_id"]
    assert integrated["installed_path_id"] == integrated["selected_path_id"]

    assertion = artifacts.fallback_assertions.iloc[0]
    assert assertion["fallback_policy_id"] == FALLBACK_POLICY_ID
    assert bool(assertion["input_guard_triggered"])
    assert bool(assertion["fallback_executed"])
    assert not bool(assertion["learned_release_used"])
    assert not bool(assertion["unsupported_learned_choice_used"])
    assert (
        assertion["selected_path_id"]
        != assertion["counterfactual_unsupported_learned_path_id"]
    )
    assert bool(assertion["traceability_preserved"])
    assert assertion["bytes_delivered"] >= assertion["required_bytes"]
    assert assertion["completion_time_ns"] <= assertion["deadline_ns"]

    counts = artifacts.claims_ledger["counts"]
    assert counts == {
        "exported_candidate_rows": 4,
        "unique_common_candidates": 2,
        "exported_ood_candidate_rows": 4,
        "exported_gate_released_candidate_rows": 0,
        "integrated_decisions": 1,
        "integrated_fallback_decisions": 1,
        "integrated_released_decisions": 0,
        "counterfactual_unsupported_learned_decisions": 1,
        "counterfactual_unsupported_learned_decisions_used": 0,
        "fallback_actions_installed": 1,
        "integrated_completed_on_time_outcomes": 1,
        "traceable_integrated_task_rows": 1,
    }
    assert all(artifacts.claims_ledger["checks"].values())
    assert "field validation" in artifacts.claims_ledger["unsupported_claims"]


def test_writer_is_fresh_versioned_and_hash_manifest_is_verifiable(tmp_path) -> None:
    artifacts = run_synthetic_integrated_fallback()
    output = write_integrated_fallback_artifacts(artifacts, tmp_path / "fallback_v1")

    required = {
        "campaign_manifest.csv",
        "candidate_path_observation.csv",
        "task.csv",
        "decision_action.csv",
        "eo_resource_state.csv",
        "outcome.csv",
        "full_chain_audit.csv",
        "task_ledger.csv",
        "fallback_assertions.csv",
        "claims_ledger.json",
        "CLAIMS_LEDGER.md",
        "scenario_fixture.json",
        "profile_provenance.json",
        "run_manifest.json",
        "SHA256SUMS",
    }
    assert required == {path.name for path in output.iterdir() if path.is_file()}
    audit = pd.read_csv(output / "full_chain_audit.csv")
    assertions = pd.read_csv(output / "fallback_assertions.csv")
    assert audit["evidence_tier"].eq(EVIDENCE_TIER).all()
    assert assertions["evidence_tier"].eq(EVIDENCE_TIER).all()

    claims = json.loads((output / "claims_ledger.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert claims["evidence_tier"] == EVIDENCE_TIER
    assert manifest["evidence_tier"] == EVIDENCE_TIER
    assert manifest["all_fallback_checks_passed"]

    recorded = {}
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        recorded[name] = digest
    for name, digest in recorded.items():
        assert sha256((output / name).read_bytes()).hexdigest() == digest

    with pytest.raises(FileExistsError, match="not empty"):
        write_integrated_fallback_artifacts(artifacts, output)
