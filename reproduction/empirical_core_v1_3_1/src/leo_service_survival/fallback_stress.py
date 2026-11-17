"""Deterministic synthetic integration stress for the EOS input guard.

This module deliberately evaluates a network profile far outside a separately
fitted synthetic reference domain.  It then exercises the real EOS full-chain
gate, scheduler, installation and outcome code paths.  The resulting artifact
can establish implementation behaviour only: it is always labelled
``synthetic_simulation`` and is not field, operational or comparative evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import pandas as pd

from .eos_full_chain import (
    EVIDENCE_TIER,
    POLICY_BASELINE,
    POLICY_INTEGRATED,
    EOSFullChainConfig,
    NetworkPathProfile,
    _arm_tables,
    _build_common_actions,
    _concat_arm_tables,
    _fit_simulation_domain_gate,
    _hash_json,
    _seed_task_table,
    parse_eos_bench_scenario,
)
from .integration import (
    FullChainAudit,
    build_full_chain_ledger,
    validate_full_chain_bundle,
)


FALLBACK_STRESS_PROTOCOL = "eos-integrated-fallback-synthetic-v2"
FALLBACK_POLICY_ID = POLICY_BASELINE
FALLBACK_GATE_REASON = "gate_failed_or_released_actions_conflicted"


REFERENCE_PROFILE = NetworkPathProfile(
    path_profile_id="supported_reference",
    interface_type="synthetic_ntn",
    operator_id="synthetic_reference_operator",
    gateway_id="synthetic_gateway",
    mean_snr_db=10.0,
    snr_std_db=1.0,
    mean_rate_mbps=0.05,
    rate_log_sd=0.10,
    jitter_ms=10.0,
    outage_probability=0.0,
    handover_probability=0.02,
    source="declared_supported_synthetic_reference",
)


UNSUPPORTED_LEARNED_PROFILE = NetworkPathProfile(
    path_profile_id="unsupported_learned_preferred",
    interface_type="synthetic_ntn",
    operator_id="synthetic_shifted_operator_a",
    gateway_id="synthetic_gateway",
    mean_snr_db=40.0,
    snr_std_db=0.01,
    mean_rate_mbps=10.0,
    rate_log_sd=0.01,
    jitter_ms=20.0,
    outage_probability=0.0,
    handover_probability=0.01,
    source="deliberately_shifted_synthetic_learned_preferred_profile",
)


UNSUPPORTED_REACTIVE_PROFILE = NetworkPathProfile(
    path_profile_id="unsupported_reactive_preferred",
    interface_type="synthetic_ntn",
    operator_id="synthetic_shifted_operator_b",
    gateway_id="synthetic_gateway",
    mean_snr_db=80.0,
    snr_std_db=0.01,
    mean_rate_mbps=10.0,
    rate_log_sd=0.01,
    jitter_ms=1_000.0,
    outage_probability=0.0,
    handover_probability=0.90,
    source="deliberately_shifted_synthetic_reactive_preferred_profile",
)


UNSUPPORTED_PROFILES = (
    UNSUPPORTED_LEARNED_PROFILE,
    UNSUPPORTED_REACTIVE_PROFILE,
)


@dataclass(frozen=True)
class IntegratedFallbackArtifacts:
    """Audited tables and explicit assertions for one fallback stress run."""

    tables: Mapping[str, pd.DataFrame]
    audit: FullChainAudit
    ledger: pd.DataFrame
    fallback_assertions: pd.DataFrame
    claims_ledger: Mapping[str, Any]
    run_manifest: Mapping[str, Any]
    scenario_payload: Mapping[str, Any]
    profile_payload: Mapping[str, Any]


def synthetic_fallback_scenario_payload() -> dict[str, Any]:
    """Return the one-task fixture used to isolate gate-to-fallback semantics."""

    return {
        "scenario_id": "Fallback_Stress_S1",
        "scenario_type": "deterministic_synthetic_integration_stress",
        "evidence_tier": EVIDENCE_TIER,
        "metadata": {
            "name": "Fallback_Stress_S1",
            "creation_time": "2026-01-01T00:00:00+00:00",
            "duration": 86_400.0,
            "time_step": 1.0,
        },
        "satellites": [
            {
                "id": "SYNTH_SAT_A",
                "orbital_type": "LEO",
                "orbital_params": {
                    "semi_major_axis_km": 7_000.0,
                    "inclination_deg": 98.0,
                },
                "satellite_specs": {"max_data_storage_GB": 1.0},
                "observation_capability": {
                    "sensors": [
                        {
                            "sensor_id": "SYNTH_SENSOR_A",
                            "sensor_mode": "synthetic_visible",
                            "data_rate_Mbps": 0.001,
                        }
                    ]
                },
                "maneuverability_capability": {"stabilization_time_s": 1.0},
            }
        ],
        "missions": [
            {
                "id": "SYNTH_TASK_1",
                "target_location": {"latitude": 0.0, "longitude": 0.0},
                "priority": 10.0,
                "observation_requirement": {"duration_s": 1.0},
            }
        ],
        "observation_windows": [
            {
                "satellite_id": "SYNTH_SAT_A",
                "sensor_id": "SYNTH_SENSOR_A",
                "mission_id": "SYNTH_TASK_1",
                "time_windows": [
                    {
                        "start_time": "2026-01-01T01:00:00+00:00",
                        "end_time": "2026-01-01T01:01:00+00:00",
                        "orbit_number": 1,
                        "agile_data": {},
                        "non_agile_data": {},
                    }
                ],
            }
        ],
        "claim_boundary": (
            "Purpose-built synthetic fixture for gate-to-fallback implementation testing; "
            "not measured geometry, an operational workload, or field evidence."
        ),
    }


def fallback_stress_config(seed: int = 2026102801) -> EOSFullChainConfig:
    """Return frozen settings that isolate the fallback path deterministically."""

    return EOSFullChainConfig(
        seeds=(int(seed),),
        downlink_contact_s=90.0,
        telemetry_max_age_ms=100.0,
        storage_stress_cap_bytes=500_000,
        release_spread_hours=0.0,
        minimum_deadline_hours=12.0,
        priority_deadline_hours=0.0,
        deadline_jitter_hours=0.0,
        gate_reference_rows_per_profile=500,
        gate_reference_seed=2026102802,
        install_failure_probability=0.0,
        acquisition_failure_probability=0.0,
        product_invalid_probability=0.0,
        slew_guard_s=1.0,
        bootstrap_resamples=10,
        bootstrap_seed=2026102803,
    )


def _ledger_projection(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        name: (
            frame
            if name == "campaign_manifest"
            else frame.drop(columns=["evidence_tier"], errors="ignore")
        )
        for name, frame in tables.items()
    }


def _build_fallback_assertions(
    tables: Mapping[str, pd.DataFrame], ledger: pd.DataFrame
) -> pd.DataFrame:
    actions = tables["decision_action"]
    candidates = tables["candidate_path_observation"]
    tasks = tables["task"]
    resources = tables["eo_resource_state"]
    outcomes = tables["outcome"]

    baseline_by_logical = (
        actions.loc[actions["policy_arm"].eq(POLICY_BASELINE)]
        .set_index(["simulation_seed", "logical_task_id"])["selected_path_id"]
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    integrated = actions.loc[actions["policy_arm"].eq(POLICY_INTEGRATED)]
    for action in integrated.to_dict(orient="records"):
        decision_id = action["decision_id"]
        task_id = action["task_id"]
        selected_id = action["selected_path_id"]
        decision_candidates = candidates.loc[candidates["decision_id"].eq(decision_id)]
        selected_candidate = decision_candidates.loc[
            decision_candidates["candidate_path_id"].eq(selected_id)
        ]
        if len(selected_candidate) != 1:
            raise AssertionError("integrated selection must resolve to exactly one candidate")
        candidate = selected_candidate.iloc[0]
        task_rows = tasks.loc[tasks["task_id"].eq(task_id)]
        outcome_rows = outcomes.loc[outcomes["task_id"].eq(task_id)]
        resource_rows = resources.loc[
            resources["task_id"].eq(task_id)
            & resources["contact_id"].eq(selected_id)
        ]
        ledger_rows = ledger.loc[ledger["task_id"].eq(task_id)]
        if not (
            len(task_rows) == len(outcome_rows) == len(resource_rows) == len(ledger_rows) == 1
        ):
            raise AssertionError("task/resource/outcome/ledger trace must be one-to-one")
        task = task_rows.iloc[0]
        outcome = outcome_rows.iloc[0]
        resource = resource_rows.iloc[0]
        baseline_selected = baseline_by_logical[
            (int(action["simulation_seed"]), action["logical_task_id"])
        ]
        feasible_candidates = decision_candidates.loc[
            decision_candidates["resource_feasible"].astype(bool)
        ].sort_values(
            ["predicted_completion_probability", "window_start_ns", "candidate_path_id"],
            ascending=[False, True, True],
        )
        if feasible_candidates.empty:
            raise AssertionError("fallback stress requires feasible unsupported candidates")
        counterfactual_learned_path_id = str(
            feasible_candidates.iloc[0]["candidate_path_id"]
        )
        all_candidates_blocked_as_ood = bool(
            decision_candidates["gate_ood_fraction"].gt(0.0).all()
            and (~decision_candidates["gate_release"].astype(bool)).all()
            and decision_candidates["gate_reasons"]
            .astype(str)
            .str.contains("out_of_domain_fraction_exceeded", regex=False)
            .all()
        )
        input_guard_triggered = bool(
            all_candidates_blocked_as_ood
            and float(candidate["gate_ood_fraction"]) > 0.0
            and not bool(candidate["gate_release"])
        )
        fallback_executed = bool(
            action["gate_state"] == "fallback"
            and action["gate_reason"] == FALLBACK_GATE_REASON
            and selected_id == action["baseline_path_id"]
            and selected_id == baseline_selected
        )
        learned_release_used = bool(action["gate_state"] == "released")
        unsupported_learned_choice_used = bool(
            selected_id == counterfactual_learned_path_id
        )
        traceability_preserved = bool(
            task["decision_id"] == decision_id
            and outcome["decision_id"] == decision_id
            and resource["decision_id"] == decision_id
            and int(outcome["bytes_delivered"]) >= int(task["required_bytes"])
            and int(outcome["completion_time_ns"]) <= int(task["deadline_ns"])
        )
        rows.append(
            {
                "evidence_tier": EVIDENCE_TIER,
                "simulation_seed": int(action["simulation_seed"]),
                "logical_task_id": action["logical_task_id"],
                "task_id": task_id,
                "decision_id": decision_id,
                "fallback_policy_id": FALLBACK_POLICY_ID,
                "input_guard_triggered": input_guard_triggered,
                "guard_reason": candidate["gate_reasons"],
                "guard_ood_fraction": float(candidate["gate_ood_fraction"]),
                "unsupported_candidate_gate_release": bool(candidate["gate_release"]),
                "all_candidates_blocked_as_ood": all_candidates_blocked_as_ood,
                "fallback_executed": fallback_executed,
                "learned_release_used": learned_release_used,
                "counterfactual_unsupported_learned_path_id": (
                    counterfactual_learned_path_id
                ),
                "unsupported_learned_choice_used": unsupported_learned_choice_used,
                "selected_path_id": selected_id,
                "reactive_reference_path_id": action["baseline_path_id"],
                "standalone_baseline_selected_path_id": baseline_selected,
                "selected_matches_reactive_baseline": bool(
                    selected_id == action["baseline_path_id"] == baseline_selected
                ),
                "installed_path_id": action["installed_path_id"],
                "action_executed": bool(outcome["action_executed"]),
                "resource_feasible": bool(resource["resource_feasible"]),
                "required_bytes": int(task["required_bytes"]),
                "bytes_delivered": int(outcome["bytes_delivered"]),
                "deadline_ns": int(task["deadline_ns"]),
                "completion_time_ns": int(outcome["completion_time_ns"]),
                "met_deadline": bool(outcome["met_deadline"]),
                "traceability_preserved": traceability_preserved,
            }
        )
    return pd.DataFrame(rows)


def _claims_ledger(
    tables: Mapping[str, pd.DataFrame],
    audit: FullChainAudit,
    assertions: pd.DataFrame,
) -> dict[str, Any]:
    candidates = tables["candidate_path_observation"]
    actions = tables["decision_action"]
    outcomes = tables["outcome"]
    integrated_actions = actions.loc[actions["policy_arm"].eq(POLICY_INTEGRATED)]
    checks = {
        "full_chain_relational_audit_passed": bool(audit.passed),
        "all_exported_table_rows_are_synthetic_simulation": bool(
            all(
                "evidence_tier" in frame
                and frame["evidence_tier"].eq(EVIDENCE_TIER).all()
                for frame in tables.values()
            )
        ),
        "unsupported_input_guard_triggered": bool(
            len(assertions) and assertions["input_guard_triggered"].all()
        ),
        "unsupported_candidate_not_gate_released": bool(
            len(assertions) and (~assertions["unsupported_candidate_gate_release"]).all()
        ),
        "named_reactive_fallback_executed": bool(
            len(assertions) and assertions["fallback_executed"].all()
        ),
        "unsupported_learned_release_not_used": bool(
            len(assertions) and (~assertions["learned_release_used"]).all()
        ),
        "counterfactual_unsupported_learned_choice_not_used": bool(
            len(assertions)
            and (~assertions["unsupported_learned_choice_used"]).all()
        ),
        "fallback_choice_matches_standalone_baseline": bool(
            len(assertions) and assertions["selected_matches_reactive_baseline"].all()
        ),
        "task_resource_deadline_outcome_trace_preserved": bool(
            len(assertions) and assertions["traceability_preserved"].all()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"fallback stress assertions failed: {failed}")

    unique_candidates = candidates[
        ["simulation_seed", "logical_task_id", "candidate_path_id"]
    ].drop_duplicates()
    return {
        "protocol_version": FALLBACK_STRESS_PROTOCOL,
        "evidence_tier": EVIDENCE_TIER,
        "fallback_policy_id": FALLBACK_POLICY_ID,
        "counts": {
            "exported_candidate_rows": int(len(candidates)),
            "unique_common_candidates": int(len(unique_candidates)),
            "exported_ood_candidate_rows": int(
                candidates["gate_ood_fraction"].gt(0.0).sum()
            ),
            "exported_gate_released_candidate_rows": int(
                candidates["gate_release"].astype(bool).sum()
            ),
            "integrated_decisions": int(len(integrated_actions)),
            "integrated_fallback_decisions": int(
                integrated_actions["gate_state"].eq("fallback").sum()
            ),
            "integrated_released_decisions": int(
                integrated_actions["gate_state"].eq("released").sum()
            ),
            "counterfactual_unsupported_learned_decisions": int(len(assertions)),
            "counterfactual_unsupported_learned_decisions_used": int(
                assertions["unsupported_learned_choice_used"].sum()
            ),
            "fallback_actions_installed": int(
                integrated_actions.loc[
                    integrated_actions["gate_state"].eq("fallback"),
                    "installed_path_id",
                ].notna().sum()
            ),
            "integrated_completed_on_time_outcomes": int(
                outcomes.loc[
                    outcomes["campaign_id"].isin(integrated_actions["campaign_id"]),
                    "application_complete",
                ].astype(bool).sum()
            ),
            "traceable_integrated_task_rows": int(
                assertions["traceability_preserved"].sum()
            ),
        },
        "checks": checks,
        "semantics": {
            "input_guard": (
                "A frozen Ledoit-Wolf Mahalanobis detector is fitted only on the "
                "supported synthetic reference profile. The deliberately shifted "
                "evaluation candidate is blocked from learned release."
            ),
            "fallback": (
                "When no resource-feasible candidate is gate-released, the integrated "
                "scheduler executes its reactive order. For this one-task isolation "
                "test, that choice must equal both baseline_path_id and the independently "
                "executed edf_reactive arm."
            ),
            "learned_choice_separation": (
                "Both candidates are outside the frozen reference domain. The candidate "
                "with the greatest predicted completion probability is recorded as the "
                "counterfactual unsupported learned choice. The reactive fallback instead "
                "selects by earliest window and current SNR, and the two choices are required "
                "to differ in this isolation test."
            ),
        },
        "supported_claim": (
            "The deterministic synthetic full-chain implementation fails closed for the "
            "deliberately unsupported input, executes the named reactive fallback, and "
            "retains task/resource/deadline/outcome traceability."
        ),
        "unsupported_claims": [
            "field validation",
            "operational deployment safety",
            "external-validity or independent-campaign evidence",
            "performance superiority over a baseline",
        ],
        "claim_boundary": (
            "Synthetic implementation test only; no measured trace, operator campaign, "
            "operational EO workload, or field action is used."
        ),
    }


def run_integrated_fallback_stress(
    scenario_path: str | Path,
    *,
    config: EOSFullChainConfig | None = None,
) -> IntegratedFallbackArtifacts:
    """Exercise the real EOS guard-to-reactive-fallback full-chain path."""

    config = config or fallback_stress_config()
    if len(config.seeds) != 1:
        raise ValueError("fallback stress requires exactly one deterministic seed")
    seed = int(config.seeds[0])
    scenario = parse_eos_bench_scenario(scenario_path)
    domain_gate, gate_evidence, gate_sha256 = _fit_simulation_domain_gate(
        (REFERENCE_PROFILE,), config
    )
    config_payload = {
        "protocol_version": FALLBACK_STRESS_PROTOCOL,
        "config": asdict(config),
        "reference_profile": asdict(REFERENCE_PROFILE),
        "evaluation_profiles": [asdict(profile) for profile in UNSUPPORTED_PROFILES],
        "source_scenario_sha256": scenario.source_sha256,
    }
    config_sha256 = _hash_json(config_payload)
    model_sha256 = _hash_json(
        {
            "scheduler": "existing_eos_full_chain_schedule_policy",
            "fallback_policy": FALLBACK_POLICY_ID,
            "gate": "frozen_ledoit_wolf_mahalanobis",
            "gate_sha256": gate_sha256,
            "protocol_version": FALLBACK_STRESS_PROTOCOL,
        }
    )
    task_specs = _seed_task_table(scenario, seed, config)
    common_actions = _build_common_actions(
        scenario,
        task_specs,
        UNSUPPORTED_PROFILES,
        seed,
        config,
        domain_gate,
        gate_evidence,
    )
    arms = [
        _arm_tables(
            scenario=scenario,
            task_specs=task_specs,
            common_actions=common_actions,
            policy_arm=policy_arm,
            seed=seed,
            config=config,
            config_sha256=config_sha256,
            model_sha256=model_sha256,
        )
        for policy_arm in (POLICY_BASELINE, POLICY_INTEGRATED)
    ]
    tables = _concat_arm_tables(arms)
    audit = validate_full_chain_bundle(tables)
    audit.raise_for_errors()
    ledger = build_full_chain_ledger(_ledger_projection(tables))
    assertions = _build_fallback_assertions(tables, ledger)
    claims = _claims_ledger(tables, audit, assertions)
    profile_payload = {
        "evidence_tier": EVIDENCE_TIER,
        "reference_profile": asdict(REFERENCE_PROFILE),
        "deliberately_unsupported_evaluation_profiles": [
            asdict(profile) for profile in UNSUPPORTED_PROFILES
        ],
        "reference_and_evaluation_are_disjoint": True,
        "claim_boundary": "declared synthetic distributions, not measured network data",
    }
    run_manifest = {
        "protocol_version": FALLBACK_STRESS_PROTOCOL,
        "evidence_tier": EVIDENCE_TIER,
        "simulation_seed": seed,
        "scenario_id": scenario.scenario_id,
        "source_scenario_sha256": scenario.source_sha256,
        "config_sha256": config_sha256,
        "model_sha256": model_sha256,
        "gate_sha256": gate_sha256,
        "fallback_policy_id": FALLBACK_POLICY_ID,
        "rows": {name: int(len(frame)) for name, frame in tables.items()},
        "audit_passed": bool(audit.passed),
        "all_fallback_checks_passed": bool(all(claims["checks"].values())),
        "claim_boundary": claims["claim_boundary"],
    }
    payload = json.loads(Path(scenario_path).read_text(encoding="utf-8"))
    return IntegratedFallbackArtifacts(
        tables=tables,
        audit=audit,
        ledger=ledger,
        fallback_assertions=assertions,
        claims_ledger=claims,
        run_manifest=run_manifest,
        scenario_payload=payload,
        profile_payload=profile_payload,
    )


def run_synthetic_integrated_fallback(
    *, seed: int = 2026102801
) -> IntegratedFallbackArtifacts:
    """Run the frozen one-task fixture without relying on an external input file."""

    payload = synthetic_fallback_scenario_payload()
    with TemporaryDirectory(prefix="eos_fallback_stress_") as temp_directory:
        scenario_path = Path(temp_directory) / "scenario_fixture.json"
        scenario_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return run_integrated_fallback_stress(
            scenario_path, config=fallback_stress_config(seed)
        )


def _claims_markdown(claims: Mapping[str, Any]) -> str:
    counts = claims["counts"]
    checks = claims["checks"]
    lines = [
        "# Integrated fallback synthetic evidence ledger",
        "",
        f"Evidence tier: `{claims['evidence_tier']}`.",
        "",
        "## Supported implementation claim",
        "",
        str(claims["supported_claim"]),
        "",
        "## Exact counts",
        "",
    ]
    lines.extend(f"- `{name}`: {value}" for name, value in counts.items())
    lines.extend(["", "## Required checks", ""])
    lines.extend(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(claims["semantics"]["learned_choice_separation"]),
            "",
            str(claims["claim_boundary"]),
            "",
            "This artifact does **not** establish:",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in claims["unsupported_claims"])
    return "\n".join(lines) + "\n"


def write_integrated_fallback_artifacts(
    artifacts: IntegratedFallbackArtifacts, output_directory: str | Path
) -> Path:
    """Write a fresh, checksummed fallback artifact without overwriting prior runs."""

    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; choose a fresh run path: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in artifacts.tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    audit = artifacts.audit.checks.copy()
    audit.insert(0, "evidence_tier", EVIDENCE_TIER)
    audit.to_csv(destination / "full_chain_audit.csv", index=False)
    artifacts.ledger.to_csv(destination / "task_ledger.csv", index=False)
    artifacts.fallback_assertions.to_csv(
        destination / "fallback_assertions.csv", index=False
    )
    (destination / "claims_ledger.json").write_text(
        json.dumps(artifacts.claims_ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "CLAIMS_LEDGER.md").write_text(
        _claims_markdown(artifacts.claims_ledger), encoding="utf-8"
    )
    (destination / "scenario_fixture.json").write_text(
        json.dumps(artifacts.scenario_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "profile_provenance.json").write_text(
        json.dumps(artifacts.profile_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pre_manifest_hashes = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = dict(artifacts.run_manifest)
    manifest["artifact_sha256_before_manifest"] = pre_manifest_hashes
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_hashes = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (destination / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in final_hashes.items()),
        encoding="utf-8",
    )
    return destination
