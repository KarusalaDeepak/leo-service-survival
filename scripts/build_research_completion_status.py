#!/usr/bin/env python3
"""Build an evidence-backed status ledger for the six requested completion criteria.

The ledger deliberately distinguishes an implemented engineering mechanism from a
scientifically satisfied field-validation criterion.  It reads only canonical,
versioned result directories and refuses to overwrite an existing status bundle.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "research_completion_status_v7"


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(root: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    eos_dir = root / "results" / "eos_full_chain_synthetic_v2"
    public_dir = root / "results" / "public_trace_validation_v3"
    gate_dir = root / "completion_run_2026-10-27" / "gate_stress"
    fallback_dir = root / "results" / "eos_integrated_fallback_synthetic_v2"

    eos_status_path = eos_dir / "completion_status.json"
    eos_inference_path = eos_dir / "paired_inference.json"
    eos_candidates_path = eos_dir / "candidate_path_observation.csv"
    eos_actions_path = eos_dir / "decision_action.csv"
    public_manifest_path = public_dir / "validation_manifest.json"
    public_metrics_path = public_dir / "cross_campaign_replay_proxy_metrics.csv"
    public_uncertainty_path = public_dir / "cross_campaign_replay_proxy_uncertainty.csv"
    public_coverage_path = public_dir / "campaign_site_operator_coverage.csv"
    gate_manifest_path = gate_dir / "run_manifest.json"
    gate_immediate_path = gate_dir / "gate_immediate_summary.csv"
    gate_delayed_path = gate_dir / "gate_delayed_summary.csv"
    fallback_claims_path = fallback_dir / "claims_ledger.json"
    fallback_assertions_path = fallback_dir / "fallback_assertions.csv"
    fallback_manifest_path = fallback_dir / "run_manifest.json"
    matched_dir = root / "results" / "matched_telemetry_audit_v1"
    matched_status_path = matched_dir / "criterion_status.json"
    matched_summary_path = matched_dir / "matched_telemetry_summary.csv"
    oneweb_transitions_path = (
        root
        / "data/external/lens_oneweb_iowa_2024/canonical/transition_descriptives.csv"
    )
    stock_ns3_path = (
        root
        / "simulation/ns3_5g_lena/results/stock_vsat_realistic_final/summary.json"
    )
    uw_ns3_path = root / "simulation/ns3_ntn/results/final/summary.json"

    required = [
        eos_status_path,
        eos_inference_path,
        eos_candidates_path,
        eos_actions_path,
        public_manifest_path,
        public_metrics_path,
        public_uncertainty_path,
        public_coverage_path,
        gate_manifest_path,
        gate_immediate_path,
        gate_delayed_path,
        fallback_claims_path,
        fallback_assertions_path,
        fallback_manifest_path,
        matched_status_path,
        matched_summary_path,
        oneweb_transitions_path,
        stock_ns3_path,
        uw_ns3_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing canonical evidence:\n" + "\n".join(missing))

    eos_status = _read_json(eos_status_path)
    eos_inference = _read_json(eos_inference_path)
    eos_gate = pd.read_csv(
        eos_candidates_path,
        usecols=[
            "campaign_id",
            "decision_id",
            "candidate_path_id",
            "gate_method",
            "gate_ood_fraction",
            "gate_release",
            "gate_scope",
        ],
    )
    eos_actions = pd.read_csv(eos_actions_path)
    public_manifest = _read_json(public_manifest_path)
    public_metrics = pd.read_csv(public_metrics_path)
    public_uncertainty = pd.read_csv(public_uncertainty_path)
    coverage = pd.read_csv(public_coverage_path)
    gate_manifest = _read_json(gate_manifest_path)
    gate_immediate = pd.read_csv(gate_immediate_path)
    gate_delayed = pd.read_csv(gate_delayed_path)
    fallback_claims = _read_json(fallback_claims_path)
    fallback_assertions = pd.read_csv(fallback_assertions_path)
    if not fallback_claims.get("checks") or not all(
        bool(value) for value in fallback_claims["checks"].values()
    ):
        raise AssertionError("integrated fallback checks are incomplete")
    if len(fallback_assertions) != 1:
        raise AssertionError("expected one deterministic integrated fallback decision")

    logistic = public_metrics.loc[public_metrics["model"].eq("logistic_lag_model")].iloc[0]
    persistence = public_metrics.loc[
        public_metrics["model"].eq("current_state_persistence")
    ].iloc[0]
    replay_delta = public_uncertainty.loc[
        public_uncertainty["model"].eq("logistic_lag_model")
        & public_uncertainty["reference"].eq("current_state_persistence")
    ].iloc[0]

    campaign_contexts = int(coverage["campaign_or_site"].nunique())
    operator_contexts = sorted(
        str(value)
        for value in coverage["operator_id"].dropna().astype(str).unique()
    )
    immediate = gate_immediate.iloc[0]
    delayed = gate_delayed.iloc[0]
    integrated_gate = eos_gate.loc[
        eos_gate["campaign_id"].str.endswith("::integrated_risk_gated")
    ].copy()
    integrated_actions = eos_actions.loc[
        eos_actions["policy_arm"].eq("integrated_risk_gated")
    ].copy()
    selected_gate = integrated_actions.loc[
        integrated_actions["selected_path_id"].notna(),
        ["decision_id", "selected_path_id"],
    ].merge(
        integrated_gate[
            ["decision_id", "candidate_path_id", "gate_release", "gate_ood_fraction"]
        ],
        left_on=["decision_id", "selected_path_id"],
        right_on=["decision_id", "candidate_path_id"],
        how="left",
        validate="one_to_one",
    )
    if selected_gate["gate_release"].isna().any():
        raise AssertionError("selected integrated actions do not resolve to candidate rows")
    selected_released = int(selected_gate["gate_release"].sum())
    selected_ood = int(selected_gate["gate_ood_fraction"].gt(0).sum())
    fallback_decisions = int(integrated_actions["gate_state"].eq("fallback").sum())
    no_action_decisions = int(integrated_actions["gate_state"].eq("no_action").sum())

    rows = [
        {
            "criterion_id": "C1",
            "criterion": "continuous_geometry_path_scheduling_actual_completion",
            "engineering_status": "complete_at_synthetic_simulation_tier",
            "research_completion_status": "not_satisfied_at_field_tier",
            "best_evidence_tier": "synthetic_simulation",
            "evidence": (
                f"One EOS-Bench task identifier is preserved across geometry, path choice, "
                f"scheduling, installation, realized simulated bytes, deadline, and completion. "
                f"Across the full denominator of {eos_inference['paired_tasks']} paired simulated "
                f"tasks, including no-action and failed-delivery cases as failures, completion was "
                f"{eos_inference['baseline_completion_rate']:.3f} for combined EDF scheduling plus "
                f"reactive highest-current-SNR path ranking and "
                f"{eos_inference['integrated_completion_rate']:.3f} for the integrated policy "
                f"(difference {100 * eos_inference['paired_completion_difference']:.1f} pp; "
                f"2,000-resample seed-cluster percentile interval "
                f"[{100 * eos_inference['cluster_bootstrap_95_ci'][0]:.1f}, "
                f"{100 * eos_inference['cluster_bootstrap_95_ci'][1]:.1f}] pp across ten "
                f"simulation seeds). The 36/36 audit establishes tested traceability and "
                f"data-contract invariants, not physical-model validity. "
                f"Separate stock 5G-LENA and UW ns-3-NTN executions verify packet-level NTN code "
                f"paths, but are not treated as field outcomes."
            ),
            "why_not_complete": (
                "The workload, network overlay, installed action, delivered bytes, and completion "
                "outcome are simulated rather than measured in one live chain."
            ),
            "required_next_evidence": (
                "Run the frozen controller on a live task stream and retain the complete "
                "intention-to-treat denominator from sensed geometry through application receipt."
            ),
        },
        {
            "criterion_id": "C2",
            "criterion": "independent_multisite_or_multioperator_validation",
            "engineering_status": "public_adapters_and_cross_campaign_replay_complete",
            "research_completion_status": "not_satisfied_for_live_policy",
            "best_evidence_tier": "field_observational_and_measured_trace_replay",
            "evidence": (
                f"The audited public bundle covers {campaign_contexts} campaign/site contexts and "
                f"operator labels {', '.join(operator_contexts)}. An Alaska-to-Hawaii fixed-payload "
                f"replay used {int(logistic['test_run_clusters'])} Hawaii run clusters; logistic "
                f"Brier was {logistic['brier_score']:.4f} versus {persistence['brier_score']:.4f} "
                f"for persistence, with delta interval [{replay_delta['delta_brier_ci_low']:.4f}, "
                f"{replay_delta['delta_brier_ci_high']:.4f}]."
            ),
            "why_not_complete": (
                "These sources are observational, use different feature contracts, and do not "
                "contain randomized installed actions plus application outcomes."
            ),
            "required_next_evidence": (
                "Collect prospectively frozen, opportunity-bearing live clusters at independent "
                "sites and operator/terminal contexts with the same controller and endpoint."
            ),
        },
        {
            "criterion_id": "C3",
            "criterion": "matched_satellite_beam_gateway_handover_rf_observations",
            "engineering_status": "partial_public_matched_telemetry_acquired",
            "research_completion_status": "not_satisfied",
            "best_evidence_tier": "field_observational",
            "evidence": (
                "SatInView supplies post-processed Starlink satellite/geometry/RTT matches. The "
                "LENS OneWeb Iowa interval supplies terminal assignment identifiers, source-reported "
                "SINR, RTT, and TLE association around two assignment transitions."
            ),
            "why_not_complete": (
                "Neither public source records measured beam identity and gateway identity; Starlink "
                "serving identity is inferred and OneWeb transitions are not operator-confirmed handovers."
            ),
            "required_next_evidence": (
                "Instrument one campaign with operator-confirmed serving satellite, beam, gateway, "
                "handover start/end/success, numeric RF telemetry, geometry, and service outcomes."
            ),
        },
        {
            "criterion_id": "C4",
            "criterion": "operational_eo_workload_validation",
            "engineering_status": "controlled_benchmark_execution_complete",
            "research_completion_status": "not_satisfied",
            "best_evidence_tier": "synthetic_simulation",
            "evidence": (
                "The continuous runner executes EOS-Bench opportunities and a strict task ledger, "
                "and SPOT5 instances are available for scheduler sanity tests."
            ),
            "why_not_complete": (
                "EOS-Bench and SPOT5 are generated research benchmarks, not an operator mission log "
                "with executed observation, product validation, downlink, and delivery acknowledgments."
            ),
            "required_next_evidence": (
                "Obtain or collect a real EO mission-day workload and independently verified "
                "observation-to-delivery outcomes under common resource constraints."
            ),
        },
        {
            "criterion_id": "C5",
            "criterion": "data_driven_gate_or_distribution_shift_detector",
            "engineering_status": "implemented_and_synthetically_exercised",
            "research_completion_status": "not_satisfied_for_deployment",
            "best_evidence_tier": "synthetic_simulation",
            "evidence": (
                f"Three safeguards have separate scopes: a candidate-level Mahalanobis input-domain "
                f"guard, a delayed outcome stream monitor, and an offline campaign-level policy-"
                f"admission gate. In the synthetic stress, immediate false alarms were "
                f"{int(immediate['false_alarm_campaigns'])}/{int(immediate['stable_campaigns'])} "
                f"stable-stream trials, and injected shifts were detected in "
                f"{int(immediate['detected_shift_campaigns'])}/{int(immediate['shift_campaigns'])}; "
                f"median immediate delay was {int(immediate['median_detection_delay_rows'])} rows. "
                f"The delayed monitor detected {int(delayed['detected_shift_campaigns'])}/"
                f"{int(delayed['shift_campaigns'])} injected shifts with median delay "
                f"{int(delayed['median_detection_delay_outcomes'])} matured outcomes. These are "
                f"observed synthetic counts, not perfect field rates. The EOS export flags "
                f"{int(eos_gate['gate_ood_fraction'].gt(0).sum())}/{len(eos_gate)} candidate rows "
                f"across both arms ({int(integrated_gate['gate_ood_fraction'].gt(0).sum())}/"
                f"{len(integrated_gate)} in the integrated arm). Action logs show "
                f"{selected_released} selected released actions, {selected_ood} selected OOD actions, "
                f"{no_action_decisions} no-action tasks, and {fallback_decisions} executed reactive "
                f"fallbacks; therefore the locked comparative EOS run demonstrates screening but "
                f"not fallback execution. A separate purpose-built synthetic integration stress then "
                f"rejected all {fallback_claims['counts']['unique_common_candidates']} unsupported "
                f"candidates from learned release, gate-released zero, rejected the counterfactual "
                f"synthetic integrated-ranking choice, and installed an OOD candidate through the "
                f"declared reactive fallback for "
                f"{fallback_claims['counts']['integrated_fallback_decisions']}/"
                f"{fallback_claims['counts']['integrated_decisions']} decision. Its 9/9 explicit "
                f"fallback assertions and 36/36 relational checks passed, and the task completed on "
                f"time with task/resource/deadline/outcome identity preserved. The guard therefore "
                f"fails closed for learned-action release, not for all action. This verifies only "
                f"synthetic branch execution and traceability, not a safety rate, EDF effect, "
                f"comparative benefit, or field behavior."
            ),
            "why_not_complete": (
                "False-alarm and detection-delay behavior has not been prospectively measured on "
                "independent live campaigns, and gate evidence is not yet disjoint from a deployment trial."
            ),
            "required_next_evidence": (
                "Freeze thresholds on development campaigns, then measure false alarms, detection "
                "delay, rollback, and safety on untouched live campaigns."
            ),
        },
        {
            "criterion_id": "C6",
            "criterion": "consistent_outperformance_of_baselines",
            "engineering_status": "positive_only_in_declared_controlled_stress_and_proxy_replay",
            "research_completion_status": "not_satisfied",
            "best_evidence_tier": "synthetic_simulation_and_measured_trace_replay",
            "evidence": (
                f"All {eos_inference['synthetic_replications']} EOS synthetic seeds favored the "
                f"integrated policy and the Alaska-to-Hawaii replay favored the lag model over "
                f"persistence. In contrast, prior held-out evidence did not establish survival-model "
                f"superiority over Cox, and the prior survival scheduler did not beat EDF."
            ),
            "why_not_complete": (
                "Benefits are not yet reproduced on a prespecified primary endpoint across independent "
                "live campaigns; universal dominance is neither demonstrated nor scientifically required."
            ),
            "required_next_evidence": (
                "Use the strongest frozen deployable baseline and require the cluster-level confidence "
                "bound for a preregistered practical completion gain to clear zero in external trials."
            ),
        },
    ]

    inputs = {str(path.relative_to(root)): _hash(path) for path in required}
    inputs["results/eos_full_chain_synthetic_v2/completion_status.json"] = _hash(
        eos_status_path
    )
    inputs["results/public_trace_validation_v3/validation_manifest.json"] = _hash(
        public_manifest_path
    )
    inputs["completion_run_2026-10-27/gate_stress/run_manifest.json"] = _hash(
        gate_manifest_path
    )
    return rows, dict(sorted(inputs.items()))


def _write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Research completion status — 28 October 2026",
        "",
        "## Verdict",
        "",
        "The implementation, experiments, tests, documentation, and reproducibility package specified "
        "in the current research scope are complete. The planned integrated framework and its "
        "controlled and trace-based evaluation have been completed. Prospective field validation "
        "remains outside the completed evaluation. The table below separates those statements so "
        "that a software result is never presented as operator evidence.",
        "",
        "| ID | Required criterion | Engineering status | Research status | Best tier |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['criterion_id']} | {str(row['criterion']).replace('_', ' ')} | "
            f"{str(row['engineering_status']).replace('_', ' ')} | "
            f"**{str(row['research_completion_status']).replace('_', ' ')}** | "
            f"{str(row['best_evidence_tier']).replace('_', ' ')} |"
        )
    lines += ["", "## Evidence and remaining closure condition", ""]
    for row in rows:
        lines += [
            f"### {row['criterion_id']}. {str(row['criterion']).replace('_', ' ').title()}",
            "",
            f"**What is complete:** {row['evidence']}",
            "",
            f"**Why the research criterion remains open:** {row['why_not_complete']}",
            "",
            f"**Evidence needed to close it:** {row['required_next_evidence']}",
            "",
        ]
    lines += [
        "## Claim rule",
        "",
        "The implementation, experiments, tests, documentation, and reproducibility package specified "
        "in the current research scope are complete. Whether that body of work is sufficient for the "
        "degree is a judgment for the supervisor and examination committee. A claim that the "
        "method *improves* operational completion is stronger: it requires a frozen practical effect "
        "threshold and an external cluster-level confidence bound that clears that threshold.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows, inputs = build_rows(root)
    frame = pd.DataFrame(rows)
    csv_path = output / "research_completion_status.csv"
    json_path = output / "research_completion_status.json"
    md_path = output / "research_completion_status.md"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "as_of_date": "2026-10-28",
                "all_six_field_research_criteria_satisfied": False,
                "engineering_framework_continuous_at_synthetic_tier": True,
                "criteria": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(rows, md_path)
    artifacts = [csv_path, json_path, md_path]
    manifest = {
        "inputs_sha256": inputs,
        "artifacts_sha256_before_manifest": {
            path.name: _hash(path) for path in artifacts
        },
        "claim_boundary": (
            "Engineering implementation and evidence audit. A status of not satisfied cannot be "
            "upgraded by simulation, trace replay, or cross-source joining."
        ),
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(frame[["criterion_id", "engineering_status", "research_completion_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
