#!/usr/bin/env python3
"""Build a frozen audit of EOS OOD/fallback and COMMECT protocol claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leo_service_survival.evidence_audit import (  # noqa: E402
    audit_commect_paper2,
    audit_commect_paper4,
    audit_eos_candidate_fallback,
    audit_eos_effect,
    describe_gate_mechanisms,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "evidence_clarification_v1",
    )
    parser.add_argument(
        "--eos-dir",
        type=Path,
        default=ROOT / "results" / "eos_full_chain_synthetic_v2",
    )
    parser.add_argument(
        "--paper4-results-dir",
        type=Path,
        default=ROOT / "manuscript_run_2026-10-23_tnsm_v1" / "results",
    )
    parser.add_argument(
        "--gate-stress-dir",
        type=Path,
        default=ROOT / "completion_run_2026-10-27" / "gate_stress",
    )
    parser.add_argument(
        "--paper2-root",
        type=Path,
        default=ROOT.parent / "leo-merged-disagreement-evidence-14-page",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_record(path: Path, role: str) -> dict:
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _render_markdown(audit: dict) -> str:
    eos_gate = audit["eos_v2_candidate_ood_and_fallback"]
    eos_effect = audit["eos_v2_effect"]
    p4 = audit["commect_experiments"]["P4-COMMECT-5S-CS"]
    p2 = audit["commect_experiments"]["P2-COMMECT-1S-CC-FTR"]
    baseline = eos_effect["policy_results"]["edf_reactive"]
    integrated = eos_effect["policy_results"]["integrated_risk_gated"]
    ci = eos_effect["recorded_cluster_bootstrap_95_ci"]

    mechanism_rows = []
    for item in audit["gate_mechanisms"]:
        mechanism_rows.append(
            f"| `{item['mechanism_id']}` | {item['unit']} | {item['timing']} | "
            f"{item['not_this']} |"
        )

    return f"""# Evidence clarification: EOS v2, COMMECT protocols, and gate mechanisms

## EOS v2 candidate OOD does not imply decision fallback

The locked export contains **{eos_gate['exported_ood_candidate_rows']:,} OOD candidate rows out of {eos_gate['exported_candidate_rows']:,} exported rows**. The common candidate set is exported once for each paired policy arm, so this is {eos_gate['unique_common_action_ood_records']:,} OOD records among {eos_gate['unique_common_action_records']:,} unique common actions. The integrated arm has {eos_gate['by_policy_arm']['integrated_risk_gated']['ood_candidate_rows']:,} OOD rows among {eos_gate['by_policy_arm']['integrated_risk_gated']['candidate_rows']:,} candidate rows.

The v2 rule is:

> `{eos_gate['candidate_release_rule']}`

The export obeys that rule for **{eos_gate['release_rule_compliant_rows']:,}/{eos_gate['release_rule_total_rows']:,} rows**, with {eos_gate['release_rule_violations']} violations. Among {eos_gate['integrated_choice_sets']:,} integrated choice sets, {eos_gate['choice_sets_with_any_ood_candidate']:,} contain at least one OOD candidate and {eos_gate['choice_sets_with_any_resource_feasible_ood_candidate']:,} contain at least one resource-feasible OOD candidate. Nevertheless, every one of the {eos_gate['choice_sets_with_any_resource_feasible_candidate']:,} sets with any feasible action also has a released candidate. The decision log records {eos_gate['integrated_decision_gate_states'].get('released', 0):,} `released`, {eos_gate['integrated_decision_gate_states'].get('no_action', 0):,} `no_action`, and **{eos_gate['integrated_fallback_decisions']} `fallback`** decisions. All {eos_gate['selected_integrated_actions']:,} selected integrated actions are released and in-domain.

Therefore, **714/59,640 is a candidate-row screening count, not a fallback count**. An OOD candidate can coexist with another eligible candidate. This run verifies screening but does not exercise or validate the reactive fallback branch.

## Exact EOS +17.9 percentage-point construction

The endpoint is `conservative_on_time_completion`. It is one only when the action executes, is not no-action, has no failed or late install, has feasible geometry, produces a valid observation and product, delivers at least the declared bytes, completes, and meets the deadline. The exported endpoint matches this conjunction for {eos_effect['endpoint_rule_compliant_rows']:,}/{eos_effect['endpoint_rule_total_rows']:,} task-arm rows.

| Arm | Successes | Denominator | Rate |
|---|---:|---:|---:|
| EDF + reactive | {baseline['successful_tasks']} | {baseline['denominator_tasks']} | {baseline['completion_rate']:.3f} |
| Integrated risk-gated | {integrated['successful_tasks']} | {integrated['denominator_tasks']} | {integrated['completion_rate']:.3f} |

The paired effect is ({integrated['successful_tasks']} - {baseline['successful_tasks']}) / {eos_effect['paired_task_denominator']} = **{eos_effect['paired_completion_difference']:.3f}, or {eos_effect['paired_completion_difference_percentage_points']:.1f} percentage points**. The 95% interval is **[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] percentage points**. It is a 2,000-resample percentile bootstrap over ten simulation-seed clusters; each sampled seed carries all 100 paired tasks. The ten seeds reuse one EOS-Bench scenario, so this is not an independent-campaign interval.

## The two COMMECT results are different experiments

| Experiment ID | Horizon and endpoint | Denominator | Candidate | Baseline | Meaning |
|---|---|---:|---:|---:|---|
| `{p4['experiment_id']}` | Five future 1-s means must all be <=60 ms | {p4['scheduled_test_seconds']:,} scheduled seconds | {p4['candidate_successes']}/{p4['scheduled_test_seconds']} = {100*p4['candidate_success_rate']:.2f}% | {p4['reactive_successes']}/{p4['scheduled_test_seconds']} = {100*p4['reactive_success_rate']:.2f}% | Ungated candidate; one-campaign admission keeps reactive |
| `{p2['experiment_id']}` | Exact next 1-s mean RTT <=60 ms | {p2['complete_case_decisions']:,} complete cases | {p2['candidate_successes']}/{p2['complete_case_decisions']} = {100*p2['candidate_success_rate']:.1f}% | {p2['baseline_successes']}/{p2['complete_case_decisions']} = {100*p2['baseline_success_rate']:.1f}% | Unscreened filter-then-rank ablation; not authorized |

`P4-COMMECT-5S-CS` uses a purged 60/20/20 split, a five-second survival policy, and the full scheduled test grid; missing actions or unevaluable futures are zero. `P2-COMMECT-1S-CC-FTR` uses a fixed 55/15/15/15 train/calibration/selection/test split and reports complete-case next-interval success. They share a source campaign but are neither the same target nor the same denominator, policy, or split. They must not be described as replications.

## Three distinct safeguards

| Mechanism | Unit | When it acts | Must not be conflated with |
|---|---|---|---|
{chr(10).join(mechanism_rows)}

The immediate and delayed operating-characteristic counts are from injected synthetic streams. They do not validate prospective field false-alarm, detection, or fallback-safety rates. The offline campaign-admission gate is a study-level release decision and is not inferred from a single OOD candidate or a delayed synthetic alarm.

## Claim limits

- EOS v2 is `synthetic_simulation`; the +17.9-point result is not a field effect.
- Zero fallback executions means the fallback branch was not empirically exercised in EOS v2.
- Both COMMECT experiments are one-campaign shadow replay; neither installed a forwarding action.
- Candidate screening, delayed monitoring, and campaign admission require separate evaluation.
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    eos_candidate_path = args.eos_dir / "candidate_path_observation.csv"
    eos_decision_path = args.eos_dir / "decision_action.csv"
    eos_ledger_path = args.eos_dir / "task_ledger.csv"
    eos_seed_path = args.eos_dir / "paired_seed_results.csv"
    eos_inference_path = args.eos_dir / "paired_inference.json"

    p4_summary_path = args.paper4_results_dir / "commect_policy_summary.csv"
    p4_inference_path = args.paper4_results_dir / "commect_time_series_inference.csv"
    p4_split_path = args.paper4_results_dir / "commect_split_audit.json"

    p2_result_dir = (
        args.paper2_root
        / "supplementary"
        / "primary_evidence"
        / "results"
        / "commect_1s_alignment_audit"
        / "full"
    )
    p2_summary_path = p2_result_dir / "policy_summary.csv"
    p2_metadata_path = p2_result_dir / "validation_metadata.json"
    p2_manuscript_path = args.paper2_root / "new_main.tex"

    immediate_path = args.gate_stress_dir / "gate_immediate_summary.csv"
    delayed_path = args.gate_stress_dir / "gate_delayed_summary.csv"

    candidates = pd.read_csv(
        eos_candidate_path,
        usecols=[
            "campaign_id",
            "decision_id",
            "simulation_seed",
            "logical_task_id",
            "candidate_path_id",
            "resource_feasible",
            "gate_release",
            "gate_feature_coverage",
            "gate_ood_fraction",
            "gate_reasons",
        ],
    )
    decisions = pd.read_csv(
        eos_decision_path,
        usecols=[
            "campaign_id",
            "decision_id",
            "simulation_seed",
            "logical_task_id",
            "policy_arm",
            "selected_path_id",
            "gate_state",
            "gate_reason",
            "shift_alarm_state",
        ],
    )
    ledger_columns = [
        "policy_arm",
        "simulation_seed",
        "logical_task_id",
        "conservative_on_time_completion",
        "action_executed",
        "no_action",
        "install_failed",
        "late_install",
        "geometry_allows_task",
        "observation_acquired",
        "product_valid",
        "bytes_delivered",
        "required_bytes",
        "application_complete",
        "met_deadline",
    ]
    ledger = pd.read_csv(eos_ledger_path, usecols=ledger_columns)
    paired_seed = pd.read_csv(eos_seed_path)
    eos_inference = _read_json(eos_inference_path)

    eos_gate_audit = audit_eos_candidate_fallback(candidates, decisions)
    eos_effect_audit = audit_eos_effect(ledger, paired_seed, eos_inference)
    p4_audit = audit_commect_paper4(
        pd.read_csv(p4_summary_path),
        pd.read_csv(p4_inference_path),
        _read_json(p4_split_path),
    )
    p2_audit = audit_commect_paper2(
        pd.read_csv(p2_summary_path),
        _read_json(p2_metadata_path),
    )
    mechanisms = describe_gate_mechanisms(
        pd.read_csv(immediate_path), pd.read_csv(delayed_path)
    )

    source_specs = [
        (eos_candidate_path, "EOS v2 candidate rows"),
        (eos_decision_path, "EOS v2 decision actions"),
        (eos_ledger_path, "EOS v2 endpoint ledger"),
        (eos_seed_path, "EOS v2 seed-cluster summaries"),
        (eos_inference_path, "EOS v2 recorded paired inference"),
        (p4_summary_path, "Paper 4 COMMECT policy summary"),
        (p4_inference_path, "Paper 4 COMMECT moving-block inference"),
        (p4_split_path, "Paper 4 COMMECT split audit"),
        (p2_summary_path, "Paper 2 fixed-1s COMMECT policy summary"),
        (p2_metadata_path, "Paper 2 fixed-1s COMMECT validation metadata"),
        (p2_manuscript_path, "Paper 2 protocol text"),
        (immediate_path, "synthetic immediate-guard stress summary"),
        (delayed_path, "synthetic delayed-monitor stress summary"),
        (ROOT / "src" / "leo_service_survival" / "eos_full_chain.py", "EOS v2 rules"),
        (ROOT / "src" / "leo_service_survival" / "integration.py", "EOS endpoint rule"),
        (ROOT / "src" / "leo_service_survival" / "gating.py", "gate and monitor rules"),
        (ROOT / "src" / "leo_service_survival" / "decisions.py", "Paper 4 replay policy"),
        (ROOT / "src" / "leo_service_survival" / "data.py", "Paper 4 COMMECT target/split"),
        (ROOT / "scripts" / "run_all.py", "Paper 4 execution protocol"),
    ]
    source_records = [_source_record(path, role) for path, role in source_specs]

    audit = {
        "schema_version": 1,
        "artifact_id": "evidence-clarification-v1",
        "purpose": (
            "Separate EOS candidate OOD screening from decision fallback, document "
            "the EOS paired interval, distinguish two COMMECT protocols, and keep "
            "three safeguard mechanisms non-interchangeable."
        ),
        "eos_v2_candidate_ood_and_fallback": eos_gate_audit,
        "eos_v2_effect": eos_effect_audit,
        "commect_experiments": {
            p4_audit["experiment_id"]: p4_audit,
            p2_audit["experiment_id"]: p2_audit,
        },
        "commect_results_are_replications": False,
        "gate_mechanisms": mechanisms,
        "source_artifacts": source_records,
    }

    json_path = args.output_dir / "evidence_clarification.json"
    markdown_path = args.output_dir / "EVIDENCE_CLARIFICATION.md"
    json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(audit), encoding="utf-8")

    manifest = {
        "artifact_id": "evidence-clarification-v1",
        "source_artifacts": source_records,
        "generated_artifact_sha256": {
            json_path.name: sha256_file(json_path),
            markdown_path.name: sha256_file(markdown_path),
        },
        "reran_eos_simulation": False,
        "reason_no_eos_rerun": (
            "Frozen EOS v2 candidate and action logs contain the required row-level "
            "fields; this is an audit-only derivation."
        ),
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = [json_path, markdown_path, manifest_path]
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in sorted(checksum_paths)
    )
    (args.output_dir / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")

    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "candidate_ood_rows": eos_gate_audit["exported_ood_candidate_rows"],
        "fallback_decisions": eos_gate_audit["integrated_fallback_decisions"],
        "release_rule_violations": eos_gate_audit["release_rule_violations"],
        "eos_paired_difference": eos_effect_audit["paired_completion_difference"],
        "eos_ci": eos_effect_audit["recorded_cluster_bootstrap_95_ci"],
    }, indent=2))


if __name__ == "__main__":
    main()
