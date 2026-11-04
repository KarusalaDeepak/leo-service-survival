from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from leo_service_survival.scheduler_release import (
    SchedulerReleaseCriteria,
    evaluate_scheduler_release,
    write_scheduler_release_artifacts,
)


def _write_inputs(root: Path, *, candidate_gain: bool) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    instances = ["a", "b", "c", "d", "e", "f"]
    candidate_values = [0.8, 0.7, 0.6, 0.75, 0.65, 0.55]
    baseline_values = (
        [0.7, 0.6, 0.5, 0.65, 0.55, 0.45]
        if candidate_gain
        else candidate_values
    )
    score = pd.DataFrame(
        {
            "instance": instances,
            "stress": ["combined"] * len(instances),
            "completion": candidate_values,
        }
    )
    matched = pd.DataFrame(
        {
            "instance": instances,
            "stress": ["combined"] * len(instances),
            "policy": ["full_margin"] * len(instances),
            "completion": baseline_values,
        }
    )
    rf = pd.DataFrame(
        [
            {
                "stress": "combined",
                "metric": "completion",
                "independent_units": len(instances),
                "mean_delta": 0.1 if candidate_gain else 0.0,
                "ci_low": 0.05 if candidate_gain else 0.0,
                "ci_high": 0.15 if candidate_gain else 0.0,
                "p_holm": 0.01 if candidate_gain else 1.0,
            }
        ]
    )
    transfer_summary = pd.DataFrame(
        [
            {
                "policy": "full_racs_candidate6",
                "scenario_units": len(instances),
                "completion_mean": 0.7 if candidate_gain else 0.6,
            },
            {
                "policy": "full_margin",
                "scenario_units": len(instances),
                "completion_mean": 0.6,
            },
        ]
    )
    transfer_pairs = pd.DataFrame(
        [
            {
                "reference": "full_racs_candidate6",
                "comparator": "full_margin",
                "scenario_units": len(instances),
                "mean_completion_delta": 0.1 if candidate_gain else 0.0,
                "ci_low": 0.05 if candidate_gain else 0.0,
                "ci_high": 0.15 if candidate_gain else 0.0,
                "holm_p": 0.01 if candidate_gain else 1.0,
            }
        ]
    )
    rows = []
    for instance in instances:
        rows.extend(
            [
                {
                    "instance": instance,
                    "policy": "full_racs_candidate6",
                    "completion": 0.7 if candidate_gain else 0.6,
                },
                {"instance": instance, "policy": "full_margin", "completion": 0.6},
            ]
        )
    transfer_instances = pd.DataFrame(rows)
    tables = {
        "score_outer_results": score,
        "matched_instance_results": matched,
        "paired_racs_rf_evidence": rf,
        "transfer_summary": transfer_summary,
        "transfer_paired_statistics": transfer_pairs,
        "transfer_instance_results": transfer_instances,
    }
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = root / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    manifest = root / "transfer_manifest.json"
    manifest.write_text(json.dumps({"validated_schedule_rows": 12}) + "\n")
    paths["transfer_manifest"] = manifest
    return paths


def test_scheduler_release_fails_closed_when_candidate_has_no_gain(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path / "inputs", candidate_gain=False)
    criteria = SchedulerReleaseCriteria(
        minimum_benchmark_units=6,
        minimum_transfer_scenarios=6,
        minimum_transfer_schedule_rows=12,
        bootstrap_resamples=200,
    )
    artifacts = evaluate_scheduler_release(**paths, criteria=criteria)
    assert artifacts.decision["decision"] == "RETAIN_MATCHED_SIMPLE_BASELINE"
    assert artifacts.decision["installed_scheduler"] == "full_margin"
    assert not artifacts.release_checks["passed"].all()
    assert artifacts.release_ledger["path_policy"].eq("reactive_baseline_retained").all()


def test_scheduler_release_can_authorize_complete_positive_fixture(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path / "inputs", candidate_gain=True)
    criteria = SchedulerReleaseCriteria(
        minimum_benchmark_units=6,
        minimum_transfer_scenarios=6,
        minimum_transfer_schedule_rows=12,
        bootstrap_resamples=200,
    )
    artifacts = evaluate_scheduler_release(**paths, criteria=criteria)
    assert artifacts.decision["decision"] == "AUTHORIZE_RACS"
    assert artifacts.release_checks["passed"].all()
    output = write_scheduler_release_artifacts(artifacts, tmp_path / "output")
    assert (output / "scheduler_release_decision.json").is_file()
    assert (output / "run_manifest.json").is_file()
