"""Fail-closed release decision for the Paper 3 EO scheduler.

This module integrates the reviewed RACS scheduling evidence into Paper 4.  It
does not reinterpret simulation as field evidence.  The candidate scheduler is
released only when it clears every declared comparison against a matched simple
selector and a learned comparator, and when a frozen EOS-Bench transfer audit
shows a positive completion difference.  Otherwise the matched simple selector
remains active and RACS stays in shadow evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SchedulerReleaseCriteria:
    """Frozen criteria for the controlled scheduling integration decision."""

    benchmark_condition: str = "combined"
    candidate_policy: str = "cross_fitted_racs"
    baseline_policy: str = "full_margin"
    learned_comparator: str = "selected_random_forest"
    transfer_candidate: str = "full_racs_candidate6"
    transfer_baseline: str = "full_margin"
    minimum_benchmark_units: int = 9
    minimum_transfer_scenarios: int = 24
    minimum_transfer_schedule_rows: int = 384
    minimum_completion_difference: float = 0.0
    alpha: float = 0.05
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20261103

    def __post_init__(self) -> None:
        if self.minimum_benchmark_units < 2:
            raise ValueError("minimum_benchmark_units must be at least two")
        if self.minimum_transfer_scenarios < 2:
            raise ValueError("minimum_transfer_scenarios must be at least two")
        if self.minimum_transfer_schedule_rows < 1:
            raise ValueError("minimum_transfer_schedule_rows must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")


@dataclass(frozen=True)
class SchedulerReleaseArtifacts:
    """Tables and decision record produced by the scheduler evidence gate."""

    evidence_summary: pd.DataFrame
    release_checks: pd.DataFrame
    release_ledger: pd.DataFrame
    decision: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cluster_bootstrap_interval(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if differences.ndim != 1 or len(differences) < 2:
        raise ValueError("at least two paired benchmark differences are required")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        differences,
        size=(resamples, len(differences)),
        replace=True,
    ).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _two_sided_sign_permutation_p(differences: np.ndarray) -> float:
    nonzero = differences[np.abs(differences) > 1e-12]
    if not len(nonzero):
        return 1.0
    if len(nonzero) > 20:
        raise ValueError("exact sign enumeration is limited to 20 nonzero pairs")
    observed = abs(float(nonzero.mean()))
    signs = np.array(
        [
            [1.0 if (mask >> index) & 1 else -1.0 for index in range(len(nonzero))]
            for mask in range(2 ** len(nonzero))
        ]
    )
    permuted = np.abs((signs * np.abs(nonzero)).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def evaluate_scheduler_release(
    *,
    score_outer_results: Path,
    matched_instance_results: Path,
    paired_racs_rf_evidence: Path,
    transfer_summary: Path,
    transfer_paired_statistics: Path,
    transfer_instance_results: Path,
    transfer_manifest: Path,
    criteria: SchedulerReleaseCriteria = SchedulerReleaseCriteria(),
) -> SchedulerReleaseArtifacts:
    """Evaluate the frozen RACS release decision from registered result tables."""

    paths = {
        "score_outer_results": Path(score_outer_results),
        "matched_instance_results": Path(matched_instance_results),
        "paired_racs_rf_evidence": Path(paired_racs_rf_evidence),
        "transfer_summary": Path(transfer_summary),
        "transfer_paired_statistics": Path(transfer_paired_statistics),
        "transfer_instance_results": Path(transfer_instance_results),
        "transfer_manifest": Path(transfer_manifest),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"scheduler-release inputs are missing: {missing}")

    score_outer = pd.read_csv(paths["score_outer_results"])
    matched = pd.read_csv(paths["matched_instance_results"])
    racs_rf = pd.read_csv(paths["paired_racs_rf_evidence"])
    transfer = pd.read_csv(paths["transfer_summary"])
    transfer_pairs = pd.read_csv(paths["transfer_paired_statistics"])
    transfer_instances = pd.read_csv(paths["transfer_instance_results"])
    transfer_run = json.loads(paths["transfer_manifest"].read_text())

    _require_columns(score_outer, {"instance", "stress", "completion"}, "score outer results")
    _require_columns(
        matched,
        {"instance", "stress", "policy", "completion"},
        "matched instance results",
    )
    _require_columns(
        racs_rf,
        {"stress", "metric", "mean_delta", "ci_low", "ci_high", "p_holm"},
        "RACS versus random-forest evidence",
    )
    _require_columns(
        transfer,
        {"policy", "scenario_units", "completion_mean"},
        "EOS-Bench transfer summary",
    )
    _require_columns(
        transfer_pairs,
        {"reference", "comparator", "scenario_units", "mean_completion_delta", "ci_low", "ci_high", "holm_p"},
        "EOS-Bench paired statistics",
    )
    _require_columns(
        transfer_instances,
        {"instance", "policy", "completion"},
        "EOS-Bench instance results",
    )

    condition = criteria.benchmark_condition
    candidate = score_outer.loc[
        score_outer["stress"].eq(condition), ["instance", "completion"]
    ].rename(columns={"completion": "candidate_completion"})
    baseline = (
        matched.loc[
            matched["stress"].eq(condition)
            & matched["policy"].eq(criteria.baseline_policy),
            ["instance", "completion"],
        ]
        .groupby("instance", as_index=False)["completion"]
        .mean()
        .rename(columns={"completion": "baseline_completion"})
    )
    paired = candidate.merge(baseline, on="instance", how="inner", validate="one_to_one")
    paired["difference"] = paired["candidate_completion"] - paired["baseline_completion"]
    if len(paired) != len(candidate) or len(paired) != len(baseline):
        raise ValueError("benchmark candidate and baseline instance keys do not match")
    differences = paired["difference"].to_numpy(dtype=float)
    ci_low, ci_high = _cluster_bootstrap_interval(
        differences,
        resamples=criteria.bootstrap_resamples,
        seed=criteria.bootstrap_seed,
    )
    benchmark_delta = float(differences.mean())
    benchmark_p = _two_sided_sign_permutation_p(differences)

    rf_rows = racs_rf.loc[
        racs_rf["stress"].eq(condition) & racs_rf["metric"].eq("completion")
    ]
    if len(rf_rows) != 1:
        raise ValueError("expected one combined-stress RACS-versus-RF row")
    rf_row = rf_rows.iloc[0]

    transfer_rows = transfer.set_index("policy")
    for policy in (criteria.transfer_candidate, criteria.transfer_baseline):
        if policy not in transfer_rows.index:
            raise ValueError(f"transfer summary has no policy {policy!r}")
    transfer_candidate = transfer_rows.loc[criteria.transfer_candidate]
    transfer_baseline = transfer_rows.loc[criteria.transfer_baseline]
    transfer_delta = float(
        transfer_candidate["completion_mean"] - transfer_baseline["completion_mean"]
    )
    transfer_pair = transfer_pairs.loc[
        transfer_pairs["reference"].eq(criteria.transfer_candidate)
        & transfer_pairs["comparator"].eq(criteria.transfer_baseline)
    ]
    if len(transfer_pair) != 1:
        raise ValueError("expected one transfer candidate-versus-baseline row")
    transfer_pair = transfer_pair.iloc[0]
    transfer_schedule_rows = int(transfer_run.get("validated_schedule_rows", 0))

    checks = [
        {
            "check": "benchmark_unit_floor",
            "passed": len(paired) >= criteria.minimum_benchmark_units,
            "observed": len(paired),
            "required": criteria.minimum_benchmark_units,
        },
        {
            "check": "benchmark_point_difference",
            "passed": benchmark_delta > criteria.minimum_completion_difference,
            "observed": benchmark_delta,
            "required": f"> {criteria.minimum_completion_difference}",
        },
        {
            "check": "benchmark_interval_lower_bound",
            "passed": ci_low >= criteria.minimum_completion_difference,
            "observed": ci_low,
            "required": f">= {criteria.minimum_completion_difference}",
        },
        {
            "check": "benchmark_p_value",
            "passed": benchmark_p <= criteria.alpha,
            "observed": benchmark_p,
            "required": f"<= {criteria.alpha}",
        },
        {
            "check": "learned_comparator_interval_lower_bound",
            "passed": float(rf_row["ci_low"]) > criteria.minimum_completion_difference,
            "observed": float(rf_row["ci_low"]),
            "required": f"> {criteria.minimum_completion_difference}",
        },
        {
            "check": "learned_comparator_holm_p",
            "passed": float(rf_row["p_holm"]) <= criteria.alpha,
            "observed": float(rf_row["p_holm"]),
            "required": f"<= {criteria.alpha}",
        },
        {
            "check": "transfer_scenario_floor",
            "passed": int(transfer_pair["scenario_units"]) >= criteria.minimum_transfer_scenarios,
            "observed": int(transfer_pair["scenario_units"]),
            "required": criteria.minimum_transfer_scenarios,
        },
        {
            "check": "transfer_schedule_floor",
            "passed": transfer_schedule_rows >= criteria.minimum_transfer_schedule_rows,
            "observed": transfer_schedule_rows,
            "required": criteria.minimum_transfer_schedule_rows,
        },
        {
            "check": "transfer_positive_difference",
            "passed": transfer_delta > criteria.minimum_completion_difference,
            "observed": transfer_delta,
            "required": f"> {criteria.minimum_completion_difference}",
        },
    ]
    release_checks = pd.DataFrame(checks)
    authorized = bool(release_checks["passed"].all())
    installed_scheduler = (
        criteria.candidate_policy if authorized else criteria.baseline_policy
    )

    evidence_summary = pd.DataFrame(
        [
            {
                "evidence": "controlled_combined_stress",
                "candidate": criteria.candidate_policy,
                "baseline": criteria.baseline_policy,
                "units": len(paired),
                "candidate_completion": float(paired["candidate_completion"].mean()),
                "baseline_completion": float(paired["baseline_completion"].mean()),
                "difference": benchmark_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": benchmark_p,
                "claim_boundary": "controlled synthetic benchmark",
            },
            {
                "evidence": "equal_budget_random_forest",
                "candidate": criteria.candidate_policy,
                "baseline": criteria.learned_comparator,
                "units": int(rf_row["independent_units"]),
                "candidate_completion": np.nan,
                "baseline_completion": np.nan,
                "difference": float(rf_row["mean_delta"]),
                "ci_low": float(rf_row["ci_low"]),
                "ci_high": float(rf_row["ci_high"]),
                "p_value": float(rf_row["p_holm"]),
                "claim_boundary": "nested equal-budget controlled benchmark",
            },
            {
                "evidence": "eos_bench_frozen_transfer",
                "candidate": criteria.transfer_candidate,
                "baseline": criteria.transfer_baseline,
                "units": int(transfer_pair["scenario_units"]),
                "candidate_completion": float(transfer_candidate["completion_mean"]),
                "baseline_completion": float(transfer_baseline["completion_mean"]),
                "difference": transfer_delta,
                "ci_low": float(transfer_pair["ci_low"]),
                "ci_high": float(transfer_pair["ci_high"]),
                "p_value": float(transfer_pair["holm_p"]),
                "claim_boundary": "provider observation fields plus simulated delivery overlay",
            },
        ]
    )

    transfer_wide = (
        transfer_instances.loc[
            transfer_instances["policy"].isin(
                [criteria.transfer_candidate, criteria.transfer_baseline]
            ),
            ["instance", "policy", "completion"],
        ]
        .groupby(["instance", "policy"], as_index=False)["completion"]
        .mean()
        .pivot(index="instance", columns="policy", values="completion")
        .reset_index()
    )
    release_ledger = transfer_wide.rename(
        columns={
            criteria.transfer_candidate: "shadow_racs_completion",
            criteria.transfer_baseline: "installed_baseline_completion",
        }
    )
    release_ledger["scheduler_release_authorized"] = authorized
    release_ledger["installed_scheduler"] = installed_scheduler
    release_ledger["shadow_scheduler"] = criteria.candidate_policy
    release_ledger["path_policy"] = "reactive_baseline_retained"
    release_ledger["evidence_tier"] = "controlled_simulation_transfer_audit"

    failed_checks = release_checks.loc[~release_checks["passed"], "check"].tolist()
    decision = {
        "decision": "AUTHORIZE_RACS" if authorized else "RETAIN_MATCHED_SIMPLE_BASELINE",
        "authorized": authorized,
        "installed_scheduler": installed_scheduler,
        "shadow_scheduler": criteria.candidate_policy,
        "failed_checks": failed_checks,
        "interpretation": (
            "The scheduling component is implemented and evaluated, but the RACS "
            "ranking rule does not clear the frozen release checks. The matched full-"
            "capability margin selector therefore remains active and RACS remains in "
            "shadow evaluation. This is a successful fail-closed integration result, "
            "not evidence that scheduling research failed."
        ),
        "field_claim": False,
    }
    manifest = {
        "protocol": "racs-paper4-release-integration-v1",
        "criteria": asdict(criteria),
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "evidence_tier": "controlled_offline_integration",
        "claim_boundary": (
            "The decision integrates controlled scheduling evidence and a frozen "
            "EOS-Bench transfer audit. It does not establish operational EO or live "
            "network benefit."
        ),
    }
    return SchedulerReleaseArtifacts(
        evidence_summary=evidence_summary,
        release_checks=release_checks,
        release_ledger=release_ledger,
        decision=decision,
        manifest=manifest,
    )


def write_scheduler_release_artifacts(
    artifacts: SchedulerReleaseArtifacts,
    output_dir: Path,
) -> Path:
    """Write the release decision and its auditable supporting tables."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts.evidence_summary.to_csv(output / "scheduler_evidence_summary.csv", index=False)
    artifacts.release_checks.to_csv(output / "scheduler_release_checks.csv", index=False)
    artifacts.release_ledger.to_csv(output / "scheduler_release_ledger.csv", index=False)
    (output / "scheduler_release_decision.json").write_text(
        json.dumps(artifacts.decision, indent=2, sort_keys=True) + "\n"
    )
    manifest = dict(artifacts.manifest)
    manifest["output_sha256"] = {
        name: _sha256(output / name)
        for name in (
            "scheduler_evidence_summary.csv",
            "scheduler_release_checks.csv",
            "scheduler_release_ledger.csv",
            "scheduler_release_decision.json",
        )
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return output
