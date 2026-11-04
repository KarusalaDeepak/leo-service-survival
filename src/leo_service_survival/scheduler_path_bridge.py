"""Audited bridge from the released EO scheduler to the Paper 4 path layer.

The scheduler and path experiments use the same EOS-Bench geometry, but they
add different synthetic service and workload overlays.  This module therefore
does not reinterpret one experiment's completion outcome as the other's.  It
verifies the frozen artifacts, maps every released ``full_margin`` observation
to the exact Paper 4 geometry contact, and records the path candidates exposed
at that interface.  The resulting ledger is interface evidence only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import pandas as pd


BRIDGE_PROTOCOL = "racs-full-margin-to-paper4-path-handoff-v1"
EVIDENCE_TIER = "synthetic_interface_audit"
REQUIRED_RELEASE_DECISION = "RETAIN_MATCHED_SIMPLE_BASELINE"
REQUIRED_SCHEDULER = "full_margin"


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return observed


@dataclass(frozen=True)
class SchedulerPathBridgeArtifacts:
    handoff_candidates: pd.DataFrame
    plan_summary: pd.DataFrame
    boundary: Mapping[str, Any]
    run_manifest: Mapping[str, Any]


def validate_scheduler_release(release: Mapping[str, Any]) -> None:
    """Require the fail-closed scheduler decision used by the integrated run."""

    if release.get("decision") != REQUIRED_RELEASE_DECISION:
        raise ValueError("scheduler release decision is not the frozen baseline-retention decision")
    if release.get("installed_scheduler") != REQUIRED_SCHEDULER:
        raise ValueError("installed scheduler is not full_margin")
    if bool(release.get("authorized")):
        raise ValueError("RACS must not be marked authorized in this release artifact")
    if release.get("shadow_scheduler") != "cross_fitted_racs":
        raise ValueError("shadow scheduler provenance is missing or unexpected")


def find_schedule_entry(
    schedule_path: str | Path,
    *,
    instance: str,
    policy: str = REQUIRED_SCHEDULER,
) -> Mapping[str, Any]:
    """Return one unique frozen schedule row from a JSONL transfer audit."""

    matches: list[Mapping[str, Any]] = []
    with Path(schedule_path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("instance") == instance and row.get("policy") == policy:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"expected one {policy!r} schedule for {instance!r}, found {len(matches)}"
        )
    return matches[0]


def rerun_authoritative_full_margin(
    *,
    paper3_root: str | Path,
    scenario_path: str | Path,
    expected_plans: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the authoritative Paper 3 generator and match the frozen plans.

    This is deliberately a direct import of the hash-verified Paper 3 source,
    not a reimplementation of its planning rules in Paper 4.
    """

    paper3_root = Path(paper3_root)
    scripts_dir = str(paper3_root / "scripts")
    added = scripts_dir not in sys.path
    module_names = (
        "parameter_registry",
        "matched_scheduler",
        "run_eos_bench_transfer_audit",
    )
    prior_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)
    if added:
        sys.path.insert(0, scripts_dir)
    try:
        matched = importlib.import_module("matched_scheduler")
        adapter = importlib.import_module("run_eos_bench_transfer_audit")
        if Path(matched.__file__).resolve() != (
            paper3_root / "scripts" / "matched_scheduler.py"
        ).resolve():
            raise ValueError("did not import the requested authoritative scheduler source")
        protocol = adapter.load_protocol(
            paper3_root / "parameters" / "eos_bench_transfer_protocol.json"
        )
        instance, _source_manifest = adapter.adapt_eos_scenario(
            Path(scenario_path), protocol
        )
        evaluation, plans = matched.evaluate(
            instance,
            policy="margin",
            single=False,
            seed=0,
        )
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in prior_modules.items():
            if module is not None:
                sys.modules[name] = module
        if added:
            sys.path.remove(scripts_dir)
    actual = json.loads(json.dumps([asdict(plan) for plan in plans], sort_keys=True))
    expected = json.loads(json.dumps(list(expected_plans), sort_keys=True))
    if actual != expected:
        raise ValueError(
            "authoritative full_margin rerun does not match the frozen schedule artifact"
        )
    return {
        "policy": "full_margin",
        "completed": int(evaluation["completed"]),
        "tasks": int(evaluation["tasks"]),
        "completion": float(evaluation["completion"]),
        "validated_by_matched_scheduler": True,
        "frozen_plan_match": True,
    }


def _plan_delivery_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    deliveries = list(plan.get("deliveries", []))
    if not deliveries:
        raise ValueError(f"plan {plan.get('task_id')} has no delivery")
    if len(deliveries) != 1:
        return {
            "delivery_satellite_id": "MULTIPLE",
            "delivery_ground_station_id": "MULTIPLE",
            "delivery_start_s": min(int(row[2]) for row in deliveries),
            "delivery_end_s": max(int(row[3]) for row in deliveries),
            "delivery_rate_gbps": None,
            "delivery_count": len(deliveries),
        }
    sat, station, begin, end, rate, _amount = deliveries[0]
    return {
        "delivery_satellite_id": str(sat),
        "delivery_ground_station_id": str(station),
        "delivery_start_s": int(begin),
        "delivery_end_s": int(end),
        "delivery_rate_gbps": float(rate),
        "delivery_count": 1,
    }


def map_schedule_to_path_candidates(
    schedule_entry: Mapping[str, Any],
    candidates: pd.DataFrame,
    *,
    scenario_creation_ns: int,
    simulation_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map released plans to exact geometry contacts and path-layer candidates.

    A mapping is valid when the scheduled observation lies within the same
    provider observation window for the same mission and satellite.  Candidate
    rows duplicated across Paper 4 policy arms are collapsed by path ID.
    ``resource_feasible`` and ``gate_release`` remain diagnostic columns from
    Paper 4's *separate* synthetic workload overlay; they do not authorize
    execution of the Paper 3 plan.
    """

    required = {
        "mission_id",
        "simulation_seed",
        "candidate_path_id",
        "path_profile_id",
        "satellite_id",
        "geometry_contact_id",
        "window_start_ns",
        "window_end_ns",
        "resource_feasible",
        "gate_release",
        "predicted_completion_probability",
        "rf_snr_db",
        "gateway_id",
        "evidence_tier",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"path candidate table is missing columns: {sorted(missing)}")
    seed_candidates = candidates.loc[
        candidates["simulation_seed"].astype(int) == int(simulation_seed)
    ].copy()
    if seed_candidates.empty:
        raise ValueError(f"no Paper 4 candidates for simulation seed {simulation_seed}")
    seed_candidates = seed_candidates.drop_duplicates(
        subset=["mission_id", "candidate_path_id"], keep="first"
    )

    candidate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for plan in schedule_entry.get("plans", []):
        task_id = str(plan["task_id"])
        if not task_id.startswith("req::"):
            raise ValueError(f"unexpected released task identifier: {task_id}")
        mission_id = task_id.removeprefix("req::")
        observations = list(plan.get("observations", []))
        if not observations:
            raise ValueError(f"released plan {task_id} has no observation")
        delivery = _plan_delivery_summary(plan)
        mapped_ids: set[str] = set()
        task_rows_start = len(candidate_rows)
        all_observations_mapped = True
        for observation_index, observation in enumerate(observations):
            atom_id, satellite_id, begin_s, end_s, volume_gb = observation
            if str(atom_id) != f"obs::{mission_id}":
                raise ValueError(
                    f"plan {task_id} observation {atom_id!r} does not match its mission"
                )
            begin_ns = int(scenario_creation_ns + int(begin_s) * 1_000_000_000)
            end_ns = int(scenario_creation_ns + int(end_s) * 1_000_000_000)
            matches = seed_candidates.loc[
                (seed_candidates["mission_id"].astype(str) == mission_id)
                & (seed_candidates["satellite_id"].astype(str) == str(satellite_id))
                & (seed_candidates["window_start_ns"].astype("int64") <= begin_ns)
                & (seed_candidates["window_end_ns"].astype("int64") >= end_ns)
            ].copy()
            if matches.empty:
                all_observations_mapped = False
                candidate_rows.append(
                    {
                        "task_id": task_id,
                        "mission_id": mission_id,
                        "observation_index": observation_index,
                        "scheduled_satellite_id": str(satellite_id),
                        "scheduled_observation_start_s": int(begin_s),
                        "scheduled_observation_end_s": int(end_s),
                        "scheduled_volume_gb": float(volume_gb),
                        "geometry_handoff_valid": False,
                        "handoff_status": "NO_EXACT_GEOMETRY_MAPPING",
                        **delivery,
                    }
                )
                continue
            for candidate in matches.to_dict(orient="records"):
                candidate_id = str(candidate["candidate_path_id"])
                mapped_ids.add(candidate_id)
                candidate_rows.append(
                    {
                        "task_id": task_id,
                        "mission_id": mission_id,
                        "observation_index": observation_index,
                        "scheduled_satellite_id": str(satellite_id),
                        "scheduled_observation_start_s": int(begin_s),
                        "scheduled_observation_end_s": int(end_s),
                        "scheduled_volume_gb": float(volume_gb),
                        "scheduler_variant": str(plan.get("variant", "")),
                        "scheduler_margin": float(plan.get("margin", float("nan"))),
                        "scheduler_completion_s": int(plan.get("completion", 0)),
                        **delivery,
                        "geometry_handoff_valid": True,
                        "p4_simulation_seed": int(simulation_seed),
                        "geometry_contact_id": str(candidate["geometry_contact_id"]),
                        "candidate_path_id": candidate_id,
                        "path_profile_id": str(candidate["path_profile_id"]),
                        "path_gateway_id": str(candidate["gateway_id"]),
                        "path_rf_snr_db": float(candidate["rf_snr_db"]),
                        "path_predicted_completion_probability": float(
                            candidate["predicted_completion_probability"]
                        ),
                        "resource_feasible_under_separate_p4_overlay": bool(
                            candidate["resource_feasible"]
                        ),
                        "gate_release_under_separate_p4_overlay": bool(
                            candidate["gate_release"]
                        ),
                        "p4_candidate_evidence_tier": str(candidate["evidence_tier"]),
                        "handoff_status": "PATH_CANDIDATES_EXPOSED",
                    }
                )
        task_slice = candidate_rows[task_rows_start:]
        valid_rows = [row for row in task_slice if row.get("geometry_handoff_valid")]
        feasible_count = sum(
            bool(row.get("resource_feasible_under_separate_p4_overlay"))
            for row in valid_rows
        )
        released_count = sum(
            bool(row.get("gate_release_under_separate_p4_overlay"))
            for row in valid_rows
        )
        summary_rows.append(
            {
                "task_id": task_id,
                "mission_id": mission_id,
                "installed_scheduler": REQUIRED_SCHEDULER,
                "scheduled_observations": len(observations),
                "all_observations_geometry_mapped": all_observations_mapped,
                "unique_path_candidates_exposed": len(mapped_ids),
                "resource_feasible_candidates_under_separate_p4_overlay": feasible_count,
                "gate_released_candidates_under_separate_p4_overlay": released_count,
                "bridge_action": (
                    "EXPOSE_PATH_CANDIDATES_ONLY"
                    if all_observations_mapped
                    else "FAIL_CLOSED_NO_PATH_HANDOFF"
                ),
                "outcome_inference_permitted": False,
            }
        )
    return pd.DataFrame(candidate_rows), pd.DataFrame(summary_rows)


def build_scheduler_path_bridge(
    *,
    transfer_dir: str | Path,
    paper3_root: str | Path,
    scheduler_release_path: str | Path,
    p4_full_chain_dir: str | Path,
    scenario_path: str | Path,
    instance: str,
    simulation_seed: int,
) -> SchedulerPathBridgeArtifacts:
    """Verify frozen inputs and build the controlled schedule-to-path ledger."""

    transfer_dir = Path(transfer_dir)
    paper3_root = Path(paper3_root)
    p4_full_chain_dir = Path(p4_full_chain_dir)
    scenario_path = Path(scenario_path)
    release_path = Path(scheduler_release_path)

    release = _read_json(release_path)
    validate_scheduler_release(release)

    transfer_manifest = _read_json(transfer_dir / "run_manifest.json")
    for name, expected in transfer_manifest["output_sha256"].items():
        _verify_hash(transfer_dir / name, expected, f"Paper 3 transfer output {name}")
    if transfer_manifest.get("validated_schedule_rows") != transfer_manifest.get(
        "evaluated_schedule_rows"
    ):
        raise ValueError("Paper 3 transfer bundle contains unvalidated schedules")
    source_files = {
        "matched_scheduler.py": paper3_root / "scripts" / "matched_scheduler.py",
        "run_eos_bench_transfer_audit.py": paper3_root
        / "scripts"
        / "run_eos_bench_transfer_audit.py",
        "eos_bench_transfer_protocol.json": paper3_root
        / "parameters"
        / "eos_bench_transfer_protocol.json",
    }
    for name, path in source_files.items():
        _verify_hash(path, transfer_manifest["source_sha256"][name], f"Paper 3 source {name}")

    input_manifest = _read_json(transfer_dir / "input_manifest.json")
    source_hash = sha256_file(scenario_path)
    input_rows = [row for row in input_manifest if row.get("sha256") == source_hash]
    if len(input_rows) != 1:
        raise ValueError("scenario is not uniquely recorded in the frozen Paper 3 input manifest")

    p4_manifest = _read_json(p4_full_chain_dir / "run_manifest.json")
    if p4_manifest.get("source_scenario_sha256") != source_hash:
        raise ValueError("Paper 3 and Paper 4 artifacts do not use the same scenario bytes")
    candidate_path = p4_full_chain_dir / "candidate_path_observation.csv"
    expected_candidate_hash = p4_manifest["artifact_sha256_before_manifest"][
        "candidate_path_observation.csv"
    ]
    _verify_hash(candidate_path, expected_candidate_hash, "Paper 4 candidate path table")

    schedule_entry = find_schedule_entry(
        transfer_dir / "schedules.jsonl", instance=instance
    )
    rerun = rerun_authoritative_full_margin(
        paper3_root=paper3_root,
        scenario_path=scenario_path,
        expected_plans=schedule_entry["plans"],
    )
    scenario_payload = _read_json(scenario_path)
    creation = pd.Timestamp(scenario_payload["metadata"]["creation_time"])
    if creation.tzinfo is None:
        creation = creation.tz_localize("UTC")
    else:
        creation = creation.tz_convert("UTC")
    candidates = pd.read_csv(candidate_path)
    handoff, plan_summary = map_schedule_to_path_candidates(
        schedule_entry,
        candidates,
        scenario_creation_ns=int(creation.value),
        simulation_seed=simulation_seed,
    )

    mapping_complete = bool(
        len(plan_summary)
        and plan_summary["all_observations_geometry_mapped"].astype(bool).all()
    )
    boundary = {
        "protocol": BRIDGE_PROTOCOL,
        "evidence_tier": EVIDENCE_TIER,
        "installed_scheduler": REQUIRED_SCHEDULER,
        "shadow_scheduler": release["shadow_scheduler"],
        "scheduled_plans": int(len(plan_summary)),
        "plans_with_exact_geometry_mapping": int(
            plan_summary["all_observations_geometry_mapped"].astype(bool).sum()
        ),
        "path_candidate_rows_exposed": int(
            handoff["geometry_handoff_valid"].fillna(False).astype(bool).sum()
        ),
        "mapping_complete": mapping_complete,
        "continuous_outcome_experiment_completed": False,
        "reason_outcome_not_combined": (
            "Paper 3 uses a deterministic simulated ground-service overlay with release zero "
            "and a scenario-horizon deadline. Paper 4 uses independently generated releases, "
            "deadlines, rates, gateway identifiers, and post-decision outcomes. The exact "
            "geometry handoff is valid, but completion outcomes cannot be merged without "
            "changing either frozen experiment."
        ),
        "permitted_claim": (
            "The frozen scheduler-release outcome is applied at the integration boundary: "
            "full_margin supplies validated EO plans, and every selected observation is "
            "traceable to the Paper 4 path-candidate interface on the same EOS-Bench geometry."
        ),
        "prohibited_claims": [
            "The bridge is an operational EO experiment.",
            "Paper 3 and Paper 4 completion rates are directly comparable.",
            "The RACS score is deployed or superior.",
            "The path candidates caused the Paper 3 completion outcomes.",
        ],
    }
    if not mapping_complete:
        raise ValueError("at least one released plan lacks an exact Paper 4 geometry handoff")

    run_manifest = {
        "protocol": BRIDGE_PROTOCOL,
        "evidence_tier": EVIDENCE_TIER,
        "instance": instance,
        "simulation_seed": int(simulation_seed),
        "source_scenario_sha256": source_hash,
        "verified_scheduler_release_sha256": sha256_file(release_path),
        "verified_paper3_transfer_manifest_sha256": sha256_file(
            transfer_dir / "run_manifest.json"
        ),
        "verified_paper3_schedules_sha256": sha256_file(
            transfer_dir / "schedules.jsonl"
        ),
        "verified_paper3_source_sha256": {
            name: sha256_file(path) for name, path in source_files.items()
        },
        "verified_p4_manifest_sha256": sha256_file(
            p4_full_chain_dir / "run_manifest.json"
        ),
        "verified_p4_candidate_table_sha256": sha256_file(candidate_path),
        "authoritative_scheduler_rerun": rerun,
        "rows": {
            "handoff_candidates": int(len(handoff)),
            "plan_summary": int(len(plan_summary)),
        },
    }
    return SchedulerPathBridgeArtifacts(handoff, plan_summary, boundary, run_manifest)


def write_scheduler_path_bridge(
    artifacts: SchedulerPathBridgeArtifacts, output_dir: str | Path
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts.handoff_candidates.to_csv(output / "schedule_to_path_handoff.csv", index=False)
    artifacts.plan_summary.to_csv(output / "plan_handoff_summary.csv", index=False)
    (output / "scientific_boundary.json").write_text(
        json.dumps(artifacts.boundary, indent=2, sort_keys=True) + "\n"
    )
    manifest = dict(artifacts.run_manifest)
    manifest["output_sha256"] = {
        name: sha256_file(output / name)
        for name in (
            "schedule_to_path_handoff.csv",
            "plan_handoff_summary.csv",
            "scientific_boundary.json",
        )
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return output
