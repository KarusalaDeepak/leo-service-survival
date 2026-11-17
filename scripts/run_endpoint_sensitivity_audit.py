#!/usr/bin/env python3
"""Execute rebuilt endpoint sensitivity without modifying manuscript freezes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from leo_service_survival.data import build_wetlinks_episodes
from leo_service_survival.endpoint_sensitivity_audit import (
    cadence_support, episode_fingerprint, finalize_export_schema, fit_and_score_profile, prepare_regular_series,
    profile_metadata, rebuild_episodes, registered_profiles, sha256_file,
    source_cadence_audit, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wetlinks-dir", type=Path,
                        default=ROOT / "upstream/paper2/data/external/wetlinks_dataset/Preprocessed_Data")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/endpoint_parameter_sensitivity_v1")
    parser.add_argument("--profile", action="append", help="Optional focused smoke-run profile ids")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty audit freeze: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    paths = sorted(args.wetlinks_dir.resolve().glob("analysis_data_*.csv"))
    if len(paths) != 2:
        raise SystemExit("Expected exactly two released WetLinks analysis tables")
    cadence = source_cadence_audit(paths)
    cadence.to_csv(output / "source_cadence.csv", index=False)
    profiles = registered_profiles()
    if args.profile:
        wanted = set(args.profile)
        if not wanted.issubset({p.profile_id for p in profiles}):
            raise SystemExit(f"Unknown profile(s): {wanted}")
        profiles = tuple(p for p in profiles if p.profile_id in wanted)
    rows, support_rows, regular_cache, score_cache = [], [], {}, {}
    baseline_fingerprint = None
    baseline_match = None
    for profile in profiles:
        row = profile_metadata(profile)
        row["registered_alternative"] = True
        supported, reason = cadence_support(profile, cadence)
        row["cadence_status"] = reason
        print(f"Starting {profile.profile_id}: {reason}", flush=True)
        if not supported:
            row.update(status="not_supported", executed=False, reason=reason)
            rows.append(row)
            continue
        profile_output = output / profile.profile_id
        profile_output.mkdir()
        stage_start = time.monotonic()
        try:
            if profile.bin_minutes not in regular_cache:
                regular_cache[profile.bin_minutes] = prepare_regular_series(paths, profile.bin_minutes)
            episodes = rebuild_episodes(regular_cache[profile.bin_minutes], profile)
            fingerprint = episode_fingerprint(episodes)
            row.update(episodes=len(episodes), events=int(episodes.event.sum()),
                       episode_fingerprint=fingerprint,
                       cause_loss=int(episodes.cause.eq("loss").sum()),
                       cause_latency=int(episodes.cause.eq("latency").sum()),
                       cause_joint=int(episodes.cause.eq("joint").sum()))
            if profile.profile_id == "primary":
                baseline_fingerprint = fingerprint
                original = build_wetlinks_episodes(args.wetlinks_dir).frame
                baseline_match = fingerprint == episode_fingerprint(original)
                if not baseline_match:
                    raise AssertionError("Audit primary episodes differ from existing core builder")
            cache_key = (fingerprint, profile.analysis_horizon_bins)
            if cache_key in score_cache:
                origin, result = score_cache[cache_key]
                row["deterministic_score_reused_from"] = origin
            else:
                result = fit_and_score_profile(episodes, profile)
                score_cache[cache_key] = (profile.profile_id, result)
                row["deterministic_score_reused_from"] = ""
            row["same_endpoints_as_primary"] = fingerprint == baseline_fingerprint
            row["gap_rule_effect"] = ("structurally_invariant_missing_bin_immediately_censors"
                                      if profile.profile_id.startswith("maximum_gap_bins") else "not_a_gap_contrast")
            result["selection"].to_csv(profile_output / "development_selection.csv", index=False)
            result["support"].to_csv(profile_output / "support.csv", index=False)
            result["calibration_support"].to_csv(profile_output / "calibration_support.csv", index=False)
            result["test_predictions"].to_csv(profile_output / "test_predictions.csv", index=False)
            write_json(profile_output / "split_audit.json", result["split_audit"])
            write_json(profile_output / "selection_failures.json", result["selection_failures"])
            write_json(profile_output / "scores.json", result["scores"])
            for name, metrics in result["scores"].items():
                for metric, value in metrics.items():
                    row[f"{name}_{metric}"] = value
            row["calibrated_minus_km_brier"] = (result["scores"]["calibrated"]["integrated_brier"]
                                                 - result["scores"]["kaplan_meier"]["integrated_brier"])
            row["selected_configuration"] = json.dumps(result["selected_configuration"], sort_keys=True)
            row["selected_calibration_c"] = result["selected_calibration_c"]
            row["test_joint_events"] = int(result["support"].query("role == 'test'").iloc[0].cause_joint)
            row["status"] = "executed"
            row["executed"] = True
            row["reason"] = "all_cause_scores_estimable_cause_specific_power_not_established"
            support = result["support"].copy()
            support.insert(0, "profile_id", profile.profile_id)
            support_rows.append(support)
        except Exception as error:
            # Retain failures as explicit scientific non-estimability; never
            # silently omit a difficult endpoint or replace it with fake data.
            row.update(status="not_estimable", executed=True,
                       reason=f"{type(error).__name__}: {error}")
            write_json(profile_output / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        row["runtime_seconds"] = time.monotonic() - stage_start
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / "endpoint_parameter_sensitivity.csv", index=False)
        print(f"Finished {profile.profile_id}: {row['status']} ({row['runtime_seconds']:.1f}s)", flush=True)
    table = pd.DataFrame(rows)
    for column in ("episodes", "events", "calibrated_minus_km_brier"):
        if column not in table:
            table[column] = np.nan
    table.to_csv(output / "endpoint_parameter_sensitivity.csv", index=False)
    coverage = []
    primary_row = table.loc[table.profile_id.eq("primary")]
    for axis in ("bin_minutes", "latency_limit_ms", "loss_limit_pct", "sustained_bad_bins",
                 "maximum_gap_bins", "analysis_horizon_bins"):
        contrast_rows = table.loc[table.contrast.eq(f"one_factor:{axis}")]
        compared = pd.concat([primary_row, contrast_rows], ignore_index=True)
        for _, row in compared.sort_values(axis).iterrows():
            coverage.append({"parameter": f"endpoint.{axis}", "value": row[axis],
                             "profile_id": row.profile_id, "status": row.status,
                             "endpoint_rebuild_executed": bool(row.executed),
                             "result": row.get("gap_rule_effect", row.get("reason", "")),
                             "note": ("Repeated primary row is the same reference, not another independent run"
                                      if row.profile_id == "primary" else "")})
    pd.DataFrame(coverage).to_csv(output / "sensitivity_coverage.csv", index=False)
    if support_rows:
        pd.concat(support_rows, ignore_index=True).to_csv(output / "all_profile_support.csv", index=False)
    manifest = {
        "audit_version": "endpoint-parameter-sensitivity-v1",
        "source_inputs": cadence.to_dict(orient="records"),
        "profiles": [profile_metadata(profile) for profile in profiles],
        "primary_episode_matches_frozen_core_builder": baseline_match,
        "status_counts": table.status.value_counts().to_dict(),
        "runtime_seconds": time.monotonic() - started,
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name)
                             for name in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")},
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "scope": {
            "not_new_independent_validation": True,
            "endpoint_values_not_selected_by_test_results": True,
            "one_factor_all_registered_levels": not bool(args.profile),
            "one_minute_not_supported_if_raw_median_cadence_exceeds_one_minute": True,
            "no_interpolation_or_synthetic_probe_rows": True,
            "gap_variants_are_structurally_invariant_on_regular_grid": True,
            "model_family": "competing-risk boosted discrete hazard; automatic row stopping disabled",
            "development_selection": "three focal configurations times four calibration C values; same purged calendar roles",
            "test_role": "retrospective endpoint robustness audit, not an untouched new field campaign",
            "selection_metric": "lowest inner-validation mean IPCW Brier; deterministic documented tie-breaks",
            "comparability": "compare models within a profile; different endpoints/horizons are different estimands",
            "rare_causes": "all counts retained; all-cause scoring does not establish cause-specific power",
            "gap_interpretation": "any explicitly unobserved regular-grid row immediately censors; gap tolerance does not bridge missing observations",
            "core_results_modified": False,
        },
    }
    code_paths = [Path(__file__), ROOT / "src/leo_service_survival/endpoint_sensitivity_audit.py",
                  ROOT / "src/leo_service_survival/data.py", ROOT / "src/leo_service_survival/models.py",
                  ROOT / "src/leo_service_survival/metrics.py", ROOT / "src/leo_service_survival/parameters.py"]
    manifest["code_sha256"] = {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths}
    manifest["artifact_sha256"] = {str(path.relative_to(output)): sha256_file(path)
                                  for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "manifest.json", manifest)
    successful = table.loc[table.status.eq("executed")]
    deltas = successful["calibrated_minus_km_brier"].dropna()
    primary_summary = (primary_row.replace({np.nan: None}).to_dict(orient="records")[0]
                       if len(primary_row) else None)
    write_json(output / "summary.json", {
        "executed_profiles": table.loc[table.status.eq("executed"), "profile_id"].tolist(),
        "unsupported_profiles": table.loc[table.status.eq("not_supported"), ["profile_id", "reason"]].to_dict(orient="records"),
        "non_estimable_profiles": table.loc[table.status.eq("not_estimable"), ["profile_id", "reason"]].to_dict(orient="records"),
        "interpretation": "This is executed robustness evidence, not a search for a best endpoint or physical validation. All-cause model ordering can change with the service definition. One-minute continuity is not recoverable from these burst summaries. Regular-grid gap levels are equivalent under immediate censoring; they are not three independent successful validations.",
        "primary_episode_matches_existing_builder": baseline_match,
        "status_counts": manifest["status_counts"],
        "primary_result": primary_summary,
        "calibrated_minus_km_brier_range_across_executed_profiles": (
            {"minimum": float(deltas.min()), "maximum": float(deltas.max())} if len(deltas) else None),
        "profiles_with_lower_calibrated_than_km_brier": successful.loc[
            successful.calibrated_minus_km_brier.lt(0), "profile_id"].tolist() if len(successful) else [],
        "profiles_with_higher_or_equal_calibrated_than_km_brier": successful.loc[
            successful.calibrated_minus_km_brier.ge(0), "profile_id"].tolist() if len(successful) else [],
        "primary_shortfall": "Joint and latency-cause scarcity must be reported per profile, not hidden by aggregate all-cause metrics.",
    })
    finalize_export_schema(output)
    print(json.dumps({"output": str(output), "status_counts": manifest["status_counts"],
                      "runtime_seconds": manifest["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
