#!/usr/bin/env python3
"""Run add-on training-only search and measured feature-only guard audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from leo_service_survival.data import build_wetlinks_episodes, calendar_block_split  # noqa: E402
from leo_service_survival.parameter_defensibility import (  # noqa: E402
    CALIBRATION_MINIMUM_SUPPORT, EXPANDED_CALIBRATION_C_GRID, EXPANDED_FOCAL_GRID,
    FEATURE_COVERAGE_BUDGETS, GUARD_QUANTILES, HAZARD_CAPS, OOD_FRACTION_BUDGETS, PROBABILITY_CLIPS,
    coverage_budget_sensitivity, feature_guard_sensitivity, governance_logic_sensitivity,
    monitor_logic_sensitivity,
    hazard_cap_sensitivity, parameter_coverage_addendum,
    run_search_boundary_audit, stable_json_hash,
)
from run_all import truncate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wetlinks-dir", type=Path, default=ROOT / "upstream/paper2/data/external/wetlinks_dataset/Preprocessed_Data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/parameter_defensibility_v1")
    parser.add_argument("--allow-overwrite", action="store_true", help="Regenerate this diagnostic run's known derived outputs; never use on a frozen manuscript run")
    parser.add_argument("--hazard-cap-only", action="store_true", help="Append the declared hazard-cap audit to this existing, still-unfrozen diagnostic run")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not (args.allow_overwrite or args.hazard_cap_only):
        raise FileExistsError(f"refusing to mix runs in {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.hazard_cap_only:
        selection = json.loads((output / "development_selection_freeze.json").read_text())
        cap_protocol = {"caps": list(HAZARD_CAPS), "lower_floor": "1e-8 except probability-preserving [0,1] alternative", "evidence_role": "development_sensitivity_only_not_retuning", "frozen_model_modified": False, "selected_configurations": selection["folds"]}
        (output / "declared_hazard_cap_protocol.json").write_text(json.dumps(cap_protocol, indent=2) + "\n")
        dataset = build_wetlinks_episodes(args.wetlinks_dir.resolve())
        train, _, _ = calendar_block_split(truncate(dataset.frame, 12))
        cap_table, hit_table = hazard_cap_sensitivity(train, dataset.feature_columns, selection["folds"])
        cap_table.to_csv(output / "hazard_cap_sensitivity.csv", index=False)
        hit_table.to_csv(output / "hazard_cap_hits.csv", index=False)
        parameter_coverage_addendum().to_csv(output / "parameter_coverage_addendum.csv", index=False)
        manifest = json.loads((output / "run_manifest.json").read_text())
        manifest["hazard_cap_protocol_sha256"] = stable_json_hash(cap_protocol)
        manifest["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT / "scripts/run_parameter_defensibility.py", ROOT / "src/leo_service_survival/parameter_defensibility.py")}
        manifest["outputs_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.is_file() and path.name != "run_manifest.json"}
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"hazard_cap_rows": len(cap_table), "hazard_hit_rows": len(hit_table), "output_dir": str(output)}, indent=2))
        return 0
    protocol = {
        "analysis_id": "parameter_defensibility_v1",
        "status": "retrospective_add_on_not_preregistered",
        "primary_model_replaced": False,
        "primary_run_modified": False,
        "search_scope": "nested_calendar_block_folds_inside_original_training_partition_only",
        "random_person_period_early_stopping": False,
        "expanded_focal_grid": list(EXPANDED_FOCAL_GRID),
        "expanded_calibration_c_grid": list(EXPANDED_CALIBRATION_C_GRID),
        "minimum_calibration_support_grid": list(CALIBRATION_MINIMUM_SUPPORT),
        "probability_clip_grid": list(PROBABILITY_CLIPS),
        "hazard_cap_grid": list(HAZARD_CAPS),
        "guard_quantile_grid": list(GUARD_QUANTILES),
        "ood_fraction_budget_grid": list(OOD_FRACTION_BUDGETS),
        "feature_coverage_budget_grid": list(FEATURE_COVERAGE_BUDGETS),
        "monitor_logic_grid": {"null_mean": [0.05, 0.10, 0.20], "alpha": [0.005, 0.01, 0.05], "betting_fraction": [0.5, 1.0, 2.0]},
        "risk_budget_selection": "none; all declared alternatives reported; not learned from test",
        "heldout_outcome_scoring": False,
        "heldout_use": "features only, after declaration; descriptive shift sensitivity, no threshold selection",
        "inference": "two overlapping temporal development folds; no independent-campaign CI or significance claim",
        "guard_scope": "two measured sites from one acquisition family; not false-alarm calibration or deployment authorization",
        "governance_scope": "risk-budget grid uses supplied logic fixture, not campaign-derived statistical bounds",
        "search_budget_rationale": "append two lower learning rates and two stronger calibration penalties to probe prior lower boundaries while holding other settings fixed; finite grid never establishes universal optimum",
    }
    # Written before fitting/search and before any guard held-out feature score.
    (output / "declared_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    dataset = build_wetlinks_episodes(args.wetlinks_dir.resolve())
    train, calibration, test = calendar_block_split(truncate(dataset.frame, 12))
    audit = run_search_boundary_audit(train, dataset.feature_columns)
    for name in ("search", "scores", "summary", "paired", "calibration_sensitivity"):
        audit[name].to_csv(output / f"{name}.csv", index=False)
    selection = {
        "scope": "original_training_only_no_frozen_model_replacement",
        "folds": audit["selection_freeze"],
        "heldout_outcomes_used": False,
    }
    (output / "development_selection_freeze.json").write_text(json.dumps(selection, indent=2) + "\n")
    (output / "temporal_fold_audit.json").write_text(json.dumps(audit["fold_audit"], indent=2) + "\n")
    cap_table, hit_table = hazard_cap_sensitivity(train, dataset.feature_columns, selection["folds"])
    cap_table.to_csv(output / "hazard_cap_sensitivity.csv", index=False)
    hit_table.to_csv(output / "hazard_cap_hits.csv", index=False)
    parameter_coverage_addendum().to_csv(output / "parameter_coverage_addendum.csv", index=False)
    guard, batches = feature_guard_sensitivity(
        train, {"train": train, "calibration": calibration, "test": test}, dataset.feature_columns
    )
    guard.to_csv(output / "feature_guard_sensitivity.csv", index=False)
    batches.to_csv(output / "feature_guard_site_calendar_batches.csv", index=False)
    coverage_budget_sensitivity(guard).to_csv(output / "feature_coverage_budget_sensitivity.csv", index=False)
    governance_logic_sensitivity(train, dataset.feature_columns).to_csv(output / "governance_logic_sensitivity.csv", index=False)
    monitor_logic_sensitivity().to_csv(output / "monitor_logic_sensitivity.csv", index=False)
    result = {
        "protocol_sha256": stable_json_hash(protocol),
        "selection_sha256": stable_json_hash(selection),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "partitions": {name: {"rows": len(frame), "sites": sorted(frame["site"].unique().tolist()), "calendar_blocks": sorted(frame["calendar_block"].unique().tolist())} for name, frame in (("train", train), ("calibration", calibration), ("test", test))},
        "boundary_flags": [{"fold": row["fold"], "search": row["search"], "learning_rate_at_expanded_lower_boundary": row["learning_rate_at_expanded_lower_boundary"], "calibration_c_at_expanded_lower_boundary": row["calibration_c_at_expanded_lower_boundary"], "calibration_c_at_upper_boundary": row["calibration_c_at_upper_boundary"]} for row in audit["selection_freeze"]],
        "caution": "Boundary selection remains possible; broader grids reduce one concern but cannot prove unique coefficients, physical calibration, field false-alarm rates, or operational benefit.",
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(args.wetlinks_dir.resolve().glob("analysis_data_*.csv"))},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT / "scripts/run_parameter_defensibility.py", ROOT / "src/leo_service_survival/parameter_defensibility.py")},
        "outputs_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.is_file() and path.name != "run_manifest.json"},
    }
    (output / "run_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "search_rows": len(audit["search"]), "outer_scores": len(audit["scores"]), "calibration_sensitivity_rows": len(audit["calibration_sensitivity"]), "guard_sensitivity_rows": len(guard), "boundary_flags": result["boundary_flags"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
