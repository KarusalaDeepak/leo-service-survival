#!/usr/bin/env python3
"""Build the reviewer-facing all-pipeline WetLinks prediction release.

This focused runner refits the twelve frozen manuscript pipelines on the same
chronological training/calibration partitions and exports every held-out
prediction and proper-score input at horizons 1 through 12.  It deliberately
does not pickle or otherwise serialize fitted model objects.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leo_service_survival.data import (  # noqa: E402
    build_wetlinks_episodes,
    calendar_block_split,
    merge_sparse_terminal_inference_block,
)
from leo_service_survival.diagnostics import calibration_fit_audit  # noqa: E402
from leo_service_survival.metrics import evaluate_survival_predictions  # noqa: E402
from leo_service_survival.models import (  # noqa: E402
    CalibratedSurvivalModel,
    build_model_suite,
    select_baseline_model_configurations,
    select_focal_model_configuration,
)
from leo_service_survival.release_artifacts import (  # noqa: E402
    EXPECTED_PIPELINES,
    build_pipeline_execution_status,
    build_prediction_score_release,
    preimputation_feature_coverage,
)

# Reuse the primary runner's administrative-horizon convention rather than
# maintaining a second subtly different endpoint implementation.
from run_all import development_tripartition, truncate  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    clean = not subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "base_commit": commit,
        "tracked_worktree_clean": clean,
        "interpretation": (
            "code-file hashes below are authoritative when the run includes "
            "uncommitted release-builder changes"
        ),
    }


def _fit_and_predict(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    horizons: tuple[int, ...],
    *,
    focal_hyperparameters: dict[str, float | int],
    focal_calibration_c: float,
    baseline_hyperparameters: dict[str, object],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[dict[str, str]]]:
    suite = build_model_suite(
        features,
        max_horizon=max(horizons),
        random_state=17,
        focal_hyperparameters=focal_hyperparameters,
        calibration_c=focal_calibration_c,
        baseline_hyperparameters=baseline_hyperparameters,
    )
    predictions: dict[str, np.ndarray] = {}
    calibration_audits: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for pipeline in EXPECTED_PIPELINES:
        model = suite.models[pipeline]
        try:
            model.fit(train)
        except Exception as error:  # fail-closed audit trail before raising
            failures.append(
                {
                    "pipeline": pipeline,
                    "stage": "fit",
                    "status": "failed",
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                }
            )
            continue
        try:
            if isinstance(model, CalibratedSurvivalModel):
                model.calibrate(calibration)
                audit = calibration_fit_audit(
                    model,
                    calibration,
                    horizons,
                    bin_seconds=300,
                )
                audit.insert(0, "pipeline", pipeline)
                calibration_audits.append(audit)
            predictions[pipeline] = model.predict_survival(test, horizons)
        except Exception as error:  # fail-closed audit trail before raising
            failures.append(
                {
                    "pipeline": pipeline,
                    "stage": "calibrate_or_predict",
                    "status": "failed",
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                }
            )
    calibration_table = (
        pd.concat(calibration_audits, ignore_index=True)
        if calibration_audits
        else pd.DataFrame()
    )
    return predictions, calibration_table, failures


def _assert_canonical_metrics(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    horizons: tuple[int, ...],
    expected_path: Path,
) -> None:
    if not expected_path.is_file():
        raise FileNotFoundError(f"canonical metric table is absent: {expected_path}")
    expected = pd.read_csv(expected_path).set_index("model")
    if set(expected.index.astype(str)) != set(EXPECTED_PIPELINES):
        raise ValueError("canonical metric table does not contain exactly 12 pipelines")
    checked_metrics = (
        "integrated_brier",
        "mean_calibration_error",
        "binary_cross_entropy",
        "restricted_mean_time_mae_bins",
        "c_index",
    )
    for pipeline in EXPECTED_PIPELINES:
        observed = evaluate_survival_predictions(frame, predictions[pipeline], horizons)
        for metric in checked_metrics:
            if not np.isclose(
                float(observed[metric]),
                float(expected.loc[pipeline, metric]),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                raise AssertionError(
                    f"{pipeline} {metric} does not reproduce canonical table: "
                    f"{observed[metric]} versus {expected.loc[pipeline, metric]}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wetlinks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=ROOT / "results/manuscript/comparator_configurations.csv",
    )
    parser.add_argument(
        "--canonical-metrics",
        type=Path,
        default=ROOT / "results/manuscript/wetlinks_model_metrics.csv",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    targets = {
        "predictions": output / "wetlinks_all_pipeline_horizon_predictions.csv.gz",
        "scores": output / "wetlinks_all_pipeline_horizon_scores.csv",
        "status": output / "wetlinks_pipeline_execution_status.csv",
        "calibration": output / "wetlinks_pipeline_calibration_fit_audit.csv",
        "coverage": output / "wetlinks_preimputation_feature_coverage.csv",
        "alignment": output / "wetlinks_prediction_alignment_audit.json",
        "manifest": output / "wetlinks_prediction_release_manifest.json",
        "failures": output / "wetlinks_pipeline_build_failures.csv",
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not args.allow_overwrite:
        raise FileExistsError(
            "refusing to overwrite release artifacts; use a fresh directory or "
            f"--allow-overwrite: {existing}"
        )
    output.mkdir(parents=True, exist_ok=True)

    dataset = build_wetlinks_episodes(args.wetlinks_dir.expanduser().resolve())
    full = truncate(dataset.frame, 12)
    train, calibration, test = calendar_block_split(full)
    test, block_audit = merge_sparse_terminal_inference_block(
        test,
        group_column="calendar_block",
        output_column="analysis_block",
        minimum_fraction_of_prior_median=0.25,
    )
    horizons = tuple(range(1, 13))

    # Reproduce the development-only configuration selection used by the
    # primary manuscript runner.  The held-out test partition is never used
    # to choose a model setting.  Keeping this logic here prevents the
    # reviewer derivative from silently reverting to constructor defaults.
    inner_fit, inner_calibration, inner_validation = development_tripartition(
        train,
        group_column="calendar_block",
    )
    focal_hyperparameters, focal_calibration_c, focal_selection = (
        select_focal_model_configuration(
            inner_fit,
            inner_calibration,
            inner_validation,
            tuple(dataset.feature_columns),
            max_horizon=max(horizons),
            random_state=17,
        )
    )
    baseline_hyperparameters, baseline_selection = (
        select_baseline_model_configurations(
            inner_fit,
            inner_validation,
            tuple(dataset.feature_columns),
            max_horizon=max(horizons),
            random_state=17,
        )
    )

    predictions, calibration_table, failures = _fit_and_predict(
        train,
        calibration,
        test,
        tuple(dataset.feature_columns),
        horizons,
        focal_hyperparameters=focal_hyperparameters,
        focal_calibration_c=focal_calibration_c,
        baseline_hyperparameters=baseline_hyperparameters,
    )
    if failures:
        pd.DataFrame(failures).to_csv(targets["failures"], index=False)
        raise RuntimeError(
            "one or more pipelines failed; failure reasons were written to "
            f"{targets['failures']}"
        )
    if targets["failures"].exists():
        targets["failures"].unlink()

    try:
        _assert_canonical_metrics(test, predictions, horizons, args.canonical_metrics)
    except Exception as error:
        pd.DataFrame(
            [
                {
                    "pipeline": "all",
                    "stage": "canonical_metric_check",
                    "status": "failed",
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                }
            ]
        ).to_csv(targets["failures"], index=False)
        raise
    prediction_table, score_table, alignment = build_prediction_score_release(
        test,
        predictions,
        horizons,
        bin_seconds=dataset.bin_seconds,
    )
    configuration = pd.read_csv(args.pipeline_config)
    status = build_pipeline_execution_status(
        configuration,
        predictions,
        alignment,
        calibration_fit_audit=calibration_table,
    )
    coverage = preimputation_feature_coverage(
        {"train": train, "calibration": calibration, "test": test},
        tuple(dataset.feature_columns),
    )

    prediction_table.to_csv(targets["predictions"], index=False, compression="gzip")
    score_table.to_csv(targets["scores"], index=False)
    status.to_csv(targets["status"], index=False)
    calibration_table.to_csv(targets["calibration"], index=False)
    coverage.to_csv(targets["coverage"], index=False)
    targets["alignment"].write_text(
        json.dumps(alignment, indent=2) + "\n",
        encoding="utf-8",
    )

    serialized = [
        targets[key]
        for key in (
            "predictions",
            "scores",
            "status",
            "calibration",
            "coverage",
            "alignment",
        )
    ]
    code_paths = [
        ROOT / "scripts/build_prediction_release_artifacts.py",
        ROOT / "scripts/run_all.py",
        ROOT / "src/leo_service_survival/data.py",
        ROOT / "src/leo_service_survival/diagnostics.py",
        ROOT / "src/leo_service_survival/metrics.py",
        ROOT / "src/leo_service_survival/models.py",
        ROOT / "src/leo_service_survival/release_artifacts.py",
    ]
    source_paths = sorted(
        args.wetlinks_dir.expanduser().resolve().glob("analysis_data_*.csv")
    )
    manifest = {
        "schema_version": "1.0",
        "builder": "scripts/build_prediction_release_artifacts.py",
        "producer_function": "main",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_state": _git_state(),
        "command_profile": (
            "python scripts/build_prediction_release_artifacts.py "
            "--wetlinks-dir <lawfully-obtained-WetLinks/Preprocessed_Data> "
            "--output-dir <fresh-output-directory>"
        ),
        "random_seed": 17,
        "test_set_id": alignment["test_set_id"],
        "test_episode_order_sha256": alignment["episode_order_sha256"],
        "pipelines": list(EXPECTED_PIPELINES),
        "horizons": list(horizons),
        "model_object_policy": (
            "no executable model object is released; refit from cited source data "
            "and frozen configuration"
        ),
        "forbidden_model_object_extensions": [
            ".pkl",
            ".pickle",
            ".joblib",
            ".onnx",
            ".ubj",
            ".model",
        ],
        "derivative_data_license": {
            "name": "Creative Commons Attribution-ShareAlike 4.0 International",
            "spdx": "CC-BY-SA-4.0",
            "url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "upstream": "WetLinks TMA 2024 analysis tables",
            "modification_notice": (
                "source measurements were binned into service episodes, split "
                "chronologically, scored by twelve pipelines, and exported as "
                "derived predictions and censoring-aware score inputs"
            ),
        },
        "canonical_metric_check": {
            "path": str(args.canonical_metrics.name),
            "rtol": 1e-10,
            "atol": 1e-12,
            "status": "passed",
        },
        "development_only_configuration_selection": {
            "partition_rule": (
                "chronological 60/20/20 tripartition of retained training "
                "calendar blocks; final calibration and held-out test rows excluded"
            ),
            "test_outcomes_used": False,
            "focal_hyperparameters": focal_hyperparameters,
            "focal_calibration_c": focal_calibration_c,
            "baseline_hyperparameters": baseline_hyperparameters,
            "focal_candidate_rows": int(len(focal_selection)),
            "baseline_candidate_rows": int(len(baseline_selection)),
        },
        "terminal_inference_block_rule": block_audit,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "source_inputs": [
            {
                "name": path.name,
                "sha256": _sha256(path),
                "redistributed": False,
            }
            for path in source_paths
        ],
        "code_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for path in code_paths
        ],
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in serialized
        ],
    }
    targets["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
