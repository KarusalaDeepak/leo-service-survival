from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leo_service_survival.release_artifacts import (
    CALIBRATED_PIPELINES,
    EXPECTED_PIPELINES,
)


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "results/reviewer_prediction_release"
EXPECTED_ORDER_DIGEST = (
    "d4317eff6d0f1d1b497175b0a9fb43a52ecac3a9b51d998f72955df0253866f9"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier_digest(values: pd.Series) -> str:
    payload = json.dumps(
        values.astype(str).tolist(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_complete_prediction_grid_and_ordered_row_identity() -> None:
    prediction = pd.read_csv(
        RELEASE / "wetlinks_all_pipeline_horizon_predictions.csv.gz"
    )
    assert len(prediction) == 821 * 12 * 12
    assert set(prediction["pipeline"]) == set(EXPECTED_PIPELINES)
    assert set(prediction["horizon_bins"]) == set(range(1, 13))
    key = ["test_set_id", "test_row_index", "pipeline", "horizon_bins"]
    assert not prediction.duplicated(key).any()
    assert prediction["predicted_survival"].between(0.0, 1.0).all()
    assert np.isfinite(prediction["predicted_survival"]).all()
    assert prediction["score_probability_clipped_1e6"].between(
        1e-6, 1.0 - 1e-6
    ).all()

    for (_pipeline, _horizon), group in prediction.groupby(
        ["pipeline", "horizon_bins"], sort=False
    ):
        ordered = group.sort_values("test_row_index")
        assert ordered["test_row_index"].tolist() == list(range(821))
        assert _identifier_digest(ordered["episode_id"]) == EXPECTED_ORDER_DIGEST

    for _pipeline, group in prediction.groupby("pipeline", sort=False):
        matrix = (
            group.pivot(
                index="test_row_index",
                columns="horizon_bins",
                values="predicted_survival",
            )
            .sort_index()
            .sort_index(axis=1)
            .to_numpy(float)
        )
        assert np.all(np.diff(matrix, axis=1) <= 1e-12)

    unknown = prediction["outcome_status"].eq("censored_unknown")
    assert prediction.loc[unknown, "observed_survival_target"].isna().all()
    assert prediction.loc[unknown, "ipcw_weight"].eq(0.0).all()
    assert prediction.loc[unknown, "evaluable_at_horizon"].eq(False).all()


def test_released_row_contributions_recompute_horizon_and_canonical_scores() -> None:
    prediction = pd.read_csv(
        RELEASE / "wetlinks_all_pipeline_horizon_predictions.csv.gz"
    )
    score = pd.read_csv(
        RELEASE / "wetlinks_all_pipeline_horizon_scores.csv"
    ).set_index(["pipeline", "horizon_bins"])

    for key, group in prediction.groupby(["pipeline", "horizon_bins"]):
        expected_brier = float(group["ipcw_brier_contribution"].mean())
        expected_log_loss = float(
            group["ipcw_log_loss_contribution"].sum()
            / group["ipcw_weight"].sum()
        )
        assert np.isclose(
            expected_brier,
            float(score.loc[key, "pointwise_ipcw_brier"]),
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.isclose(
            expected_log_loss,
            float(score.loc[key, "pointwise_ipcw_log_loss"]),
            rtol=1e-12,
            atol=1e-12,
        )

    canonical = pd.read_csv(
        ROOT / "results/manuscript/wetlinks_model_metrics.csv"
    ).set_index("model")
    integrated = score.groupby(level="pipeline")["pointwise_ipcw_brier"].mean()
    for pipeline in EXPECTED_PIPELINES:
        assert np.isclose(
            float(integrated.loc[pipeline]),
            float(canonical.loc[pipeline, "integrated_brier"]),
            rtol=1e-10,
            atol=1e-12,
        )


def test_calibration_status_coverage_and_preimputation_counts() -> None:
    calibration = pd.read_csv(
        RELEASE / "wetlinks_pipeline_calibration_fit_audit.csv"
    )
    assert len(calibration) == len(CALIBRATED_PIPELINES) * 12
    assert set(calibration["pipeline"]) == CALIBRATED_PIPELINES
    assert not calibration.duplicated(["pipeline", "horizon_bins"]).any()
    assert set(calibration["horizon_bins"]) == set(range(1, 13))
    assert not calibration["raw_fallback_used"].astype(bool).any()

    status = pd.read_csv(RELEASE / "wetlinks_pipeline_execution_status.csv")
    assert len(status) == 12
    assert set(status["pipeline"]) == set(EXPECTED_PIPELINES)
    assert status["fit_status"].eq("success").all()
    assert status["prediction_status"].eq("success").all()
    assert not status["model_object_released"].astype(bool).any()

    coverage = pd.read_csv(
        RELEASE / "wetlinks_preimputation_feature_coverage.csv"
    )
    assert len(coverage) == 3 * 16
    assert (
        coverage["finite_before_imputation"]
        + coverage["missing_or_nonfinite_before_imputation"]
        == coverage["episodes"]
    ).all()
    interaction = coverage["feature"].eq("rain_latency_interaction")
    assert coverage.loc[interaction, "structural_fill_rule_before_episode_construction"].str.contains(
        "missing rain_mm replaced by 0", regex=False
    ).all()
    assert coverage["imputer_source"].eq(
        "training_partition_feature_median"
    ).all()


def test_release_manifest_hashes_are_portable_and_complete() -> None:
    manifest = json.loads(
        (RELEASE / "wetlinks_prediction_release_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["test_episode_order_sha256"] == EXPECTED_ORDER_DIGEST
    assert manifest["pipelines"] == list(EXPECTED_PIPELINES)
    assert manifest["horizons"] == list(range(1, 13))
    assert manifest["canonical_metric_check"]["status"] == "passed"
    assert manifest["derivative_data_license"]["spdx"] == "CC-BY-SA-4.0"
    assert "/Users/" not in json.dumps(manifest)

    for item in manifest["files"]:
        path = RELEASE / item["path"]
        assert path.is_file()
        assert path.stat().st_size == item["bytes"]
        assert _sha256(path) == item["sha256"]
    for item in manifest["code_files"]:
        path = ROOT / item["path"]
        assert path.is_file()
        assert _sha256(path) == item["sha256"]

