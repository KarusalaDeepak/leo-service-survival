from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leo_service_survival.eos_full_chain_predictor import (
    FEATURE_NAMES,
    fit_full_chain_predictor,
    predictor_design,
    audit_full_chain_regularization_grid,
)


def _partitions():
    frames = {}
    for index, role in enumerate(("training", "tuning", "calibration", "validation")):
        seed = 1_000 + index
        rng = np.random.default_rng(seed)
        rows = 200
        rate = rng.uniform(0.02, 0.04, rows)
        outage = rng.uniform(0.01, 0.3, rows)
        target_probability = np.clip(0.3 + 10.0 * rate - outage, 0.05, 0.95)
        frames[role] = pd.DataFrame({
            "forecast_capacity_ratio": rng.uniform(1.0, 3.0, rows),
            "profile_mean_rate_mbps": rate,
            "required_bytes": rng.integers(100_000, 250_000, rows),
            "planned_downlink_duration_s": 90.0,
            "profile_rate_log_sd": rng.uniform(0.1, 0.3, rows),
            "profile_outage_probability": outage,
            "forecast_handover_probability": rng.uniform(0.01, 0.3, rows),
            "forecast_jitter_ms": rng.uniform(2.0, 30.0, rows),
            "technical_completion": rng.binomial(1, target_probability),
            "partition": role,
            "reference_seed": seed,
            "reference_row_id": [f"{role}:{row}" for row in range(rows)],
            "path_available": True,
        })
    return frames


def test_validation_labels_cannot_select_or_fit_predictor_or_calibration():
    partitions = _partitions()
    first = fit_full_chain_predictor(partitions)
    changed = {role: frame.copy() for role, frame in partitions.items()}
    changed["validation"]["technical_completion"] = 1 - changed["validation"]["technical_completion"]
    second = fit_full_chain_predictor(changed)
    assert first.model.sha256 == second.model.sha256
    assert first.model == second.model
    pd.testing.assert_frame_equal(
        first.tables["predictor_regularization_selection"],
        second.tables["predictor_regularization_selection"],
    )
    assert not first.tables["predictor_probability_diagnostics"].equals(
        second.tables["predictor_probability_diagnostics"]
    )


def test_predictor_ignores_all_extra_future_outcomes_and_respects_path_availability():
    partitions = _partitions()
    bundle = fit_full_chain_predictor(partitions)
    frame = partitions["validation"].copy()
    first = bundle.model.predict(frame)
    frame["technical_completion"] = 1 - frame["technical_completion"]
    frame["realized_outage"] = True
    frame["realized_handover"] = True
    frame["realized_rate_mbps"] = 0.0
    frame["bytes_delivered"] = 0
    frame["application_complete"] = False
    np.testing.assert_array_equal(first, bundle.model.predict(frame))
    frame["path_available"] = False
    np.testing.assert_array_equal(bundle.model.predict(frame), np.zeros(len(frame)))
    assert not any("realized" in feature for feature in FEATURE_NAMES)
    assert bundle.manifest["validation_role"].startswith("report_only")
    assert bundle.model.calibration_slope >= 0


def test_reference_roles_require_disjoint_seeds_and_row_identifiers():
    partitions = _partitions()
    partitions["validation"]["reference_seed"] = partitions["training"]["reference_seed"].iloc[0]
    with pytest.raises(ValueError, match="disjoint"):
        fit_full_chain_predictor(partitions)
    partitions = _partitions()
    partitions["validation"]["reference_row_id"] = partitions["training"]["reference_row_id"]
    with pytest.raises(ValueError, match="disjoint"):
        fit_full_chain_predictor(partitions)


def test_predictor_rejects_missing_nonfinite_or_invalid_forecast_inputs():
    validation = _partitions()["validation"]
    with pytest.raises(ValueError, match="missing inputs"):
        predictor_design(validation.drop(columns="forecast_handover_probability"))
    bad = validation.copy()
    bad.loc[0, "forecast_jitter_ms"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predictor_design(bad)
    bad = validation.copy()
    bad.loc[0, "required_bytes"] = 0
    with pytest.raises(ValueError, match="positive"):
        predictor_design(bad)


def test_regularization_range_audit_cannot_use_validation_labels():
    partitions = _partitions()
    reference = pd.concat(partitions.values(), ignore_index=True)
    first = audit_full_chain_regularization_grid(reference, frozen_regularization_c=0.01)
    changed = reference.copy()
    changed.loc[changed["partition"].isin(["validation", "calibration"]), "technical_completion"] = 0
    second = audit_full_chain_regularization_grid(changed, frozen_regularization_c=0.01)
    pd.testing.assert_frame_equal(first, second)
    assert not first["validation_or_evaluation_outcomes_used"].any()
