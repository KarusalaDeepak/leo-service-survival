"""Hand-calculated checks for the discrete censoring convention."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import leo_service_survival.metrics as metric_module
from leo_service_survival.data import (
    CENSOR_REASON_ADMINISTRATIVE_HORIZON,
    CENSOR_REASON_UNRESOLVED_BAD_ONSET,
    _episodes_from_regular_series,
)
from leo_service_survival.metrics import (
    _censoring_survival,
    evaluate_survival_predictions,
    ipcw_restricted_duration_targets,
    ipcw_survival_targets,
)
from leo_service_survival.models import FeatureEncoder, _expand_person_period
from scripts.audit_ipcw_boundaries import (
    build_ipcw_boundary_audit,
    write_ipcw_boundary_audit,
)
from scripts.run_all import truncate


def _regular_episode(causes: list[str]) -> dict:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=len(causes), freq="5min", tz="UTC"
            ),
            "observed": [True] * len(causes),
            "usable": [cause == "unknown" for cause in causes],
            "cause": causes,
            "x": np.arange(len(causes), dtype=float),
        }
    )
    return _episodes_from_regular_series(
        frame,
        site="fixture",
        timestamp_column="timestamp",
        observed_column="observed",
        usable_column="usable",
        cause_column="cause",
        feature_columns=("x",),
        bin_seconds=300,
        sustained_bad_bins=2,
        max_gap_bins=2,
    )[0]


def test_hand_calculated_reverse_km_ties_horizon_and_clipping():
    # At t=2, one censoring boundary among five at-risk rows gives G(2)=4/5;
    # both tied rows then leave.  At t=3, one of three at-risk rows is censored,
    # so G(3)=(4/5)(2/3)=8/15.  The final censor drives G(4) to zero, which is
    # clipped to the declared numerical floor of 1e-3.
    frame = pd.DataFrame(
        {
            "duration_bins": [2, 2, 3, 3, 4],
            "event": [1, 0, 1, 0, 0],
        }
    )
    query = np.asarray([1, 2, 3, 4])
    assert np.allclose(
        _censoring_survival(frame, query, left_limit=False),
        [1.0, 4.0 / 5.0, 8.0 / 15.0, 1e-3],
    )
    assert np.allclose(
        _censoring_survival(frame, query, left_limit=True),
        [1.0, 1.0, 4.0 / 5.0, 8.0 / 15.0],
    )

    known, target, weights = ipcw_survival_targets(frame, horizon=3)
    assert known.tolist() == [True, False, True, False, True]
    assert target.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert np.allclose(weights, [5.0 / 4.0, 0.0, 15.0 / 8.0, 0.0, 15.0 / 8.0])

    duration_known, reference, duration_weights = ipcw_restricted_duration_targets(
        frame, horizon=3
    )
    assert duration_known.tolist() == known.tolist()
    assert np.allclose(reference[duration_known], [2.0, 3.0, 3.0])
    assert np.isnan(reference[~duration_known]).all()
    assert np.allclose(duration_weights, weights)


def test_horizon_confirmation_admin_encoding_and_person_period_exposure():
    # The event begins at H=2 but is known only after the confirming H+1 bin.
    confirmed = _regular_episode(["unknown", "unknown", "loss", "loss"])
    unresolved = _regular_episode(["unknown", "unknown", "loss"])
    ended_good = _regular_episode(["unknown", "unknown", "unknown", "unknown"])
    administrative = truncate(pd.DataFrame([ended_good]), max_horizon=2).iloc[0].to_dict()

    assert confirmed["event"] == 1
    assert confirmed["duration_bins"] == 2
    assert confirmed["event_onset_time"] == pd.Timestamp("2026-01-01 00:10:00Z")
    assert confirmed["event_confirmation_time"] == pd.Timestamp(
        "2026-01-01 00:15:00Z"
    )
    assert unresolved["event"] == 0
    assert unresolved["duration_bins"] == 2
    assert unresolved["censor_reason"] == CENSOR_REASON_UNRESOLVED_BAD_ONSET
    assert administrative["event"] == 0
    assert administrative["duration_bins"] == 3
    assert administrative["censor_reason"] == CENSOR_REASON_ADMINISTRATIVE_HORIZON

    episodes = pd.DataFrame([confirmed, unresolved, administrative])
    known, target, weights = ipcw_survival_targets(episodes, horizon=2)
    assert known.tolist() == [True, False, True]
    assert target.tolist() == [0.0, 0.0, 1.0]
    assert np.allclose(weights, [1.5, 0.0, 1.5])

    encoder = FeatureEncoder(("x",)).fit(episodes)
    confirmed_x, confirmed_y, _, _ = _expand_person_period(
        episodes.iloc[[0]], encoder, max_horizon=2, competing=False
    )
    unresolved_x, unresolved_y, _, _ = _expand_person_period(
        episodes.iloc[[1]], encoder, max_horizon=2, competing=False
    )
    administrative_x, administrative_y, _, _ = _expand_person_period(
        episodes.iloc[[2]], encoder, max_horizon=2, competing=False
    )
    assert len(confirmed_x) == 2
    assert confirmed_y.tolist() == [0, 1]
    # Censoring at the possible onset H excludes the unresolved bad bin itself
    # but retains the single fully observed pre-boundary hazard row.
    assert len(unresolved_x) == 1
    assert unresolved_y.tolist() == [0]
    # Administrative H+1 encoding contributes every analysed bin through H.
    assert len(administrative_x) == 2
    assert administrative_y.tolist() == [0, 0]


def test_restricted_duration_masks_undefined_references_before_arithmetic(monkeypatch):
    frame = pd.DataFrame(
        {
            "duration_bins": [2, 4, 2, 3],
            "event": [1, 0, 0, 0],
        }
    )
    survival = np.asarray(
        [
            [0.80, 0.20, 0.10],
            [0.95, 0.90, 0.85],
            [0.85, 0.70, 0.55],
            [0.75, 0.50, 0.25],
        ]
    )
    original_abs = np.abs

    def finite_only_abs(values, *args, **kwargs):
        assert np.isfinite(np.asarray(values, dtype=float)).all()
        return original_abs(values, *args, **kwargs)

    monkeypatch.setattr(metric_module.np, "abs", finite_only_abs)
    result = evaluate_survival_predictions(frame, survival, horizons=(1, 2, 3))
    assert np.isfinite(result["restricted_mean_time_mae_bins"])


def test_machine_readable_boundary_audit_covers_declared_quantities():
    checks, summary = build_ipcw_boundary_audit()

    assert len(checks) == 23
    assert checks["pass_flag"].all()
    assert set(checks["check_family"]) == {
        "reverse_km_post_boundary_g",
        "reverse_km_left_limit_g",
        "ipcw_weight_h3",
        "evaluable_at_h3",
        "episode_status_h3",
    }
    assert summary["all_passed"] is True
    assert summary["passed_checks"] == summary["total_checks"] == 23
    assert summary["failed_checks"] == 0
    assert summary["fixture"]["duration_bins"] == [2, 2, 3, 3, 4]
    assert summary["fixture"]["event"] == [1, 0, 1, 0, 0]


def test_machine_readable_boundary_audit_writes_deterministic_csv_and_json(tmp_path):
    csv_path, json_path, summary = write_ipcw_boundary_audit(tmp_path)
    first_csv = csv_path.read_bytes()
    first_json = json_path.read_bytes()

    write_ipcw_boundary_audit(tmp_path)
    assert csv_path.read_bytes() == first_csv
    assert json_path.read_bytes() == first_json

    written = pd.read_csv(csv_path)
    assert written["pass_flag"].all()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded == summary
    assert loaded["checks_csv"] == csv_path.name

