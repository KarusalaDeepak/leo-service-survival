from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from leo_service_survival.endpoint_sensitivity_audit import (
    AuditHazardModel, EndpointAuditProfile, cadence_support,
    confirmation_safe_calendar_split, prepare_regular_series, rebuild_episodes, registered_profiles,
    truncate_confirmation_aware,
)
from leo_service_survival.data import WETLINKS_FEATURES


def _regular():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC"),
        "latency_mean_ms": [40, 90, 90, np.nan, 40, 40, 90, 90],
        "packet_loss_pct": [0, 0, 0, np.nan, 0, 0, 0, 0],
        "observed": [True, True, True, False, True, True, True, True],
    })
    for column in WETLINKS_FEATURES:
        if column not in frame:
            frame[column] = 0.0
    return {"fixture": frame}


def test_registry_covers_every_declared_endpoint_level_and_interactions():
    profiles = registered_profiles()
    for column, values in {
        "bin_minutes": {1, 5, 10}, "latency_limit_ms": {70, 80, 90},
        "loss_limit_pct": {0.5, 1, 2}, "sustained_bad_bins": {1, 2, 3},
        "maximum_gap_bins": {1, 2, 3}, "analysis_horizon_bins": {6, 9, 12},
    }.items():
        assert {getattr(profile, column) for profile in profiles} == values
    assert sum(profile.contrast.startswith("interaction") for profile in profiles) == 3


def test_one_minute_is_unsupported_not_interpolated():
    cadence = pd.DataFrame({"median_start_gap_seconds": [211.0, 185.0]})
    assert cadence_support(EndpointAuditProfile(), cadence)[0]
    supported, reason = cadence_support(replace(EndpointAuditProfile(), bin_minutes=1), cadence)
    assert not supported
    assert reason == "not_supported_by_source_cadence_no_upsampling"


def test_unknown_cadence_is_not_claimed_supported():
    supported, reason = cadence_support(EndpointAuditProfile(), pd.DataFrame())
    assert not supported
    assert "missing_cadence" in reason


def test_regularization_keeps_unmeasured_bins_missing(tmp_path):
    timestamps = ["2026-01-01 00:00:00", "2026-01-01 00:05:00", "2026-01-01 00:15:00"]
    raw = pd.DataFrame({"site_name": "fixture", "timestamp_start": timestamps})
    for name in ("ping_avg", "ping_worst", "ping_stddev", "ping_packet_loss", "download", "upload",
                 "temp", "humidity", "rain", "windspeed", "barom"):
        raw[name] = 0.0
    raw["ping_avg"] = [40.0, 45.0, 50.0]
    path = tmp_path / "analysis_data_fixture.csv"
    raw.to_csv(path, index=False)
    frame = prepare_regular_series([path], 5)["fixture"]
    missing = frame.iloc[2]
    assert not missing.observed
    assert pd.isna(missing.latency_mean_ms)
    assert frame.latency_mean_ms.notna().sum() == 3


def test_missing_regular_bin_is_censored_at_all_registered_gap_levels():
    profiles = [replace(EndpointAuditProfile(), maximum_gap_bins=gap) for gap in (1, 2, 3)]
    frames = [rebuild_episodes(_regular(), profile) for profile in profiles]
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], frame)


def test_confirmation_lookahead_includes_persistence_after_horizon():
    frame = rebuild_episodes(_regular(), replace(EndpointAuditProfile(), sustained_bad_bins=3))
    frame.loc[0, "duration_bins"] = 30
    truncated = truncate_confirmation_aware(frame, 6)
    expected = truncated.loc[0, "start_time"] + pd.Timedelta(minutes=5 * (6 + 3 - 1))
    assert truncated.loc[0, "analysis_label_resolution_time"] == expected
    assert truncated.loc[0, "duration_bins"] == 7
    assert truncated.loc[0, "event"] == 0


def _calendar_frame():
    start = pd.date_range("2025-01-06", periods=20, freq="7D", tz="UTC")
    return pd.DataFrame({
        "start_time": start, "calendar_block": start.strftime("%G-W%V"),
        "site": "fixture", "analysis_label_resolution_time": start + pd.Timedelta(hours=1),
    })


def test_confirmation_safe_split_has_no_shared_week_or_future_label():
    frame = _calendar_frame()
    frame.loc[11, "analysis_label_resolution_time"] = frame.loc[14, "start_time"]
    parts, audit = confirmation_safe_calendar_split(frame)
    assert audit[0]["resolution_purge_rows"] == 1
    for earlier, later in zip(parts, parts[1:]):
        assert earlier.analysis_label_resolution_time.max() < later.start_time.min()
        assert not set(earlier.calendar_block) & set(later.calendar_block)


def test_sparse_calendar_support_is_retained_as_non_estimable_error():
    with pytest.raises(ValueError, match="eight calendar"):
        confirmation_safe_calendar_split(_calendar_frame().iloc[:6])


def test_audit_hazard_disables_person_period_random_early_stopping():
    frame = rebuild_episodes(_regular(), EndpointAuditProfile())
    model = AuditHazardModel(WETLINKS_FEATURES, max_horizon=3, competing=True).fit(frame)
    assert model.model_.early_stopping is False


def test_no_event_support_is_an_explicit_fit_error():
    frame = rebuild_episodes(_regular(), EndpointAuditProfile())
    frame["event"] = 0
    frame["cause"] = "censored"
    with pytest.raises(ValueError, match="no event class"):
        AuditHazardModel(WETLINKS_FEATURES, max_horizon=3, competing=True).fit(frame)
