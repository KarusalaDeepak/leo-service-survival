from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from leo_service_survival.data import (
    CENSOR_REASON_ADMINISTRATIVE_HORIZON,
    CENSOR_REASON_INTERNAL_MEASUREMENT_GAP,
    CENSOR_REASON_QUALIFYING_EVENT,
    CENSOR_REASON_RECORDING_ENDED_GOOD,
    CENSOR_REASON_UNRESOLVED_BAD_ONSET,
    _episodes_from_regular_series,
    build_commect_landmarks,
    calendar_block_split,
    purged_epoch_split,
)
from scripts.run_all import (
    truncate,
    wetlinks_censor_reason_breakdown,
    write_wetlinks_endpoint_audits,
)


def _regular_episode(causes, *, observed=None, timestamps=None):
    causes = list(causes)
    if observed is None:
        observed = [True] * len(causes)
    if timestamps is None:
        timestamps = pd.date_range(
            "2026-01-01", periods=len(causes), freq="5min", tz="UTC"
        )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "observed": observed,
            "usable": [
                bool(is_observed and cause == "unknown")
                for cause, is_observed in zip(causes, observed)
            ],
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


def test_regular_endpoint_records_qualifying_event_reason():
    episode = _regular_episode(["unknown", "latency", "latency"])

    assert episode["event"] == 1
    assert episode["censor_reason"] == CENSOR_REASON_QUALIFYING_EVENT


def test_regular_endpoint_distinguishes_record_end_from_unresolved_bad_onset():
    ended_good = _regular_episode(["unknown", "unknown", "unknown"])
    unresolved = _regular_episode(["unknown", "unknown", "loss"])

    assert ended_good["event"] == 0
    assert ended_good["censor_reason"] == CENSOR_REASON_RECORDING_ENDED_GOOD
    assert unresolved["event"] == 0
    assert unresolved["censor_reason"] == CENSOR_REASON_UNRESOLVED_BAD_ONSET


def test_regular_endpoint_distinguishes_gap_from_missing_confirmation():
    gap = _regular_episode(
        ["unknown", "unknown", "unknown"], observed=[True, True, False]
    )
    missing_confirmation = _regular_episode(
        ["unknown", "latency", "unknown"], observed=[True, True, False]
    )

    assert gap["censor_reason"] == CENSOR_REASON_INTERNAL_MEASUREMENT_GAP
    assert (
        missing_confirmation["censor_reason"]
        == CENSOR_REASON_UNRESOLVED_BAD_ONSET
    )


def test_regular_endpoint_marks_timestamp_discontinuity_as_internal_gap():
    timestamps = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
            "2026-01-01T00:30:00Z",
        ],
        utc=True,
    )
    episode = _regular_episode(
        ["unknown", "unknown", "unknown"], timestamps=timestamps
    )

    assert episode["censor_reason"] == CENSOR_REASON_INTERNAL_MEASUREMENT_GAP


def test_commect_complete_horizon_is_administrative_not_record_end(tmp_path):
    path = tmp_path / "commect.csv"
    pd.DataFrame(
        {
            "relative_path": ["sat"] * 4,
            "bin_start_utc": pd.date_range(
                "2026-01-01", periods=4, freq="s", tz="UTC"
            ),
            "bin_epoch": [100, 101, 102, 103],
            "latency_mean_ms": [40.0, 41.0, 42.0, 43.0],
            "latency_std_ms": [1.0] * 4,
            "latency_max_ms": [45.0, 46.0, 47.0, 48.0],
            "observed_replies": [1.0] * 4,
            "observation_age_ms": [0.0] * 4,
            "rsrp_mean_dbm": [np.nan] * 4,
            "rsrq_mean_db": [np.nan] * 4,
            "snr_mean_db": [np.nan] * 4,
            "access_technology": ["satellite"] * 4,
        }
    ).to_csv(path, index=False)

    frame = build_commect_landmarks(path, max_horizon_seconds=2).frame
    complete = frame.loc[frame["decision_epoch"].eq(100)].iloc[0]
    ended = frame.loc[frame["decision_epoch"].eq(103)].iloc[0]

    assert complete["censor_reason"] == CENSOR_REASON_ADMINISTRATIVE_HORIZON
    assert ended["censor_reason"] == CENSOR_REASON_RECORDING_ENDED_GOOD


def test_calendar_embargo_is_reported_as_split_exclusion_not_censoring():
    rows = []
    for block in range(10):
        rows.append(
            {
                "calendar_block": f"2026-W{block + 1:02d}",
                "start_time": pd.Timestamp("2026-01-01", tz="UTC")
                + pd.Timedelta(weeks=block),
                "site": "fixture",
                "censor_reason": CENSOR_REASON_RECORDING_ENDED_GOOD,
            }
        )
    train, calibration, test = calendar_block_split(pd.DataFrame(rows))

    audit = train.attrs["split_exclusion_audit"]
    assert audit["reason"] == "calendar_block_embargo"
    assert audit["endpoint_censoring"] is False
    assert audit["excluded_episode_count"] == 2
    assert calibration.attrs["split_exclusion_audit"] == audit
    assert test.attrs["split_exclusion_audit"] == audit
    assert set(pd.concat([train, calibration, test])["censor_reason"]) == {
        CENSOR_REASON_RECORDING_ENDED_GOOD
    }


def test_epoch_purge_is_reported_as_split_exclusion_not_censoring():
    frame = pd.DataFrame(
        {
            "decision_epoch": np.arange(100),
            "censor_reason": [CENSOR_REASON_ADMINISTRATIVE_HORIZON] * 100,
        }
    )
    train, calibration, test = purged_epoch_split(frame, purge_bins=3)

    audit = train.attrs["split_exclusion_audit"]
    assert audit["reason"] == "overlapping_label_purge"
    assert audit["endpoint_censoring"] is False
    assert audit["excluded_episode_count"] == 6
    assert calibration.attrs["split_exclusion_audit"] == audit
    assert test.attrs["split_exclusion_audit"] == audit


def test_horizon_truncation_marks_only_beyond_horizon_rows_administrative():
    frame = pd.DataFrame(
        {
            "duration_bins": [2, 7, 8],
            "event": [1, 1, 0],
            "cause": ["loss", "latency", "censored"],
            "censor_reason": [
                CENSOR_REASON_QUALIFYING_EVENT,
                CENSOR_REASON_QUALIFYING_EVENT,
                CENSOR_REASON_RECORDING_ENDED_GOOD,
            ],
        }
    )

    truncated = truncate(frame, max_horizon=5)

    assert truncated.loc[0, "duration_bins"] == 2
    assert truncated.loc[0, "event"] == 1
    assert truncated.loc[0, "cause"] == "loss"
    assert truncated.loc[0, "censor_reason"] == CENSOR_REASON_QUALIFYING_EVENT
    assert truncated.loc[1:, "duration_bins"].eq(6).all()
    assert truncated.loc[1:, "event"].eq(0).all()
    assert truncated.loc[1:, "cause"].eq("censored").all()
    assert truncated.loc[1:, "censor_reason"].eq(
        CENSOR_REASON_ADMINISTRATIVE_HORIZON
    ).all()


def _endpoint_audit_partitions():
    audit = {
        "reason": "calendar_block_embargo",
        "excluded_blocks": ["2026-W03"],
        "excluded_episode_count": 4,
        "endpoint_censoring": False,
    }
    partitions = {}
    for name, records in {
        "train": [
            (2, 1, "loss", CENSOR_REASON_QUALIFYING_EVENT),
            (5, 1, "latency", CENSOR_REASON_QUALIFYING_EVENT),
            (2, 0, "censored", CENSOR_REASON_RECORDING_ENDED_GOOD),
            (13, 0, "censored", CENSOR_REASON_ADMINISTRATIVE_HORIZON),
        ],
        "calibration": [
            (2, 0, "censored", CENSOR_REASON_INTERNAL_MEASUREMENT_GAP)
        ],
        "test": [
            (2, 0, "censored", CENSOR_REASON_UNRESOLVED_BAD_ONSET)
        ],
    }.items():
        durations, events, causes, reasons = zip(*records)
        frame = pd.DataFrame(
            {
                "site": ["site-a", "site-b", "site-a", "site-b"][: len(records)],
                "duration_bins": durations,
                "event": events,
                "cause": causes,
                "censor_reason": reasons,
            }
        )
        frame.attrs["split_exclusion_audit"] = audit.copy()
        partitions[name] = frame
    return partitions


def test_wetlinks_endpoint_audits_serialize_reasons_and_split_exclusions(tmp_path):
    partitions = _endpoint_audit_partitions()

    expected = wetlinks_censor_reason_breakdown(partitions)
    breakdown, exclusion = write_wetlinks_endpoint_audits(partitions, tmp_path)

    pd.testing.assert_frame_equal(breakdown, expected)
    written = pd.read_csv(tmp_path / "wetlinks_censor_reason_breakdown.csv")
    assert {"all_sites", "site-a", "site-b"}.issubset(
        set(written["site_scope"])
    )
    pooled = written.loc[written["site_scope"].eq("all_sites")]
    observed_totals = pooled.groupby(
        ["partition", "horizon_minutes"]
    )["episode_count"].sum()
    for (partition, horizon_minutes), count in observed_totals.items():
        assert horizon_minutes in {15.0, 30.0, 45.0, 60.0}
        assert count == len(partitions[partition])
    train_15 = pooled.loc[
        pooled["partition"].eq("train")
        & pooled["horizon_minutes"].eq(15.0)
    ].set_index("censor_reason")["episode_count"]
    train_30 = pooled.loc[
        pooled["partition"].eq("train")
        & pooled["horizon_minutes"].eq(30.0)
    ].set_index("censor_reason")["episode_count"]
    assert train_15[CENSOR_REASON_QUALIFYING_EVENT] == 1
    assert train_15[CENSOR_REASON_ADMINISTRATIVE_HORIZON] == 2
    assert train_30[CENSOR_REASON_QUALIFYING_EVENT] == 2
    assert train_30[CENSOR_REASON_ADMINISTRATIVE_HORIZON] == 1
    assert set(written["administrative_censoring_rule"]) == {
        "outcomes beyond H are censored at H+1"
    }
    serialized = json.loads(
        (tmp_path / "wetlinks_split_exclusion_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert serialized == exclusion
    assert serialized["consistent_across_partitions"] is True
    assert serialized["partition_audits"]["test"]["endpoint_censoring"] is False


def test_wetlinks_censor_reason_breakdown_rejects_event_reason_mismatch():
    partitions = _endpoint_audit_partitions()
    partitions["test"].loc[:, "event"] = 1

    with pytest.raises(ValueError, match="event/censor-reason mismatch"):
        wetlinks_censor_reason_breakdown(partitions)
