import json

import pandas as pd

from leo_service_survival.data import (
    WETLINKS_FEATURES,
    build_nexasphere_athens_episodes,
)


def test_nexasphere_builder_uses_metadata_window_and_preserves_missing_features(
    tmp_path,
):
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "start_timestamp_utc": "2026-05-11T00:00:00Z",
                "end_timestamp_utc": "2026-05-11T00:30:00Z",
            }
        ),
        encoding="utf-8",
    )
    times = pd.date_range("2026-05-11", periods=6, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": [
                "2026-04-28T00:00:00Z",
                *times.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
            "avg_ms": [150.0, 30.0, 30.0, 100.0, 100.0, 30.0, 30.0],
            "max_ms": [160.0, 40.0, 40.0, 110.0, 110.0, 40.0, 40.0],
            "loss_pct": [50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "speedtest_dl_mbps": [1.0, 100.0, 100.0, 80.0, 80.0, 90.0, 90.0],
            "speedtest_ul_mbps": [1.0, 20.0, 20.0, 18.0, 18.0, 19.0, 19.0],
        }
    )
    frame.to_csv(tmp_path / "all_metrics.csv", index=False)

    dataset = build_nexasphere_athens_episodes(tmp_path)

    assert dataset.feature_columns == WETLINKS_FEATURES
    assert dataset.frame["start_time"].min() >= pd.Timestamp(
        "2026-05-11T00:00:00Z"
    )
    assert int(dataset.frame["event"].sum()) == 1
    assert "latency" in set(dataset.frame["cause"])
    assert dataset.frame["temperature_c"].isna().all()
    assert dataset.frame["latency_std_ms"].isna().all()
