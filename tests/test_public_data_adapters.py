from __future__ import annotations

import gzip
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from leo_service_survival.lens_adapter import (
    align_lens_irtt_to_dish,
    build_lens_connectivity_episodes,
    load_lens_dish_status,
    load_lens_irtt,
    load_lens_ping,
)
from leo_service_survival.fjordlink_adapter import (
    FjordLinkConfig,
    audit_fjordlink_concurrency,
    load_fjordlink_twamp,
)
from leo_service_survival.nuwins_adapter import (
    MainlandAlignmentConfig,
    align_nuwins_mainland_pair,
    load_nuwins_pre_aligned_panel,
)
from leo_service_survival.nexasphere_adapter import load_nexasphere_athens
from leo_service_survival.sat_in_view_adapter import load_sat_in_view_matches
from scripts.run_public_trace_validation import run_cross_campaign_proxy_benchmark


def _twamp_fixture(start: str, rtts: list[float]) -> str:
    base = pd.Timestamp(start)
    lines: list[str] = []
    for index, rtt in enumerate(rtts):
        sent = base + pd.Timedelta(seconds=index)
        reply = sent + pd.Timedelta(milliseconds=rtt)
        lines.append(
            f"{sent.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - twampy - "
            f"DEBUG: Sent to 192.0.2.1 [sseq={index}]\n"
        )
        lines.append(
            f"{reply.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - twampy - "
            "DEBUG: Reply from 192.0.2.1 "
            f"[rseq={index} sseq={index} rtt={rtt:.2f}ms "
            f"outbound={rtt / 2:.2f}ms inbound={rtt / 2:.2f}ms]\n"
        )
    return "".join(lines)


def test_fjordlink_requires_measured_clock_overlap_and_preserves_clock_basis(tmp_path):
    archive = tmp_path / "fjordlink.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "out-client-wyk/starlink_2025-01-01_12-00-00/twamp.log",
            _twamp_fixture("2025-01-01 12:00:00", [40.0, 42.0, 44.0]),
        )
        bundle.writestr(
            "out-client-wyk/CAU-0C_2025-01-01_12-00-01/twamp.log",
            _twamp_fixture("2025-01-01 12:00:01", [30.0, 31.0, 32.0]),
        )
        bundle.writestr(
            "out-client-wyk/CAU-4329_2025-01-02_12-00-00/twamp.log",
            _twamp_fixture("2025-01-02 12:00:00", [35.0, 36.0]),
        )

    sessions, replies = load_fjordlink_twamp(
        archive,
        config=FjordLinkConfig(
            max_sessions=10,
            max_replies_per_session=10,
            minimum_overlap_s=1.0,
        ),
    )
    assert set(sessions["operator_id"]) == {"starlink", "telekom", "vodafone"}
    assert len(replies) == 8
    assert sessions["timestamp_basis"].eq("logger_clock_timezone_unspecified").all()
    assert replies["source_clock_time"].dt.tz is None
    assert sessions["packet_count_consistent"].all()
    assert sessions["source_id"].str.contains("sha256:").all()

    audit = audit_fjordlink_concurrency(sessions, minimum_overlap_s=1.0)
    assert len(audit) == 1
    assert bool(audit.loc[0, "pairing_eligible"])
    assert audit.loc[0, "terrestrial_operator_id"] == "telekom"
    assert audit.loc[0, "clock_overlap_s"] >= 1.0


def test_fjordlink_separate_sessions_fail_closed(tmp_path):
    archive = tmp_path / "fjordlink.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "out-client-wyk/starlink_2025-01-01_12-00-00/twamp.log",
            _twamp_fixture("2025-01-01 12:00:00", [40.0, 42.0]),
        )
        bundle.writestr(
            "out-client-wyk/CAU-0C_2025-01-02_12-00-00/twamp.log",
            _twamp_fixture("2025-01-02 12:00:00", [30.0, 31.0]),
        )

    sessions, _ = load_fjordlink_twamp(archive)
    audit = audit_fjordlink_concurrency(sessions, minimum_overlap_s=1.0)
    assert not bool(audit.loc[0, "pairing_eligible"])
    assert audit.loc[0, "exclusion_reason"] == "no_synchronized_clock_overlap"
    assert audit.loc[0, "nearest_interval_gap_s"] > 80_000


def test_nuwins_pre_aligned_panel_is_long_separate_and_provenanced(tmp_path):
    path = tmp_path / "paired.csv"
    pd.DataFrame(
        {
            "A": ["starlink", "starlink"],
            "A_run": ["run-s", "run-s"],
            "A_time": [
                "2024-06-21T20:34:28.860Z",
                "2024-06-21T20:34:29.360Z",
            ],
            "A_throughput_mbps": [0.0, 12.5],
            "A_actual_tech": [pd.NA, pd.NA],
            "B": ["att", "att"],
            "B_run": ["run-a", "run-a"],
            "B_time": [
                "2024-06-21T20:34:28.932Z",
                "2024-06-21T20:34:29.432Z",
            ],
            "B_throughput_mbps": [8.0, 20.0],
            "B_actual_tech": ["5G-low", "5G-low"],
        }
    ).to_csv(path, index=False)

    result = load_nuwins_pre_aligned_panel(
        path,
        campaign_id="alaska_202406",
        panel_kind="downlink_throughput",
    )
    assert len(result) == 4
    assert result.groupby("decision_epoch_id").size().eq(2).all()
    assert result["rtt_ms"].isna().all()
    assert result["throughput_mbps"].notna().all()
    assert result.loc[result["path_id"].eq("starlink"), "throughput_mbps"].iloc[0] == 0.0
    assert set(result["source_row_number"]) == {2, 3}
    assert result["evidence_tier"].eq("field_observational").all()
    assert result["alignment_method"].eq("provider_pre_aligned_row").all()
    assert result["source_id"].str.contains("sha256:").all()
    assert "satellite_id" not in result
    assert "beam_id" not in result
    assert "eo_task_id" not in result

    with pytest.raises(ValueError, match="missing required columns"):
        load_nuwins_pre_aligned_panel(
            path,
            campaign_id="alaska_202406",
            panel_kind="rtt",
        )


def test_mainland_alignment_is_one_to_one_deterministic_and_fail_closed(tmp_path):
    anchor_path = tmp_path / "starlink.csv"
    candidate_path = tmp_path / "att.csv"
    pd.DataFrame(
        {
            "utc_ts": [2000, 2001, 3000, 4000],
            "run_id": [1000, 1000, 9000, 1000],
            "throughput_mbps": [10.0, 11.0, 12.0, np.nan],
        }
    ).to_csv(anchor_path, index=False)
    pd.DataFrame(
        {
            "utc_ts": [2000],
            "run_id": [1003],
            "throughput_mbps": [20.0],
        }
    ).to_csv(candidate_path, index=False)

    result = align_nuwins_mainland_pair(
        anchor_path,
        candidate_path,
        anchor_operator="starlink",
        candidate_operator="att",
        panel_kind="downlink_throughput",
        config=MainlandAlignmentConfig(
            run_tolerance_ms=5,
            sample_tolerance_ms=5,
        ),
    )
    assert len(result.observations) == 2
    assert result.observations["decision_epoch_id"].nunique() == 1
    assert result.observations["inter_path_skew_ms"].eq(0).all()
    assert result.observations["source_time_utc"].notna().all()
    assert result.observations["paired_source_time_utc"].notna().all()
    assert result.observations["alignment_eligible"].all()
    assert result.observations["evidence_tier"].eq("field_observational").all()
    assert len(result.audit) == 4
    assert result.audit["alignment_eligible"].sum() == 1
    assert set(result.audit["alignment_exclusion_reason"].dropna()) == {
        "no_unused_candidate_within_tolerance",
        "run_tolerance_exceeded",
        "missing_anchor_outcome",
    }

    repeated = align_nuwins_mainland_pair(
        anchor_path,
        candidate_path,
        anchor_operator="starlink",
        candidate_operator="att",
        panel_kind="downlink_throughput",
        config=MainlandAlignmentConfig(
            run_tolerance_ms=5,
            sample_tolerance_ms=5,
        ),
    )
    assert repeated.observations["decision_epoch_id"].tolist() == result.observations[
        "decision_epoch_id"
    ].tolist()


def _write_irtt_fixture(path, base_ns: int) -> None:
    def received(seqno: int, offset_ms: int, rtt_ms: int) -> dict:
        send_ns = base_ns + offset_ms * 1_000_000
        return {
            "seqno": seqno,
            "lost": "false",
            "timestamps": {
                "client": {
                    "send": {"wall": send_ns},
                    "receive": {"wall": send_ns + rtt_ms * 1_000_000},
                }
            },
            "delay": {
                "rtt": rtt_ms * 1_000_000,
                "send": 12_000_000,
                "receive": (rtt_ms - 12) * 1_000_000,
            },
        }

    payload = {
        "version": {"irtt": "fixture"},
        "config": {"params": {"interval": 10_000_000}},
        "stats": {"packets_sent": 4},
        "round_trips": [
            received(0, 0, 25),
            {
                "seqno": 1,
                "lost": "true_down",
                "timestamps": {
                    "client": {"send": {"wall": base_ns + 10_000_000}, "receive": {}}
                },
                "delay": {},
            },
            received(3, 40, 30),
            received(4, 50, 28),
        ],
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def _write_dish_fixture(path, base_time: pd.Timestamp) -> None:
    columns = [
        "",
        "result",
        "table",
        "_time",
        "node",
        "dish_pop_ping_latency_ms",
        "dish_pop_ping_drop_rate",
        "dish_downlink_throughput_bps",
        "dish_uplink_throughput_bps",
        "dish_gps_sats",
        "dish_gps_valid",
        "dish_obstruction_currently_obstructed",
        "dish_obstruction_fraction_obstructed",
        "dish_is_snr_above_noise_floor",
        "dish_is_snr_persistently_low",
        "dish_actuator_state",
        "dish_attitude_estimation_state",
        "dish_software_version",
        "dish_uptime",
    ]
    lines = [
        "#group,false\n",
        "#datatype,string\n",
        "#default,last\n",
        ",".join(columns) + "\n",
    ]
    for minute, latency in ((-1, 21.0), (1, 22.0)):
        timestamp = base_time + pd.Timedelta(minutes=minute)
        values = [
            "",
            "",
            "0",
            timestamp.isoformat().replace("+00:00", "Z"),
            "starlink-victoria",
            str(latency),
            "0",
            "1000",
            "200",
            "12",
            "true",
            "false",
            "0",
            "true",
            "false",
            "OK",
            "FILTER_CONVERGED",
            "fixture",
            "10",
        ]
        lines.append(",".join(values) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_lens_streaming_alignment_gaps_and_connectivity_episodes(tmp_path):
    base = pd.Timestamp("2025-11-14T19:00:00Z")
    base_ns = int(base.value)
    irtt_path = tmp_path / "irtt.json.gz"
    dish_path = tmp_path / "dish.csv"
    _write_irtt_fixture(irtt_path, base_ns)
    _write_dish_fixture(dish_path, base)

    probes = load_lens_irtt(irtt_path, max_records=3, gap_multiplier=2.0)
    assert len(probes) == 3
    assert probes["source_subset_truncated"].all()
    assert probes["probe_received"].tolist() == [True, False, True]
    assert probes.loc[2, "sequence_gap_count"] == 1
    assert probes.loc[2, "trace_gap_detected"]
    assert probes.loc[0, "rtt_ms"] == 25.0

    telemetry = load_lens_dish_status(dish_path, max_records=2)
    assert len(telemetry) == 2
    assert telemetry["dish_gps_valid"].eq(True).all()
    aligned = align_lens_irtt_to_dish(
        probes, telemetry, max_telemetry_age_s=75.0
    )
    assert aligned["telemetry_aligned"].all()
    assert aligned["telemetry_age_ms"].between(60_000, 60_050).all()
    assert aligned["aligned_source_ids"].str.contains(";").all()
    episodes = build_lens_connectivity_episodes(
        aligned, maximum_contiguous_gap_ms=25.0
    )
    assert episodes["connectivity_state"].tolist() == [
        "reachable",
        "probe_loss",
        "reachable",
    ]
    assert episodes["probe_count"].sum() == 3
    assert episodes["evidence_tier"].eq("field_observational").all()
    for forbidden in ("satellite_id", "beam_id", "gateway_id", "eo_task_id"):
        assert forbidden not in aligned
        assert forbidden not in episodes


def test_lens_ping_retains_reply_sequence_gaps_without_fabricating_loss(tmp_path):
    path = tmp_path / "ping-testnode-100.64.0.1-10ms-1h-fixture.txt"
    path.write_text(
        "PING 100.64.0.1 (100.64.0.1) 56(84) bytes of data.\n"
        "[1763146800.010000] 64 bytes from 100.64.0.1: icmp_seq=1 ttl=64 time=20.1 ms\n"
        "[1763146800.020000] 64 bytes from 100.64.0.1: icmp_seq=2 ttl=64 time=21.2 ms\n"
        "[1763146800.060000] 64 bytes from 100.64.0.1: icmp_seq=5 ttl=64 time=23.4 ms\n",
        encoding="utf-8",
    )
    result = load_lens_ping(path, max_records=10)
    assert len(result) == 3
    assert result["probe_received"].all()
    assert result["sequence_gap_count"].tolist() == [0, 0, 2]
    assert result["trace_gap_detected"].tolist() == [False, False, True]
    assert "missing_ping_reply_sequences" in result.loc[2, "trace_gap_reason"]
    assert result["node_id"].eq("testnode").all()
    assert result["source_id"].str.contains("member:").all()
    assert result["evidence_tier"].eq("field_observational").all()


def test_sat_in_view_marks_identity_as_inferred_and_respects_gaps(tmp_path):
    path = tmp_path / "matched.csv"
    pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                [
                    "2024-06-22T00:00:00Z",
                    "2024-06-22T00:00:01Z",
                    "2024-06-22T00:00:02Z",
                    "2024-06-22T00:00:10Z",
                ]
            ),
            "Y": [71, 71, 72, 72],
            "X": [76, 76, 77, 78],
            "Elevation": [47.2, 47.1, 46.8, 45.0],
            "Azimuth": [25.1, 25.2, 26.0, 28.0],
            "PingTime": [40.0, 41.0, 42.0, 43.0],
            "Connected_Satellite": ["SAT-A", "SAT-A", "SAT-B", "SAT-C"],
            "Distance": [740.0, 741.0, 742.0, 750.0],
        }
    ).to_csv(path, index=False)

    result = load_sat_in_view_matches(
        path, max_rows=4, maximum_transition_gap_s=2.0
    )
    assert result["inferred_handover_event"].tolist() == [False, False, True, False]
    assert result["identity_change_unresolved_across_gap"].tolist() == [
        False,
        False,
        False,
        True,
    ]
    assert result["identity_evidence"].str.contains("not_direct_telemetry").all()
    assert result["source_id"].str.contains("sha256:").all()
    assert result["evidence_tier"].eq("field_observational").all()
    assert "satellite_id" not in result
    assert "beam_id" not in result
    assert "gateway_id" not in result
    assert "rf_snr_db" not in result


def test_cross_campaign_proxy_benchmark_is_labeled_and_cluster_paired():
    def campaign(name: str, offset: float) -> pd.DataFrame:
        records = []
        for path_id in ("starlink", "att"):
            for run in range(3):
                for sample in range(20):
                    throughput = (
                        0.0
                        if (sample + run) % 7 in (0, 1)
                        else 10.0 + offset + (path_id == "att") * 2.0
                    )
                    records.append(
                        {
                            "campaign_id": name,
                            "path_id": path_id,
                            "source_run_id": f"run-{run}",
                            "source_time_utc": pd.Timestamp("2025-01-01", tz="UTC")
                            + pd.Timedelta(seconds=run * 100 + sample / 2),
                            "throughput_mbps": throughput,
                        }
                    )
        return pd.DataFrame.from_records(records)

    metrics, intervals = run_cross_campaign_proxy_benchmark(
        {
            "train": campaign("train", 0.0),
            "test": campaign("test", 1.0),
        },
        train_campaign="train",
        test_campaign="test",
        payload_bytes=2_000_000,
        future_rows=4,
        assumed_interval_s=0.5,
        bootstrap_repetitions=10,
        seed=11,
    )
    assert set(metrics["model"]) == {
        "train_prevalence",
        "current_state_persistence",
        "logistic_lag_model",
    }
    assert metrics["evidence_tier"].eq("measured_trace_replay").all()
    assert metrics["endpoint"].eq(
        "fixed_payload_future_capacity_trace_replay_proxy"
    ).all()
    assert (
        (intervals["model"] == "logistic_lag_model")
        & (intervals["reference"] == "current_state_persistence")
    ).any()
    assert intervals["cluster_unit"].eq("source_run_id").all()


def test_nexasphere_excludes_april_pilot_and_marks_private_analysis(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "start_timestamp_utc": "2026-05-11T12:06:16Z",
                "end_timestamp_utc": "2026-05-18T12:07:03Z",
            }
        ),
        encoding="utf-8",
    )
    timestamps = ["2026-04-28T11:44:38Z", "2026-05-11T12:06:17Z"]
    pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "cycle": [0, 1],
            "target": ["8.8.8.8", "8.8.8.8"],
            "min_ms": [20.0, 21.0],
            "avg_ms": [25.0, 26.0],
            "max_ms": [30.0, 31.0],
            "loss_pct": [0.0, 0.0],
        }
    ).to_csv(tmp_path / "latency.csv", index=False)
    pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "cycle": [0, 1],
            "dish_dl_bps": [1.0, 2.0],
            "dish_ul_bps": [1.0, 2.0],
            "pop_ping_ms": [20.0, 21.0],
            "pop_drop_rate": [0.0, 0.0],
            "snr": [True, True],
            "obstruction_pct": [0.0, 0.0],
            "obstructed": [False, False],
            "uptime_s": [1, 2],
            "state": ["CONNECTED", "CONNECTED"],
            "error": [pd.NA, pd.NA],
        }
    ).to_csv(tmp_path / "antenna_metrics.csv", index=False)
    pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "cycle": [0, 1],
            "iperf3_dl_mbps": [100.0, 110.0],
            "iperf3_dl_error": [pd.NA, pd.NA],
            "speedtest_dl_mbps": [120.0, 130.0],
            "speedtest_ping_ms": [30.0, 31.0],
            "iperf3_server_used": ["server", "server"],
            "speedtest_server": ["Athens", "Athens"],
            "speedtest_error": [pd.NA, pd.NA],
        }
    ).to_csv(tmp_path / "throughput_Down.csv", index=False)
    pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "cycle": [0, 1],
            "iperf3_ul_mbps": [10.0, 11.0],
            "iperf3_ul_error": [pd.NA, pd.NA],
            "speedtest_ul_mbps": [12.0, 13.0],
            "iperf3_server_used": ["server", "server"],
            "speedtest_server": ["Athens", "Athens"],
            "speedtest_error": [pd.NA, pd.NA],
        }
    ).to_csv(tmp_path / "throughput_Up.csv", index=False)

    bundle = load_nexasphere_athens(tmp_path)
    for table in (bundle.latency, bundle.antenna, bundle.downlink, bundle.uplink):
        assert len(table) == 1
        assert table["source_time_utc"].min() >= pd.Timestamp(
            "2026-05-11T12:06:16Z"
        )
        assert table["usage_restriction"].eq(
            "private_analysis_license_unresolved_no_raw_redistribution"
        ).all()
        assert table["evidence_tier"].eq("field_observational").all()
        assert "satellite_id" not in table
