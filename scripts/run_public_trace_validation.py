#!/usr/bin/env python3
"""Bounded multi-campaign/site summaries for acquired public LEO traces.

This runner reports observational coverage and QoS distributions.  Its optional
cross-campaign benchmark is a fixed-payload *trace-replay proxy*, not an
operational EO completion outcome and not a policy trial.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from leo_service_survival.lens_adapter import (  # noqa: E402
    align_lens_irtt_to_dish,
    load_lens_dish_status,
    load_lens_irtt,
    load_lens_ping,
)
from leo_service_survival.fjordlink_adapter import (  # noqa: E402
    FjordLinkConfig,
    audit_fjordlink_concurrency,
    load_fjordlink_twamp,
)
from leo_service_survival.nuwins_adapter import (  # noqa: E402
    load_nuwins_pre_aligned_panel,
)
from leo_service_survival.nexasphere_adapter import (  # noqa: E402
    USAGE_RESTRICTION,
    load_nexasphere_athens,
)
from leo_service_survival.sat_in_view_adapter import (  # noqa: E402
    load_sat_in_view_matches,
)


EVIDENCE_TIER = "field_observational"
REPLAY_TIER = "measured_trace_replay"


def _quantiles(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"mean": np.nan, "p10": np.nan, "median": np.nan, "p90": np.nan}
    return {
        "mean": float(numeric.mean()),
        "p10": float(numeric.quantile(0.10)),
        "median": float(numeric.quantile(0.50)),
        "p90": float(numeric.quantile(0.90)),
    }


def _source_ids(frame: pd.DataFrame) -> str:
    """Return a deterministic, nonempty provenance token for a derived row."""

    if frame.empty:
        return "unavailable"
    column = (
        "source_id"
        if "source_id" in frame
        else "aligned_source_ids"
        if "aligned_source_ids" in frame
        else None
    )
    if column is None:
        return "unavailable"
    tokens: set[str] = set()
    for value in frame[column].dropna().astype(str):
        tokens.update(token.strip() for token in value.split(";") if token.strip())
    values = sorted(tokens)
    return ";".join(values) if values else "unavailable"


def _file_source_id(dataset: str, path: Path, context: str) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{dataset}:{context}:sha256:{digest.hexdigest()}"


def _campaign_from_path(path: Path) -> str:
    # RTT pair filenames omit the campaign, so inspect the full extracted path.
    text = str(path).lower()
    if "alaska" in text:
        return "alaska_202406"
    if "hawaii" in text:
        return "hawaii_202408"
    return "mainland_202411"


def _panel_from_path(path: Path) -> str:
    return "rtt" if "ping" in path.name.lower() or "latency" in path.parts else "downlink_throughput"


def summarize_nuwins(
    root: Path, *, max_rows_per_file: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    coverage: list[dict[str, object]] = []
    qos: list[dict[str, object]] = []
    benchmark_panels: dict[str, pd.DataFrame] = {}
    extracted = root / "datasets" / "extracted"

    paired_files = sorted(extracted.glob("*_road_test_*/processed/mptcp/*.csv"))
    paired_files += sorted(
        extracted.glob("*_road_test_*/processed/mpshell/raw_traces/*.csv")
    )
    for path in paired_files:
        campaign = _campaign_from_path(path)
        panel = _panel_from_path(path)
        try:
            frame = load_nuwins_pre_aligned_panel(
                path,
                campaign_id=campaign,
                panel_kind=panel,
                max_rows=max_rows_per_file,
            )
        except Exception as exc:  # Audit row is preferable to silent omission.
            coverage.append(
                {
                    "source_family": "NUWiNS",
                    "campaign_or_site": campaign,
                    "operator_id": pd.NA,
                    "panel_kind": panel,
                    "pair_id": pd.NA,
                    "records_loaded": 0,
                    "first_time_utc": pd.NaT,
                    "last_time_utc": pd.NaT,
                    "evidence_tier": EVIDENCE_TIER,
                    "observation_design": "publisher_pre_aligned_observational_panel",
                    "ingestion_status": f"error:{type(exc).__name__}:{exc}",
                    "source_file": str(path),
                }
            )
            continue
        if "starlink_att" in path.name and panel == "downlink_throughput":
            benchmark_panels[campaign] = frame.copy()
        for (pair, path_id), group in frame.groupby(["pair_id", "path_id"]):
            coverage.append(
                {
                    "source_family": "NUWiNS",
                    "campaign_or_site": campaign,
                    "operator_id": path_id,
                    "panel_kind": panel,
                    "pair_id": pair,
                    "records_loaded": len(group),
                    "first_time_utc": group["source_time_utc"].min(),
                    "last_time_utc": group["source_time_utc"].max(),
                    "evidence_tier": EVIDENCE_TIER,
                    "observation_design": "publisher_pre_aligned_observational_panel",
                    "ingestion_status": "ok",
                    "source_file": str(path),
                    "source_id": _source_ids(group),
                }
            )
            metric = "throughput_mbps" if panel == "downlink_throughput" else "rtt_ms"
            summary = _quantiles(group[metric])
            qos.append(
                {
                    "source_family": "NUWiNS",
                    "campaign_or_site": campaign,
                    "operator_id": path_id,
                    "panel_kind": panel,
                    "pair_id": pair,
                    "metric": metric,
                    "records": len(group),
                    "observed_fraction": float(group["outcome_observed"].mean()),
                    "positive_throughput_fraction": float(
                        group["throughput_mbps"].gt(0).mean()
                    )
                    if panel == "downlink_throughput"
                    else np.nan,
                    **summary,
                    "evidence_tier": EVIDENCE_TIER,
                    "availability_definition": (
                        "measured_throughput_gt_0_in_released_row"
                        if panel == "downlink_throughput"
                        else "not_estimated_from_paired_RTT_rows"
                    ),
                    "source_id": _source_ids(group),
                }
            )

    mainland_files = sorted(
        extracted.glob("la_to_omaha_road_test_*/processed/throughput/*.csv")
    ) + sorted(extracted.glob("la_to_omaha_road_test_*/processed/latency/*.csv"))
    for path in mainland_files:
        panel = _panel_from_path(path)
        metric = "throughput_mbps" if panel == "downlink_throughput" else "rtt_ms"
        try:
            raw = pd.read_csv(path, nrows=max_rows_per_file)
            if metric not in raw or "utc_ts" not in raw:
                raise ValueError(f"missing {metric} or utc_ts")
            operator = (
                str(raw["operator"].dropna().iloc[0]).lower()
                if "operator" in raw and raw["operator"].notna().any()
                else path.name.split("_")[0].split(".")[1]
                if path.name.startswith("icmp_ping.")
                else path.name.split("_")[0]
            )
            times = pd.to_datetime(
                pd.to_numeric(raw["utc_ts"], errors="coerce"),
                unit="ms",
                utc=True,
                errors="coerce",
            )
            values = pd.to_numeric(raw[metric], errors="coerce")
            source_id = _file_source_id(
                "nuwins_sigmetrics26", path, "mainland_202411"
            )
            coverage.append(
                {
                    "source_family": "NUWiNS",
                    "campaign_or_site": "mainland_202411",
                    "operator_id": operator,
                    "panel_kind": panel,
                    "pair_id": pd.NA,
                    "records_loaded": len(raw),
                    "first_time_utc": times.min(),
                    "last_time_utc": times.max(),
                    "evidence_tier": EVIDENCE_TIER,
                    "observation_design": "independent_per_operator_observational_trace_not_pair_aligned",
                    "ingestion_status": "ok",
                    "source_file": str(path),
                    "source_id": source_id,
                }
            )
            qos.append(
                {
                    "source_family": "NUWiNS",
                    "campaign_or_site": "mainland_202411",
                    "operator_id": operator,
                    "panel_kind": panel,
                    "pair_id": pd.NA,
                    "metric": metric,
                    "records": len(raw),
                    "observed_fraction": float(values.notna().mean()),
                    "positive_throughput_fraction": float(values.gt(0).mean())
                    if panel == "downlink_throughput"
                    else np.nan,
                    **_quantiles(values),
                    "evidence_tier": EVIDENCE_TIER,
                    "availability_definition": (
                        "measured_throughput_gt_0_in_source_row"
                        if panel == "downlink_throughput"
                        else "not_estimated_from_RTT_reply_rows"
                    ),
                    "source_id": source_id,
                }
            )
        except Exception as exc:
            coverage.append(
                {
                    "source_family": "NUWiNS",
                    "campaign_or_site": "mainland_202411",
                    "operator_id": pd.NA,
                    "panel_kind": panel,
                    "pair_id": pd.NA,
                    "records_loaded": 0,
                    "first_time_utc": pd.NaT,
                    "last_time_utc": pd.NaT,
                    "evidence_tier": EVIDENCE_TIER,
                    "observation_design": "independent_per_operator_observational_trace_not_pair_aligned",
                    "ingestion_status": f"error:{type(exc).__name__}:{exc}",
                    "source_file": str(path),
                }
            )
    return (
        pd.DataFrame.from_records(coverage),
        pd.DataFrame.from_records(qos),
        benchmark_panels,
    )


def summarize_fjordlink(
    root: Path,
    *,
    max_sessions: int,
    max_replies_per_session: int,
    minimum_overlap_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarise FjordLink without assuming its paths were concurrent."""

    archive = root / "out-client-wyk.zip"
    try:
        sessions, replies = load_fjordlink_twamp(
            archive,
            config=FjordLinkConfig(
                max_sessions=max_sessions,
                max_replies_per_session=max_replies_per_session,
                minimum_overlap_s=minimum_overlap_s,
            ),
        )
        pairing = audit_fjordlink_concurrency(
            sessions,
            minimum_overlap_s=minimum_overlap_s,
        )
    except Exception as exc:
        coverage = pd.DataFrame.from_records(
            [
                {
                    "source_family": "FjordLink",
                    "campaign_or_site": "kiel_fjord_2024_2025",
                    "operator_id": pd.NA,
                    "panel_kind": "rtt",
                    "pair_id": pd.NA,
                    "records_loaded": 0,
                    "first_time_utc": pd.NaT,
                    "last_time_utc": pd.NaT,
                    "evidence_tier": EVIDENCE_TIER,
                    "observation_design": "operator_specific_sessions_pairing_not_established",
                    "ingestion_status": f"error:{type(exc).__name__}:{exc}",
                    "source_file": str(archive),
                }
            ]
        )
        return coverage, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    coverage_rows: list[dict[str, object]] = []
    qos_rows: list[dict[str, object]] = []
    for operator, group in sessions.groupby("operator_id", dropna=False):
        operator_replies = replies.loc[replies["operator_id"].eq(operator)]
        consistent = group.loc[group["packet_count_consistent"]]
        sent = int(consistent["sent_packets"].sum())
        received = int(consistent["received_packets"].sum())
        reply_fraction = received / sent if sent else np.nan
        coverage_rows.append(
            {
                "source_family": "FjordLink",
                "campaign_or_site": "kiel_fjord_2024_2025",
                "operator_id": operator,
                "panel_kind": "rtt",
                "pair_id": pd.NA,
                "records_loaded": len(operator_replies),
                "first_time_utc": pd.NaT,
                "last_time_utc": pd.NaT,
                "first_source_clock_time": group["first_source_clock_time"].min(),
                "last_source_clock_time": group["last_source_clock_time"].max(),
                "timestamp_basis": "logger_clock_timezone_unspecified",
                "session_count": len(group),
                "evidence_tier": EVIDENCE_TIER,
                "observation_design": "operator_specific_sessions_concurrency_audited_not_assumed",
                "ingestion_status": "ok",
                "source_file": str(archive),
                "source_id": _source_ids(group),
            }
        )
        qos_rows.append(
            {
                "source_family": "FjordLink",
                "campaign_or_site": "kiel_fjord_2024_2025",
                "operator_id": operator,
                "panel_kind": "rtt",
                "pair_id": pd.NA,
                "metric": "rtt_ms",
                "records": len(operator_replies),
                "observed_fraction": reply_fraction,
                "positive_throughput_fraction": np.nan,
                **_quantiles(operator_replies["rtt_ms"]),
                "evidence_tier": EVIDENCE_TIER,
                "availability_definition": "twamp_reply_fraction_for_count_consistent_sessions",
                "source_id": _source_ids(group),
            }
        )
    return (
        pd.DataFrame.from_records(coverage_rows),
        pd.DataFrame.from_records(qos_rows),
        sessions,
        pairing,
    )


def summarize_lens(
    root: Path,
    *,
    max_probe_records: int,
    max_probe_members: int,
    max_telemetry_records: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coverage: list[dict[str, object]] = []
    qos: list[dict[str, object]] = []
    alignment: list[dict[str, object]] = []
    for site_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        site = site_dir.name.split("_2025", 1)[0]
        ping_files = sorted(site_dir.glob("*ping*.tar.zst"))
        irtt_files = sorted(site_dir.glob("irtt*.json.gz"))
        dish_files = sorted(site_dir.glob("*dish_status*.tar.zst"))
        probe_path = irtt_files[0] if irtt_files else (ping_files[0] if ping_files else None)
        probe_kind = "irtt" if irtt_files else "ping"
        if probe_path is None or not dish_files:
            coverage.append(
                {
                    "source_family": "LENS",
                    "campaign_or_site": site,
                    "operator_id": "starlink",
                    "panel_kind": "probe_plus_dish_status",
                    "pair_id": pd.NA,
                    "records_loaded": 0,
                    "first_time_utc": pd.NaT,
                    "last_time_utc": pd.NaT,
                    "evidence_tier": EVIDENCE_TIER,
                    "observation_design": "single_operator_observational_trace",
                    "ingestion_status": "error:missing_probe_or_dish_archive",
                    "source_file": str(site_dir),
                }
            )
            continue
        try:
            telemetry = load_lens_dish_status(
                dish_files[0],
                max_records=max_telemetry_records,
                campaign_id=f"lens_{site}",
            )
            telemetry_start = (
                telemetry["telemetry_time_utc"].min()
                if len(telemetry)
                else None
            )
            telemetry_end = (
                telemetry["telemetry_time_utc"].max() + pd.Timedelta(minutes=1)
                if len(telemetry)
                else None
            )
            probes = (
                load_lens_irtt(
                    probe_path,
                    max_records=max_probe_records,
                    campaign_id=f"lens_{site}",
                    node_id=site,
                )
                if probe_kind == "irtt"
                else load_lens_ping(
                    probe_path,
                    max_records=max_probe_records,
                    max_members=max_probe_members,
                    start=telemetry_start,
                    end=telemetry_end,
                    campaign_id=f"lens_{site}",
                    node_id=site,
                )
            )
            aligned = align_lens_irtt_to_dish(
                probes, telemetry, max_telemetry_age_s=75.0
            )
            missing_sequences = int(
                pd.to_numeric(probes["sequence_gap_count"], errors="coerce")
                .fillna(0)
                .sum()
            )
            reply_denominator = len(probes) + missing_sequences
            coverage.append(
                {
                    "source_family": "LENS",
                    "campaign_or_site": site,
                    "operator_id": "starlink",
                    "panel_kind": f"{probe_kind}_plus_dish_status",
                    "pair_id": pd.NA,
                    "records_loaded": len(probes),
                    "first_time_utc": probes["source_time_utc"].min(),
                    "last_time_utc": probes["source_time_utc"].max(),
                    "evidence_tier": EVIDENCE_TIER,
                    "observation_design": "single_operator_observational_trace",
                    "ingestion_status": "ok",
                    "source_file": str(probe_path),
                    "source_id": _source_ids(probes),
                }
            )
            qos.append(
                {
                    "source_family": "LENS",
                    "campaign_or_site": site,
                    "operator_id": "starlink",
                    "panel_kind": probe_kind,
                    "pair_id": pd.NA,
                    "metric": "rtt_ms",
                    "records": len(probes),
                    "observed_fraction": float(probes["rtt_ms"].notna().mean()),
                    "positive_throughput_fraction": np.nan,
                    **_quantiles(probes["rtt_ms"]),
                    "evidence_tier": EVIDENCE_TIER,
                    "availability_definition": "not_estimated_from_reply_only_rows",
                    "reply_sequence_coverage_fraction": (
                        len(probes) / reply_denominator if reply_denominator else np.nan
                    ),
                    "inferred_missing_reply_sequences": missing_sequences,
                    "source_id": _source_ids(probes),
                }
            )
            for metric in (
                "dish_pop_ping_latency_ms",
                "dish_pop_ping_drop_rate",
                "dish_downlink_throughput_bps",
            ):
                qos.append(
                    {
                        "source_family": "LENS",
                        "campaign_or_site": site,
                        "operator_id": "starlink",
                        "panel_kind": "dish_status",
                        "pair_id": pd.NA,
                        "metric": metric,
                        "records": len(telemetry),
                        "observed_fraction": float(telemetry[metric].notna().mean()),
                        "positive_throughput_fraction": np.nan,
                        **_quantiles(telemetry[metric]),
                        "evidence_tier": EVIDENCE_TIER,
                        "availability_definition": "not_a_path_availability_estimate",
                        "source_id": _source_ids(telemetry),
                    }
                )
            alignment.append(
                {
                    "source_family": "LENS",
                    "campaign_or_site": site,
                    "probe_kind": probe_kind,
                    "probe_records": len(probes),
                    "dish_records": len(telemetry),
                    "aligned_records": int(aligned["telemetry_aligned"].sum()),
                    "alignment_fraction": float(aligned["telemetry_aligned"].mean()),
                    "max_telemetry_age_s": 75.0,
                    "alignment_method": "causal_backward_asof",
                    "evidence_tier": EVIDENCE_TIER,
                    "source_id": _source_ids(aligned),
                }
            )
        except Exception as exc:
            coverage.append(
                {
                    "source_family": "LENS",
                    "campaign_or_site": site,
                    "operator_id": "starlink",
                    "panel_kind": f"{probe_kind}_plus_dish_status",
                    "pair_id": pd.NA,
                    "records_loaded": 0,
                    "first_time_utc": pd.NaT,
                    "last_time_utc": pd.NaT,
                    "evidence_tier": EVIDENCE_TIER,
                    "observation_design": "single_operator_observational_trace",
                    "ingestion_status": f"error:{type(exc).__name__}:{exc}",
                    "source_file": str(probe_path),
                }
            )
    return (
        pd.DataFrame.from_records(coverage),
        pd.DataFrame.from_records(qos),
        pd.DataFrame.from_records(alignment),
    )


def summarize_sat_in_view(
    path: Path, *, max_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = load_sat_in_view_matches(path, max_rows=max_rows)
    coverage = pd.DataFrame.from_records(
        [
            {
                "source_family": "SatInView",
                "campaign_or_site": "public_site_id_withheld",
                "operator_id": "starlink",
                "panel_kind": "post_processed_trajectory_match_plus_ping",
                "pair_id": pd.NA,
                "records_loaded": len(matches),
                "first_time_utc": matches["source_time_utc"].min(),
                "last_time_utc": matches["source_time_utc"].max(),
                "evidence_tier": EVIDENCE_TIER,
                "observation_design": "observational_artifact_with_inferred_not_direct_serving_identity",
                "ingestion_status": "ok",
                "source_file": str(path),
                "source_id": _source_ids(matches),
            }
        ]
    )
    summary = pd.DataFrame.from_records(
        [
            {
                "records": len(matches),
                "unique_inferred_satellite_labels": matches[
                    "inferred_satellite_id"
                ].nunique(),
                "consecutive_inferred_identity_changes": int(
                    matches["inferred_handover_event"].sum()
                ),
                "identity_changes_unresolved_across_gaps": int(
                    matches["identity_change_unresolved_across_gap"].sum()
                ),
                "ping_rtt_median_ms": float(matches["ping_rtt_ms"].median()),
                "ping_rtt_p90_ms": float(matches["ping_rtt_ms"].quantile(0.90)),
                "matched_elevation_median_deg": float(
                    matches["matched_elevation_deg"].median()
                ),
                "matched_distance_median_km": float(
                    matches["matched_slant_distance_km"].median()
                ),
                "identity_evidence": matches["identity_evidence"].iloc[0],
                "handover_evidence": matches["handover_evidence"].iloc[0],
                "site_id_status": "withheld_not_available_in_public_csv",
                "evidence_tier": EVIDENCE_TIER,
                "source_id": matches["source_id"].iloc[0],
            }
        ]
    )
    return coverage, summary


def summarize_nexasphere_athens(
    extracted_directory: Path, *, max_records_per_table: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bundle = load_nexasphere_athens(
        extracted_directory, max_records_per_table=max_records_per_table
    )
    tables = {
        "latency": bundle.latency,
        "antenna": bundle.antenna,
        "downlink": bundle.downlink,
        "uplink": bundle.uplink,
    }
    coverage: list[dict[str, object]] = []
    for panel, frame in tables.items():
        coverage.append(
            {
                "source_family": "NEXASPHERE",
                "campaign_or_site": "athens_202605",
                "operator_id": "starlink",
                "panel_kind": panel,
                "pair_id": pd.NA,
                "records_loaded": len(frame),
                "first_time_utc": frame["source_time_utc"].min(),
                "last_time_utc": frame["source_time_utc"].max(),
                "evidence_tier": EVIDENCE_TIER,
                "observation_design": "single_operator_private_analysis_observational_trace",
                "ingestion_status": "ok",
                "source_file": frame["source_file"].iloc[0],
                "source_id": _source_ids(frame),
                "usage_restriction": USAGE_RESTRICTION,
            }
        )

    qos: list[dict[str, object]] = []
    metric_definitions = (
        ("latency", "latency_avg_ms", bundle.latency),
        ("latency", "packet_loss_pct", bundle.latency),
        ("antenna", "dish_pop_ping_latency_ms", bundle.antenna),
        ("antenna", "dish_pop_drop_rate_fraction", bundle.antenna),
        ("downlink", "iperf3_dl_mbps", bundle.downlink),
        ("downlink", "speedtest_dl_mbps", bundle.downlink),
        ("uplink", "iperf3_ul_mbps", bundle.uplink),
        ("uplink", "speedtest_ul_mbps", bundle.uplink),
    )
    for panel, metric, frame in metric_definitions:
        groupings = frame.groupby("target") if panel == "latency" else [(None, frame)]
        for target, group in groupings:
            values = pd.to_numeric(group[metric], errors="coerce")
            qos.append(
                {
                    "source_family": "NEXASPHERE",
                    "campaign_or_site": "athens_202605",
                    "operator_id": "starlink",
                    "panel_kind": panel,
                    "pair_id": pd.NA,
                    "target": target,
                    "metric": metric,
                    "records": len(group),
                    "observed_fraction": float(values.notna().mean()),
                    "positive_throughput_fraction": float(values.gt(0).mean())
                    if metric.endswith("_mbps")
                    else np.nan,
                    **_quantiles(values),
                    "evidence_tier": EVIDENCE_TIER,
                    "availability_definition": "not_an_operational_availability_estimate",
                    "usage_restriction": USAGE_RESTRICTION,
                    "source_id": _source_ids(group),
                }
            )
    summary = pd.DataFrame.from_records(
        [
            {
                "metadata_start_utc": bundle.metadata["start_timestamp_utc"],
                "metadata_end_utc": bundle.metadata["end_timestamp_utc"],
                "latency_rows": len(bundle.latency),
                "antenna_rows": len(bundle.antenna),
                "downlink_rows": len(bundle.downlink),
                "uplink_rows": len(bundle.uplink),
                "latency_targets": ";".join(
                    sorted(bundle.latency["target"].astype(str).unique())
                ),
                "connected_state_fraction": float(
                    bundle.antenna["state"].eq("CONNECTED").mean()
                ),
                "median_pop_ping_ms": float(
                    bundle.antenna["dish_pop_ping_latency_ms"].median()
                ),
                "median_speedtest_downlink_mbps": float(
                    bundle.downlink["speedtest_dl_mbps"].median()
                ),
                "median_speedtest_uplink_mbps": float(
                    bundle.uplink["speedtest_ul_mbps"].median()
                ),
                "evidence_tier": EVIDENCE_TIER,
                "usage_restriction": USAGE_RESTRICTION,
                "source_id": _source_ids(
                    pd.concat(tables.values(), ignore_index=True)
                ),
            }
        ]
    )
    return (
        pd.DataFrame.from_records(coverage),
        pd.DataFrame.from_records(qos),
        summary,
    )


def _make_replay_examples(
    frame: pd.DataFrame,
    *,
    payload_bytes: int,
    future_rows: int,
    assumed_interval_s: float,
) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["path_id", "source_run_id", "source_time_utc"], kind="mergesort"
    ).copy()
    records: list[pd.DataFrame] = []
    for (_, _), group in ordered.groupby(["path_id", "source_run_id"], sort=False):
        group = group.sort_values("source_time_utc", kind="mergesort").copy()
        throughput = pd.to_numeric(group["throughput_mbps"], errors="coerce")
        group["rolling_mean_mbps"] = throughput.shift(1).rolling(3, min_periods=1).mean()
        group["current_throughput_mbps"] = throughput
        group["current_positive"] = throughput.gt(0).astype(float)
        future_capacity = sum(
            throughput.shift(-offset).fillna(0.0)
            for offset in range(1, future_rows + 1)
        )
        future_bytes = future_capacity * 1_000_000.0 / 8.0 * assumed_interval_s
        group["replay_complete"] = future_bytes.ge(payload_bytes).astype(int)
        group["future_rows_available"] = len(group) - np.arange(len(group)) - 1
        group = group.loc[group["future_rows_available"].ge(future_rows)]
        records.append(group)
    result = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    if not result.empty:
        result["path_is_starlink"] = result["path_id"].eq("starlink").astype(float)
        result["rolling_mean_mbps"] = result["rolling_mean_mbps"].fillna(
            result["current_throughput_mbps"]
        )
    return result


def _score(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return (
        float(np.mean((probability - y) ** 2)),
        float(log_loss(y, probability, labels=[0, 1])),
    )


def run_cross_campaign_proxy_benchmark(
    panels: dict[str, pd.DataFrame],
    *,
    train_campaign: str,
    test_campaign: str,
    payload_bytes: int,
    future_rows: int,
    assumed_interval_s: float,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_campaign not in panels or test_campaign not in panels:
        return pd.DataFrame(), pd.DataFrame()
    train = _make_replay_examples(
        panels[train_campaign],
        payload_bytes=payload_bytes,
        future_rows=future_rows,
        assumed_interval_s=assumed_interval_s,
    )
    test = _make_replay_examples(
        panels[test_campaign],
        payload_bytes=payload_bytes,
        future_rows=future_rows,
        assumed_interval_s=assumed_interval_s,
    )
    features = [
        "current_throughput_mbps",
        "rolling_mean_mbps",
        "current_positive",
        "path_is_starlink",
    ]
    if train.empty or test.empty or train["replay_complete"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000, random_state=seed),
    )
    model.fit(train[features], train["replay_complete"])
    prevalence = float(train["replay_complete"].mean())
    predictions = {
        "train_prevalence": np.full(len(test), prevalence),
        "current_state_persistence": test["current_positive"].to_numpy(float),
        "logistic_lag_model": model.predict_proba(test[features])[:, 1],
    }
    train_source_ids = _source_ids(train)
    test_source_ids = _source_ids(test)
    y = test["replay_complete"].to_numpy(int)
    metrics: list[dict[str, object]] = []
    for name, probability in predictions.items():
        brier, cross_entropy = _score(y, probability)
        metrics.append(
            {
                "model": name,
                "train_campaign": train_campaign,
                "test_campaign": test_campaign,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_run_clusters": test["source_run_id"].nunique(),
                "payload_bytes": payload_bytes,
                "future_rows": future_rows,
                "assumed_interval_s": assumed_interval_s,
                "brier_score": brier,
                "binary_cross_entropy": cross_entropy,
                "completion_prevalence_test": float(y.mean()),
                "evidence_tier": REPLAY_TIER,
                "endpoint": "fixed_payload_future_capacity_trace_replay_proxy",
                "train_source_ids": train_source_ids,
                "test_source_ids": test_source_ids,
            }
        )

    rng = np.random.default_rng(seed)
    cluster_labels = test["source_run_id"].astype(str).to_numpy()
    clusters = np.unique(cluster_labels)
    comparison_pairs = (
        ("current_state_persistence", "train_prevalence"),
        ("logistic_lag_model", "train_prevalence"),
        ("logistic_lag_model", "current_state_persistence"),
    )
    draws: list[dict[str, object]] = []
    if len(clusters) >= 2 and bootstrap_repetitions > 0:
        for repetition in range(bootstrap_repetitions):
            selected = rng.choice(clusters, size=len(clusters), replace=True)
            indices = np.concatenate(
                [np.flatnonzero(cluster_labels == cluster) for cluster in selected]
            )
            repetition_scores: dict[str, float] = {}
            for name, probability in predictions.items():
                brier, _ = _score(y[indices], probability[indices])
                repetition_scores[name] = brier
            for name, reference in comparison_pairs:
                draws.append(
                    {
                        "repetition": repetition,
                        "model": name,
                        "reference": reference,
                        "delta_brier": repetition_scores[name]
                        - repetition_scores[reference],
                    }
                )
    intervals: list[dict[str, object]] = []
    draw_frame = pd.DataFrame.from_records(draws)
    point_brier = {
        row["model"]: float(row["brier_score"])
        for row in metrics
    }
    for name, reference in comparison_pairs:
        values = (
            draw_frame.loc[
                draw_frame["model"].eq(name)
                & draw_frame["reference"].eq(reference),
                "delta_brier",
            ]
            if not draw_frame.empty
            else pd.Series(dtype=float)
        )
        intervals.append(
            {
                "model": name,
                "reference": reference,
                "point_delta_brier": point_brier[name] - point_brier[reference],
                "cluster_unit": "source_run_id",
                "test_clusters": len(clusters),
                "bootstrap_repetitions": bootstrap_repetitions,
                "delta_brier_ci_low": float(values.quantile(0.025))
                if len(values)
                else np.nan,
                "delta_brier_ci_high": float(values.quantile(0.975))
                if len(values)
                else np.nan,
                "uncertainty_status": "cluster_bootstrap"
                if len(clusters) >= 2
                else "not_estimable_fewer_than_two_runs",
                "evidence_tier": REPLAY_TIER,
                "train_source_ids": train_source_ids,
                "test_source_ids": test_source_ids,
            }
        )
    return pd.DataFrame.from_records(metrics), pd.DataFrame.from_records(intervals)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default=str(ROOT / "data" / "external")
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results" / "public_trace_validation")
    )
    parser.add_argument("--nuwins-max-rows-per-file", type=int, default=100_000)
    parser.add_argument("--lens-max-probe-records", type=int, default=100_000)
    parser.add_argument("--lens-max-probe-members", type=int, default=1)
    parser.add_argument("--lens-max-telemetry-records", type=int, default=20_000)
    parser.add_argument("--fjordlink-max-sessions", type=int, default=200)
    parser.add_argument("--fjordlink-max-replies-per-session", type=int, default=100_000)
    parser.add_argument("--fjordlink-minimum-overlap-s", type=float, default=1.0)
    parser.add_argument("--sat-in-view-max-rows", type=int, default=100_000)
    parser.add_argument("--nexasphere-max-records-per-table", type=int, default=100_000)
    parser.add_argument("--payload-bytes", type=int, default=5_000_000)
    parser.add_argument("--future-rows", type=int, default=10)
    parser.add_argument("--assumed-interval-s", type=float, default=0.5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--skip-proxy-benchmark", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(
        args.nuwins_max_rows_per_file,
        args.lens_max_probe_records,
        args.lens_max_probe_members,
        args.lens_max_telemetry_records,
        args.fjordlink_max_sessions,
        args.fjordlink_max_replies_per_session,
        args.sat_in_view_max_rows,
        args.nexasphere_max_records_per_table,
    ) <= 0:
        raise ValueError("all source bounds must be positive")
    if args.fjordlink_minimum_overlap_s < 0:
        raise ValueError("fjordlink minimum overlap must be nonnegative")
    data_root = Path(args.data_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    nuwins_coverage, nuwins_qos, panels = summarize_nuwins(
        data_root / "nuwins_sigmetrics26",
        max_rows_per_file=args.nuwins_max_rows_per_file,
    )
    fjordlink_coverage, fjordlink_qos, fjordlink_sessions, fjordlink_pairing = (
        summarize_fjordlink(
            data_root / "fjordlink",
            max_sessions=args.fjordlink_max_sessions,
            max_replies_per_session=args.fjordlink_max_replies_per_session,
            minimum_overlap_s=args.fjordlink_minimum_overlap_s,
        )
    )
    lens_coverage, lens_qos, lens_alignment = summarize_lens(
        data_root / "lens" / "public_measurement_subset",
        max_probe_records=args.lens_max_probe_records,
        max_probe_members=args.lens_max_probe_members,
        max_telemetry_records=args.lens_max_telemetry_records,
    )
    sat_coverage, sat_summary = summarize_sat_in_view(
        data_root / "sat_in_view" / "paper_results" / "matched_satellite_data.csv",
        max_rows=args.sat_in_view_max_rows,
    )
    nexa_coverage, nexa_qos, nexa_summary = summarize_nexasphere_athens(
        data_root / "nexasphere_athens_2026" / "extracted",
        max_records_per_table=args.nexasphere_max_records_per_table,
    )
    coverage = pd.concat(
        [
            nuwins_coverage,
            fjordlink_coverage,
            lens_coverage,
            sat_coverage,
            nexa_coverage,
        ],
        ignore_index=True,
    )
    qos = pd.concat(
        [nuwins_qos, fjordlink_qos, lens_qos, nexa_qos],
        ignore_index=True,
    )
    coverage.to_csv(output / "campaign_site_operator_coverage.csv", index=False)
    qos.to_csv(output / "path_qos_distributions.csv", index=False)
    lens_alignment.to_csv(output / "lens_telemetry_alignment.csv", index=False)
    fjordlink_sessions.to_csv(output / "fjordlink_session_summary.csv", index=False)
    fjordlink_pairing.to_csv(output / "fjordlink_pairing_audit.csv", index=False)
    sat_summary.to_csv(output / "sat_in_view_inference_summary.csv", index=False)
    nexa_summary.to_csv(output / "nexasphere_athens_private_summary.csv", index=False)

    benchmark = pd.DataFrame()
    intervals = pd.DataFrame()
    if not args.skip_proxy_benchmark:
        benchmark, intervals = run_cross_campaign_proxy_benchmark(
            panels,
            train_campaign="alaska_202406",
            test_campaign="hawaii_202408",
            payload_bytes=args.payload_bytes,
            future_rows=args.future_rows,
            assumed_interval_s=args.assumed_interval_s,
            bootstrap_repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
    benchmark.to_csv(output / "cross_campaign_replay_proxy_metrics.csv", index=False)
    intervals.to_csv(output / "cross_campaign_replay_proxy_uncertainty.csv", index=False)

    manifest = {
        "arguments": vars(args),
        "row_counts": {
            "coverage": len(coverage),
            "qos": len(qos),
            "lens_alignment": len(lens_alignment),
            "fjordlink_sessions": len(fjordlink_sessions),
            "fjordlink_pairing_audit": len(fjordlink_pairing),
            "fjordlink_pairing_eligible": int(
                fjordlink_pairing.get("pairing_eligible", pd.Series(dtype=bool))
                .fillna(False)
                .sum()
            ),
            "sat_in_view_summary": len(sat_summary),
            "nexasphere_athens_summary": len(nexa_summary),
            "proxy_metrics": len(benchmark),
            "proxy_uncertainty": len(intervals),
        },
        "evidence_labels": {
            "coverage_and_qos": EVIDENCE_TIER,
            "fixed_payload_proxy": REPLAY_TIER,
        },
        "claim_limits": [
            "The public traces are observational and are not randomized policy trials.",
            "LENS samples one Starlink path per site and does not provide a concurrent 5G alternative.",
            "FjordLink is counted as paired multi-access evidence only for source-clock-overlapping sessions; a combined-dataset label alone is insufficient.",
            "FjordLink timestamps retain the source logger clock because the public release does not declare a timezone.",
            "NUWiNS mainland per-operator files have no exact cross-operator key; coverage rows do not imply alignment.",
            "The benchmark endpoint is replayed future capacity for a fixed payload, not observed EO task completion.",
            "No adapter supplies serving-satellite, beam, gateway, RF, scheduler action, or operational EO acknowledgment fields.",
            "SatInView satellite labels are post-processed trajectory matches; its identity-change counts are not direct handover observations.",
            "The NEXASPHERE license is unresolved: only private aggregate analysis is permitted here and no raw/row-level output may be redistributed.",
            "A better point estimate is not evidence of consistent superiority unless the paired cluster interval excludes zero on the declared loss scale.",
        ],
    }
    (output / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["row_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
