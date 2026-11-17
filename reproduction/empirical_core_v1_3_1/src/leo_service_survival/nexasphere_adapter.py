"""Private-analysis adapter for the NEXASPHERE Athens Starlink release.

The archive contains an April pilot before its declared seven-day run.  Every
table is therefore hard-filtered to the start/end timestamps in ``metadata.json``.
The public record inspected locally does not resolve a redistribution license;
outputs carry an explicit private-analysis/no-raw-redistribution restriction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


USAGE_RESTRICTION = "private_analysis_license_unresolved_no_raw_redistribution"


@dataclass(frozen=True)
class NexaSphereBundle:
    latency: pd.DataFrame
    antenna: pd.DataFrame
    downlink: pd.DataFrame
    uplink: pd.DataFrame
    metadata: dict


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metadata(path: Path) -> tuple[dict, pd.Timestamp, pd.Timestamp]:
    with path.open("rt", encoding="utf-8") as handle:
        metadata = json.load(handle)
    try:
        start = pd.Timestamp(metadata["start_timestamp_utc"])
        end = pd.Timestamp(metadata["end_timestamp_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metadata must declare valid start/end UTC timestamps") from exc
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if start >= end:
        raise ValueError("metadata start_timestamp_utc must precede end_timestamp_utc")
    return metadata, start, end


def _windowed_csv(
    path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_records: int,
) -> pd.DataFrame:
    if max_records <= 0:
        raise ValueError("max_records_per_table must be positive")
    frames: list[pd.DataFrame] = []
    source_offset = 0
    retained = 0
    for chunk in pd.read_csv(path, chunksize=25_000, low_memory=False):
        if "timestamp_utc" not in chunk:
            raise ValueError(f"{path.name} lacks timestamp_utc")
        chunk = chunk.copy()
        chunk["_source_row_number"] = (
            np.arange(len(chunk), dtype=int) + source_offset + 2
        )
        source_offset += len(chunk)
        timestamp = pd.to_datetime(chunk["timestamp_utc"], utc=True, errors="coerce")
        if timestamp.isna().any():
            bad = chunk.loc[timestamp.isna(), "_source_row_number"].tolist()[:5]
            raise ValueError(f"{path.name} has invalid timestamps at rows {bad}")
        keep = timestamp.ge(start) & timestamp.le(end)
        selected = chunk.loc[keep].copy()
        selected["source_time_utc"] = timestamp.loc[keep]
        if retained + len(selected) > max_records:
            selected = selected.iloc[: max_records - retained].copy()
        if len(selected):
            frames.append(selected)
            retained += len(selected)
        if retained >= max_records:
            break
        # Source files are chronological; no later chunk can re-enter the window.
        if len(timestamp) and timestamp.min() > end:
            break
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(result) and not result["source_time_utc"].between(start, end).all():
        raise AssertionError("metadata-window filter failed closed")
    return result


def _provenance(frame: pd.DataFrame, path: Path, table: str) -> pd.DataFrame:
    digest = _file_sha256(path)
    result = frame.copy()
    result["campaign_id"] = "nexasphere_athens_202605"
    result["site_id"] = "athens_greece"
    result["operator_id"] = "starlink"
    result["evidence_tier"] = "field_observational"
    result["usage_restriction"] = USAGE_RESTRICTION
    result["source_dataset"] = "nexasphere_athens_zenodo_20796667"
    result["source_id"] = f"nexasphere:{table}:sha256:{digest}"
    result["source_file"] = str(path)
    result["source_sha256"] = digest
    result["source_row_number"] = result.pop("_source_row_number").astype(int)
    return result


def load_nexasphere_athens(
    extracted_directory: str | Path,
    *,
    max_records_per_table: int = 100_000,
) -> NexaSphereBundle:
    """Load four distinct tables inside the metadata-declared May window."""

    root = Path(extracted_directory).expanduser().resolve()
    metadata_path = root / "metadata.json"
    metadata, start, end = _load_metadata(metadata_path)
    expected_start = pd.Timestamp("2026-05-11T12:06:16Z")
    expected_end = pd.Timestamp("2026-05-18T12:07:03Z")
    if start != expected_start or end != expected_end:
        raise ValueError(
            "NEXASPHERE metadata window differs from the frozen acquisition protocol"
        )

    latency_path = root / "latency.csv"
    antenna_path = root / "antenna_metrics.csv"
    downlink_path = root / "throughput_Down.csv"
    uplink_path = root / "throughput_Up.csv"
    for path in (latency_path, antenna_path, downlink_path, uplink_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    latency_raw = _windowed_csv(
        latency_path, start=start, end=end, max_records=max_records_per_table
    )
    antenna_raw = _windowed_csv(
        antenna_path, start=start, end=end, max_records=max_records_per_table
    )
    downlink_raw = _windowed_csv(
        downlink_path, start=start, end=end, max_records=max_records_per_table
    )
    uplink_raw = _windowed_csv(
        uplink_path, start=start, end=end, max_records=max_records_per_table
    )

    latency = latency_raw.rename(
        columns={
            "min_ms": "latency_min_ms",
            "avg_ms": "latency_avg_ms",
            "max_ms": "latency_max_ms",
            "loss_pct": "packet_loss_pct",
        }
    )[
        [
            "source_time_utc",
            "cycle",
            "target",
            "latency_min_ms",
            "latency_avg_ms",
            "latency_max_ms",
            "packet_loss_pct",
            "_source_row_number",
        ]
    ]
    antenna = antenna_raw.rename(
        columns={
            "dish_dl_bps": "dish_downlink_bps",
            "dish_ul_bps": "dish_uplink_bps",
            "pop_ping_ms": "dish_pop_ping_latency_ms",
            "pop_drop_rate": "dish_pop_drop_rate_fraction",
            "snr": "dish_snr_above_noise_floor",
            "uptime_s": "dish_uptime_s",
        }
    )[
        [
            "source_time_utc",
            "cycle",
            "dish_downlink_bps",
            "dish_uplink_bps",
            "dish_pop_ping_latency_ms",
            "dish_pop_drop_rate_fraction",
            "dish_snr_above_noise_floor",
            "obstruction_pct",
            "obstructed",
            "dish_uptime_s",
            "state",
            "error",
            "_source_row_number",
        ]
    ]
    downlink = downlink_raw[
        [
            "source_time_utc",
            "cycle",
            "iperf3_dl_mbps",
            "iperf3_dl_error",
            "speedtest_dl_mbps",
            "speedtest_ping_ms",
            "iperf3_server_used",
            "speedtest_server",
            "speedtest_error",
            "_source_row_number",
        ]
    ]
    uplink = uplink_raw[
        [
            "source_time_utc",
            "cycle",
            "iperf3_ul_mbps",
            "iperf3_ul_error",
            "speedtest_ul_mbps",
            "iperf3_server_used",
            "speedtest_server",
            "speedtest_error",
            "_source_row_number",
        ]
    ]
    return NexaSphereBundle(
        latency=_provenance(latency, latency_path, "latency"),
        antenna=_provenance(antenna, antenna_path, "antenna"),
        downlink=_provenance(downlink, downlink_path, "downlink"),
        uplink=_provenance(uplink, uplink_path, "uplink"),
        metadata={
            **metadata,
            "metadata_source_sha256": _file_sha256(metadata_path),
            "usage_restriction": USAGE_RESTRICTION,
        },
    )
