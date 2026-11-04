"""Fail-closed adapter for the public FjordLink Starlink/5G campaign.

The FjordLink release contains operator-specific TWAMP sessions in a ZIP
archive.  A dataset-level description of both Starlink and 5G is not evidence
that two particular sessions were observed concurrently.  This module keeps
the source logger clock (whose timezone is not declared in the release),
summarises each session, and admits a cross-path pair only when the measured
session intervals overlap by a predeclared minimum duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from zipfile import ZipFile

import pandas as pd


_LINE_TIME = re.compile(r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
_SENT = re.compile(r"Sent to .*\[sseq=(?P<sseq>\d+)\]")
_REPLY = re.compile(
    r"Reply from .*\[rseq=(?P<rseq>\d+) sseq=(?P<sseq>\d+) "
    r"rtt=(?P<rtt>[0-9.]+)ms outbound=(?P<outbound>[0-9.]+)ms "
    r"inbound=(?P<inbound>[0-9.]+)ms\]"
)


@dataclass(frozen=True)
class FjordLinkConfig:
    """Bounded, predeclared parsing and pairing settings."""

    max_sessions: int = 200
    max_replies_per_session: int = 100_000
    minimum_overlap_s: float = 1.0

    def validate(self) -> None:
        if self.max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        if self.max_replies_per_session <= 0:
            raise ValueError("max_replies_per_session must be positive")
        if self.minimum_overlap_s < 0:
            raise ValueError("minimum_overlap_s must be nonnegative")


def _archive_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_session(name: str) -> tuple[str, str]:
    lowered = name.lower()
    mobility = "stationary" if "stationary" in lowered else "mobile_or_unspecified"
    if lowered.startswith("starlink"):
        return "starlink", mobility
    if lowered.startswith("cau-4329"):
        return "vodafone", mobility
    if lowered.startswith("cau-0c"):
        return "telekom", mobility
    if lowered.startswith("router5gaddix"):
        return "cellular_unspecified", mobility
    return "unclassified", mobility


def load_fjordlink_twamp(
    archive_path: str | Path,
    *,
    config: FjordLinkConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return session summaries and bounded reply observations.

    Timestamps remain on the source logger clock because the public release
    does not declare a timezone.  They are valid for overlap checks within the
    same acquisition system but are not relabelled as UTC.
    """

    cfg = config or FjordLinkConfig()
    cfg.validate()
    archive = Path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(archive)

    archive_hash = _archive_sha256(archive)
    session_rows: list[dict[str, object]] = []
    reply_rows: list[dict[str, object]] = []

    with ZipFile(archive) as bundle:
        members = sorted(
            (
                info
                for info in bundle.infolist()
                if info.filename.endswith("/twamp.log")
                and not info.filename.startswith("__MACOSX/")
            ),
            key=lambda info: info.filename,
        )[: cfg.max_sessions]
        for info in members:
            parts = Path(info.filename).parts
            session_name = parts[-2]
            operator, mobility = _classify_session(session_name)
            session_id = f"fjordlink:{session_name}"
            source_id = (
                f"fjordlink:sha256:{archive_hash}:member_crc32:{info.CRC:08x}:"
                f"{info.filename}"
            )
            first_time: pd.Timestamp | None = None
            last_time: pd.Timestamp | None = None
            sent_packets = 0
            received_packets = 0
            retained_replies = 0

            with bundle.open(info) as raw:
                for encoded in raw:
                    line = encoded.decode("utf-8", errors="replace")
                    time_match = _LINE_TIME.match(line)
                    if time_match is None:
                        continue
                    timestamp = pd.to_datetime(
                        time_match.group("time"),
                        format="%Y-%m-%d %H:%M:%S,%f",
                        errors="raise",
                    )
                    first_time = timestamp if first_time is None else min(first_time, timestamp)
                    last_time = timestamp if last_time is None else max(last_time, timestamp)
                    if _SENT.search(line):
                        sent_packets += 1
                    reply_match = _REPLY.search(line)
                    if reply_match is None:
                        continue
                    received_packets += 1
                    if retained_replies >= cfg.max_replies_per_session:
                        continue
                    reply_rows.append(
                        {
                            "campaign_id": "fjordlink_kiel_fjord_2024_2025",
                            "session_id": session_id,
                            "operator_id": operator,
                            "mobility_class": mobility,
                            "source_clock_time": timestamp,
                            "timestamp_basis": "logger_clock_timezone_unspecified",
                            "rtt_ms": float(reply_match.group("rtt")),
                            "outbound_ms": float(reply_match.group("outbound")),
                            "inbound_ms": float(reply_match.group("inbound")),
                            "source_id": source_id,
                            "evidence_tier": "field_observational",
                        }
                    )
                    retained_replies += 1

            count_consistent = sent_packets > 0 and received_packets <= sent_packets
            loss_fraction = (
                1.0 - received_packets / sent_packets
                if count_consistent
                else float("nan")
            )
            session_rows.append(
                {
                    "campaign_id": "fjordlink_kiel_fjord_2024_2025",
                    "session_id": session_id,
                    "session_name": session_name,
                    "operator_id": operator,
                    "access_family": "leo" if operator == "starlink" else "terrestrial"
                    if operator != "unclassified"
                    else "unknown",
                    "mobility_class": mobility,
                    "first_source_clock_time": first_time,
                    "last_source_clock_time": last_time,
                    "timestamp_basis": "logger_clock_timezone_unspecified",
                    "sent_packets": sent_packets,
                    "received_packets": received_packets,
                    "reply_minus_sent_count": received_packets - sent_packets,
                    "packet_count_consistent": count_consistent,
                    "retained_replies": retained_replies,
                    "reply_retention_truncated": received_packets > retained_replies,
                    "loss_fraction": loss_fraction,
                    "source_member": info.filename,
                    "source_id": source_id,
                    "evidence_tier": "field_observational",
                }
            )

    return pd.DataFrame.from_records(session_rows), pd.DataFrame.from_records(reply_rows)


def audit_fjordlink_concurrency(
    sessions: pd.DataFrame,
    *,
    minimum_overlap_s: float = 1.0,
) -> pd.DataFrame:
    """Audit whether Starlink and terrestrial sessions truly overlap.

    The result retains the nearest terrestrial session for every Starlink
    session, including rejected pairs.  This prevents a combined-dataset label
    from being silently converted into synchronized counterfactual evidence.
    """

    if minimum_overlap_s < 0:
        raise ValueError("minimum_overlap_s must be nonnegative")
    required = {
        "session_id",
        "operator_id",
        "first_source_clock_time",
        "last_source_clock_time",
        "source_id",
    }
    missing = required.difference(sessions.columns)
    if missing:
        raise ValueError(f"sessions missing required columns: {sorted(missing)}")

    leo = sessions.loc[sessions["operator_id"].eq("starlink")].copy()
    terrestrial = sessions.loc[
        sessions["operator_id"].isin(["vodafone", "telekom", "cellular_unspecified"])
    ].copy()
    columns = [
        "starlink_session_id",
        "terrestrial_session_id",
        "terrestrial_operator_id",
        "clock_overlap_s",
        "nearest_interval_gap_s",
        "minimum_required_overlap_s",
        "pairing_eligible",
        "exclusion_reason",
        "starlink_source_id",
        "terrestrial_source_id",
    ]
    rows: list[dict[str, object]] = []
    for starlink in leo.itertuples(index=False):
        candidates: list[tuple[float, float, object]] = []
        for cellular in terrestrial.itertuples(index=False):
            if any(
                pd.isna(value)
                for value in (
                    starlink.first_source_clock_time,
                    starlink.last_source_clock_time,
                    cellular.first_source_clock_time,
                    cellular.last_source_clock_time,
                )
            ):
                continue
            overlap_s = max(
                0.0,
                (
                    min(starlink.last_source_clock_time, cellular.last_source_clock_time)
                    - max(starlink.first_source_clock_time, cellular.first_source_clock_time)
                ).total_seconds(),
            )
            if overlap_s > 0:
                gap_s = 0.0
            else:
                gap_s = min(
                    abs(
                        (
                            starlink.first_source_clock_time
                            - cellular.last_source_clock_time
                        ).total_seconds()
                    ),
                    abs(
                        (
                            cellular.first_source_clock_time
                            - starlink.last_source_clock_time
                        ).total_seconds()
                    ),
                )
            candidates.append((gap_s, -overlap_s, cellular))
        if not candidates:
            rows.append(
                {
                    "starlink_session_id": starlink.session_id,
                    "terrestrial_session_id": pd.NA,
                    "terrestrial_operator_id": pd.NA,
                    "clock_overlap_s": 0.0,
                    "nearest_interval_gap_s": float("nan"),
                    "minimum_required_overlap_s": minimum_overlap_s,
                    "pairing_eligible": False,
                    "exclusion_reason": "no_terrestrial_session",
                    "starlink_source_id": starlink.source_id,
                    "terrestrial_source_id": pd.NA,
                }
            )
            continue
        gap_s, negative_overlap_s, cellular = min(candidates, key=lambda item: (item[0], item[1]))
        overlap_s = -negative_overlap_s
        eligible = overlap_s >= minimum_overlap_s
        rows.append(
            {
                "starlink_session_id": starlink.session_id,
                "terrestrial_session_id": cellular.session_id,
                "terrestrial_operator_id": cellular.operator_id,
                "clock_overlap_s": overlap_s,
                "nearest_interval_gap_s": gap_s,
                "minimum_required_overlap_s": minimum_overlap_s,
                "pairing_eligible": eligible,
                "exclusion_reason": pd.NA if eligible else "no_synchronized_clock_overlap",
                "starlink_source_id": starlink.source_id,
                "terrestrial_source_id": cellular.source_id,
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns)
