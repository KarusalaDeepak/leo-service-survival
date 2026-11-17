"""Bounded, provenance-preserving adapters for the public LENS traces.

The downloaded LENS subset is a field observation of one Starlink terminal.
It contains probe and dish telemetry, not serving-satellite, beam, gateway, RF,
5G, geometry, policy-action, or EO-task evidence.  This module therefore emits
only fields present in, or transparently derived from, those source records.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import signal
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from fnmatch import fnmatch
from hashlib import sha256
from pathlib import Path
from typing import IO, Iterator

import numpy as np
import pandas as pd


def _positive_limit(value: int | None, name: str) -> int | None:
    if value is not None and int(value) <= 0:
        raise ValueError(f"{name} must be positive or None")
    return None if value is None else int(value)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: str | Path, measurement: str) -> dict[str, str]:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    digest = _file_sha256(source_path)
    return {
        "source_dataset": "lens",
        "source_id": f"lens:{measurement}:sha256:{digest}",
        "source_file": str(source_path),
        "source_sha256": digest,
    }


@contextmanager
def _json_text(path: Path) -> Iterator[IO[str]]:
    if path.name.endswith(".gz"):
        with gzip.open(path, mode="rt", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open(mode="rt", encoding="utf-8") as handle:
            yield handle


def _read_json_prefix_before_array(
    path: Path,
    key: str,
    *,
    chunk_size: int = 64 * 1024,
    maximum_prefix_bytes: int = 16 * 1024 * 1024,
) -> dict:
    marker = json.dumps(key)
    buffer = ""
    with _json_text(path) as handle:
        while marker not in buffer:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"JSON key {key!r} was not found")
            buffer += chunk
            if len(buffer.encode("utf-8")) > maximum_prefix_bytes:
                raise ValueError(f"JSON prefix before {key!r} exceeds safety bound")
    prefix = buffer[: buffer.index(marker)].rstrip()
    if prefix.endswith(","):
        prefix = prefix[:-1]
    try:
        value = json.loads(prefix + "}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"unable to parse JSON metadata before {key!r}") from exc
    if not isinstance(value, dict):
        raise ValueError("IRTT JSON root must be an object")
    return value


def _iter_json_array_objects(
    path: Path,
    key: str,
    *,
    chunk_size: int = 64 * 1024,
    maximum_object_chars: int = 16 * 1024 * 1024,
) -> Iterator[dict]:
    """Stream objects from a named top-level JSON array without loading it."""

    marker = json.dumps(key)
    decoder = json.JSONDecoder()
    with _json_text(path) as handle:
        buffer = ""
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"JSON key {key!r} was not found")
            buffer += chunk
            marker_position = buffer.find(marker)
            if marker_position < 0:
                if len(buffer) > maximum_object_chars:
                    raise ValueError(f"JSON prefix before {key!r} exceeds safety bound")
                continue
            array_position = buffer.find("[", marker_position + len(marker))
            while array_position < 0:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(f"JSON key {key!r} is not followed by an array")
                buffer += chunk
                array_position = buffer.find("[", marker_position + len(marker))
            position = array_position + 1
            break

        end_of_stream = False
        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer):
                    break
                buffer = ""
                position = 0
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(f"unterminated JSON array {key!r}")
                buffer = chunk

            if buffer[position] == "]":
                return

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as exc:
                    if end_of_stream:
                        raise ValueError(f"malformed object in JSON array {key!r}") from exc
                    if position:
                        buffer = buffer[position:]
                        position = 0
                    if len(buffer) > maximum_object_chars:
                        raise ValueError("one JSON array object exceeds safety bound")
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        end_of_stream = True
                    else:
                        buffer += chunk
            if not isinstance(value, dict):
                raise ValueError(f"all elements in {key!r} must be JSON objects")
            yield value
            position = end
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def _nested(mapping: dict, *keys: str) -> object | None:
    value: object = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _as_float(value: object, scale: float = 1.0) -> float:
    try:
        return float(value) / scale
    except (TypeError, ValueError):
        return np.nan


def load_lens_irtt(
    path: str | Path,
    *,
    max_records: int | None = 100_000,
    gap_multiplier: float = 5.0,
    campaign_id: str | None = None,
    node_id: str | None = None,
    operator_id: str = "starlink",
) -> pd.DataFrame:
    """Stream a bounded number of LENS IRTT probes into a canonical table.

    ``max_records`` defaults to 100,000 to prevent an accidental 414-MB JSON
    expansion.  Pass ``None`` deliberately for the full file.  Gap flags use
    both sequence discontinuity and an inter-send interval greater than
    ``gap_multiplier`` times the source-declared probe interval.
    """

    max_records = _positive_limit(max_records, "max_records")
    if float(gap_multiplier) <= 1.0:
        raise ValueError("gap_multiplier must be greater than 1")
    path = Path(path).expanduser().resolve()
    metadata = _metadata(path, "irtt")
    filename_node_match = re.match(r"irtt-([^-]+)-", path.name)
    resolved_node = node_id or (
        filename_node_match.group(1) if filename_node_match else "unknown_lens_node"
    )
    resolved_campaign = campaign_id or f"lens_{resolved_node}_irtt"
    header = _read_json_prefix_before_array(path, "round_trips")
    interval_ns = _nested(header, "config", "params", "interval")
    try:
        interval_ns = int(interval_ns)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        interval_ns = 0
    total_records = _nested(header, "stats", "packets_sent")
    try:
        total_records = int(total_records)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        total_records = None

    records: list[dict[str, object]] = []
    previous_seqno: int | None = None
    previous_send_ns: int | None = None
    for record_number, item in enumerate(
        _iter_json_array_objects(path, "round_trips"), start=1
    ):
        if max_records is not None and record_number > max_records:
            break
        seq_value = item.get("seqno")
        try:
            seqno = int(seq_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"IRTT record {record_number} has no integer seqno") from exc
        send_value = _nested(item, "timestamps", "client", "send", "wall")
        try:
            send_ns = int(send_value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            send_ns = None
        receive_value = _nested(item, "timestamps", "client", "receive", "wall")
        try:
            receive_ns = int(receive_value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            receive_ns = None

        raw_loss = item.get("lost", "unknown")
        if raw_loss is False:
            loss_label = "false"
        elif raw_loss is True:
            loss_label = "true_unknown"
        else:
            loss_label = str(raw_loss).strip().lower()
        probe_received = loss_label == "false"
        sequence_gap = (
            max(seqno - previous_seqno - 1, 0) if previous_seqno is not None else 0
        )
        sequence_nonmonotone = bool(
            previous_seqno is not None and seqno <= previous_seqno
        )
        send_interval_ms = (
            (send_ns - previous_send_ns) / 1_000_000.0
            if send_ns is not None and previous_send_ns is not None
            else np.nan
        )
        timestamp_gap = bool(
            interval_ns > 0
            and np.isfinite(send_interval_ms)
            and send_interval_ms > gap_multiplier * interval_ns / 1_000_000.0
        )
        reasons: list[str] = []
        if sequence_gap:
            reasons.append("missing_sequence_numbers")
        if sequence_nonmonotone:
            reasons.append("nonmonotone_sequence")
        if timestamp_gap:
            reasons.append("timestamp_gap")
        if send_ns is None:
            reasons.append("missing_client_send_timestamp")

        records.append(
            {
                "campaign_id": resolved_campaign,
                "operator_id": operator_id,
                "node_id": resolved_node,
                "source_record_number": record_number,
                "source_seqno": seqno,
                "source_time_ns": send_ns,
                "source_time_utc": pd.to_datetime(
                    send_ns, unit="ns", utc=True, errors="coerce"
                ),
                "client_receive_time_ns": receive_ns,
                "loss_label": loss_label,
                "probe_received": probe_received,
                "rtt_ms": _as_float(_nested(item, "delay", "rtt"), 1_000_000.0),
                "uplink_delay_ms": _as_float(
                    _nested(item, "delay", "send"), 1_000_000.0
                ),
                "downlink_delay_ms": _as_float(
                    _nested(item, "delay", "receive"), 1_000_000.0
                ),
                "send_interval_ms": send_interval_ms,
                "expected_interval_ms": interval_ns / 1_000_000.0
                if interval_ns > 0
                else np.nan,
                "sequence_gap_count": sequence_gap,
                "trace_gap_detected": bool(reasons),
                "trace_gap_reason": ";".join(reasons) if reasons else pd.NA,
                "source_total_records": total_records,
                "source_subset_limit_records": max_records,
                "source_subset_truncated": bool(
                    max_records is not None
                    and total_records is not None
                    and max_records < total_records
                ),
                "evidence_tier": "field_observational",
                **metadata,
            }
        )
        previous_seqno = seqno
        if send_ns is not None:
            previous_send_ns = send_ns
    return pd.DataFrame.from_records(records)


@contextmanager
def _dish_csv_text(path: Path) -> Iterator[IO[str]]:
    if path.suffix.lower() == ".csv":
        with path.open("rt", encoding="utf-8", newline="") as handle:
            yield handle
        return
    if not path.name.endswith(".tar.zst"):
        raise ValueError("dish status input must be an annotated CSV or .tar.zst")

    listing = subprocess.run(
        ["tar", "--zstd", "-tf", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    members = [line for line in listing.stdout.splitlines() if line.endswith(".csv")]
    if len(members) != 1:
        raise ValueError(
            f"dish archive must contain exactly one CSV member, found {members}"
        )
    process = subprocess.Popen(
        ["tar", "--zstd", "-xOf", str(path), members[0]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    text_handle = io.TextIOWrapper(process.stdout, encoding="utf-8", newline="")
    try:
        yield text_handle
    finally:
        text_handle.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.returncode not in (0, -signal.SIGTERM, -signal.SIGPIPE):
            raise RuntimeError(
                f"unable to stream dish archive: {stderr.decode('utf-8', 'replace')}"
            )


def _bool_or_na(value: object) -> bool | object:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return pd.NA


def _integer_or_na(value: object) -> int | object:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return pd.NA


def load_lens_dish_status(
    path: str | Path,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    max_records: int | None = 10_000,
    campaign_id: str | None = None,
    operator_id: str = "starlink",
) -> pd.DataFrame:
    """Load a bounded UTC interval from LENS Flux-annotated dish telemetry."""

    max_records = _positive_limit(max_records, "max_records")
    start_time = pd.to_datetime(start, utc=True) if start is not None else None
    end_time = pd.to_datetime(end, utc=True) if end is not None else None
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("start must precede end")
    path = Path(path).expanduser().resolve()
    metadata = _metadata(path, "dish_status")

    records: list[dict[str, object]] = []
    with _dish_csv_text(path) as handle:
        reader = csv.reader(handle)
        header: list[str] | None = None
        for physical_line, fields in enumerate(reader, start=1):
            if header is None:
                if "_time" in fields and "node" in fields:
                    header = [name if name else "_flux_empty" for name in fields]
                continue
            if not fields or all(not value for value in fields):
                continue
            if len(fields) != len(header):
                raise ValueError(
                    f"dish CSV line {physical_line} has {len(fields)} fields; expected {len(header)}"
                )
            row = dict(zip(header, fields))
            try:
                timestamp = pd.Timestamp(row.get("_time"))
                timestamp = (
                    timestamp.tz_localize("UTC")
                    if timestamp.tzinfo is None
                    else timestamp.tz_convert("UTC")
                )
            except (TypeError, ValueError):
                raise ValueError(f"dish CSV line {physical_line} has invalid _time")
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp >= end_time:
                continue
            if max_records is not None and len(records) >= max_records:
                break
            records.append(
                {
                    "campaign_id": campaign_id
                    or f"lens_{str(row.get('node', 'unknown')).replace('starlink-', '')}_dish_status",
                    "operator_id": operator_id,
                    "node_id": row.get("node"),
                    "telemetry_time_utc": timestamp,
                    "dish_pop_ping_latency_ms": _as_float(
                        row.get("dish_pop_ping_latency_ms")
                    ),
                    "dish_pop_ping_drop_rate": _as_float(
                        row.get("dish_pop_ping_drop_rate")
                    ),
                    "dish_downlink_throughput_bps": _as_float(
                        row.get("dish_downlink_throughput_bps")
                    ),
                    "dish_uplink_throughput_bps": _as_float(
                        row.get("dish_uplink_throughput_bps")
                    ),
                    "dish_gps_sats": _integer_or_na(row.get("dish_gps_sats")),
                    "dish_gps_valid": _bool_or_na(row.get("dish_gps_valid")),
                    "dish_currently_obstructed": _bool_or_na(
                        row.get("dish_obstruction_currently_obstructed")
                    ),
                    "dish_fraction_obstructed": _as_float(
                        row.get("dish_obstruction_fraction_obstructed")
                    ),
                    "dish_snr_above_noise_floor": _bool_or_na(
                        row.get("dish_is_snr_above_noise_floor")
                    ),
                    "dish_snr_persistently_low": _bool_or_na(
                        row.get("dish_is_snr_persistently_low")
                    ),
                    "dish_actuator_state": row.get("dish_actuator_state"),
                    "dish_attitude_estimation_state": row.get(
                        "dish_attitude_estimation_state"
                    ),
                    "dish_software_version": row.get("dish_software_version"),
                    "dish_uptime_s": _integer_or_na(row.get("dish_uptime")),
                    "source_record_number": physical_line,
                    "evidence_tier": "field_observational",
                    **metadata,
                }
            )
    if header is None:
        raise ValueError("dish CSV header containing _time and node was not found")
    return pd.DataFrame.from_records(records)


def _list_tar_members(path: Path) -> list[str]:
    try:
        listing = subprocess.run(
            ["tar", "--zstd", "-tf", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"unable to list {path}; the archive may be incomplete: {exc.stderr.strip()}"
        ) from exc
    return [line for line in listing.stdout.splitlines() if line and not line.endswith("/")]


def _stop_process(process: subprocess.Popen[bytes]) -> tuple[int, bytes]:
    if process.poll() is None:
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    error = process.stderr.read() if process.stderr is not None else b""
    return int(process.returncode), error


@contextmanager
def _ping_member_text(
    archive: Path, member: str, *, nested_archive: bool
) -> Iterator[IO[str]]:
    """Stream one plain or one nested-hour ping member from a Zstd tar."""

    if archive.suffix.lower() == ".txt":
        with archive.open("rt", encoding="utf-8", newline="") as handle:
            yield handle
        return

    outer = subprocess.Popen(
        ["tar", "--zstd", "-xOf", str(archive), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert outer.stdout is not None
    inner: subprocess.Popen[bytes] | None = None
    if nested_archive:
        inner = subprocess.Popen(
            ["tar", "--zstd", "-xOf", "-"],
            stdin=outer.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        outer.stdout.close()
        assert inner.stdout is not None
        binary_stream = inner.stdout
    else:
        binary_stream = outer.stdout
    text_handle = io.TextIOWrapper(binary_stream, encoding="utf-8", newline="")
    try:
        yield text_handle
    finally:
        text_handle.close()
        inner_status = (0, b"")
        if inner is not None:
            inner_status = _stop_process(inner)
        outer_status = _stop_process(outer)
        # A bounded reader intentionally closes the pipe before the member's
        # aggregate footer, so bsdtar may return 1 after its downstream closes.
        allowed = (0, 1, -signal.SIGTERM, -signal.SIGPIPE)
        if inner_status[0] not in allowed:
            raise RuntimeError(
                "unable to stream nested ping member: "
                + inner_status[1].decode("utf-8", "replace")
            )
        if outer_status[0] not in allowed:
            raise RuntimeError(
                "unable to stream ping member: "
                + outer_status[1].decode("utf-8", "replace")
            )


_PING_REPLY = re.compile(
    r"^\[(?P<epoch>\d+(?:\.\d+)?)\]\s+"
    r"(?P<bytes>\d+)\s+bytes from\s+(?P<target>\S+):\s+"
    r"icmp_seq=(?P<seq>\d+)\s+ttl=(?P<ttl>\d+)\s+"
    r"time=<?(?P<rtt>\d+(?:\.\d+)?)\s+ms\s*$"
)


def _epoch_decimal_to_ns(value: str) -> int:
    seconds, separator, fraction = value.partition(".")
    fraction_ns = int((fraction + "000000000")[:9]) if separator else 0
    return int(seconds) * 1_000_000_000 + fraction_ns


def load_lens_ping(
    path: str | Path,
    *,
    max_records: int | None = 100_000,
    max_members: int | None = 1,
    member_glob: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    campaign_id: str | None = None,
    node_id: str | None = None,
    operator_id: str = "starlink",
    gap_multiplier: float = 5.0,
) -> pd.DataFrame:
    """Stream replies from a one-hour or weekly LENS ping archive.

    Weekly portal bundles contain hourly ``.txt.tar.zst`` members nested in an
    outer ``.tar.zst``.  This function streams both layers and defaults to one
    lexicographically selected hour and 100,000 replies.  Missing replies are
    not fabricated; sequence-number gaps are retained in
    ``sequence_gap_count`` and break downstream episodes.
    """

    max_records = _positive_limit(max_records, "max_records")
    max_members = _positive_limit(max_members, "max_members")
    start_time = pd.to_datetime(start, utc=True) if start is not None else None
    end_time = pd.to_datetime(end, utc=True) if end is not None else None
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("start must precede end")
    if gap_multiplier <= 1.0:
        raise ValueError("gap_multiplier must be greater than 1")
    path = Path(path).expanduser().resolve()
    metadata = _metadata(path, "ping")
    if path.suffix.lower() == ".txt":
        plain_members = [path.name]
        nested_members: list[str] = []
    else:
        all_members = _list_tar_members(path)
        plain_members = sorted(
            member for member in all_members if member.endswith(".txt")
        )
        nested_members = sorted(
            member for member in all_members if member.endswith(".txt.tar.zst")
        )
    if plain_members and nested_members:
        raise ValueError("ping archive mixes plain and nested hourly members")
    nested = bool(nested_members)
    members = nested_members if nested else plain_members
    if member_glob is not None:
        members = [member for member in members if fnmatch(member, member_glob)]
    if start_time is not None or end_time is not None:
        filtered_members: list[str] = []
        for member in members:
            match = re.search(
                r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})",
                Path(member).name,
            )
            if match is None:
                # A nonstandard filename is retained and filtered record-wise.
                filtered_members.append(member)
                continue
            nominal_start = pd.to_datetime(
                match.group(1), format="%Y-%m-%d-%H-%M-%S", utc=True
            )
            nominal_end = nominal_start + pd.Timedelta(hours=1)
            if start_time is not None and nominal_end <= start_time:
                continue
            if end_time is not None and nominal_start >= end_time:
                continue
            filtered_members.append(member)
        members = filtered_members
    if not members:
        raise ValueError("no ping text members matched the requested archive/filter")
    if max_members is not None:
        members = members[:max_members]

    records: list[dict[str, object]] = []
    for member_number, member in enumerate(members, start=1):
        basename = Path(member).name
        node_match = re.match(r"ping-([^-]+)-", basename)
        resolved_node = node_id or (
            node_match.group(1) if node_match else "unknown_lens_node"
        )
        interval_match = re.search(r"-(\d+)ms-", basename)
        expected_interval_ms = (
            float(interval_match.group(1)) if interval_match else np.nan
        )
        member_source_id = (
            f"{metadata['source_id']}:member:{sha256(member.encode('utf-8')).hexdigest()[:16]}"
        )
        previous_seq: int | None = None
        previous_time_ns: int | None = None
        reply_number = 0
        with _ping_member_text(path, member, nested_archive=nested) as handle:
            for physical_line, line in enumerate(handle, start=1):
                match = _PING_REPLY.match(line.strip())
                if match is None:
                    continue
                if max_records is not None and len(records) >= max_records:
                    break
                reply_number += 1
                seqno = int(match.group("seq"))
                source_time_ns = _epoch_decimal_to_ns(match.group("epoch"))
                if start_time is not None and source_time_ns < int(start_time.value):
                    continue
                if end_time is not None and source_time_ns >= int(end_time.value):
                    continue
                sequence_gap = (
                    max(seqno - previous_seq - 1, 0) if previous_seq is not None else 0
                )
                sequence_nonmonotone = bool(
                    previous_seq is not None and seqno <= previous_seq
                )
                interval_ms = (
                    (source_time_ns - previous_time_ns) / 1_000_000.0
                    if previous_time_ns is not None
                    else np.nan
                )
                timestamp_gap = bool(
                    np.isfinite(expected_interval_ms)
                    and np.isfinite(interval_ms)
                    and interval_ms > gap_multiplier * expected_interval_ms
                )
                reasons: list[str] = []
                if sequence_gap:
                    reasons.append("missing_ping_reply_sequences")
                if sequence_nonmonotone:
                    reasons.append("nonmonotone_ping_sequence")
                if timestamp_gap:
                    reasons.append("reply_timestamp_gap")
                records.append(
                    {
                        "campaign_id": campaign_id or f"lens_{resolved_node}_ping",
                        "operator_id": operator_id,
                        "node_id": resolved_node,
                        "source_member": member,
                        "source_member_number": member_number,
                        "source_record_number": reply_number,
                        "source_physical_line": physical_line,
                        "source_seqno": seqno,
                        "source_time_ns": source_time_ns,
                        "probe_received": True,
                        "loss_label": "reply_received",
                        "rtt_ms": float(match.group("rtt")),
                        "reply_bytes": int(match.group("bytes")),
                        "reply_ttl": int(match.group("ttl")),
                        "ping_target": match.group("target"),
                        "send_interval_ms": interval_ms,
                        "expected_interval_ms": expected_interval_ms,
                        "sequence_gap_count": sequence_gap,
                        "trace_gap_detected": bool(reasons),
                        "trace_gap_reason": ";".join(reasons) if reasons else pd.NA,
                        "source_subset_limit_records": max_records,
                        "source_subset_limit_members": max_members,
                        "evidence_tier": "field_observational",
                        **metadata,
                        "source_id": member_source_id,
                    }
                )
                previous_seq = seqno
                previous_time_ns = source_time_ns
        if max_records is not None and len(records) >= max_records:
            break
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        result["source_time_utc"] = pd.to_datetime(
            result["source_time_ns"], unit="ns", utc=True, errors="coerce"
        )
    return result


def align_lens_irtt_to_dish(
    probes: pd.DataFrame,
    telemetry: pd.DataFrame,
    *,
    max_telemetry_age_s: float = 75.0,
) -> pd.DataFrame:
    """Causally align each probe to the most recent bounded-age dish sample.

    Unmatched probes are retained with ``telemetry_aligned=False``.  No future
    telemetry is used, which avoids leaking post-probe state into decisions.
    """

    if max_telemetry_age_s < 0:
        raise ValueError("max_telemetry_age_s must be non-negative")
    required_probe = {"source_time_utc", "source_id", "source_seqno"}
    required_telemetry = {"telemetry_time_utc", "source_id"}
    missing_probe = sorted(required_probe.difference(probes.columns))
    missing_telemetry = sorted(required_telemetry.difference(telemetry.columns))
    if missing_probe or missing_telemetry:
        raise ValueError(
            f"alignment inputs missing probe={missing_probe}, telemetry={missing_telemetry}"
        )
    if probes.empty:
        return probes.copy()

    left = probes.copy()
    left["_input_order"] = np.arange(len(left), dtype=int)
    left["source_time_utc"] = pd.to_datetime(
        left["source_time_utc"], utc=True, errors="coerce"
    )
    right = telemetry.copy()
    right["telemetry_time_utc"] = pd.to_datetime(
        right["telemetry_time_utc"], utc=True, errors="coerce"
    )
    rename = {
        column: f"telemetry_{column}"
        for column in (
            "source_id",
            "source_file",
            "source_sha256",
            "source_record_number",
            "evidence_tier",
            "source_dataset",
            "campaign_id",
            "operator_id",
            "node_id",
        )
        if column in right.columns
    }
    right = right.rename(columns=rename)
    valid_left = left.loc[left["source_time_utc"].notna()].sort_values(
        "source_time_utc", kind="mergesort"
    )
    valid_right = right.loc[right["telemetry_time_utc"].notna()].sort_values(
        "telemetry_time_utc", kind="mergesort"
    )
    if valid_right.empty:
        merged = valid_left.copy()
        merged["telemetry_time_utc"] = pd.NaT
        merged["telemetry_source_id"] = pd.NA
    else:
        merged = pd.merge_asof(
            valid_left,
            valid_right,
            left_on="source_time_utc",
            right_on="telemetry_time_utc",
            direction="backward",
            tolerance=pd.Timedelta(seconds=float(max_telemetry_age_s)),
        )
    invalid_left = left.loc[left["source_time_utc"].isna()].copy()
    for column in merged.columns:
        if column not in invalid_left.columns:
            invalid_left[column] = pd.NA
    combined = pd.concat([merged, invalid_left[merged.columns]], ignore_index=True)
    combined = combined.sort_values("_input_order", kind="mergesort").drop(
        columns="_input_order"
    )
    # An all-unmatched timezone-aware merge can be coerced to object dtype by
    # pandas 2.0; restore explicit UTC datetimes before age arithmetic.
    combined["source_time_utc"] = pd.to_datetime(
        combined["source_time_utc"], utc=True, errors="coerce"
    )
    combined["telemetry_time_utc"] = pd.to_datetime(
        combined["telemetry_time_utc"], utc=True, errors="coerce"
    )
    combined["telemetry_aligned"] = combined["telemetry_time_utc"].notna()
    combined["telemetry_age_ms"] = (
        combined["source_time_utc"] - combined["telemetry_time_utc"]
    ).dt.total_seconds() * 1000.0
    combined["telemetry_alignment_method"] = "causal_backward_asof"
    combined["telemetry_max_age_s"] = float(max_telemetry_age_s)
    combined["telemetry_alignment_reason"] = np.where(
        combined["source_time_utc"].isna(),
        "missing_probe_timestamp",
        np.where(
            combined["telemetry_aligned"],
            pd.NA,
            "no_prior_telemetry_within_tolerance",
        ),
    )
    combined["aligned_source_ids"] = np.where(
        combined["telemetry_aligned"],
        combined["source_id"].astype(str)
        + ";"
        + combined["telemetry_source_id"].astype(str),
        combined["source_id"],
    )
    return combined.reset_index(drop=True)


def build_lens_connectivity_episodes(
    aligned_probes: pd.DataFrame,
    *,
    maximum_contiguous_gap_ms: float = 50.0,
) -> pd.DataFrame:
    """Collapse probes into reachable/lost run-length episodes.

    Episodes break on a received/lost state transition, a sequence gap, or an
    excessive timestamp gap.  They represent observed probe connectivity, not
    satellite visibility windows or operational EO-task completion.
    """

    required = {
        "source_seqno",
        "source_time_utc",
        "probe_received",
        "rtt_ms",
        "source_id",
        "evidence_tier",
    }
    missing = sorted(required.difference(aligned_probes.columns))
    if missing:
        raise ValueError(f"episode input is missing columns: {missing}")
    if maximum_contiguous_gap_ms <= 0:
        raise ValueError("maximum_contiguous_gap_ms must be positive")
    if aligned_probes.empty:
        return pd.DataFrame()

    frame = aligned_probes.copy().sort_values(
        ["source_time_utc", "source_seqno"], kind="mergesort"
    )
    frame["source_time_utc"] = pd.to_datetime(
        frame["source_time_utc"], utc=True, errors="coerce"
    )
    prior_time = frame["source_time_utc"].shift()
    delta_ms = (frame["source_time_utc"] - prior_time).dt.total_seconds() * 1000.0
    prior_seq = pd.to_numeric(frame["source_seqno"], errors="coerce").shift()
    sequence_break = (
        pd.to_numeric(frame["source_seqno"], errors="coerce") - prior_seq
    ).ne(1)
    state_break = frame["probe_received"].astype(bool).ne(
        frame["probe_received"].astype(bool).shift()
    )
    time_break = delta_ms.gt(float(maximum_contiguous_gap_ms)) | delta_ms.lt(0)
    frame["_episode"] = (state_break | sequence_break | time_break).cumsum()
    frame["_preceded_by_gap"] = sequence_break | time_break

    records: list[dict[str, object]] = []
    for episode_number, (_, group) in enumerate(frame.groupby("_episode"), start=1):
        received = bool(group["probe_received"].iloc[0])
        rtt = pd.to_numeric(group["rtt_ms"], errors="coerce").dropna()
        telemetry_coverage = (
            float(group["telemetry_aligned"].mean())
            if "telemetry_aligned" in group
            else 0.0
        )
        telemetry_ids = (
            sorted(
                str(value)
                for value in group.get("telemetry_source_id", pd.Series(dtype=object))
                .dropna()
                .unique()
            )
            if "telemetry_source_id" in group
            else []
        )
        source_id = str(group["source_id"].iloc[0])
        records.append(
            {
                "campaign_id": group.get(
                    "campaign_id", pd.Series(["lens_victoria_2025_11_14_19z"])
                ).iloc[0],
                "episode_id": sha256(
                    f"{source_id}|{int(group['source_seqno'].iloc[0])}|{int(group['source_seqno'].iloc[-1])}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:24],
                "connectivity_state": "reachable" if received else "probe_loss",
                "episode_start_utc": group["source_time_utc"].iloc[0],
                "episode_end_utc": group["source_time_utc"].iloc[-1],
                "observed_span_ms": (
                    group["source_time_utc"].iloc[-1]
                    - group["source_time_utc"].iloc[0]
                ).total_seconds()
                * 1000.0,
                "probe_count": len(group),
                "received_probe_count": int(group["probe_received"].sum()),
                "lost_probe_count": int((~group["probe_received"].astype(bool)).sum()),
                "rtt_median_ms": float(rtt.median()) if len(rtt) else np.nan,
                "rtt_p95_ms": float(rtt.quantile(0.95)) if len(rtt) else np.nan,
                "first_source_seqno": int(group["source_seqno"].iloc[0]),
                "last_source_seqno": int(group["source_seqno"].iloc[-1]),
                "preceded_by_trace_gap": bool(group["_preceded_by_gap"].iloc[0]),
                "telemetry_coverage_fraction": telemetry_coverage,
                "source_id": source_id,
                "telemetry_source_ids": ";".join(telemetry_ids)
                if telemetry_ids
                else pd.NA,
                "episode_derivation": "run_length_probe_received_break_on_sequence_or_time_gap",
                "evidence_tier": "field_observational",
            }
        )
    return pd.DataFrame.from_records(records)
