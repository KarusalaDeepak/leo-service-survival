"""Dataset construction with explicit event and censoring semantics.

TLE records are intentionally absent from the empirical label builders.  They
describe geometry, not service failures.  A measured service event is the first
task-relevant QoS violation observed after a usable interval begins.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Stable endpoint-resolution provenance.  ``censor_reason`` intentionally has
# a value for events as well as censored records so every episode explains why
# observation stopped.  Split embargoes and purges are not endpoint censoring;
# those are reported separately in ``DataFrame.attrs`` by the split builders.
CENSOR_REASON_QUALIFYING_EVENT = "qualifying_event"
CENSOR_REASON_ADMINISTRATIVE_HORIZON = "administrative_horizon"
CENSOR_REASON_RECORDING_ENDED_GOOD = "recording_ended_good"
CENSOR_REASON_UNRESOLVED_BAD_ONSET = (
    "unresolved_bad_onset_missing_confirmation"
)
CENSOR_REASON_INTERNAL_MEASUREMENT_GAP = "internal_measurement_gap"

CENSOR_REASONS = (
    CENSOR_REASON_QUALIFYING_EVENT,
    CENSOR_REASON_ADMINISTRATIVE_HORIZON,
    CENSOR_REASON_RECORDING_ENDED_GOOD,
    CENSOR_REASON_UNRESOLVED_BAD_ONSET,
    CENSOR_REASON_INTERNAL_MEASUREMENT_GAP,
)


@dataclass(frozen=True)
class EpisodeDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    bin_seconds: int
    dataset_name: str


WETLINKS_FEATURES = (
    "latency_mean_ms",
    "latency_std_ms",
    "latency_max_ms",
    "packet_loss_pct",
    "download_mbps",
    "upload_mbps",
    "temperature_c",
    "humidity_pct",
    "rain_mm",
    "wind_mps",
    "pressure_hpa",
    "latency_roll3",
    "loss_roll3",
    "rain_latency_interaction",
    "hour_sin",
    "hour_cos",
)

COMMECT_FEATURES = (
    "latency_mean_ms",
    "latency_std_ms",
    "latency_max_ms",
    "observed_replies",
    "observation_age_ms",
    "rsrp_mean_dbm",
    "rsrq_mean_db",
    "snr_mean_db",
    "is_leo",
    "latency_roll3",
    "reply_roll3",
    "hour_sin",
    "hour_cos",
)

LENS_FEATURES = (
    "latency_mean_ms",
    "burst_indicator",
    "reply_pressure_score",
    "latency_roll3",
    "hour_sin",
    "hour_cos",
)


def _time_features(frame: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    out = frame.copy()
    timestamp = pd.to_datetime(out[timestamp_column], errors="coerce", utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    return out


def _cause_from_row(row: pd.Series, latency_limit_ms: float, loss_limit_pct: float) -> str:
    latency_bad = bool(row["latency_mean_ms"] > latency_limit_ms)
    loss_bad = bool(row["packet_loss_pct"] > loss_limit_pct)
    if latency_bad and loss_bad:
        return "joint"
    if latency_bad:
        return "latency"
    if loss_bad:
        return "loss"
    return "unknown"


def _persistent_cause(causes: Iterable[str]) -> str:
    """Return the QoS limit that persists across every confirmation bin.

    The endpoint is defined per limit, not merely as two consecutive rows for
    which *some* limit is violated.  Thus ``latency`` followed by ``loss`` is
    not a sustained event, while ``latency`` followed by ``joint`` is a
    sustained latency event.  ``joint`` is reserved for the case in which both
    limits persist throughout the confirmation window.
    """

    cause_sets = {
        "latency": {"latency"},
        "loss": {"loss"},
        "joint": {"latency", "loss"},
    }
    sequence = [cause_sets.get(str(cause), set()) for cause in causes]
    if not sequence:
        return "unknown"
    persistent = set.intersection(*sequence)
    if persistent == {"latency", "loss"}:
        return "joint"
    if persistent == {"latency"}:
        return "latency"
    if persistent == {"loss"}:
        return "loss"
    return "unknown"


def _episodes_from_regular_series(
    series: pd.DataFrame,
    *,
    site: str,
    timestamp_column: str,
    observed_column: str,
    usable_column: str,
    cause_column: str,
    feature_columns: Iterable[str],
    bin_seconds: int,
    sustained_bad_bins: int,
    max_gap_bins: int,
) -> list[dict]:
    """Construct runs under the frozen sustained-violation endpoint.

    An event begins at the first bin of ``sustained_bad_bins`` consecutive,
    contiguous, observed bins that all violate the *same* QoS limit.  The
    event time is the onset bin; ``event_confirmation_time`` records the last
    bin needed to confirm persistence.  If observation stops before a
    candidate onset can be confirmed, the episode is censored at that onset,
    because its event status is unknown there.  Episodes resume only after the
    confirmation window, which prevents one observation from belonging to two
    labelled episodes.
    """

    feature_columns = tuple(feature_columns)
    series = series.sort_values(timestamp_column).reset_index(drop=True)
    observed = series[observed_column].fillna(False).to_numpy(bool)
    usable = series[usable_column].fillna(False).to_numpy(bool)
    times = pd.to_datetime(series[timestamp_column], utc=True)
    episodes: list[dict] = []
    n = len(series)
    i = 0
    episode_index = 0
    while i < n:
        if not observed[i] or not usable[i]:
            i += 1
            continue
        start = i
        event = 0
        cause = "censored"
        event_confirmation_index: int | None = None
        censor_index: int | None = None
        censor_reason: str | None = None
        label_resolution_time: pd.Timestamp | None = None
        j = i + 1
        while j < n:
            gap_seconds = (times.iloc[j] - times.iloc[j - 1]).total_seconds()
            if gap_seconds > bin_seconds * max_gap_bins or not observed[j]:
                censor_index = j
                censor_reason = CENSOR_REASON_INTERNAL_MEASUREMENT_GAP
                # The gap or explicitly unobserved row is the observation that
                # resolves the run as censored.  Keep this distinct from the
                # censor boundary itself, which can be an earlier possible
                # event onset under the persistence rule below.
                label_resolution_time = times.iloc[j]
                break
            if not usable[j]:
                stop = j + sustained_bad_bins
                if stop > n:
                    # The possible event would begin now, but the record ends
                    # before persistence can be confirmed.  Record the nominal
                    # final confirmation time so split audits include the full
                    # label look-ahead even though that row is unavailable.
                    censor_index = j
                    censor_reason = CENSOR_REASON_UNRESOLVED_BAD_ONSET
                    label_resolution_time = times.iloc[j] + pd.Timedelta(
                        seconds=bin_seconds * (sustained_bad_bins - 1)
                    )
                    break
                confirmation_observed = True
                confirmation_failure_index: int | None = None
                for k in range(j, stop):
                    if not observed[k]:
                        confirmation_observed = False
                        confirmation_failure_index = k
                        break
                    if k > j:
                        confirmation_gap = (
                            times.iloc[k] - times.iloc[k - 1]
                        ).total_seconds()
                        if confirmation_gap > bin_seconds * max_gap_bins:
                            confirmation_observed = False
                            confirmation_failure_index = k
                            break
                if not confirmation_observed:
                    censor_index = j
                    censor_reason = CENSOR_REASON_UNRESOLVED_BAD_ONSET
                    if confirmation_failure_index is None:
                        raise AssertionError("confirmation failure index was not recorded")
                    label_resolution_time = times.iloc[confirmation_failure_index]
                    break
                persistent_cause = _persistent_cause(
                    series.iloc[j:stop][cause_column].astype(str).tolist()
                )
                if persistent_cause != "unknown":
                    event = 1
                    cause = persistent_cause
                    censor_reason = CENSOR_REASON_QUALIFYING_EVENT
                    event_confirmation_index = stop - 1
                    label_resolution_time = times.iloc[event_confirmation_index]
                    break
            j += 1
        if not event and censor_index is None:
            # Boundary immediately after the last fully observed bin.
            censor_index = n
            censor_reason = CENSOR_REASON_RECORDING_ENDED_GOOD
        if censor_reason is None:
            raise AssertionError("endpoint resolution reason was not recorded")
        boundary_index = j if event else int(censor_index)
        duration_bins = max(1, boundary_index - start)
        start_row = series.iloc[start]
        boundary_time = (
            times.iloc[boundary_index]
            if boundary_index < n
            else times.iloc[-1] + pd.Timedelta(seconds=bin_seconds)
        )
        if label_resolution_time is None:
            # A run ending after a fully usable bin is resolved when the next
            # expected observation boundary is found to be outside the record.
            label_resolution_time = boundary_time
        record = {
            "episode_id": f"{site}:{episode_index}",
            "site": site,
            "start_time": times.iloc[start],
            "duration_bins": int(duration_bins),
            "event": int(event),
            "cause": cause,
            "censor_reason": censor_reason,
            "event_onset_time": times.iloc[j] if event else pd.NaT,
            "event_confirmation_time": (
                times.iloc[event_confirmation_index]
                if event_confirmation_index is not None
                else pd.NaT
            ),
            "censoring_time": pd.NaT if event else boundary_time,
            "label_resolution_time": label_resolution_time,
            "bin_seconds": int(bin_seconds),
            "sustained_bad_bins": int(sustained_bad_bins),
            "label_semantics_version": "same-limit-onset-v1",
            # WetLinks is one longitudinal acquisition per site.  Calendar
            # weeks are retained separately as serial-dependence blocks; they
            # must not be misrepresented as independent field campaigns.
            "campaign_id": site,
            "block_id": f"{site}:{times.iloc[start].strftime('%G-W%V')}",
            "calendar_block": times.iloc[start].strftime("%G-W%V"),
        }
        for column in feature_columns:
            record[column] = start_row.get(column, np.nan)
        episodes.append(record)
        episode_index += 1
        if event:
            i = j + sustained_bad_bins
        else:
            i = max(int(censor_index) + 1, start + 1)
    return episodes


def build_wetlinks_episodes(
    input_dir: str | Path,
    *,
    bin_minutes: int = 5,
    latency_limit_ms: float = 80.0,
    loss_limit_pct: float = 1.0,
    sustained_bad_bins: int = 2,
) -> EpisodeDataset:
    """Build measured Starlink service episodes from WetLinks.

    The primary event is sustained scheduling-level unusability, not proven RF
    link loss.  A violation requires two consecutive five-minute bins above the
    RTT or loss threshold.  Missing measurement gaps administratively censor a
    run.  Weather is used as a covariate; it is not treated as a labelled cause.
    """

    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob("analysis_data_*.csv"))
    if len(paths) < 2:
        raise FileNotFoundError(f"expected two WetLinks analysis tables in {input_dir}")
    all_episodes: list[dict] = []
    for path in paths:
        usecols = [
            "site_name",
            "timestamp_start",
            "ping_avg",
            "ping_worst",
            "ping_stddev",
            "ping_packet_loss",
            "download",
            "upload",
            "temp",
            "humidity",
            "rain",
            "windspeed",
            "barom",
        ]
        raw = pd.read_csv(path, usecols=usecols)
        raw["timestamp"] = pd.to_datetime(raw["timestamp_start"], errors="coerce", utc=True)
        numeric = [column for column in usecols if column not in {"site_name", "timestamp_start"}]
        raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
        # The released weather tables contain finite sentinel values (for
        # example -9999).  Convert physically impossible values to missing
        # before aggregation so they cannot become predictive artefacts.
        valid_ranges = {
            "temp": (-80.0, 60.0),
            "humidity": (0.0, 100.0),
            "rain": (0.0, 500.0),
            "windspeed": (0.0, 100.0),
            "barom": (800.0, 1100.0),
        }
        for column, (lower, upper) in valid_ranges.items():
            raw.loc[~raw[column].between(lower, upper), column] = np.nan
        site = str(raw["site_name"].dropna().iloc[0])
        indexed = raw.set_index("timestamp")
        regular = indexed.resample(f"{bin_minutes}min").agg(
            latency_mean_ms=("ping_avg", "mean"),
            latency_std_ms=("ping_stddev", "mean"),
            latency_max_ms=("ping_worst", "max"),
            packet_loss_pct=("ping_packet_loss", "mean"),
            download_bps=("download", "mean"),
            upload_bps=("upload", "mean"),
            temperature_c=("temp", "mean"),
            humidity_pct=("humidity", "mean"),
            rain_mm=("rain", "mean"),
            wind_mps=("windspeed", "mean"),
            pressure_hpa=("barom", "mean"),
        ).reset_index()
        regular["download_mbps"] = regular["download_bps"] / 1e6
        regular["upload_mbps"] = regular["upload_bps"] / 1e6
        regular["observed"] = regular["latency_mean_ms"].notna() & regular["packet_loss_pct"].notna()
        regular["usable"] = (
            regular["observed"]
            & regular["latency_mean_ms"].le(latency_limit_ms)
            & regular["packet_loss_pct"].le(loss_limit_pct)
        )
        regular["cause"] = regular.apply(
            _cause_from_row,
            axis=1,
            latency_limit_ms=latency_limit_ms,
            loss_limit_pct=loss_limit_pct,
        )
        regular["latency_roll3"] = regular["latency_mean_ms"].rolling(3, min_periods=1).mean()
        regular["loss_roll3"] = regular["packet_loss_pct"].rolling(3, min_periods=1).mean()
        regular["rain_latency_interaction"] = (
            regular["rain_mm"].fillna(0.0) * regular["latency_mean_ms"]
        )
        regular = _time_features(regular, "timestamp")
        all_episodes.extend(
            _episodes_from_regular_series(
                regular,
                site=site,
                timestamp_column="timestamp",
                observed_column="observed",
                usable_column="usable",
                cause_column="cause",
                feature_columns=WETLINKS_FEATURES,
                bin_seconds=bin_minutes * 60,
                sustained_bad_bins=sustained_bad_bins,
                max_gap_bins=2,
            )
        )
    frame = pd.DataFrame(all_episodes).sort_values(["start_time", "site"]).reset_index(drop=True)
    return EpisodeDataset(frame, WETLINKS_FEATURES, bin_minutes * 60, "WetLinks")


def build_palo_alto_starlink_episodes(
    input_dir: str | Path,
    *,
    bin_minutes: int = 5,
    latency_limit_ms: float = 80.0,
    loss_limit_pct: float = 1.0,
    sustained_bad_bins: int = 2,
) -> EpisodeDataset:
    """Build an external test set from the Palo Alto Starlink ping record.

    The public record contains eleven days of ten-second ICMP burst summaries
    (RTT and loss) but no throughput, weather, serving-satellite, or stated
    timestamp time zone.  Unavailable covariates and cyclical local-time
    features are intentionally left missing and are imputed only by the
    training-fitted encoder.  This is cross-geography service-level validation,
    not physical-link or cause attribution.
    """

    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob("ping_metrics_2024-05-*.*"))
    if len(paths) != 12:
        raise FileNotFoundError(
            f"expected 12 Palo Alto daily ping tables in {input_dir}; found {len(paths)}"
        )
    daily = []
    expected = {
        "Timestamp",
        "PacketTransmit",
        "PacketReceive",
        "PacketLossRate",
        "RTTAvg",
        "RTTMax",
        "RTTMdev",
    }
    for path in paths:
        separator = "\t" if path.suffix.lower() == ".tsv" else ";"
        frame = pd.read_csv(path, sep=separator)
        missing = expected.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        daily.append(frame)
    raw = pd.concat(daily, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce", utc=True)
    numeric = [
        "PacketTransmit",
        "PacketReceive",
        "PacketLossRate",
        "RTTAvg",
        "RTTMax",
        "RTTMdev",
    ]
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
    indexed = raw.set_index("timestamp")
    regular = indexed.resample(f"{bin_minutes}min").agg(
        latency_mean_ms=("RTTAvg", "mean"),
        latency_std_ms=("RTTMdev", "mean"),
        latency_max_ms=("RTTMax", "max"),
        packet_loss_pct=("PacketLossRate", "mean"),
        transmitted_packets=("PacketTransmit", "sum"),
        received_packets=("PacketReceive", "sum"),
    ).reset_index()
    regular["observed"] = (
        regular["transmitted_packets"].gt(0)
        & regular["packet_loss_pct"].notna()
    )
    regular["usable"] = (
        regular["observed"]
        & regular["latency_mean_ms"].notna()
        & regular["latency_mean_ms"].le(latency_limit_ms)
        & regular["packet_loss_pct"].le(loss_limit_pct)
    )
    regular["cause"] = regular.apply(
        _cause_from_row,
        axis=1,
        latency_limit_ms=latency_limit_ms,
        loss_limit_pct=loss_limit_pct,
    )
    regular["latency_roll3"] = regular["latency_mean_ms"].rolling(3, min_periods=1).mean()
    regular["loss_roll3"] = regular["packet_loss_pct"].rolling(3, min_periods=1).mean()
    # These variables are unavailable in the released external record.  Do not
    # synthesize them: the fitted WetLinks encoder supplies training medians.
    for column in (
        "download_mbps",
        "upload_mbps",
        "temperature_c",
        "humidity_pct",
        "rain_mm",
        "wind_mps",
        "pressure_hpa",
        "rain_latency_interaction",
        "hour_sin",
        "hour_cos",
    ):
        regular[column] = np.nan
    episodes = _episodes_from_regular_series(
        regular,
        site="palo_alto_us",
        timestamp_column="timestamp",
        observed_column="observed",
        usable_column="usable",
        cause_column="cause",
        feature_columns=WETLINKS_FEATURES,
        bin_seconds=bin_minutes * 60,
        sustained_bad_bins=sustained_bad_bins,
        max_gap_bins=2,
    )
    frame = pd.DataFrame(episodes).sort_values("start_time").reset_index(drop=True)
    return EpisodeDataset(
        frame,
        WETLINKS_FEATURES,
        bin_minutes * 60,
        "Palo Alto Starlink external record",
    )


def build_nexasphere_athens_episodes(
    input_dir: str | Path,
    *,
    bin_minutes: int = 5,
    latency_limit_ms: float = 80.0,
    loss_limit_pct: float = 1.0,
    sustained_bad_bins: int = 2,
) -> EpisodeDataset:
    """Build aggregate-only external episodes from NEXASPHERE Athens.

    The archive also contains an April pilot.  Only the inclusive seven-day
    interval declared by ``metadata.json`` is used.  Weather and a
    WetLinks-equivalent within-burst RTT standard deviation are unavailable;
    those covariates remain missing and can only be imputed by a training-fitted
    encoder.  No Athens value is used for model fitting or calibration.
    """

    input_dir = Path(input_dir)
    metadata_path = input_dir / "metadata.json"
    metrics_path = input_dir / "all_metrics.csv"
    if not metadata_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"expected metadata.json and all_metrics.csv under {input_dir}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    start = pd.to_datetime(metadata["start_timestamp_utc"], utc=True)
    end = pd.to_datetime(metadata["end_timestamp_utc"], utc=True)

    usecols = [
        "timestamp_utc",
        "avg_ms",
        "max_ms",
        "loss_pct",
        "speedtest_dl_mbps",
        "speedtest_ul_mbps",
    ]
    raw = pd.read_csv(metrics_path, usecols=usecols, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp_utc"], errors="coerce", utc=True)
    raw = raw.loc[raw["timestamp"].between(start, end)].copy()
    numeric = [column for column in usecols if column != "timestamp_utc"]
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
    if raw.empty:
        raise ValueError("metadata-window filter produced no Athens measurements")

    regular = (
        raw.set_index("timestamp")
        .resample(f"{bin_minutes}min")
        .agg(
            latency_mean_ms=("avg_ms", "mean"),
            latency_max_ms=("max_ms", "max"),
            packet_loss_pct=("loss_pct", "mean"),
            download_mbps=("speedtest_dl_mbps", "mean"),
            upload_mbps=("speedtest_ul_mbps", "mean"),
        )
        .reset_index()
    )
    for column in (
        "latency_std_ms",
        "temperature_c",
        "humidity_pct",
        "rain_mm",
        "wind_mps",
        "pressure_hpa",
    ):
        regular[column] = np.nan
    regular["observed"] = regular["latency_mean_ms"].notna() & regular[
        "packet_loss_pct"
    ].notna()
    regular["usable"] = (
        regular["observed"]
        & regular["latency_mean_ms"].le(latency_limit_ms)
        & regular["packet_loss_pct"].le(loss_limit_pct)
    )
    regular["cause"] = regular.apply(
        _cause_from_row,
        axis=1,
        latency_limit_ms=latency_limit_ms,
        loss_limit_pct=loss_limit_pct,
    )
    regular["latency_roll3"] = regular["latency_mean_ms"].rolling(
        3, min_periods=1
    ).mean()
    regular["loss_roll3"] = regular["packet_loss_pct"].rolling(
        3, min_periods=1
    ).mean()
    regular["rain_latency_interaction"] = np.nan
    regular = _time_features(regular, "timestamp")
    episodes = _episodes_from_regular_series(
        regular,
        site="nexasphere_athens_gr",
        timestamp_column="timestamp",
        observed_column="observed",
        usable_column="usable",
        cause_column="cause",
        feature_columns=WETLINKS_FEATURES,
        bin_seconds=bin_minutes * 60,
        sustained_bad_bins=sustained_bad_bins,
        max_gap_bins=2,
    )
    frame = pd.DataFrame(episodes).sort_values("start_time").reset_index(drop=True)
    return EpisodeDataset(
        frame,
        WETLINKS_FEATURES,
        bin_minutes * 60,
        "NEXASPHERE Athens Starlink external record",
    )


def build_commect_landmarks(
    path: str | Path,
    *,
    latency_limit_ms: float = 60.0,
    max_horizon_seconds: int = 30,
) -> EpisodeDataset:
    """Create one-second landmark event records from the concurrent COMMECT trace."""

    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["bin_start_utc"], errors="coerce", utc=True)
    numeric = [
        "latency_mean_ms",
        "latency_std_ms",
        "latency_max_ms",
        "observed_replies",
        "observation_age_ms",
        "rsrp_mean_dbm",
        "rsrq_mean_db",
        "snr_mean_db",
        "bin_epoch",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame["is_leo"] = frame["access_technology"].astype(str).str.contains("sat", case=False).astype(int)
    records: list[dict] = []
    for path_name, group in frame.groupby("relative_path", sort=True):
        group = group.sort_values("bin_epoch").reset_index(drop=True)
        group["latency_roll3"] = group["latency_mean_ms"].rolling(3, min_periods=1).mean()
        group["reply_roll3"] = group["observed_replies"].rolling(3, min_periods=1).mean()
        group = _time_features(group, "timestamp")
        epochs = group["bin_epoch"].to_numpy(float)
        latency = group["latency_mean_ms"].to_numpy(float)
        replies = group["observed_replies"].to_numpy(float)
        for i, row in group.iterrows():
            if not np.isfinite(latency[i]) or replies[i] <= 0:
                continue
            # H+1 distinguishes an administratively truncated record known to
            # survive every analysed step 1..H from a measurement gap at the
            # boundary H, whose status at H is unknown.
            duration = max_horizon_seconds + 1
            event = 0
            cause = "censored"
            censor_reason = CENSOR_REASON_ADMINISTRATIVE_HORIZON
            for step in range(1, max_horizon_seconds + 1):
                j = i + step
                if j >= len(group):
                    duration = step
                    event = 0
                    cause = "censored"
                    censor_reason = CENSOR_REASON_RECORDING_ENDED_GOOD
                    break
                if epochs[j] - epochs[i] != step:
                    duration = step
                    event = 0
                    cause = "censored"
                    censor_reason = CENSOR_REASON_INTERNAL_MEASUREMENT_GAP
                    break
                if not np.isfinite(latency[j]) or replies[j] <= 0:
                    duration = step
                    event = 1
                    cause = "availability"
                    censor_reason = CENSOR_REASON_QUALIFYING_EVENT
                    break
                if latency[j] > latency_limit_ms:
                    duration = step
                    event = 1
                    cause = "latency"
                    censor_reason = CENSOR_REASON_QUALIFYING_EVENT
                    break
            future_five: list[float] = []
            future_five_known = True
            for step in range(1, 6):
                j = i + step
                if (
                    j >= len(group)
                    or epochs[j] - epochs[i] != step
                    or not np.isfinite(latency[j])
                    or replies[j] <= 0
                ):
                    future_five_known = False
                    break
                future_five.append(float(latency[j]))
            record = {
                "episode_id": f"{path_name}:{int(row['bin_epoch'])}",
                "site": str(path_name),
                "start_time": row["timestamp"],
                "decision_epoch": int(row["bin_epoch"]),
                "relative_path": str(path_name),
                "duration_bins": int(duration),
                "event": int(event),
                "cause": cause,
                "censor_reason": censor_reason,
                "campaign_id": "commect_denmark_2023_10_11",
                "block_id": f"commect:{int(row['bin_epoch']) // 60}",
                "future_five_known": int(future_five_known),
                "future_mean_latency_h5": float(np.mean(future_five)) if future_five_known else np.nan,
                "future_success_h5": int(max(future_five) <= latency_limit_ms) if future_five_known else np.nan,
            }
            for column in COMMECT_FEATURES:
                record[column] = row.get(column, np.nan)
            records.append(record)
    out = pd.DataFrame(records).sort_values(["decision_epoch", "relative_path"]).reset_index(drop=True)
    return EpisodeDataset(out, COMMECT_FEATURES, 1, "COMMECT")


def build_lens_episodes(
    prediction_path: str | Path,
    *,
    latency_limit_ms: float = 80.0,
) -> EpisodeDataset:
    """Build a secondary multi-site episode table from released LENS-derived artifacts."""

    frame = pd.read_csv(prediction_path)
    if "model_name" in frame:
        frame = frame[frame["model_name"].eq("persistence")].copy()
    frame = frame.drop_duplicates("row_id") if "row_id" in frame else frame.drop_duplicates(
        ["relative_path", "session_bin_index"]
    )
    frame["timestamp"] = pd.to_datetime(frame["bin_start_utc"], errors="coerce", utc=True)
    frame["latency_mean_ms"] = pd.to_numeric(frame["latency_mean_ms"], errors="coerce")
    frame["burst_indicator"] = pd.to_numeric(frame.get("burst_indicator"), errors="coerce")
    frame["reply_pressure_score"] = pd.to_numeric(frame.get("reply_pressure_score"), errors="coerce")
    records: list[dict] = []
    for session, group in frame.groupby("relative_path", sort=True):
        group = group.sort_values("session_bin_index").reset_index(drop=True)
        group["latency_roll3"] = group["latency_mean_ms"].rolling(3, min_periods=1).mean()
        group = _time_features(group, "timestamp")
        values = group["latency_mean_ms"].to_numpy(float)
        indices = group["session_bin_index"].to_numpy(int)
        for i, row in group.iterrows():
            if not np.isfinite(values[i]) or values[i] > latency_limit_ms:
                continue
            # The default means that all 30 future bins were observed usable.
            # A censoring boundary inside the loop is stored at the first
            # unknown bin instead.
            duration = 31
            event = 0
            censor_reason = CENSOR_REASON_ADMINISTRATIVE_HORIZON
            for step in range(1, 31):
                j = i + step
                if j >= len(group):
                    duration = step
                    censor_reason = CENSOR_REASON_RECORDING_ENDED_GOOD
                    break
                if indices[j] - indices[i] != step:
                    duration = step
                    censor_reason = CENSOR_REASON_INTERNAL_MEASUREMENT_GAP
                    break
                if not np.isfinite(values[j]) or values[j] > latency_limit_ms:
                    duration = step
                    event = 1
                    censor_reason = CENSOR_REASON_QUALIFYING_EVENT
                    break
            record = {
                "episode_id": f"{session}:{int(row['session_bin_index'])}",
                "site": str(row["location"]),
                "start_time": row["timestamp"],
                "duration_bins": int(duration),
                "event": int(event),
                "cause": "latency" if event else "censored",
                "censor_reason": censor_reason,
                "campaign_id": str(session),
                "block_id": str(session),
            }
            for column in LENS_FEATURES:
                record[column] = row.get(column, np.nan)
            records.append(record)
    out = pd.DataFrame(records).sort_values(["start_time", "site"]).reset_index(drop=True)
    return EpisodeDataset(out, LENS_FEATURES, 60, "LENS-derived subset")


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split within site by complete episode start time, never by expanded rows."""

    train_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in frame.sort_values("start_time").groupby("site", sort=True):
        n = len(group)
        a = max(1, int(np.floor(n * train_fraction)))
        b = max(a + 1, int(np.floor(n * (train_fraction + calibration_fraction))))
        b = min(b, n)
        train_parts.append(group.iloc[:a])
        calibration_parts.append(group.iloc[a:b])
        test_parts.append(group.iloc[b:])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(calibration_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def calendar_block_split(
    frame: pd.DataFrame,
    *,
    block_column: str = "calendar_block",
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    embargo_blocks: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prospective split using common complete calendar blocks across sites.

    A block is assigned wholly to one partition and the same cutoffs apply to
    every site.  Entire blocks around each boundary are discarded as an
    embargo, preventing cross-site calendar-future leakage and shared weeks.
    This is a split exclusion, not endpoint censoring; an audit is attached to
    each returned frame as ``frame.attrs["split_exclusion_audit"]``.
    """

    if block_column not in frame:
        raise KeyError(block_column)
    blocks = sorted(frame[block_column].dropna().astype(str).unique())
    if len(blocks) < 8:
        raise ValueError("at least eight calendar blocks are required")
    train_count = max(1, int(np.floor(len(blocks) * train_fraction)))
    calibration_end = max(
        train_count + embargo_blocks + 1,
        int(np.floor(len(blocks) * (train_fraction + calibration_fraction))),
    )
    calibration_end = min(calibration_end, len(blocks) - embargo_blocks - 1)
    train_blocks = set(blocks[:train_count])
    calibration_blocks = set(blocks[train_count + embargo_blocks : calibration_end])
    test_blocks = set(blocks[calibration_end + embargo_blocks :])
    if not train_blocks or not calibration_blocks or not test_blocks:
        raise ValueError("calendar split produced an empty partition")
    train = frame[frame[block_column].astype(str).isin(train_blocks)].copy()
    calibration = frame[frame[block_column].astype(str).isin(calibration_blocks)].copy()
    test = frame[frame[block_column].astype(str).isin(test_blocks)].copy()
    train = train.sort_values(["start_time", "site"]).reset_index(drop=True)
    calibration = calibration.sort_values(["start_time", "site"]).reset_index(
        drop=True
    )
    test = test.sort_values(["start_time", "site"]).reset_index(drop=True)
    included_blocks = train_blocks | calibration_blocks | test_blocks
    excluded_blocks = sorted(set(blocks).difference(included_blocks))
    exclusion_audit = {
        "reason": "calendar_block_embargo",
        "excluded_blocks": excluded_blocks,
        "excluded_episode_count": int(
            frame[block_column].astype(str).isin(excluded_blocks).sum()
        ),
        "endpoint_censoring": False,
    }
    for part, name in (
        (train, "train"),
        (calibration, "calibration"),
        (test, "test"),
    ):
        part.attrs["split_name"] = name
        part.attrs["split_exclusion_audit"] = exclusion_audit.copy()
    return train, calibration, test


def purged_epoch_split(
    frame: pd.DataFrame,
    *,
    epoch_column: str = "decision_epoch",
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
    purge_bins: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Common-wall-clock split with an embargo for overlapping landmarks.

    All paths use the same epoch boundaries.  The purge is at least the
    maximum label look-ahead, preventing a future observation from appearing
    in labels on both sides of a split.  Purged landmarks are split
    exclusions, not censored endpoints; an audit is attached to each returned
    frame as ``frame.attrs["split_exclusion_audit"]``.
    """

    epochs = np.sort(frame[epoch_column].dropna().astype(int).unique())
    if len(epochs) < 3:
        raise ValueError("not enough epochs for a purged split")
    train_position = min(max(1, int(np.floor(len(epochs) * train_fraction))), len(epochs) - 2)
    calibration_position = min(
        max(train_position + 1, int(np.floor(len(epochs) * (train_fraction + calibration_fraction)))),
        len(epochs) - 1,
    )
    train_end = int(epochs[train_position - 1])
    calibration_start = train_end + int(purge_bins) + 1
    calibration_end = int(epochs[calibration_position - 1])
    test_start = calibration_end + int(purge_bins) + 1
    train = frame[frame[epoch_column].le(train_end)].copy()
    calibration = frame[
        frame[epoch_column].ge(calibration_start) & frame[epoch_column].le(calibration_end)
    ].copy()
    test = frame[frame[epoch_column].ge(test_start)].copy()
    if min(len(train), len(calibration), len(test)) == 0:
        raise ValueError("purge removed an entire split")
    included_mask = (
        frame[epoch_column].le(train_end)
        | (
            frame[epoch_column].ge(calibration_start)
            & frame[epoch_column].le(calibration_end)
        )
        | frame[epoch_column].ge(test_start)
    )
    exclusion_audit = {
        "reason": "overlapping_label_purge",
        "excluded_episode_count": int((~included_mask).sum()),
        "train_calibration_purge": [train_end + 1, calibration_start - 1],
        "calibration_test_purge": [calibration_end + 1, test_start - 1],
        "endpoint_censoring": False,
    }
    train = train.reset_index(drop=True)
    calibration = calibration.reset_index(drop=True)
    test = test.reset_index(drop=True)
    for part, name in ((train, "train"), (calibration, "calibration"), (test, "test")):
        part.attrs["split_name"] = name
        part.attrs["train_end_epoch"] = train_end
        part.attrs["calibration_start_epoch"] = calibration_start
        part.attrs["calibration_end_epoch"] = calibration_end
        part.attrs["test_start_epoch"] = test_start
        part.attrs["split_exclusion_audit"] = exclusion_audit.copy()
    return train, calibration, test


def merge_sparse_terminal_inference_block(
    frame: pd.DataFrame,
    *,
    group_column: str = "calendar_block",
    output_column: str = "analysis_block",
    minimum_fraction_of_prior_median: float = 0.25,
) -> tuple[pd.DataFrame, dict]:
    """Merge an objectively incomplete terminal block for block inference.

    The rule uses only block sizes and chronology, never outcomes or model
    performance.  It protects equal-block inference from assigning a handful
    of terminal observations the same weight as a complete calendar week.
    Point prediction metrics remain evaluated on every episode.
    """

    if group_column not in frame:
        raise ValueError(f"missing grouping column {group_column!r}")
    if not 0.0 < float(minimum_fraction_of_prior_median) < 1.0:
        raise ValueError("minimum_fraction_of_prior_median must lie in (0, 1)")
    out = frame.copy()
    if "start_time" in out:
        chronology = (
            out.groupby(group_column, sort=False)["start_time"]
            .min()
            .sort_values()
            .index.astype(str)
            .tolist()
        )
    else:
        chronology = sorted(out[group_column].astype(str).unique().tolist())
    counts = out[group_column].astype(str).value_counts().to_dict()
    audit = {
        "source_column": group_column,
        "output_column": output_column,
        "minimum_fraction_of_prior_median": float(minimum_fraction_of_prior_median),
        "original_blocks": chronology,
        "original_block_sizes": {block: int(counts[block]) for block in chronology},
        "merged": False,
        "merged_terminal_block": None,
        "merged_into_block": None,
    }
    out[output_column] = out[group_column].astype(str)
    if len(chronology) >= 2:
        terminal = chronology[-1]
        preceding = chronology[:-1]
        prior_median = float(np.median([counts[block] for block in preceding]))
        threshold = prior_median * float(minimum_fraction_of_prior_median)
        audit["prior_block_size_median"] = prior_median
        audit["terminal_merge_threshold"] = threshold
        if counts[terminal] < threshold:
            destination = preceding[-1]
            out.loc[out[group_column].astype(str).eq(terminal), output_column] = destination
            audit.update(
                {
                    "merged": True,
                    "merged_terminal_block": terminal,
                    "merged_into_block": destination,
                }
            )
    audit["analysis_blocks"] = sorted(out[output_column].astype(str).unique().tolist())
    audit["analysis_block_sizes"] = {
        str(key): int(value)
        for key, value in out[output_column].astype(str).value_counts().sort_index().items()
    }
    return out.reset_index(drop=True), audit
