"""Auditable adapters for the public NUWiNS SIGMETRICS'26 traces.

The publisher releases two materially different data products:

* Alaska and Hawaii contain publisher-pre-aligned two-path panels.
* The LA--Omaha campaign contains independent per-operator files without an
  exact cross-operator key.

The APIs below keep those cases separate.  In particular, the mainland
adapter uses an explicit, deterministic, fail-closed nearest-time rule and
returns an alignment audit in addition to eligible long-form observations.
RTT and throughput are never joined by this module.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


_PANEL_METRIC: Final[dict[str, tuple[str, float]]] = {
    "downlink_throughput": ("throughput_mbps", 250.0),
    "rtt": ("rtt_ms", 100.0),
}

_PROVENANCE_COLUMNS: Final[tuple[str, ...]] = (
    "source_dataset",
    "source_id",
    "source_file",
    "source_sha256",
    "source_row_number",
)


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


def _source_metadata(path: str | Path, campaign_id: str) -> dict[str, str]:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    digest = _file_sha256(source_path)
    return {
        "source_dataset": "nuwins_sigmetrics26",
        "source_id": f"nuwins_sigmetrics26:{campaign_id}:sha256:{digest}",
        "source_file": str(source_path),
        "source_sha256": digest,
    }


def _panel_definition(panel_kind: str) -> tuple[str, float]:
    try:
        return _PANEL_METRIC[panel_kind]
    except KeyError as exc:
        raise ValueError(
            f"panel_kind must be one of {sorted(_PANEL_METRIC)}, got {panel_kind!r}"
        ) from exc


def _epoch_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return sha256(payload).hexdigest()[:24]


def _pair_id(first: object, second: object) -> str:
    return "__".join(sorted((str(first).strip().lower(), str(second).strip().lower())))


def load_nuwins_pre_aligned_panel(
    path: str | Path,
    *,
    campaign_id: str,
    panel_kind: str,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Normalize an Alaska/Hawaii publisher-paired panel to long form.

    ``max_rows`` bounds source rows before the two-row long-form expansion.
    A measured zero remains an observed outcome.  Missing outcomes are retained
    but the entire pair is marked ineligible.  The function deliberately has no
    facility for combining this output with the other panel kind.
    """

    metric, _ = _panel_definition(panel_kind)
    max_rows = _positive_limit(max_rows, "max_rows")
    path = Path(path).expanduser().resolve()
    metadata = _source_metadata(path, campaign_id)
    frame = pd.read_csv(path, nrows=max_rows)

    value_columns = {"A": f"A_{metric}", "B": f"B_{metric}"}
    required = {
        "A",
        "A_run",
        "A_time",
        value_columns["A"],
        "A_actual_tech",
        "B",
        "B_run",
        "B_time",
        value_columns["B"],
        "B_actual_tech",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{panel_kind} pre-aligned panel is missing required columns: {missing}"
        )

    times = {
        slot: pd.to_datetime(frame[f"{slot}_time"], utc=True, errors="coerce")
        for slot in ("A", "B")
    }
    if times["A"].isna().any() or times["B"].isna().any():
        raise ValueError("pre-aligned panel contains an unparseable path timestamp")

    key = pd.DataFrame(
        {
            "A_run": frame["A_run"].astype("string"),
            "B_run": frame["B_run"].astype("string"),
            "A_time": times["A"],
            "B_time": times["B"],
        }
    )
    if key.duplicated().any():
        raise ValueError("publisher pre-aligned composite key is not unique")

    epoch_time = pd.concat(times, axis=1).max(axis=1)
    skew_ms = (times["A"] - times["B"]).abs().dt.total_seconds() * 1000.0
    values = {
        slot: pd.to_numeric(frame[value_columns[slot]], errors="coerce")
        for slot in ("A", "B")
    }
    pair_eligible = values["A"].notna() & values["B"].notna()

    records: list[dict[str, object]] = []
    for source_index in range(len(frame)):
        physical_row = source_index + 2  # CSV header occupies physical line 1.
        pair = _pair_id(frame.at[source_index, "A"], frame.at[source_index, "B"])
        epoch = _epoch_id(
            metadata["source_sha256"], panel_kind, physical_row, pair
        )
        for slot in ("A", "B"):
            value = values[slot].iat[source_index]
            record: dict[str, object] = {
                "campaign_id": str(campaign_id),
                "panel_kind": panel_kind,
                "pair_id": pair,
                "decision_epoch_id": epoch,
                "path_id": str(frame.at[source_index, slot]).strip().lower(),
                "path_slot": slot,
                "source_run_id": str(frame.at[source_index, f"{slot}_run"]),
                "source_time_utc": times[slot].iat[source_index],
                "paired_source_time_utc": times["B" if slot == "A" else "A"].iat[
                    source_index
                ],
                "epoch_time_utc": epoch_time.iat[source_index],
                "inter_path_skew_ms": float(skew_ms.iat[source_index]),
                "throughput_mbps": value
                if panel_kind == "downlink_throughput"
                else np.nan,
                "rtt_ms": value if panel_kind == "rtt" else np.nan,
                "actual_tech": frame.at[source_index, f"{slot}_actual_tech"],
                "outcome_observed": bool(pd.notna(value)),
                "measurement_role": "observed_outcome_not_predecision_covariate",
                "alignment_method": "provider_pre_aligned_row",
                "alignment_eligible": bool(pair_eligible.iat[source_index]),
                "alignment_exclusion_reason": pd.NA
                if pair_eligible.iat[source_index]
                else "missing_path_outcome",
                "evidence_tier": "field_observational",
                **metadata,
                "source_row_number": physical_row,
            }
            records.append(record)
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class MainlandAlignmentConfig:
    """Frozen tolerances for an explicitly derived mainland pair panel."""

    run_tolerance_ms: float = 5_000.0
    sample_tolerance_ms: float | None = None
    max_anchor_rows: int | None = None
    max_candidate_rows: int | None = None

    def validated(self, panel_kind: str) -> "MainlandAlignmentConfig":
        _, default_sample_tolerance = _panel_definition(panel_kind)
        sample_tolerance = (
            default_sample_tolerance
            if self.sample_tolerance_ms is None
            else float(self.sample_tolerance_ms)
        )
        if float(self.run_tolerance_ms) < 0 or sample_tolerance < 0:
            raise ValueError("alignment tolerances must be non-negative")
        return MainlandAlignmentConfig(
            run_tolerance_ms=float(self.run_tolerance_ms),
            sample_tolerance_ms=sample_tolerance,
            max_anchor_rows=_positive_limit(self.max_anchor_rows, "max_anchor_rows"),
            max_candidate_rows=_positive_limit(
                self.max_candidate_rows, "max_candidate_rows"
            ),
        )


@dataclass(frozen=True)
class MainlandAlignmentResult:
    """Eligible long-form pairs plus one audit row per anchor observation."""

    observations: pd.DataFrame
    audit: pd.DataFrame
    config: MainlandAlignmentConfig


def _load_mainland_source(
    path: Path,
    panel_kind: str,
    max_rows: int | None,
) -> pd.DataFrame:
    metric, _ = _panel_definition(panel_kind)
    frame = pd.read_csv(path, nrows=max_rows)
    required = {"utc_ts", "run_id", metric}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"mainland {panel_kind} file is missing columns: {missing}")
    result = frame.copy()
    result["_source_row_number"] = np.arange(len(result), dtype=int) + 2
    result["_run_numeric"] = pd.to_numeric(result["run_id"], errors="coerce")
    result["_time_numeric_ms"] = pd.to_numeric(result["utc_ts"], errors="coerce")
    result["_source_time_utc"] = pd.to_datetime(
        result["_time_numeric_ms"], unit="ms", utc=True, errors="coerce"
    )
    result["_metric"] = pd.to_numeric(result[metric], errors="coerce")
    return result


def _nearest_run_map(
    anchor: pd.DataFrame,
    candidate: pd.DataFrame,
    tolerance_ms: float,
) -> dict[float, tuple[float | None, float | None, str | None]]:
    candidate_runs = np.sort(candidate["_run_numeric"].dropna().unique())
    mapping: dict[float, tuple[float | None, float | None, str | None]] = {}
    for anchor_run in np.sort(anchor["_run_numeric"].dropna().unique()):
        if len(candidate_runs) == 0:
            mapping[float(anchor_run)] = (None, None, "no_candidate_runs")
            continue
        insertion = int(np.searchsorted(candidate_runs, anchor_run))
        positions = [
            position
            for position in (insertion - 1, insertion)
            if 0 <= position < len(candidate_runs)
        ]
        # Numeric run ID then source order provides a stable tie break.
        best = min(
            (float(candidate_runs[position]) for position in positions),
            key=lambda value: (abs(value - float(anchor_run)), value),
        )
        skew = abs(best - float(anchor_run))
        if skew > tolerance_ms:
            mapping[float(anchor_run)] = (
                best,
                skew,
                "run_tolerance_exceeded",
            )
        else:
            mapping[float(anchor_run)] = (best, skew, None)
    return mapping


def align_nuwins_mainland_pair(
    anchor_path: str | Path,
    candidate_path: str | Path,
    *,
    campaign_id: str = "mainland_202411",
    anchor_operator: str,
    candidate_operator: str,
    panel_kind: str,
    config: MainlandAlignmentConfig | None = None,
) -> MainlandAlignmentResult:
    """Derive a deterministic mainland pair with fail-closed eligibility.

    The anchor is normally Starlink.  Each anchor run is mapped to the nearest
    candidate run start, then anchor samples are processed in timestamp/source
    order and greedily matched to the nearest *unused* candidate sample.  Ties
    choose the earlier candidate timestamp and then its physical source row.
    Rows that fail either frozen tolerance never appear in ``observations``;
    they remain visible in ``audit`` with an exclusion reason.
    """

    config = (config or MainlandAlignmentConfig()).validated(panel_kind)
    metric, _ = _panel_definition(panel_kind)
    anchor_path = Path(anchor_path).expanduser().resolve()
    candidate_path = Path(candidate_path).expanduser().resolve()
    anchor_meta = _source_metadata(anchor_path, campaign_id)
    candidate_meta = _source_metadata(candidate_path, campaign_id)
    anchor = _load_mainland_source(anchor_path, panel_kind, config.max_anchor_rows)
    candidate = _load_mainland_source(
        candidate_path, panel_kind, config.max_candidate_rows
    )
    run_map = _nearest_run_map(anchor, candidate, config.run_tolerance_ms)

    observations: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    used_candidate_rows: set[int] = set()

    ordered_anchor = anchor.sort_values(
        ["_time_numeric_ms", "_source_row_number"], kind="mergesort", na_position="last"
    )
    for anchor_index, anchor_row in ordered_anchor.iterrows():
        reason: str | None = None
        candidate_index: int | None = None
        run_skew_ms: float | None = None
        sample_skew_ms: float | None = None
        mapped_run: float | None = None

        if pd.isna(anchor_row["_run_numeric"]):
            reason = "missing_anchor_run_id"
        elif pd.isna(anchor_row["_time_numeric_ms"]):
            reason = "missing_anchor_timestamp"
        elif pd.isna(anchor_row["_metric"]):
            reason = "missing_anchor_outcome"
        else:
            mapped_run, run_skew_ms, reason = run_map[float(anchor_row["_run_numeric"])]

        if reason is None and mapped_run is not None:
            pool = candidate.loc[candidate["_run_numeric"].eq(mapped_run)].copy()
            pool = pool.loc[pool["_time_numeric_ms"].notna() & pool["_metric"].notna()]
            pool = pool.sort_values(
                ["_time_numeric_ms", "_source_row_number"], kind="mergesort"
            )
            times = pool["_time_numeric_ms"].to_numpy(dtype=float)
            indices = pool.index.to_numpy(dtype=int)
            insertion = bisect_left(times, float(anchor_row["_time_numeric_ms"]))
            eligible: list[tuple[float, float, int, int]] = []
            left = insertion - 1
            while left >= 0:
                difference = abs(times[left] - float(anchor_row["_time_numeric_ms"]))
                if difference > float(config.sample_tolerance_ms):
                    break
                if int(indices[left]) not in used_candidate_rows:
                    eligible.append(
                        (difference, float(times[left]), int(pool.iloc[left]["_source_row_number"]), int(indices[left]))
                    )
                left -= 1
            right = insertion
            while right < len(times):
                difference = abs(times[right] - float(anchor_row["_time_numeric_ms"]))
                if difference > float(config.sample_tolerance_ms):
                    break
                if int(indices[right]) not in used_candidate_rows:
                    eligible.append(
                        (difference, float(times[right]), int(pool.iloc[right]["_source_row_number"]), int(indices[right]))
                    )
                right += 1
            if eligible:
                sample_skew_ms, _, _, candidate_index = min(eligible)
                used_candidate_rows.add(candidate_index)
            else:
                within_tolerance = bool(
                    len(times)
                    and np.min(
                        np.abs(times - float(anchor_row["_time_numeric_ms"]))
                    )
                    <= float(config.sample_tolerance_ms)
                )
                reason = (
                    "no_unused_candidate_within_tolerance"
                    if within_tolerance
                    else "sample_tolerance_exceeded"
                )

        audit_rows.append(
            {
                "campaign_id": campaign_id,
                "panel_kind": panel_kind,
                "anchor_operator": anchor_operator,
                "candidate_operator": candidate_operator,
                "anchor_source_id": anchor_meta["source_id"],
                "candidate_source_id": candidate_meta["source_id"],
                "anchor_source_row_number": int(anchor_row["_source_row_number"]),
                "candidate_source_row_number": int(candidate.loc[candidate_index, "_source_row_number"])
                if candidate_index is not None
                else pd.NA,
                "anchor_run_id": str(anchor_row["run_id"]),
                "candidate_run_id": str(candidate.loc[candidate_index, "run_id"])
                if candidate_index is not None
                else (str(int(mapped_run)) if mapped_run is not None else pd.NA),
                "anchor_source_time_utc": anchor_row["_source_time_utc"],
                "candidate_source_time_utc": candidate.loc[candidate_index, "_source_time_utc"]
                if candidate_index is not None
                else pd.NaT,
                "run_start_skew_ms": run_skew_ms,
                "inter_path_skew_ms": sample_skew_ms,
                "run_tolerance_ms": float(config.run_tolerance_ms),
                "sample_tolerance_ms": float(config.sample_tolerance_ms),
                "alignment_eligible": reason is None,
                "alignment_exclusion_reason": reason if reason is not None else pd.NA,
                "alignment_method": "nearest_run_then_nearest_unused_timestamp",
                "evidence_tier": "field_observational",
            }
        )
        if reason is not None or candidate_index is None:
            continue

        candidate_row = candidate.loc[candidate_index]
        pair = _pair_id(anchor_operator, candidate_operator)
        epoch = _epoch_id(
            anchor_meta["source_sha256"],
            candidate_meta["source_sha256"],
            panel_kind,
            int(anchor_row["_source_row_number"]),
            int(candidate_row["_source_row_number"]),
        )
        epoch_time = max(
            anchor_row["_source_time_utc"], candidate_row["_source_time_utc"]
        )
        for slot, operator, row, own_meta, paired_row in (
            ("A", anchor_operator, anchor_row, anchor_meta, candidate_row),
            ("B", candidate_operator, candidate_row, candidate_meta, anchor_row),
        ):
            value = float(row["_metric"])
            observations.append(
                {
                    "campaign_id": campaign_id,
                    "panel_kind": panel_kind,
                    "pair_id": pair,
                    "decision_epoch_id": epoch,
                    "path_id": str(operator).strip().lower(),
                    "path_slot": slot,
                    "source_run_id": str(row["run_id"]),
                    "paired_source_run_id": str(paired_row["run_id"]),
                    "source_time_utc": row["_source_time_utc"],
                    "paired_source_time_utc": paired_row["_source_time_utc"],
                    "epoch_time_utc": epoch_time,
                    "run_start_skew_ms": float(run_skew_ms),
                    "inter_path_skew_ms": float(sample_skew_ms),
                    "throughput_mbps": value
                    if panel_kind == "downlink_throughput"
                    else np.nan,
                    "rtt_ms": value if panel_kind == "rtt" else np.nan,
                    "actual_tech": row.get("actual_tech", pd.NA),
                    "outcome_observed": True,
                    "measurement_role": "observed_outcome_not_predecision_covariate",
                    "alignment_method": "nearest_run_then_nearest_unused_timestamp",
                    "alignment_eligible": True,
                    "alignment_exclusion_reason": pd.NA,
                    "run_tolerance_ms": float(config.run_tolerance_ms),
                    "sample_tolerance_ms": float(config.sample_tolerance_ms),
                    "evidence_tier": "field_observational",
                    **own_meta,
                    "source_row_number": int(row["_source_row_number"]),
                }
            )

    observation_columns = [
        "campaign_id",
        "panel_kind",
        "pair_id",
        "decision_epoch_id",
        "path_id",
        "path_slot",
        "source_run_id",
        "paired_source_run_id",
        "source_time_utc",
        "paired_source_time_utc",
        "epoch_time_utc",
        "run_start_skew_ms",
        "inter_path_skew_ms",
        "throughput_mbps",
        "rtt_ms",
        "actual_tech",
        "outcome_observed",
        "measurement_role",
        "alignment_method",
        "alignment_eligible",
        "alignment_exclusion_reason",
        "run_tolerance_ms",
        "sample_tolerance_ms",
        "evidence_tier",
        *_PROVENANCE_COLUMNS,
    ]
    return MainlandAlignmentResult(
        observations=pd.DataFrame.from_records(
            observations, columns=observation_columns
        ),
        audit=pd.DataFrame.from_records(audit_rows),
        config=config,
    )
