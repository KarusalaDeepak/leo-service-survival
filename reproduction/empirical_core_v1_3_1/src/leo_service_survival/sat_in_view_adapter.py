"""Adapter for the SatInView trajectory-matching artifact.

``Connected_Satellite`` in the public CSV is a post-processed trajectory match
using obstruction-map pixels, private terminal geometry, and TLE data.  It is
not a directly observed serving-satellite identifier.  The adapter makes that
distinction explicit and only derives a possible handover from consecutive
changes in that inferred label.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sat_in_view_matches(
    path: str | Path,
    *,
    max_rows: int | None = 100_000,
    maximum_transition_gap_s: float = 2.0,
) -> pd.DataFrame:
    """Load bounded inferred satellite matches and conservative transitions.

    Identity changes across a missing/long interval are retained but are not
    labeled as inferred handovers because the transition was not observed
    continuously.  No site coordinates, beam, gateway, or RF fields are added.
    """

    if max_rows is not None and int(max_rows) <= 0:
        raise ValueError("max_rows must be positive or None")
    if maximum_transition_gap_s <= 0:
        raise ValueError("maximum_transition_gap_s must be positive")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = _file_sha256(path)
    frame = pd.read_csv(path, nrows=None if max_rows is None else int(max_rows))
    required = {
        "Timestamp",
        "Y",
        "X",
        "Elevation",
        "Azimuth",
        "PingTime",
        "Connected_Satellite",
        "Distance",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SatInView artifact is missing columns: {missing}")

    result = pd.DataFrame(
        {
            "campaign_id": "sat_in_view_public_artifact",
            "source_time_utc": pd.to_datetime(
                frame["Timestamp"], utc=True, errors="coerce"
            ),
            "obstruction_pixel_y": pd.to_numeric(frame["Y"], errors="coerce"),
            "obstruction_pixel_x": pd.to_numeric(frame["X"], errors="coerce"),
            "matched_elevation_deg": pd.to_numeric(
                frame["Elevation"], errors="coerce"
            ),
            "matched_azimuth_deg": pd.to_numeric(
                frame["Azimuth"], errors="coerce"
            ),
            "ping_rtt_ms": pd.to_numeric(frame["PingTime"], errors="coerce"),
            "inferred_satellite_id": frame["Connected_Satellite"].astype("string"),
            "matched_slant_distance_km": pd.to_numeric(
                frame["Distance"], errors="coerce"
            ),
        }
    )
    if result["source_time_utc"].isna().any():
        raise ValueError("SatInView artifact contains an invalid timestamp")
    result["source_row_number"] = np.arange(len(result), dtype=int) + 2
    result = result.sort_values(
        ["source_time_utc", "source_row_number"], kind="mergesort"
    ).reset_index(drop=True)
    gap_s = result["source_time_utc"].diff().dt.total_seconds()
    prior_identity = result["inferred_satellite_id"].shift()
    result["inferred_identity_change"] = (
        result["inferred_satellite_id"].ne(prior_identity)
        & prior_identity.notna()
        & result["inferred_satellite_id"].notna()
    )
    result["transition_contiguous"] = (
        gap_s.gt(0) & gap_s.le(float(maximum_transition_gap_s))
    )
    result["inferred_handover_event"] = (
        result["inferred_identity_change"] & result["transition_contiguous"]
    )
    result["identity_change_unresolved_across_gap"] = (
        result["inferred_identity_change"] & ~result["transition_contiguous"]
    )
    result["time_since_prior_match_s"] = gap_s
    result["identity_evidence"] = (
        "post_processed_trajectory_obstruction_map_match_not_direct_telemetry"
    )
    result["handover_evidence"] = (
        "derived_only_from_consecutive_inferred_identity_change"
    )
    result["evidence_tier"] = "field_observational"
    result["source_dataset"] = "sat_in_view"
    result["source_id"] = f"sat_in_view:matched_satellite_data:sha256:{digest}"
    result["source_file"] = str(path)
    result["source_sha256"] = digest
    return result
