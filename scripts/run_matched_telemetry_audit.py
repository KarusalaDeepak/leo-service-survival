#!/usr/bin/env python3
"""Audit matched public LEO identity, radio, geometry, and RTT evidence.

The script deliberately does not join SatInView and LENS OneWeb. They are
different campaigns. Its output is a coverage audit plus bounded descriptive
summaries, never a synthetic full-chain or causal handover result.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leo_service_survival.sat_in_view_adapter import load_sat_in_view_matches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAT_IN_VIEW = (
    PROJECT_ROOT
    / "data/external/sat_in_view/paper_results/matched_satellite_data.csv"
)
DEFAULT_ONEWEB = (
    PROJECT_ROOT
    / "data/external/lens_oneweb_iowa_2024/canonical/"
    "oneweb_iowa_2024-12-20_track_577438120_sat_424.csv"
)
DEFAULT_ONEWEB_AUDIT = (
    PROJECT_ROOT
    / "data/external/lens_oneweb_iowa_2024/canonical/audit_manifest.json"
)
DEFAULT_ONEWEB_TRANSITIONS = (
    PROJECT_ROOT
    / "data/external/lens_oneweb_iowa_2024/canonical/transition_descriptives.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/matched_telemetry_audit_v1"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(q)) if len(numeric) else np.nan


def _sat_in_view_transition_table(frame: pd.DataFrame, half_window: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    transition_indices = np.flatnonzero(frame["inferred_handover_event"].to_numpy(bool))
    event_set = set(int(value) for value in transition_indices)
    for index in transition_indices:
        start = int(index) - half_window
        stop = int(index) + half_window
        if start < 0 or stop >= len(frame):
            continue
        window = frame.iloc[start : stop + 1]
        expected = np.arange(-half_window, half_window + 1, dtype=float)
        relative_time = (
            window["source_time_utc"] - frame.iloc[int(index)]["source_time_utc"]
        ).dt.total_seconds().to_numpy(float)
        if not np.allclose(relative_time, expected, atol=0.05):
            continue
        if any(candidate != int(index) for candidate in event_set.intersection(range(start, stop + 1))):
            continue
        pre = pd.to_numeric(frame.iloc[start:int(index)]["ping_rtt_ms"], errors="coerce")
        post = pd.to_numeric(frame.iloc[int(index) : stop + 1]["ping_rtt_ms"], errors="coerce")
        if pre.isna().any() or post.isna().any():
            continue
        rows.append(
            {
                "transition_time_utc": frame.iloc[int(index)]["source_time_utc"],
                "from_inferred_satellite_id": frame.iloc[int(index) - 1]["inferred_satellite_id"],
                "to_inferred_satellite_id": frame.iloc[int(index)]["inferred_satellite_id"],
                "pre_5s_mean_rtt_ms": float(pre.mean()),
                "transition_and_post_5s_mean_rtt_ms": float(post.mean()),
                "descriptive_rtt_change_ms": float(post.mean() - pre.mean()),
                "transition_rtt_ms": float(frame.iloc[int(index)]["ping_rtt_ms"]),
                "evidence_tier": "field_observational",
                "identity_semantics": "post_processed_trajectory_match_not_direct_operator_identity",
                "transition_semantics": "consecutive_inferred_identity_change_not_confirmed_handover",
            }
        )
    return pd.DataFrame(rows)


def _coverage_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "SatInView",
                "site_contexts": 1,
                "operator_context": "Starlink",
                "satellite_identity": "inferred trajectory/obstruction-map match",
                "rf_observation": False,
                "handover_observation": "derived consecutive inferred-ID change",
                "beam_identity": False,
                "gateway_identity": False,
                "orbital_geometry": True,
                "matched_rtt": True,
                "eo_workload": False,
                "action_or_completion": False,
                "evidence_tier": "field_observational",
            },
            {
                "source": "LENS OneWeb compact interval",
                "site_contexts": 1,
                "operator_context": "OneWeb",
                "satellite_identity": "terminal AIM SATELLITE_ID plus TLE name-key match",
                "rf_observation": True,
                "handover_observation": "adjacent AIM satellite/track assignment changes",
                "beam_identity": False,
                "gateway_identity": False,
                "orbital_geometry": "TLE association; no propagated look angles in this audit",
                "matched_rtt": True,
                "eo_workload": False,
                "action_or_completion": False,
                "evidence_tier": "field_observational",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sat-in-view", type=Path, default=DEFAULT_SAT_IN_VIEW)
    parser.add_argument("--oneweb", type=Path, default=DEFAULT_ONEWEB)
    parser.add_argument("--oneweb-audit", type=Path, default=DEFAULT_ONEWEB_AUDIT)
    parser.add_argument(
        "--oneweb-transitions", type=Path, default=DEFAULT_ONEWEB_TRANSITIONS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"choose a fresh output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    sat = load_sat_in_view_matches(args.sat_in_view, max_rows=None)
    transitions = _sat_in_view_transition_table(sat)
    oneweb = pd.read_csv(args.oneweb)
    oneweb_audit = json.loads(args.oneweb_audit.read_text(encoding="utf-8"))
    oneweb_transitions = pd.read_csv(args.oneweb_transitions)

    oneweb_joined = oneweb.loc[oneweb["ping_join_within_tolerance"].eq(1)].copy()
    oneweb_rtt = pd.to_numeric(oneweb_joined["ping_rtt_ms"], errors="coerce")
    oneweb_sinr = pd.to_numeric(
        oneweb_joined["average_sinr_reported"], errors="coerce"
    )
    joint = oneweb_rtt.notna() & oneweb_sinr.notna()
    spearman = (
        float(oneweb_rtt[joint].corr(oneweb_sinr[joint], method="spearman"))
        if int(joint.sum()) >= 3
        else np.nan
    )

    summary = pd.DataFrame(
        [
            {
                "source": "SatInView",
                "rows": int(len(sat)),
                "matched_rtt_rows": int(sat["ping_rtt_ms"].notna().sum()),
                "unique_satellite_ids": int(sat["inferred_satellite_id"].nunique()),
                "observed_or_inferred_transitions": int(sat["inferred_handover_event"].sum()),
                "unresolved_identity_changes_across_gaps": int(
                    sat["identity_change_unresolved_across_gap"].sum()
                ),
                "median_rtt_ms": _safe_quantile(sat["ping_rtt_ms"], 0.5),
                "p95_rtt_ms": _safe_quantile(sat["ping_rtt_ms"], 0.95),
                "median_sinr_reported": np.nan,
                "rtt_sinr_spearman": np.nan,
                "transition_windows_with_complete_5s_context": int(len(transitions)),
                "evidence_tier": "field_observational",
            },
            {
                "source": "LENS OneWeb compact interval",
                "rows": int(len(oneweb)),
                "matched_rtt_rows": int(len(oneweb_joined)),
                "unique_satellite_ids": int(
                    pd.concat(
                        [
                            oneweb_transitions["from_satellite_id"],
                            oneweb_transitions["to_satellite_id"],
                        ],
                        ignore_index=True,
                    ).nunique()
                ),
                "observed_or_inferred_transitions": int(len(oneweb_transitions)),
                "unresolved_identity_changes_across_gaps": 0,
                "median_rtt_ms": _safe_quantile(oneweb_joined["ping_rtt_ms"], 0.5),
                "p95_rtt_ms": _safe_quantile(oneweb_joined["ping_rtt_ms"], 0.95),
                "median_sinr_reported": _safe_quantile(
                    oneweb_joined["average_sinr_reported"], 0.5
                ),
                "rtt_sinr_spearman": spearman,
                "transition_windows_with_complete_5s_context": int(
                    (
                        oneweb_transitions["before_rtt_accepted_rows"].gt(0)
                        & oneweb_transitions["after_rtt_accepted_rows"].gt(0)
                    ).sum()
                ),
                "evidence_tier": "field_observational",
            },
        ]
    )
    coverage = _coverage_table()
    transitions.to_csv(args.output / "sat_in_view_transition_descriptives.csv", index=False)
    oneweb_transitions.to_csv(
        args.output / "oneweb_transition_descriptives.csv", index=False
    )
    summary.to_csv(args.output / "matched_telemetry_summary.csv", index=False)
    coverage.to_csv(args.output / "matched_field_coverage.csv", index=False)

    status = {
        "criterion": "matched_satellite_beam_gateway_handover_rf",
        "satisfied": False,
        "evidence_tier": "field_observational",
        "reason": (
            "The public campaigns provide complementary satellite identity, transition, RF, "
            "geometry, and RTT evidence, but neither records measured beam and gateway identity, "
            "and the Starlink identity/transition labels are inferred. Unrelated campaigns are "
            "not joined to manufacture a complete row."
        ),
        "oneweb_assignment_transition_interpretation": oneweb_audit["selection"][
            "interpretation_limit"
        ],
    }
    (args.output / "criterion_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    artifacts = sorted(path for path in args.output.iterdir() if path.is_file())
    manifest = {
        "evidence_tier": "field_observational",
        "inputs": {
            str(args.sat_in_view.resolve()): _sha256(args.sat_in_view),
            str(args.oneweb.resolve()): _sha256(args.oneweb),
            str(args.oneweb_audit.resolve()): _sha256(args.oneweb_audit),
            str(args.oneweb_transitions.resolve()): _sha256(
                args.oneweb_transitions
            ),
        },
        "artifacts_sha256_before_manifest": {
            path.name: _sha256(path) for path in artifacts
        },
        "claim_boundary": (
            "Descriptive matched-telemetry coverage audit. No cross-campaign join, causal "
            "handover effect, policy benefit, or EO completion claim."
        ),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
