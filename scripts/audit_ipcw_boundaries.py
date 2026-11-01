#!/usr/bin/env python3
"""Write a deterministic, hand-calculated audit of discrete IPCW boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leo_service_survival.metrics import _censoring_survival, ipcw_survival_targets


SCHEMA_VERSION = "ipcw-boundary-audit-v1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "review_audit"
ABSOLUTE_TOLERANCE = 1e-12


def _format_number(value: float) -> str:
    return f"{float(value):.12g}"


def build_ipcw_boundary_audit() -> tuple[pd.DataFrame, dict]:
    """Return row-level checks and a compact summary for a fixed miniature case."""

    horizon = 3
    case_ids = (
        "event_t2",
        "censor_t2",
        "event_t3",
        "censor_t3",
        "administrative_h_plus_1",
    )
    frame = pd.DataFrame(
        {
            "case_id": case_ids,
            "duration_bins": [2, 2, 3, 3, 4],
            "event": [1, 0, 1, 0, 0],
        }
    )
    query = np.asarray([1, 2, 3, 4], dtype=int)
    observed_post = _censoring_survival(frame, query, left_limit=False)
    observed_left = _censoring_survival(frame, query, left_limit=True)
    expected_post = np.asarray([1.0, 4.0 / 5.0, 8.0 / 15.0, 1e-3])
    expected_left = np.asarray([1.0, 1.0, 4.0 / 5.0, 8.0 / 15.0])

    observed_known, observed_target, observed_weight = ipcw_survival_targets(
        frame, horizon=horizon
    )
    expected_known = np.asarray([True, False, True, False, True])
    expected_target = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])
    expected_weight = np.asarray([5.0 / 4.0, 0.0, 15.0 / 8.0, 0.0, 15.0 / 8.0])

    def status(known: bool, target: float) -> str:
        if not known:
            return "censored_at_or_before_h_unknown"
        if target == 1.0:
            return "survived_beyond_h"
        return "failure_by_h"

    observed_status = [
        status(bool(known), float(target))
        for known, target in zip(observed_known, observed_target)
    ]
    expected_status = [
        "failure_by_h",
        "censored_at_or_before_h_unknown",
        "failure_by_h",
        "censored_at_or_before_h_unknown",
        "survived_beyond_h",
    ]

    rows: list[dict] = []

    def add_numeric(family: str, item: str, observed: float, expected: float) -> None:
        error = abs(float(observed) - float(expected))
        rows.append(
            {
                "check_family": family,
                "item": item,
                "observed_value": _format_number(observed),
                "expected_value": _format_number(expected),
                "absolute_error": _format_number(error),
                "tolerance": _format_number(ABSOLUTE_TOLERANCE),
                "pass_flag": bool(error <= ABSOLUTE_TOLERANCE),
            }
        )

    def add_exact(family: str, item: str, observed: object, expected: object) -> None:
        rows.append(
            {
                "check_family": family,
                "item": item,
                "observed_value": str(observed).lower()
                if isinstance(observed, (bool, np.bool_))
                else str(observed),
                "expected_value": str(expected).lower()
                if isinstance(expected, (bool, np.bool_))
                else str(expected),
                "absolute_error": "",
                "tolerance": "exact",
                "pass_flag": bool(observed == expected),
            }
        )

    for time, observed, expected in zip(query, observed_post, expected_post):
        add_numeric("reverse_km_post_boundary_g", f"time_{time}", observed, expected)
    for time, observed, expected in zip(query, observed_left, expected_left):
        add_numeric("reverse_km_left_limit_g", f"time_{time}", observed, expected)
    for case_id, observed, expected in zip(case_ids, observed_weight, expected_weight):
        add_numeric("ipcw_weight_h3", case_id, observed, expected)
    for case_id, observed, expected in zip(case_ids, observed_known, expected_known):
        add_exact("evaluable_at_h3", case_id, observed, expected)
    for case_id, observed, expected in zip(case_ids, observed_status, expected_status):
        add_exact("episode_status_h3", case_id, observed, expected)

    checks = pd.DataFrame(rows)
    passed = int(checks["pass_flag"].sum())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "fixture": {
            "horizon_bins": horizon,
            "case_id": list(case_ids),
            "duration_bins": frame["duration_bins"].astype(int).tolist(),
            "event": frame["event"].astype(int).tolist(),
        },
        "declared_convention": {
            "censor_boundary": "status in the censoring bin is unknown",
            "event_boundary": "event in the assigned bin is observed",
            "weight_denominator": "post-boundary reverse-KM G(t)=P(C>t)",
            "administrative_horizon": "known survival through H is encoded at H+1",
            "censoring_survival_floor": 1e-3,
        },
        "expected_survival_target_h3": expected_target.tolist(),
        "total_checks": int(len(checks)),
        "passed_checks": passed,
        "failed_checks": int(len(checks) - passed),
        "all_passed": bool(checks["pass_flag"].all()),
        "checks_csv": "ipcw_boundary_checks.csv",
    }
    return checks, summary


def write_ipcw_boundary_audit(output_dir: str | Path) -> tuple[Path, Path, dict]:
    """Write deterministic CSV and JSON artifacts and return their paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks, summary = build_ipcw_boundary_audit()
    csv_path = output_dir / "ipcw_boundary_checks.csv"
    json_path = output_dir / "ipcw_boundary_audit.json"
    checks.to_csv(csv_path, index=False, lineterminator="\n")
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    csv_path, json_path, summary = write_ipcw_boundary_audit(args.output_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(
        f"checks: {summary['passed_checks']}/{summary['total_checks']} passed; "
        f"all_passed={summary['all_passed']}"
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

