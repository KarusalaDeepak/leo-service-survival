#!/usr/bin/env python3
"""Build the two decision-effect panels from released aggregate tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "manuscript"
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "decision_contrasts.png"
NAVY = "#173B57"
TEAL = "#2C8C82"
GOLD = "#C9892B"
GRID = "#E2E6E9"


def require_one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    """Return one uniquely identified result row."""
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def build(results_dir: Path, output: Path) -> None:
    commect = pd.read_csv(results_dir / "commect_time_series_inference.csv")
    scheduler = pd.read_csv(results_dir / "scheduler_inference.csv")

    commect_row = require_one(
        commect,
        reference="survival_candidate",
        baseline="reactive",
        metric="conservative_success",
        block_length=60,
    )
    scheduler_row = require_one(
        scheduler,
        reference="survival_aware",
        baseline="edf",
        metric="completion_rate_all",
        uncertainty_layer="empirical_trace_block",
    )

    panels = (
        {
            "title": "(a) COMMECT candidate minus reactive control",
            "label": "Reactive",
            "point": 100 * float(commect_row["mean_difference"]),
            "low": 100 * float(commect_row["ci_low"]),
            "high": 100 * float(commect_row["ci_high"]),
            "color": TEAL,
            "footer": "60-s moving-block interval; one recorded campaign.",
        },
        {
            "title": "(b) Survival-ranked minus earliest-feasible",
            "label": "Earliest",
            "point": 100 * float(scheduler_row["mean_difference"]),
            "low": 100 * float(scheduler_row["ci_low"]),
            "high": 100 * float(scheduler_row["ci_high"]),
            "color": GOLD,
            "footer": (
                "Four-block approximate interval; exhaustive sign-flip "
                f"sensitivity p = {float(scheduler_row['p_sign_flip']):.3f}."
            ),
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 2.75), sharex=True)
    for axis, panel in zip(axes, panels, strict=True):
        axis.axvline(0, color="#6E7B87", linewidth=1.1)
        axis.errorbar(
            panel["point"],
            0,
            xerr=[
                [panel["point"] - panel["low"]],
                [panel["high"] - panel["point"]],
            ],
            fmt="o",
            color=panel["color"],
            capsize=5,
            linewidth=2.1,
            markersize=6,
        )
        axis.set_title(panel["title"], fontsize=9.5, fontweight="bold")
        axis.set_yticks([0], labels=[panel["label"]])
        axis.set_ylim(-0.7, 0.7)
        axis.grid(axis="x", color=GRID, linewidth=0.7)
        axis.text(
            0.5,
            -0.38,
            panel["footer"],
            transform=axis.transAxes,
            color=NAVY,
            fontsize=7.2,
            ha="center",
            va="top",
        )
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)

    axes[0].set_xlim(-2, 11)
    fig.supxlabel(
        "Difference in scheduled-task success (percentage points)",
        fontsize=8.5,
        y=0.03,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.32, wspace=0.28)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.results_dir.resolve(), args.output.resolve())
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
