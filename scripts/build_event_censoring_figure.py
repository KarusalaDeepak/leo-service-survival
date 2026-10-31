#!/usr/bin/env python3
"""Rebuild Fig. 2: event assignment, confirmation, and censoring paths.

The figure is intentionally self-contained so that revising it cannot overwrite
any other manuscript figure.  Run from any directory with::

    python3 scripts/build_event_censoring_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "figures" / "event_censoring.png"

NAVY = "#17365D"
TEAL = "#2A9D8F"
TEAL_LIGHT = "#D9F0EC"
RED = "#C74440"
RED_LIGHT = "#F7DAD8"
BLUE = "#3976A8"
GRAY = "#68727D"
GRAY_LIGHT = "#E8EBEE"
GOLD = "#B07A00"


def draw_bin(ax, x0: float, y: float, state: str) -> None:
    """Draw one five-minute observation bin."""
    styles = {
        "good": dict(facecolor=TEAL_LIGHT, edgecolor=TEAL, hatch=None),
        "bad": dict(facecolor=RED_LIGHT, edgecolor=RED, hatch=None),
        "missing": dict(facecolor="white", edgecolor=GRAY, hatch="////"),
        "unknown": dict(facecolor=GRAY_LIGHT, edgecolor="#B4BBC2", hatch=None),
    }
    style = styles[state]
    ax.add_patch(
        Rectangle(
            (x0, y - 0.145),
            5,
            0.29,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=0.9,
            hatch=style["hatch"],
            zorder=2,
        )
    )


def build_figure(output: Path = OUTPUT) -> None:
    """Render the four observation paths used by the manuscript definition."""
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.15, 2.65))
    ax.set_xlim(-10.8, 31.2)
    ax.set_ylim(-0.54, 4.48)
    ax.axis("off")

    ax.text(
        -10.65,
        4.24,
        "Two-bin event rule for a 30-min task",
        color=NAVY,
        fontsize=8.9,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ax.text(
        30.0,
        4.24,
        "event time = first bad-bin edge; label available after confirmation",
        color=GRAY,
        fontsize=6.65,
        ha="right",
        va="center",
    )

    rows = [
        (
            3.02,
            "Confirmed event",
            ["good", "good", "good", "bad", "bad", "unknown"],
            "onset",
        ),
        (
            2.10,
            "Later observation missing",
            ["good", "good", "good", "missing", "unknown", "unknown"],
            "gap",
        ),
        (
            1.18,
            "Confirmation missing",
            ["good", "good", "good", "bad", "missing", "unknown"],
            "unresolved",
        ),
        (
            0.26,
            "Known through $d$",
            ["good", "good", "good", "good", "good", "good"],
            "complete",
        ),
    ]

    for y, label, states, kind in rows:
        ax.text(
            -10.65,
            y,
            label,
            color=NAVY,
            fontsize=7.15,
            fontweight="bold",
            ha="left",
            va="center",
        )
        for bin_index, state in enumerate(states):
            draw_bin(ax, 5 * bin_index, y, state)

        if kind == "onset":
            # Both bins 4 and 5 violate the same limit.  Keep their red fill
            # explicit because the event marker at their shared edge can hide
            # the bin boundary at manuscript scale.
            ax.text(17.5, y, "first bad", color=RED, fontsize=5.55, fontweight="bold", ha="center", va="center", zorder=4)
            ax.text(22.5, y, "same-limit bad", color=RED, fontsize=5.35, fontweight="bold", ha="center", va="center", zorder=4)
            ax.plot(20, y, marker="X", markersize=6.7, markeredgewidth=1.1, color=RED, zorder=5)
            ax.annotate(
                "onset/event: 20 min",
                xy=(20, y + 0.02),
                xytext=(18.6, y + 0.34),
                fontsize=5.95,
                color=RED,
                ha="right",
                va="bottom",
                arrowprops=dict(arrowstyle="-", color=RED, linewidth=0.75),
            )
            ax.plot(25, y, marker="o", markersize=4.7, markerfacecolor="white", markeredgecolor=GOLD, zorder=5)
            ax.annotate(
                "confirmation: 25 min",
                xy=(25, y + 0.02),
                xytext=(26.0, y + 0.34),
                fontsize=5.95,
                color=GOLD,
                ha="left",
                va="bottom",
                arrowprops=dict(arrowstyle="-", color=GOLD, linewidth=0.75),
            )
        elif kind == "gap":
            ax.plot(15, y, marker="|", markersize=11, markeredgewidth=1.7, color=BLUE, zorder=5)
            ax.text(15.38, y + 0.23, "gap censor: 15 min", color=BLUE, fontsize=6.15, ha="left")
        elif kind == "unresolved":
            ax.plot(20, y, marker="|", markersize=11, markeredgewidth=1.7, color=BLUE, zorder=5)
            ax.text(20.38, y + 0.23, "censor: 20 min", color=BLUE, fontsize=6.15, ha="left")
            ax.text(25.0, y - 0.31, "confirmation unavailable", color=GRAY, fontsize=5.9, ha="center")
        else:
            ax.plot(30, y, marker="o", markersize=4.7, markerfacecolor="white", markeredgecolor=TEAL, zorder=5)
            ax.text(29.65, y + 0.23, "known to 30 min", color=TEAL, fontsize=6.15, ha="right")

    # Five-minute boundaries and a compact time axis.
    for tick in range(0, 31, 5):
        ax.plot([tick, tick], [-0.18, -0.08], color=GRAY, linewidth=0.7)
        ax.text(tick, -0.25, str(tick), color=GRAY, fontsize=6.2, ha="center", va="top")
    ax.plot([0, 30], [-0.13, -0.13], color=GRAY, linewidth=0.7)
    ax.text(15, -0.48, "minutes after task start", color=GRAY, fontsize=6.45, ha="center", va="top")

    # Legend uses the same visual encoding as the rows.
    legend_y = 3.82
    for x, state, text in [
        (0.1, "good", "acceptable"),
        (8.15, "bad", "limit violated"),
        (17.15, "missing", "missing"),
        (24.25, "unknown", "not used"),
    ]:
        draw_bin(ax, x, legend_y, state)
        # Legend swatches are narrower than observation bins.
        ax.patches[-1].set_width(1.15)
        ax.patches[-1].set_height(0.20)
        ax.patches[-1].set_y(legend_y - 0.10)
        ax.text(x + 1.45, legend_y, text, fontsize=5.95, color=GRAY, va="center", ha="left")

    fig.subplots_adjust(left=0.015, right=0.995, top=0.985, bottom=0.08)
    fig.savefig(output, dpi=320, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
    print(f"Wrote {OUTPUT}")
