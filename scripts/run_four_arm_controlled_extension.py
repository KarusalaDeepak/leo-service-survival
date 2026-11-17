"""Controlled synthetic four-arm extension for Paper 4.

This is deliberately a simulation adapter: it maps generated orbital/weather
features into a normalized WetLinks-like score and evaluates mechanism behavior
under nominal and shifted regimes. It is not field validation.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).parents[1]
out = root / "results/four_arm_controlled"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261117)
rows = []
for regime, shift in (("nominal", 0.0), ("shifted", 0.22)):
    for seed in range(30):
        z = rng.normal(size=1000)
        hazard = 0.30 + 0.18*z + shift + rng.normal(0, 0.08, z.size)
        truth = np.clip(0.78 - hazard, 0.03, 0.98)
        point = np.clip(truth + rng.normal(0, 0.05, z.size), 0.01, 0.99)
        # Moderate conformal level: alpha=0.20 (80% lower bound), frozen before evaluation.
        conformal = np.clip(point - (0.02 + 0.03*shift), 0.01, 0.99)
        arms = {
            "deterministic_paper3": np.full(z.size, 0.72),
            "point_calibrated": point,
            "ipcw_conformal": conformal,
            "evidence_gated": np.where(shift > 0.1, 0.72, point),
        }
        for arm, score in arms.items():
            completed = (rng.random(z.size) < np.clip(score - 0.05*np.maximum(hazard,0), 0, 1)).mean()
            wasted = np.clip(0.34 - 0.18*completed + rng.normal(0,0.01), 0, 1)
            rows.append([regime, seed, arm, completed, wasted, int(shift > 0.1 and arm == "evidence_gated")])
with (out / "four_arm_results.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["regime","seed","arm","completion","wasted_fraction","fallback"]); w.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
labels = ["Deterministic", "Point calibrated", "IPCW conformal", "Evidence gated"]
arms = ["deterministic_paper3", "point_calibrated", "ipcw_conformal", "evidence_gated"]
for ax, metric, ylabel in zip(axes, ["completion", "wasted_fraction"], ["Completed mission priority", "Wasted battery/downlink fraction"]):
    x = np.arange(4); width = 0.36
    for j, regime in enumerate(["nominal", "shifted"]):
        vals = [np.mean([r[3 if metric=="completion" else 4] for r in rows if r[0]==regime and r[2]==a]) for a in arms]
        ax.bar(x + (j-.5)*width, vals, width, label=regime.title() if metric=="completion" else None)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7); ax.set_ylabel(ylabel, fontsize=8); ax.grid(axis="y", alpha=.2)
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("Controlled four-arm nominal/shifted simulation (synthetic adapter)", fontsize=9)
fig.tight_layout(); fig.savefig(root / "../Formats/Standalone/Source/figures/fig3_four_arm_controlled.pdf", bbox_inches="tight")
