import csv, gzip
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

src = Path(__file__).parents[1] / "results/reviewer_prediction_release/wetlinks_all_pipeline_horizon_predictions.csv.gz"
out = Path(__file__).parents[1] / "results/manuscript/wetlinks_horizon_curve_summary.csv"
fig = Path(__file__).parents[1] / "../Formats/Standalone/Source/figures/fig2_wetlinks_calibration.pdf"

bins = defaultdict(lambda: [0.0, 0.0, 0])
with gzip.open(src, "rt", newline="") as fh:
    for row in csv.DictReader(fh):
        if row["pipeline"] != "calibrated_competing_risk":
            continue
        h = int(row["horizon_bins"])
        bins[h][0] += float(row["predicted_survival"])
        if row["observed_survival_target"] != "":
            bins[h][1] += float(row["observed_survival_target"])
        bins[h][2] += 1

rows = []
for h in sorted(bins):
    pred, obs, n = bins[h]
    rows.append((h, h * 5.0, pred / n, obs / n, n))

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["horizon_bins", "horizon_minutes", "predicted_survival", "observed_survival", "rows"])
    w.writerows(rows)

x = [r[1] for r in rows]
y_pred = [r[2] for r in rows]
y_obs = [r[3] for r in rows]
fig.parent.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(5.8, 3.2))
plt.plot(x, y_pred, "o-", label="Calibrated prediction", color="#1f77b4")
plt.plot(x, y_obs, "s-", label="Observed target", color="#d62728")
plt.xlabel("Forecast horizon (min)")
plt.ylabel("Survival probability")
plt.ylim(0.7, 1.01)
plt.grid(alpha=0.25)
plt.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(fig, bbox_inches="tight")
