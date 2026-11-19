#!/usr/bin/env python3
"""Paired assignment-level inference, not independent-campaign inference."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def exact_signflip(difference):
    values = np.asarray(difference)
    values = values[np.abs(values) > 1e-12]
    if not len(values):
        return 1.0
    # Symmetry permits fixing the first sign positive.
    sums = np.array([values[0]])
    for x in values[1:]:
        sums = np.concatenate((sums + x, sums - x))
    return float(np.mean(np.abs(sums) >= abs(values.sum()) - 1e-12))


def holm(values):
    p = np.asarray(values)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def run(path):
    frame = pd.read_csv(path / "runs.csv")
    rng = np.random.default_rng(343434)
    rows = []
    for (load, overhead), group in frame.groupby(["load", "overhead"]):
        for metric in ("priority", "goodput"):
            wide = group.pivot(index="seed", columns="policy", values=metric)
            for baseline in ("edf", "edf_5min", "frozen"):
                d = (wide["isotonic"] - wide[baseline]).to_numpy()
                boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                rows.append(dict(load=load, overhead=overhead, metric=metric,
                                 candidate="isotonic", baseline=baseline,
                                 mean_delta=d.mean(), ci_low=lo, ci_high=hi,
                                 exact_signflip_p=exact_signflip(d),
                                 positive=int((d > 1e-12).sum()),
                                 negative=int((d < -1e-12).sum()),
                                 tied=int((abs(d) <= 1e-12).sum())))
    results = pd.DataFrame(rows)
    results["holm_p"] = np.nan
    for metric, part in results.groupby("metric"):
        results.loc[part.index, "holm_p"] = holm(part.exact_signflip_p)
    results.to_csv(path / "paired_inference.csv", index=False)
    report = dict(units="20 paired assignment seeds conditional on fixed acquired scaffold and trace pool",
                  bootstrap="10000 paired seed resamples; nominal percentile 95% interval",
                  tests="two-sided exact sign flips, symmetric-difference null assumption",
                  multiplicity="Holm across 27 primary priority contrasts; separate 27 goodput contrasts",
                  status="exploratory previously inspected source data; not campaign-level confirmation")
    (path / "inference_scope.json").write_text(json.dumps(report, indent=2) + "\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("result_dir", type=Path)
    run(p.parse_args().result_dir)
