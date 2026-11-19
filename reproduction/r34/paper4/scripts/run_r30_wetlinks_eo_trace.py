#!/usr/bin/env python3
"""Exploratory temporal recalibration and WetLinks-to-EO trace assignment.

No model is refitted from evaluation outcomes. The first two already-inspected
later WetLinks blocks fit a weighted isotonic update; the last two blocks score
it. Recorded QoS outcomes are assigned by a fixed hash to EOS-Bench downlinks.
That assignment and the all-or-nothing QoS-to-byte rule are simulation
assumptions, NOT synchronized EO field observations or measured transfer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PH_D = ROOT.parents[1]
PAPER3 = Path(os.environ.get("PAPER3_CODE_ROOT", str(
    PH_D / "03_Paper_RACS_EO_Scheduling" / "Code_and_Manuscript")))
sys.path.insert(0, str(PAPER3 / "scripts"))
_metric_spec = importlib.util.spec_from_file_location(
    "paper4_metrics_without_model_dependencies", ROOT / "src/leo_service_survival/metrics.py"
)
_metric_module = importlib.util.module_from_spec(_metric_spec)
assert _metric_spec and _metric_spec.loader
_metric_spec.loader.exec_module(_metric_module)
ipcw_survival_targets = _metric_module.ipcw_survival_targets
import matched_scheduler as ms  # noqa: E402

DEVELOPMENT_BLOCKS = ("2024-W07", "2024-W08")
EVALUATION_BLOCKS = ("2024-W09", "2024-W10")
HORIZONS = tuple(range(1, 13))
POLICIES = ("earliest", "margin", "raw", "frozen_logistic", "temporal_isotonic", "lower_envelope")
SEEDS = tuple(range(20))


def weighted_pav(x: np.ndarray, y: np.ndarray, weight: np.ndarray):
    """Weighted increasing step-function isotonic fit with deterministic ties."""
    order = np.argsort(x, kind="mergesort")
    xs, ys, ws = x[order], y[order], weight[order]
    good = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ws) & (ws > 0)
    xs, ys, ws = xs[good], ys[good], ws[good]
    if not len(xs):
        raise ValueError("no positive-weight calibration rows")
    # Combine identical predictions before PAV; each distinct score has one
    # weighted target, so repeated rows cannot depend on input order.
    ux, inverse = np.unique(xs, return_inverse=True)
    total_w = np.bincount(inverse, weights=ws)
    total_y = np.bincount(inverse, weights=ws * ys)
    blocks = []  # [start_index, end_index, weight, weighted_target_sum]
    for j in range(len(ux)):
        blocks.append([j, j, float(total_w[j]), float(total_y[j])])
        while len(blocks) >= 2:
            a, b = blocks[-2:]
            if a[3] / a[2] <= b[3] / b[2] + 1e-14:
                break
            blocks[-2:] = [[a[0], b[1], a[2] + b[2], a[3] + b[3]]]
    fitted = np.empty(len(ux), dtype=float)
    for left, right, w, s in blocks:
        fitted[left : right + 1] = s / w
    return ux, np.clip(fitted, 0, 1)


def isotonic_predict(knots: np.ndarray, values: np.ndarray, x: np.ndarray) -> np.ndarray:
    # Step interpolation is explicit; no unavailable spline/extrapolation fit.
    index = np.searchsorted(knots, x, side="right") - 1
    return values[np.clip(index, 0, len(values) - 1)]


def load_frozen_predictions(path: Path) -> pd.DataFrame:
    keep = {
        "competing_risk_uncalibrated": "raw",
        "calibrated_competing_risk": "frozen_logistic",
    }
    frame = pd.read_csv(path)
    frame = frame[frame["pipeline"].isin(keep)].copy()
    frame["pipeline"] = frame["pipeline"].map(keep)
    if set(frame["analysis_block"]) != set(DEVELOPMENT_BLOCKS + EVALUATION_BLOCKS):
        raise ValueError("unexpected analysis blocks; frozen protocol does not apply")
    keys = ["test_row_index", "horizon_bins"]
    meta = frame[frame["pipeline"] == "raw"].copy()
    if meta.duplicated(keys).any():
        raise ValueError("duplicate raw prediction key")
    aligned = frame.pivot(index=keys, columns="pipeline", values="predicted_survival")
    meta = meta.drop(columns=["pipeline", "predicted_survival"])
    meta = meta.merge(aligned.reset_index(), on=keys, validate="one_to_one")
    if meta[["raw", "frozen_logistic"]].isna().any().any():
        raise ValueError("missing aligned model predictions")
    return meta.sort_values(keys).reset_index(drop=True)


def calibrate_and_score(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    out["temporal_isotonic"] = np.nan
    out["recent_rate"] = np.nan
    fit_rows = []
    for h in HORIZONS:
        mask = out["horizon_bins"].eq(h)
        part = out.loc[mask]
        train = part[part["analysis_block"].isin(DEVELOPMENT_BLOCKS)]
        known, target, weights = ipcw_survival_targets(train, h)
        events = int(np.sum(known & (target == 0)))
        recent_rate = float(np.sum(weights * target) / np.sum(weights))
        out.loc[mask, "recent_rate"] = recent_rate
        if known.sum() < 25 or events < 5:
            # The support rule was fixed before inspecting evaluation outcomes.
            predicted = part["frozen_logistic"].to_numpy(float)
            status, knots, levels = "frozen_logistic_fallback", 0, 0
        else:
            x, y = weighted_pav(
                train.loc[known, "raw"].to_numpy(float), target[known], weights[known]
            )
            predicted = isotonic_predict(x, y, part["raw"].to_numpy(float))
            status, knots, levels = "isotonic", len(x), len(np.unique(y))
        out.loc[mask, "temporal_isotonic"] = predicted
        fit_rows.append({"horizon_bins": h, "fit_episodes": len(train),
                         "fit_known": int(known.sum()), "fit_events": events,
                         "mapping": status, "distinct_raw_knots": knots,
                         "isotonic_levels": levels, "recent_rate_fit": recent_rate})
    # Horizon-monotone survival is required even for separately fitted maps.
    for block in DEVELOPMENT_BLOCKS + EVALUATION_BLOCKS:
        loc = out["analysis_block"].eq(block)
        wide = out.loc[loc].pivot(index="test_row_index", columns="horizon_bins",
                                  values="temporal_isotonic").reindex(columns=HORIZONS)
        projected = np.minimum.accumulate(wide.to_numpy(float), axis=1)
        lookup = {(rid, h): projected[i, j] for i, rid in enumerate(wide.index)
                  for j, h in enumerate(HORIZONS)}
        out.loc[loc, "temporal_isotonic"] = [lookup[(rid, h)] for rid, h in
            zip(out.loc[loc, "test_row_index"], out.loc[loc, "horizon_bins"])]
    out["lower_envelope"] = np.minimum(out["frozen_logistic"], out["temporal_isotonic"])
    out["recent_rate"] = np.minimum.accumulate(
        out.pivot(index="test_row_index", columns="horizon_bins", values="recent_rate")
           .reindex(columns=HORIZONS).to_numpy(float), axis=1
    )[out["test_row_index"].to_numpy(int), out["horizon_bins"].to_numpy(int) - 1]
    evaluation = out[out["analysis_block"].isin(EVALUATION_BLOCKS)]
    rows = []
    block_rows = []
    for h in HORIZONS:
        part = evaluation[evaluation["horizon_bins"].eq(h)]
        known, target, weights = ipcw_survival_targets(part, h)
        observed = float(np.sum(weights * target) / np.sum(weights))
        for name in ("raw", "frozen_logistic", "temporal_isotonic", "lower_envelope", "recent_rate"):
            prediction = part[name].to_numpy(float)
            rows.append({"horizon_bins": h, "horizon_minutes": 5 * h,
                         "method": name, "evaluation_episodes": len(part),
                         "evaluable": int(known.sum()), "events": int(np.sum(known & (target == 0))),
                         "observed_survival_ipcw": observed,
                         "mean_prediction": float(np.mean(prediction)),
                         "prediction_std": float(np.std(prediction)),
                         "prediction_min": float(np.min(prediction)),
                         "prediction_max": float(np.max(prediction)),
                         "observed_minus_predicted": observed - float(np.mean(prediction)),
                         "ipcw_brier": float(np.mean(weights * (target - prediction) ** 2))})
            for block in EVALUATION_BLOCKS:
                block_part = part[part["analysis_block"].eq(block)]
                bk, bt, bw = ipcw_survival_targets(block_part, h)
                bp = block_part[name].to_numpy(float)
                block_rows.append({"analysis_block": block, "horizon_bins": h,
                                   "method": name, "episodes": len(block_part),
                                   "evaluable": int(bk.sum()),
                                   "observed_survival_ipcw": float(np.sum(bw * bt) / np.sum(bw)),
                                   "mean_prediction": float(bp.mean()),
                                   "ipcw_brier": float(np.mean(bw * (bt - bp) ** 2))})
    return out, pd.DataFrame(rows).merge(pd.DataFrame(fit_rows), on="horizon_bins"), pd.DataFrame(block_rows)


def keyed_episode(seed: int, contact: tuple, eligible: pd.DataFrame) -> pd.Series:
    # This synthetic assignment is fixed across arms. It is not a physical
    # satellite-to-terminal match or an observed policy counterfactual.
    key = json.dumps([seed, *contact], separators=(",", ":"))
    index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % len(eligible)
    return eligible.iloc[index]


def plan_contact_keys(plan) -> tuple[tuple, ...]:
    return tuple(sorted({(sat, gs, int(b), int(e)) for sat, gs, b, e, _, _ in plan.deliveries}))


def run_one(instance, eligible: pd.DataFrame, seed: int, policy: str,
            *, horizon_seconds: int = 300) -> dict:
    ledger, accepted = ms.Ledger(), []
    completed_priority = 0.0
    successful_count = 0
    planned_priority = 0.0
    wasted_downlink_seconds = 0.0
    reserved_downlink_seconds = 0.0
    total_priority = float(sum(t["priority"] for t in instance["tasks"]))
    for task in ms.ordered_tasks(instance):
        plans = ms.candidate_plans(instance, task, ledger, single=False)
        if not plans:
            continue
        def outcomes(plan):
            records = []
            for contact in plan_contact_keys(plan):
                if contact[3] - contact[2] > horizon_seconds:
                    raise ValueError("contact exceeds evaluated service horizon")
                records.append(keyed_episode(seed, contact, eligible))
            return records
        if policy == "earliest":
            chosen = min(plans, key=lambda p: (p.completion, p.variant))
        elif policy == "margin":
            chosen = max(plans, key=lambda p: (p.margin, -p.completion, p.variant))
        else:
            chosen = max(plans, key=lambda p: (math.prod(float(r[policy]) for r in outcomes(p)),
                                               -p.completion, p.variant))
        ms.commit(chosen, ledger)
        accepted.append(chosen)
        records = outcomes(chosen)
        success = all(int(r["observed_survival_target"]) == 1 for r in records)
        planned_priority += task["priority"]
        if success:
            completed_priority += task["priority"]
            successful_count += 1
        seconds = sum(e - b for _, _, b, e in plan_contact_keys(chosen))
        reserved_downlink_seconds += seconds
        if not success:
            wasted_downlink_seconds += seconds
    ms.validate_schedule(instance, accepted)
    signature = hashlib.sha256(json.dumps(
        [(p.task_id, p.variant, plan_contact_keys(p)) for p in accepted],
        separators=(",", ":"),
    ).encode()).hexdigest()
    return {"seed": seed, "policy": policy, "tasks": len(instance["tasks"]),
            "planned": len(accepted), "successful_qos_proxy_tasks": successful_count,
            "planned_priority_fraction": planned_priority / total_priority,
            "qos_proxy_priority_fraction": completed_priority / total_priority,
            "wasted_downlink_fraction": wasted_downlink_seconds / max(reserved_downlink_seconds, 1),
            "assigned_contacts": sum(len(plan_contact_keys(p)) for p in accepted),
            "plan_signature_sha256": signature}


def run_trace_assignment(out: pd.DataFrame, instance_path: Path) -> pd.DataFrame:
    at_five = out[(out["horizon_bins"] == 1) &
                  out["analysis_block"].isin(EVALUATION_BLOCKS) &
                  out["evaluable_at_horizon"].eq(True)].sort_values("test_row_index")
    if len(at_five) < 100 or at_five["observed_survival_target"].isna().any():
        raise ValueError("insufficient evaluable five-minute outcome episodes")
    instance = ms.load_instance(instance_path, "combined")
    rows = []
    for seed in SEEDS:
        for policy in POLICIES:
            row = run_one(instance, at_five, seed, policy)
            row.update({"stress": "combined", "evaluable_trace_episodes": len(at_five),
                        "assignment": "SHA256(seed,sat,station,window) modulo complete-case episodes",
                        "outcome": "recorded five-minute QoS survival as assumed all-or-nothing delivery mask"})
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path,
        default=ROOT / "results/reviewer_prediction_release/wetlinks_all_pipeline_horizon_predictions.csv.gz")
    parser.add_argument("--instance", type=Path,
        default=PAPER3 / "data/external/eos_bench/eos_s1_overlay.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("use a new empty output directory; frozen results are never overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frozen_predictions(args.predictions)
    adjusted, calibration, calibration_blocks = calibrate_and_score(frame)
    simulation = run_trace_assignment(adjusted, args.instance)
    calibration.to_csv(args.output_dir / "calibration_horizons.csv", index=False)
    calibration_blocks.to_csv(args.output_dir / "calibration_blocks.csv", index=False)
    simulation.to_csv(args.output_dir / "trace_assignment_runs.csv", index=False)
    summary = simulation.groupby("policy", as_index=False).agg(
        mean_qos_proxy_priority=("qos_proxy_priority_fraction", "mean"),
        min_qos_proxy_priority=("qos_proxy_priority_fraction", "min"),
        max_qos_proxy_priority=("qos_proxy_priority_fraction", "max"),
        mean_wasted_downlink=("wasted_downlink_fraction", "mean"),
        mean_planned_priority=("planned_priority_fraction", "mean"))
    summary.to_csv(args.output_dir / "trace_assignment_summary.csv", index=False)
    manifest = {"protocol": "R30 exploratory; recent-rate, block and matched-margin checks added after an initial pilot; no confirmatory method selection",
                "calibration_development_blocks": DEVELOPMENT_BLOCKS,
                "evaluation_blocks": EVALUATION_BLOCKS,
                "isotonic_fit": "IPCW-weighted increasing PAV on raw predictions; >=25 known and >=5 events; horizon cumulative-min projection",
                "lower_envelope": "min(frozen logistic, temporal isotonic), not a confidence bound",
                "recent_rate": "IPCW observed survival from recalibration-development blocks only; post-pilot sanity baseline",
                "scheduler": "Paper 3 matched candidate_plans/commit/validate_schedule; combined-stress EOS-Bench 100-task overlay",
                "assignment_seeds": list(SEEDS), "trace_episode_pool": "complete cases at 5 min from evaluation blocks",
                "boundary": "unsynchronized assumed trace assignment; no EO bytes, RF, installed route or field completion measured",
                "inspection": "original later WetLinks blocks were inspected in prior manuscript revisions; exploratory only"}
    (args.output_dir / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(calibration[calibration.horizon_bins.eq(12)][["method", "observed_survival_ipcw", "mean_prediction", "ipcw_brier"]].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
