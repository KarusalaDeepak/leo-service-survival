"""Rebuilt-endpoint sensitivity, separate from the frozen empirical core.

Endpoint choices define a scientific question; they are never selected by test
score here.  Every estimable profile is retained, including unfavourable or
sparse results.  Raw burst summaries are not expanded into invented probes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import (
    CENSOR_REASON_ADMINISTRATIVE_HORIZON,
    WETLINKS_FEATURES,
    _cause_from_row,
    _episodes_from_regular_series,
    _time_features,
    calendar_block_split,
)
from .metrics import evaluate_survival_predictions, ipcw_survival_targets
from .models import (
    CalibratedSurvivalModel,
    DiscreteHazardModel,
    KaplanMeierModel,
    _expand_person_period,
)
from .parameters import CALIBRATION_C_GRID, FOCAL_HAZARD_SEARCH_GRID


@dataclass(frozen=True)
class EndpointAuditProfile:
    profile_id: str = "primary"
    bin_minutes: int = 5
    latency_limit_ms: float = 80.0
    loss_limit_pct: float = 1.0
    sustained_bad_bins: int = 2
    maximum_gap_bins: int = 2
    analysis_horizon_bins: int = 12
    contrast: str = "primary"


def registered_profiles() -> tuple[EndpointAuditProfile, ...]:
    """All registered one-factor levels, plus three stated interactions.

    The coarse time-matched profile is important: ten-minute bins with two-bin
    persistence and H=12 otherwise change both persistence and forecast time.
    Strict/lenient interactions check combined endpoint definitions rather than
    treating their consequences as independent additive effects.
    """
    baseline = EndpointAuditProfile()
    profiles = [baseline]
    alternatives = {
        "bin_minutes": (1, 10),
        "latency_limit_ms": (70.0, 90.0),
        "loss_limit_pct": (0.5, 2.0),
        "sustained_bad_bins": (1, 3),
        "maximum_gap_bins": (1, 3),
        "analysis_horizon_bins": (6, 9),
    }
    for parameter, values in alternatives.items():
        for value in values:
            profiles.append(replace(
                baseline,
                profile_id=f"{parameter}_{value:g}",
                contrast=f"one_factor:{parameter}",
                **{parameter: value},
            ))
    profiles.extend((
        replace(baseline, profile_id="joint_strict", latency_limit_ms=70.0,
                loss_limit_pct=0.5, sustained_bad_bins=1,
                contrast="interaction:strict_limits_short_persistence"),
        replace(baseline, profile_id="joint_lenient", latency_limit_ms=90.0,
                loss_limit_pct=2.0, sustained_bad_bins=3,
                contrast="interaction:lenient_limits_long_persistence"),
        replace(baseline, profile_id="bin10_time_matched", bin_minutes=10,
                sustained_bad_bins=1, analysis_horizon_bins=6,
                contrast="interaction:10_min_persistence_60_min_forecast"),
    ))
    return tuple(profiles)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_cadence_audit(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        raw = pd.read_csv(path, usecols=["site_name", "timestamp_start", "timestamp_end"])
        starts = pd.to_datetime(raw["timestamp_start"], errors="coerce", utc=True)
        ends = pd.to_datetime(raw["timestamp_end"], errors="coerce", utc=True)
        gaps = starts.dropna().drop_duplicates().sort_values().diff().dt.total_seconds().dropna()
        durations = (ends - starts).dt.total_seconds()
        rows.append({
            "source_file": str(path.resolve()), "sha256": sha256_file(path),
            "site": str(raw["site_name"].dropna().iloc[0]),
            "rows": len(raw), "start_min": starts.min().isoformat(),
            "start_max": starts.max().isoformat(),
            "minimum_start_gap_seconds": float(gaps.min()),
            "median_start_gap_seconds": float(gaps.median()),
            "median_burst_duration_seconds": float(durations.median()),
            "source_resolution": "irregular_burst_summary_not_continuous_packet_trace",
        })
    return pd.DataFrame(rows)


def cadence_support(profile: EndpointAuditProfile, cadence: pd.DataFrame) -> tuple[bool, str]:
    if cadence.empty or not np.isfinite(cadence["median_start_gap_seconds"]).all():
        return False, "not_estimable_missing_cadence_provenance"
    if (cadence["median_start_gap_seconds"] > profile.bin_minutes * 60).any():
        return False, "not_supported_by_source_cadence_no_upsampling"
    return True, "supported_aggregation_of_irregular_burst_summaries"


def prepare_regular_series(paths: list[Path], bin_minutes: int) -> dict[str, pd.DataFrame]:
    """Match existing cleaning/aggregation; no interpolation or carry-forward."""
    result = {}
    usecols = ["site_name", "timestamp_start", "ping_avg", "ping_worst", "ping_stddev",
               "ping_packet_loss", "download", "upload", "temp", "humidity", "rain",
               "windspeed", "barom"]
    for path in paths:
        raw = pd.read_csv(path, usecols=usecols)
        raw["timestamp"] = pd.to_datetime(raw["timestamp_start"], errors="coerce", utc=True)
        numeric = [name for name in usecols if name not in {"site_name", "timestamp_start"}]
        raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
        for column, (lower, upper) in {
            "temp": (-80, 60), "humidity": (0, 100), "rain": (0, 500),
            "windspeed": (0, 100), "barom": (800, 1100),
        }.items():
            raw.loc[~raw[column].between(lower, upper), column] = np.nan
        site = str(raw["site_name"].dropna().iloc[0])
        regular = raw.set_index("timestamp").resample(f"{bin_minutes}min").agg(
            latency_mean_ms=("ping_avg", "mean"), latency_std_ms=("ping_stddev", "mean"),
            latency_max_ms=("ping_worst", "max"), packet_loss_pct=("ping_packet_loss", "mean"),
            download_bps=("download", "mean"), upload_bps=("upload", "mean"),
            temperature_c=("temp", "mean"), humidity_pct=("humidity", "mean"),
            rain_mm=("rain", "mean"), wind_mps=("windspeed", "mean"), pressure_hpa=("barom", "mean"),
        ).reset_index()
        regular["download_mbps"] = regular["download_bps"] / 1e6
        regular["upload_mbps"] = regular["upload_bps"] / 1e6
        regular["observed"] = regular["latency_mean_ms"].notna() & regular["packet_loss_pct"].notna()
        regular["latency_roll3"] = regular["latency_mean_ms"].rolling(3, min_periods=1).mean()
        regular["loss_roll3"] = regular["packet_loss_pct"].rolling(3, min_periods=1).mean()
        regular["rain_latency_interaction"] = regular["rain_mm"].fillna(0) * regular["latency_mean_ms"]
        result[site] = _time_features(regular, "timestamp")
    return result


def rebuild_episodes(regular_by_site: dict[str, pd.DataFrame], profile: EndpointAuditProfile) -> pd.DataFrame:
    rows = []
    for site, original in regular_by_site.items():
        regular = original.copy()
        regular["usable"] = (regular["observed"] & regular["latency_mean_ms"].le(profile.latency_limit_ms)
                             & regular["packet_loss_pct"].le(profile.loss_limit_pct))
        regular["cause"] = regular.apply(_cause_from_row, axis=1,
                                         latency_limit_ms=profile.latency_limit_ms,
                                         loss_limit_pct=profile.loss_limit_pct)
        rows.extend(_episodes_from_regular_series(
            regular, site=site, timestamp_column="timestamp", observed_column="observed",
            usable_column="usable", cause_column="cause", feature_columns=WETLINKS_FEATURES,
            bin_seconds=60 * profile.bin_minutes, sustained_bad_bins=profile.sustained_bad_bins,
            max_gap_bins=profile.maximum_gap_bins,
        ))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["start_time", "site"]).reset_index(drop=True)


def truncate_confirmation_aware(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Preserve the exact primary truncation convention and its full lookahead."""
    out = frame.copy().reset_index(drop=True)
    resolution = pd.to_datetime(out["label_resolution_time"], utc=True).copy()
    beyond = out["duration_bins"].gt(horizon)
    resolution.loc[beyond] = (
        pd.to_datetime(out.loc[beyond, "start_time"], utc=True)
        + pd.to_timedelta(out.loc[beyond, "bin_seconds"]
                           * (horizon + out.loc[beyond, "sustained_bad_bins"] - 1), unit="s")
    )
    out["analysis_label_resolution_time"] = resolution
    out["analysis_label_end_time"] = resolution
    out.loc[beyond, "duration_bins"] = horizon + 1
    out.loc[beyond, "event"] = 0
    out.loc[beyond, "cause"] = "censored"
    out.loc[beyond, "censor_reason"] = CENSOR_REASON_ADMINISTRATIVE_HORIZON
    return out


def confirmation_safe_calendar_split(frame: pd.DataFrame) -> tuple[tuple[pd.DataFrame, ...], list[dict[str, Any]]]:
    """Whole-week embargo plus explicit purge of labels reaching later roles."""
    raw = calendar_block_split(frame, embargo_blocks=1)
    parts = [part.copy() for part in raw]
    audit = []
    for index, name in enumerate(("fit", "calibration", "test")):
        before = len(parts[index])
        cutoff = parts[index + 1]["start_time"].min() if index < 2 else None
        if cutoff is not None:
            keep = parts[index]["analysis_label_resolution_time"].lt(cutoff)
            parts[index] = parts[index].loc[keep].copy()
        audit.append({
            "role": name, "rows_before_resolution_purge": before,
            "rows_after_resolution_purge": len(parts[index]),
            "resolution_purge_rows": before - len(parts[index]),
            "next_role_start": cutoff.isoformat() if cutoff is not None else None,
            "blocks": sorted(parts[index]["calendar_block"].unique().tolist()),
            "block_embargo": raw[index].attrs.get("split_exclusion_audit", {}),
        })
        if parts[index].empty:
            raise ValueError(f"confirmation embargo removed all {name} rows")
    for earlier, later in zip(parts, parts[1:]):
        if earlier["analysis_label_resolution_time"].max() >= later["start_time"].min():
            raise AssertionError("confirmation label overlaps a later analysis role")
        if set(earlier["calendar_block"]) & set(later["calendar_block"]):
            raise AssertionError("calendar block shared across analysis roles")
    return tuple(parts), audit


class AuditHazardModel(DiscreteHazardModel):
    """Same focal hazard family, but no random person-period early-stop split.

    Complete-episode, embargoed development partitions perform all selection.
    Disabling the library's row-level stopping also avoids internal leakage.
    """
    def fit(self, frame: pd.DataFrame) -> "AuditHazardModel":
        self.encoder.fit(frame)
        x, y, weights, self.cause_to_class_ = _expand_person_period(
            frame, self.encoder, self.max_horizon, competing=True)
        if np.unique(y).size < 2:
            raise ValueError("no event class in fitting person-period data")
        self.model_ = HistGradientBoostingClassifier(
            **self.hyperparameters, random_state=self.random_state, early_stopping=False)
        self.model_.fit(x, y, sample_weight=weights)
        self.classes_ = np.asarray(self.model_.classes_, dtype=int)
        return self


def support_table(parts: tuple[pd.DataFrame, ...], horizon: int) -> pd.DataFrame:
    rows = []
    for role, part in zip(("fit", "calibration", "test"), parts):
        for h in range(1, horizon + 1):
            known, target, _ = ipcw_survival_targets(part, h)
            rows.append({
                "role": role, "horizon_bins": h, "episodes": len(part),
                "observed_events": int(part["event"].sum()),
                "known_at_horizon": int(known.sum()),
                "known_failures": int((known & (target == 0)).sum()),
                "known_survivors": int((known & (target == 1)).sum()),
                "unknown_at_horizon": int((~known).sum()),
                **{f"cause_{cause}": int(part["cause"].eq(cause).sum())
                   for cause in ("loss", "latency", "joint", "censored")},
            })
    return pd.DataFrame(rows)


def fit_and_score_profile(frame: pd.DataFrame, profile: EndpointAuditProfile) -> dict[str, Any]:
    horizon = profile.analysis_horizon_bins
    horizons = tuple(range(1, horizon + 1))
    truncated = truncate_confirmation_aware(frame, horizon)
    outer, outer_audit = confirmation_safe_calendar_split(truncated)
    inner, inner_audit = confirmation_safe_calendar_split(outer[0])
    selections = []
    failures = []
    # Fit each configuration only once; every C uses the same frozen base.
    for config_index, config in enumerate(FOCAL_HAZARD_SEARCH_GRID):
        try:
            base = AuditHazardModel(WETLINKS_FEATURES, max_horizon=horizon,
                                    competing=True, hyperparameters=config).fit(inner[0])
            for c in CALIBRATION_C_GRID:
                model = CalibratedSurvivalModel(base, horizons, calibration_c=c).calibrate(inner[1])
                scores = evaluate_survival_predictions(inner[2], model.predict_survival(inner[2], horizons), horizons)
                selections.append({"configuration_index": config_index,
                                   "calibration_c": c, **config,
                                   "development_integrated_brier": scores["integrated_brier"]})
        except (ValueError, FloatingPointError) as error:
            failures.append({"configuration_index": config_index, "error": str(error)})
    selection = pd.DataFrame(selections)
    if selection.empty or not np.isfinite(selection["development_integrated_brier"]).any():
        raise ValueError(f"no estimable development candidate: {failures}")
    selection = selection.sort_values(
        ["development_integrated_brier", "max_leaf_nodes", "l2_regularization", "calibration_c"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    winner = selection.iloc[0]
    selection["selected"] = False
    selection.loc[0, "selected"] = True
    config = dict(FOCAL_HAZARD_SEARCH_GRID[int(winner["configuration_index"])])
    base = AuditHazardModel(WETLINKS_FEATURES, max_horizon=horizon,
                            competing=True, hyperparameters=config).fit(outer[0])
    model = CalibratedSurvivalModel(base, horizons, calibration_c=float(winner["calibration_c"])).calibrate(outer[1])
    km = KaplanMeierModel().fit(outer[0])
    raw_prediction = base.predict_survival(outer[2], horizons)
    prediction = model.predict_survival(outer[2], horizons)
    km_prediction = km.predict_survival(outer[2], horizons)
    scores = {
        "calibrated": evaluate_survival_predictions(outer[2], prediction, horizons),
        "uncalibrated": evaluate_survival_predictions(outer[2], raw_prediction, horizons),
        "kaplan_meier": evaluate_survival_predictions(outer[2], km_prediction, horizons),
    }
    support = support_table(outer, horizon)
    calibration_support = support.loc[support.role.eq("calibration")].copy()
    calibration_support["calibration_fitted"] = [model.calibrators_[h] is not None for h in horizons]
    test_predictions = outer[2][["episode_id", "site", "calendar_block", "duration_bins", "event", "cause"]].copy()
    for j, h in enumerate(horizons):
        test_predictions[f"survival_h{h}"] = prediction[:, j]
        test_predictions[f"km_h{h}"] = km_prediction[:, j]
    return {"scores": scores, "selection": selection, "support": support,
            "calibration_support": calibration_support, "test_predictions": test_predictions,
            "split_audit": {"outer": outer_audit, "inner_development": inner_audit},
            "selection_failures": failures,
            "selected_configuration": config,
            "selected_calibration_c": float(winner["calibration_c"])}


def episode_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["episode_id", "site", "start_time", "duration_bins", "event", "cause",
               "censor_reason", "label_resolution_time", *WETLINKS_FEATURES]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def profile_metadata(profile: EndpointAuditProfile) -> dict[str, Any]:
    return {**asdict(profile),
            "forecast_minutes": profile.bin_minutes * profile.analysis_horizon_bins,
            "nominal_persistence_minutes": profile.bin_minutes * profile.sustained_bad_bins}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def finalize_export_schema(output: Path) -> None:
    """Add explicit count-stage labels without changing any fitted result.

    The original execution-source hashes are retained separately if an already
    completed v1 freeze receives this label-only export finalization.
    """
    table_path = output / "endpoint_parameter_sensitivity.csv"
    table = pd.read_csv(table_path)
    table["episodes_before_horizon"] = table["episodes"]
    table["events_before_horizon"] = table["events"]
    table["aggregate_count_stage"] = "rebuilt_untruncated_episodes_before_calendar_role_exclusion"
    for profile_id in table.loc[table.status.eq("executed"), "profile_id"]:
        support_path = output / profile_id / "support.csv"
        support = pd.read_csv(support_path)
        support["observed_events_within_horizon"] = support["observed_events"]
        support["count_stage"] = "after_configured_horizon_truncation_and_calendar_role_exclusion"
        support.to_csv(support_path, index=False)
        for role, group in support.groupby("role"):
            table.loc[table.profile_id.eq(profile_id), f"{role}_events_within_horizon"] = int(group.iloc[0].observed_events)
            table.loc[table.profile_id.eq(profile_id), f"{role}_episodes_after_exclusion"] = int(group.iloc[0].episodes)
    table.to_csv(table_path, index=False)
    combined_path = output / "all_profile_support.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        combined["observed_events_within_horizon"] = combined["observed_events"]
        combined["count_stage"] = "after_configured_horizon_truncation_and_calendar_role_exclusion"
        combined.to_csv(combined_path, index=False)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    primary = table.loc[table.profile_id.eq("primary")]
    summary["primary_result"] = primary.replace({np.nan: None}).to_dict(orient="records")[0] if len(primary) else None
    summary["count_semantics"] = {
        "aggregate_episodes_and_events": "before administrative horizon truncation and split exclusions",
        "support_observed_events": "within configured horizon, after calendar-role exclusions",
        "primary_raw_events": int(primary.iloc[0].events) if len(primary) else None,
        "independence": "one WetLinks acquisition family, two site streams; weeks are dependence blocks, not independent campaigns",
    }
    write_json(summary_path, summary)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("fit_run_code_sha256", manifest["code_sha256"])
    manifest["count_semantics"] = summary["count_semantics"]
    manifest["scope"]["inner_third_role"] = "development validation, not outer final test"
    manifest["export_schema_finalization"] = {
        "change": "explicit pre/post-horizon count labels only; no fits or numerical scores changed",
        "code_file": str(Path(__file__).resolve()),
        "code_sha256": sha256_file(Path(__file__)),
    }
    manifest["artifact_sha256"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path != manifest_path
    }
    write_json(manifest_path, manifest)
