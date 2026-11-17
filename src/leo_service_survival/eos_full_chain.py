"""Synthetic, trace-grounded EO full-chain experiment.

The geometry and mission opportunities in this module are parsed from an
EOS-Bench scenario.  Everything after the observation opportunity (gateway,
RF state, handover risk, downlink capacity, installation and byte delivery) is
an explicitly declared stochastic overlay.  Consequently, outputs from this
module are always labelled ``synthetic_simulation`` and must never be described
as operational or field validation.

The simulator uses common random numbers and a common feasible action set for
the two policy arms.  Decisions use only information available at the declared
decision cut-off; realized outages, byte rates and product validity are sampled
after an action has been installed and are never supplied to a policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
import math
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .integration import (
    FullChainAudit,
    build_full_chain_ledger,
    summarize_full_chain_completion,
    validate_full_chain_bundle,
)
from .gating import (
    EvidenceGate,
    FrozenMahalanobisDetector,
    GateCriteria,
    GateEvidence,
)
from .eos_full_chain_predictor import (
    FullChainCompletionPredictor,
    FullChainPredictorBundle,
    fit_full_chain_predictor,
)


EVIDENCE_TIER = "synthetic_simulation"
PROTOCOL_VERSION = "eos-full-chain-sim-v3-development-fitted"
POLICY_BASELINE = "edf_reactive"
POLICY_INTEGRATED = "integrated_risk_gated"


@dataclass(frozen=True)
class NetworkPathProfile:
    """A pre-decision distribution used to create a synthetic network path."""

    path_profile_id: str
    interface_type: str
    operator_id: str
    gateway_id: str
    mean_snr_db: float
    snr_std_db: float
    mean_rate_mbps: float
    rate_log_sd: float
    jitter_ms: float
    outage_probability: float
    handover_probability: float
    source: str = "declared_stochastic_overlay"


DEFAULT_NETWORK_PROFILES: tuple[NetworkPathProfile, ...] = (
    NetworkPathProfile(
        path_profile_id="primary_high_snr",
        interface_type="synthetic_ntn",
        operator_id="synthetic_operator_alpha",
        gateway_id="synthetic_gateway_alpha",
        mean_snr_db=14.0,
        snr_std_db=1.8,
        mean_rate_mbps=0.036,
        rate_log_sd=0.22,
        jitter_ms=22.0,
        outage_probability=0.20,
        handover_probability=0.25,
    ),
    NetworkPathProfile(
        path_profile_id="resilient_lower_snr",
        interface_type="synthetic_ntn",
        operator_id="synthetic_operator_beta",
        gateway_id="synthetic_gateway_beta",
        mean_snr_db=10.5,
        snr_std_db=1.4,
        mean_rate_mbps=0.028,
        rate_log_sd=0.15,
        jitter_ms=8.0,
        outage_probability=0.045,
        handover_probability=0.045,
    ),
)


@dataclass(frozen=True)
class EOSFullChainConfig:
    """Frozen experiment settings.

    The seeds are stochastic replications, not independent sites or operators.
    ``storage_stress_cap_bytes`` represents free storage remaining after a
    synthetic background workload; the EOS-Bench nominal capacity is retained
    separately in the resource table.
    """

    seeds: tuple[int, ...] = (
        20261101,
        20261102,
        20261103,
        20261104,
        20261105,
        20261106,
        20261107,
        20261108,
        20261109,
        20261110,
    )
    downlink_contact_s: float = 90.0
    telemetry_max_age_ms: float = 1_000.0
    storage_stress_cap_bytes: int = 350_000
    release_spread_hours: float = 4.0
    minimum_deadline_hours: float = 24.0
    priority_deadline_hours: float = 8.0
    deadline_jitter_hours: float = 30.0
    # Deprecated, inactive compatibility metadata retained for old configs.
    # The active domain screen is the fitted Mahalanobis quantile, not this z.
    gate_z_threshold: float = 2.75
    gate_reference_rows_per_profile: int = 2_000
    gate_reference_seed: int = 2026102701
    baseline_decision_ms: float = 2.0
    integrated_decision_ms: float = 8.0
    install_latency_ms: float = 20.0
    install_failure_probability: float = 0.005
    acquisition_failure_probability: float = 0.01
    product_invalid_probability: float = 0.01
    slew_guard_s: float = 10.0
    bootstrap_resamples: int = 2_000
    bootstrap_seed: int = 20261201
    predictor_training_rows: int = 6_000
    predictor_tuning_rows: int = 3_000
    predictor_calibration_rows: int = 3_000
    predictor_validation_rows: int = 3_000
    predictor_training_seed: int = 2026112101
    predictor_tuning_seed: int = 2026112102
    predictor_calibration_seed: int = 2026112103
    predictor_validation_seed: int = 2026112104
    predictor_c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if self.downlink_contact_s <= 0 or self.storage_stress_cap_bytes <= 0:
            raise ValueError("resource capacities must be positive")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        if self.gate_reference_rows_per_profile < 100:
            raise ValueError("gate_reference_rows_per_profile must be at least 100")
        predictor_seeds = (
            self.predictor_training_seed, self.predictor_tuning_seed,
            self.predictor_calibration_seed, self.predictor_validation_seed,
        )
        if len(set(predictor_seeds)) != 4 or (
            set(predictor_seeds) & (set(self.seeds) | {self.gate_reference_seed})
        ):
            raise ValueError("predictor partition seeds must be unique and disjoint from evaluation/gate seeds")
        if min(
            self.predictor_training_rows, self.predictor_tuning_rows,
            self.predictor_calibration_rows, self.predictor_validation_rows,
        ) < 100:
            raise ValueError("each synthetic predictor partition needs at least 100 rows")
        if not self.predictor_c_grid or any(
            not np.isfinite(value) or value <= 0 for value in self.predictor_c_grid
        ):
            raise ValueError("predictor regularization grid must contain positive finite values")


@dataclass(frozen=True)
class ParsedEOSScenario:
    scenario_id: str
    creation_time: pd.Timestamp
    end_time: pd.Timestamp
    tasks: pd.DataFrame
    contacts: pd.DataFrame
    satellites: pd.DataFrame
    source_sha256: str


@dataclass(frozen=True)
class EOSFullChainArtifacts:
    tables: Mapping[str, pd.DataFrame]
    audit: FullChainAudit
    ledger: pd.DataFrame
    policy_summary: pd.DataFrame
    paired_seed_results: pd.DataFrame
    paired_inference: Mapping[str, Any]
    completion_status: Mapping[str, Any]
    run_manifest: Mapping[str, Any]
    predictor_tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    predictor_manifest: Mapping[str, Any] = field(default_factory=dict)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash_json(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _rng_for(seed: int, *parts: object) -> np.random.Generator:
    token = "|".join([str(seed), *(str(part) for part in parts)])
    digest = sha256(token.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def _timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def _ns(value: pd.Timestamp) -> int:
    return int(value.value)


def parse_eos_bench_scenario(path: str | Path) -> ParsedEOSScenario:
    """Parse missions, satellites and actual EOS-Bench observation windows."""

    source_path = Path(path)
    raw_bytes = source_path.read_bytes()
    payload = json.loads(raw_bytes)
    required = {"scenario_id", "metadata", "satellites", "missions", "observation_windows"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"EOS-Bench scenario is missing keys: {sorted(missing)}")

    scenario_id = str(payload["scenario_id"])
    metadata = payload["metadata"]
    creation = _timestamp(metadata["creation_time"])
    end_time = creation + pd.to_timedelta(float(metadata["duration"]), unit="s")

    satellite_rows: list[dict[str, Any]] = []
    sensor_rates: dict[str, float] = {}
    sensor_modes: dict[str, str] = {}
    for satellite in payload["satellites"]:
        satellite_id = str(satellite["id"])
        specs = satellite.get("satellite_specs", {})
        sensors = satellite.get("observation_capability", {}).get("sensors", [])
        if not sensors:
            raise ValueError(f"satellite {satellite_id!r} has no sensor definition")
        sensor = sensors[0]
        sensor_rates[satellite_id] = float(sensor.get("data_rate_Mbps", 0.1))
        sensor_modes[satellite_id] = str(sensor.get("sensor_mode", "unspecified"))
        satellite_rows.append(
            {
                "satellite_id": satellite_id,
                "orbital_type": str(satellite.get("orbital_type", "unknown")),
                "semi_major_axis_km": float(
                    satellite.get("orbital_params", {}).get("semi_major_axis_km", np.nan)
                ),
                "inclination_deg": float(
                    satellite.get("orbital_params", {}).get("inclination_deg", np.nan)
                ),
                "nominal_storage_bytes": int(
                    float(specs.get("max_data_storage_GB", 0.0)) * 1_000_000_000
                ),
                "stabilization_time_s": float(
                    satellite.get("maneuverability_capability", {}).get(
                        "stabilization_time_s", 0.0
                    )
                ),
                "sensor_data_rate_mbps": sensor_rates[satellite_id],
                "sensor_mode": sensor_modes[satellite_id],
            }
        )
    satellites = pd.DataFrame(satellite_rows)
    if satellites["satellite_id"].duplicated().any():
        raise ValueError("satellite identifiers are not unique")

    default_rate_mbps = float(np.median(list(sensor_rates.values())))
    task_rows: list[dict[str, Any]] = []
    for mission in payload["missions"]:
        mission_id = str(mission["id"])
        requirement = mission.get("observation_requirement", {})
        duration_s = float(requirement.get("duration_s", 1.0))
        if duration_s <= 0:
            raise ValueError(f"mission {mission_id!r} has non-positive duration")
        target = mission.get("target_location", {})
        task_rows.append(
            {
                "logical_task_id": f"{scenario_id}::{mission_id}",
                "mission_id": mission_id,
                "priority": float(mission.get("priority", 1.0)),
                "required_duration_s": duration_s,
                "required_bytes": int(math.ceil(default_rate_mbps * 1_000_000 * duration_s / 8.0)),
                "target_latitude": float(target.get("latitude", np.nan)),
                "target_longitude": float(target.get("longitude", np.nan)),
                "task_type": "earth_observation_product_delivery",
            }
        )
    tasks = pd.DataFrame(task_rows)
    if tasks["logical_task_id"].duplicated().any():
        raise ValueError("mission identifiers are not unique")

    contact_rows: list[dict[str, Any]] = []
    serial = 0
    for opportunity in payload["observation_windows"]:
        satellite_id = str(opportunity["satellite_id"])
        mission_id = str(opportunity["mission_id"])
        if satellite_id not in sensor_rates:
            raise ValueError(f"window references unknown satellite {satellite_id!r}")
        for window_index, window in enumerate(opportunity.get("time_windows", [])):
            serial += 1
            start = _timestamp(window["start_time"])
            end = _timestamp(window["end_time"])
            if end <= start:
                raise ValueError("observation window end must follow start")
            orbit = int(window.get("orbit_number", -1))
            contact_rows.append(
                {
                    "logical_task_id": f"{scenario_id}::{mission_id}",
                    "mission_id": mission_id,
                    "satellite_id": satellite_id,
                    "sensor_id": str(opportunity.get("sensor_id", "unknown_sensor")),
                    "contact_id_geometry": (
                        f"{satellite_id}::{mission_id}::orbit{orbit}::w{window_index}::n{serial}"
                    ),
                    "window_start_ns": _ns(start),
                    "window_end_ns": _ns(end),
                    "window_duration_s": float((end - start).total_seconds()),
                    "orbit_number": orbit,
                    "source_geometry": "EOS-Bench observation_windows",
                }
            )
    contacts = pd.DataFrame(contact_rows)
    if not contacts.empty and contacts["contact_id_geometry"].duplicated().any():
        raise ValueError("constructed contact identifiers are not unique")

    return ParsedEOSScenario(
        scenario_id=scenario_id,
        creation_time=creation,
        end_time=end_time,
        tasks=tasks,
        contacts=contacts,
        satellites=satellites,
        source_sha256=sha256(raw_bytes).hexdigest(),
    )


def load_network_profiles(path: str | Path | None = None) -> tuple[NetworkPathProfile, ...]:
    """Load optional aggregate ns-3 profiles or use the declared default overlay.

    An optional CSV must contain the fields of :class:`NetworkPathProfile`
    except ``source``.  Supplying ns-3-derived aggregates changes the overlay
    parameters but does not change the evidence label from synthetic simulation.
    """

    if path is None:
        return DEFAULT_NETWORK_PROFILES
    frame = pd.read_csv(path)
    required = {
        "path_profile_id",
        "interface_type",
        "operator_id",
        "gateway_id",
        "mean_snr_db",
        "snr_std_db",
        "mean_rate_mbps",
        "rate_log_sd",
        "jitter_ms",
        "outage_probability",
        "handover_probability",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"network profile CSV is missing: {sorted(missing)}")
    profiles = tuple(
        NetworkPathProfile(
            **{key: row[key] for key in required},
            source=str(row.get("source", "ns3_aggregate_profile_overlay")),
        )
        for row in frame.to_dict(orient="records")
    )
    if not profiles:
        raise ValueError("network profile CSV is empty")
    return profiles


def _seed_task_table(scenario: ParsedEOSScenario, seed: int, config: EOSFullChainConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenario_start_ns = _ns(scenario.creation_time)
    scenario_end_ns = _ns(scenario.end_time)
    for task in scenario.tasks.to_dict(orient="records"):
        rng = _rng_for(seed, task["logical_task_id"], "workload")
        release_h = float(rng.uniform(0.0, config.release_spread_hours))
        priority = float(task["priority"])
        deadline_h = (
            config.minimum_deadline_hours
            + max(0.0, 10.0 - priority) * config.priority_deadline_hours
            + float(rng.uniform(0.0, config.deadline_jitter_hours))
        )
        release_ns = scenario_start_ns + int(release_h * 3_600 * 1e9)
        deadline_ns = min(
            scenario_end_ns,
            scenario_start_ns + int(deadline_h * 3_600 * 1e9),
        )
        rows.append(
            {
                **task,
                "simulation_seed": int(seed),
                "release_time_ns": int(release_ns),
                "deadline_ns": int(max(deadline_ns, release_ns + int(3_600e9))),
                "workload_source": "synthetic deadline/release overlay on EOS-Bench mission",
            }
        )
    return pd.DataFrame(rows)


def _predictor_generation_profiles(
    profiles: tuple[NetworkPathProfile, ...],
) -> tuple[NetworkPathProfile, ...]:
    """Cross declared profile attributes without evaluation outcome selection.

    Crossing rate distributions, outage, handover and jitter prevents the
    development design from making those variables perfectly collinear merely
    because the two illustrative evaluation paths have fixed identities.  The
    support is still declared by the simulator, not learned from a real network.
    """

    rates = sorted({(profile.mean_rate_mbps, profile.rate_log_sd) for profile in profiles})
    outages = sorted({profile.outage_probability for profile in profiles})
    handovers = sorted({profile.handover_probability for profile in profiles})
    jitters = sorted({profile.jitter_ms for profile in profiles})
    return tuple(
        replace(
            profiles[0], path_profile_id=f"predictor_development_profile_{index}",
            mean_rate_mbps=rate[0], rate_log_sd=rate[1],
            outage_probability=outage, handover_probability=handover,
            jitter_ms=jitter, source="crossed_declared_simulator_profile_support",
        )
        for index, (rate, outage, handover, jitter) in enumerate(
            product(rates, outages, handovers, jitters)
        )
    )


def _generate_predictor_partition(
    scenario: ParsedEOSScenario,
    profiles: tuple[NetworkPathProfile, ...],
    *,
    partition: str,
    seed: int,
    rows: int,
    config: EOSFullChainConfig,
) -> pd.DataFrame:
    """Generate forecast-feasible single-action outcomes using the exact simulator."""

    required_values = scenario.tasks["required_bytes"].to_numpy(int)
    if not len(required_values):
        raise ValueError("predictor reference needs scenario task sizes")
    catalog = _predictor_generation_profiles(profiles)
    rng = _rng_for(seed, partition, "predictor_reference_telemetry")
    records: list[dict[str, Any]] = []
    attempts = 0
    while len(records) < rows:
        attempts += 1
        if attempts > rows * 100:
            raise RuntimeError("insufficient forecast-feasible predictor reference draws")
        profile = catalog[int(rng.integers(len(catalog)))]
        required_bytes = int(rng.choice(required_values))
        forecast_rate = float(
            profile.mean_rate_mbps * math.exp(rng.normal(0.0, profile.rate_log_sd))
        )
        capacity_bytes = int(forecast_rate * 1_000_000 / 8.0 * config.downlink_contact_s)
        if capacity_bytes < required_bytes:
            continue
        row_id = f"{partition}::seed{seed}::draw{len(records)}"
        selected = {
            "candidate_path_id": row_id,
            "forecast_capacity_ratio": capacity_bytes / required_bytes,
            "profile_mean_rate_mbps": profile.mean_rate_mbps,
            "profile_rate_log_sd": profile.rate_log_sd,
            "profile_outage_probability": profile.outage_probability,
            "forecast_handover_probability": float(np.clip(
                profile.handover_probability + rng.normal(0.0, 0.015), 0.0, 0.95
            )),
            "forecast_jitter_ms": float(max(
                0.1, profile.jitter_ms * math.exp(rng.normal(0.0, 0.18))
            )),
            "planned_downlink_duration_s": config.downlink_contact_s,
            "observation_end_ns": 0,
        }
        task = {
            "logical_task_id": row_id, "required_bytes": required_bytes,
            "deadline_ns": int((config.downlink_contact_s + 1.0) * 1e9),
        }
        outcome = _realize_outcome(selected=selected, task=task, seed=seed, config=config)
        records.append({
            **selected, "required_bytes": required_bytes, "path_available": True,
            "technical_completion": int(outcome["application_complete"]),
            "reference_row_id": row_id, "reference_seed": seed,
            "partition": partition, "development_profile_id": profile.path_profile_id,
            "evidence_tier": EVIDENCE_TIER,
        })
    return pd.DataFrame(records)


def prepare_full_chain_predictor(
    scenario: ParsedEOSScenario,
    profiles: tuple[NetworkPathProfile, ...],
    config: EOSFullChainConfig,
) -> FullChainPredictorBundle:
    """Freeze a generated-data predictor before any scheduling evaluation."""

    partitions = {
        role: _generate_predictor_partition(
            scenario, profiles, partition=role, seed=seed, rows=rows, config=config,
        )
        for role, seed, rows in (
            ("training", config.predictor_training_seed, config.predictor_training_rows),
            ("tuning", config.predictor_tuning_seed, config.predictor_tuning_rows),
            ("calibration", config.predictor_calibration_seed, config.predictor_calibration_rows),
            ("validation", config.predictor_validation_seed, config.predictor_validation_rows),
        )
    }
    bundle = fit_full_chain_predictor(partitions, c_grid=config.predictor_c_grid)
    manifest = {
        **bundle.manifest,
        "source_scenario_sha256": scenario.source_sha256,
        "development_profile_catalog": [asdict(profile) for profile in _predictor_generation_profiles(profiles)],
        "evaluation_seeds": list(config.seeds),
        "gate_reference_seed": config.gate_reference_seed,
        "predictor_and_evaluation_seeds_disjoint": True,
        "generator": "exact_eos_full_chain_realize_outcome_with_forecast_feasible_draws",
        "generator_assumptions": {
            "network_profiles": [asdict(profile) for profile in profiles],
            "downlink_contact_s": config.downlink_contact_s,
            "install_failure_probability": config.install_failure_probability,
            "acquisition_failure_probability": config.acquisition_failure_probability,
            "product_invalid_probability": config.product_invalid_probability,
            "realized_handover_rate_multiplier": 0.20,
        },
    }
    return FullChainPredictorBundle(model=bundle.model, tables=bundle.tables, manifest=manifest)


GATE_FEATURES: tuple[str, ...] = (
    "rf_snr_db",
    "forecast_rate_mbps",
    "forecast_jitter_ms",
    "forecast_handover_probability",
)


def _fit_simulation_domain_gate(
    profiles: tuple[NetworkPathProfile, ...], config: EOSFullChainConfig
) -> tuple[EvidenceGate, GateEvidence, str]:
    """Fit the frozen immediate guard on disjoint synthetic reference draws.

    This makes the controlled chain exercise the same data-driven Mahalanobis
    gate implementation used by the stress suite.  The gate's evidence-count
    requirement is deliberately disabled for this simulation-only execution;
    that does not authorize a field release and never changes the evidence tier.
    """

    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(config.gate_reference_seed)
    for profile in profiles:
        for _ in range(config.gate_reference_rows_per_profile):
            rows.append(
                {
                    "rf_snr_db": float(
                        rng.normal(profile.mean_snr_db, profile.snr_std_db)
                    ),
                    "forecast_rate_mbps": float(
                        profile.mean_rate_mbps
                        * math.exp(rng.normal(0.0, profile.rate_log_sd))
                    ),
                    "forecast_jitter_ms": float(
                        max(
                            0.1,
                            profile.jitter_ms * math.exp(rng.normal(0.0, 0.18)),
                        )
                    ),
                    "forecast_handover_probability": float(
                        np.clip(
                            profile.handover_probability
                            + rng.normal(0.0, 0.015),
                            0.0,
                            0.95,
                        )
                    ),
                }
            )
    reference = pd.DataFrame(rows)
    detector = FrozenMahalanobisDetector(GATE_FEATURES, quantile=0.99).fit(reference)
    criteria = GateCriteria(
        minimum_independent_campaigns=0,
        minimum_independent_families=0,
        minimum_paired_opportunity_groups=0,
        minimum_feature_coverage=1.0,
        maximum_ood_fraction=0.0,
        maximum_calibration_gap_upper=1.0,
        minimum_safety_effect_lower=-1.0,
        # This comparison is a deterministic simulation fixture, not a field
        # latency requirement.  The generic gate correctly rejects infinity;
        # use the exact declared virtual-clock delay instead of disabling its
        # numeric validity checks with a non-finite placeholder.
        maximum_control_loop_p95_ms=config.integrated_decision_ms + config.install_latency_ms,
        maximum_endpoint_effect_ci_half_width=1.0,
        validation_tier="proof_of_method",
    )
    evidence = GateEvidence(
        schema_passed=True,
        independent_campaigns=0,
        independent_families=0,
        paired_opportunity_groups=0,
        calibration_gap_upper=0.0,
        safety_effect_lower=0.0,
        control_loop_p95_ms=config.integrated_decision_ms + config.install_latency_ms,
        endpoint_effect_ci_half_width=0.0,
        validation_tier="proof_of_method",
    )
    gate = EvidenceGate(detector, criteria)
    gate_hash = _hash_json(
        {
            "features": GATE_FEATURES,
            "quantile": detector.quantile,
            "threshold": detector.threshold_,
            "reference_rows": len(reference),
            "reference_seed": config.gate_reference_seed,
            "criteria": asdict(criteria),
            "scope": "simulation_release_only_not_field_authorization",
        }
    )
    return gate, evidence, gate_hash


def _build_common_actions(
    scenario: ParsedEOSScenario,
    task_specs: pd.DataFrame,
    profiles: tuple[NetworkPathProfile, ...],
    seed: int,
    config: EOSFullChainConfig,
    domain_gate: EvidenceGate,
    gate_evidence: GateEvidence,
    completion_predictor: FullChainCompletionPredictor | None = None,
) -> pd.DataFrame:
    satellite_lookup = scenario.satellites.set_index("satellite_id").to_dict(orient="index")
    contacts_by_task = {
        key: frame.copy()
        for key, frame in scenario.contacts.groupby("logical_task_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for task in task_specs.to_dict(orient="records"):
        logical_id = task["logical_task_id"]
        task_contacts = contacts_by_task.get(logical_id, pd.DataFrame())
        if task_contacts.empty:
            task_contacts = pd.DataFrame(
                [
                    {
                        "logical_task_id": logical_id,
                        "mission_id": task["mission_id"],
                        "satellite_id": "NO_GEOMETRY",
                        "sensor_id": "NO_SENSOR",
                        "contact_id_geometry": f"NO_CONTACT::{task['mission_id']}",
                        "window_start_ns": task["deadline_ns"],
                        "window_end_ns": task["deadline_ns"],
                        "window_duration_s": 0.0,
                        "orbit_number": -1,
                        "source_geometry": "EOS-Bench: no observation window",
                    }
                ]
            )

        task_action_rows: list[dict[str, Any]] = []
        for contact in task_contacts.to_dict(orient="records"):
            satellite_id = str(contact["satellite_id"])
            satellite = satellite_lookup.get(
                satellite_id,
                {"nominal_storage_bytes": 0, "stabilization_time_s": config.slew_guard_s},
            )
            required_duration_s = float(task["required_duration_s"])
            observation_end_ns = int(
                int(contact["window_start_ns"]) + round(required_duration_s * 1e9)
            )
            geometry_feasible = bool(
                satellite_id != "NO_GEOMETRY"
                and float(contact["window_duration_s"]) >= required_duration_s
                and int(contact["window_start_ns"]) >= int(task["release_time_ns"])
                and observation_end_ns < int(task["deadline_ns"])
            )
            storage_rng = _rng_for(seed, logical_id, contact["contact_id_geometry"], "storage")
            nominal_storage = int(satellite.get("nominal_storage_bytes", 0))
            storage_available = int(
                min(
                    nominal_storage,
                    config.storage_stress_cap_bytes * float(storage_rng.uniform(0.45, 1.35)),
                )
            )
            for profile in profiles:
                rng = _rng_for(
                    seed,
                    logical_id,
                    contact["contact_id_geometry"],
                    profile.path_profile_id,
                    "telemetry",
                )
                snr_db = float(rng.normal(profile.mean_snr_db, profile.snr_std_db))
                rate_mbps = float(
                    profile.mean_rate_mbps * math.exp(rng.normal(0.0, profile.rate_log_sd))
                )
                jitter_ms = float(max(0.1, profile.jitter_ms * math.exp(rng.normal(0.0, 0.18))))
                handover_forecast = float(
                    np.clip(profile.handover_probability + rng.normal(0.0, 0.015), 0.0, 0.95)
                )
                path_available = bool(rng.random() >= 0.015)
                remaining_s = max(0.0, (int(task["deadline_ns"]) - observation_end_ns) / 1e9)
                downlink_duration_s = min(config.downlink_contact_s, remaining_s)
                capacity_bytes = int(max(0.0, rate_mbps * 1_000_000 / 8.0 * downlink_duration_s))
                required_bytes = int(task["required_bytes"])
                capacity_ratio = capacity_bytes / max(required_bytes, 1)
                z_snr = abs(snr_db - profile.mean_snr_db) / max(profile.snr_std_db, 1e-6)
                z_rate = abs(math.log(max(rate_mbps, 1e-12) / profile.mean_rate_mbps)) / max(
                    profile.rate_log_sd, 1e-6
                )
                z_jitter = abs(math.log(max(jitter_ms, 1e-12) / profile.jitter_ms)) / 0.18
                shift_score = float(max(z_snr, z_rate, z_jitter))
                resource_feasible = bool(
                    geometry_feasible
                    and path_available
                    and storage_available >= required_bytes
                    and capacity_bytes >= required_bytes
                )
                gate_frame = pd.DataFrame(
                    [
                        {
                            "rf_snr_db": snr_db,
                            "forecast_rate_mbps": rate_mbps,
                            "forecast_jitter_ms": jitter_ms,
                            "forecast_handover_probability": handover_forecast,
                        }
                    ]
                )
                gate_decision = domain_gate.evaluate(gate_frame, gate_evidence)
                gate_release = bool(
                    resource_feasible and gate_decision.release_candidate
                )
                age_ms = float(rng.uniform(1.0, config.telemetry_max_age_ms))
                action_id = f"{contact['contact_id_geometry']}::path::{profile.path_profile_id}"
                task_action_rows.append(
                    {
                        "logical_task_id": logical_id,
                        "mission_id": task["mission_id"],
                        "simulation_seed": int(seed),
                        "candidate_path_id": action_id,
                        "path_profile_id": profile.path_profile_id,
                        "interface_type": profile.interface_type,
                        "operator_id": profile.operator_id,
                        "gateway_id": profile.gateway_id,
                        "satellite_id": satellite_id,
                        "beam_id": f"synthetic_beam::{satellite_id}::orbit{contact['orbit_number']}",
                        "contact_id": action_id,
                        "geometry_contact_id": contact["contact_id_geometry"],
                        "orbit_number": int(contact["orbit_number"]),
                        "path_available": path_available,
                        "rf_snr_db": snr_db,
                        "forecast_rate_mbps": rate_mbps,
                        "forecast_jitter_ms": jitter_ms,
                        "forecast_handover_probability": handover_forecast,
                        "profile_outage_probability": profile.outage_probability,
                        "profile_mean_rate_mbps": profile.mean_rate_mbps,
                        "profile_rate_log_sd": profile.rate_log_sd,
                        "handover_state": "elevated_risk" if handover_forecast >= 0.15 else "stable_forecast",
                        "geometry_allows_task": geometry_feasible,
                        "observation_feasible": geometry_feasible,
                        "resource_feasible": resource_feasible,
                        "predicted_completion_probability": 0.0,
                        "completion_prediction_source": "unassigned_not_a_prediction",
                        "completion_predictor_sha256": "unassigned",
                        "shift_score": shift_score,
                        "gate_release": gate_release,
                        "gate_feature_coverage": gate_decision.feature_coverage,
                        "gate_ood_fraction": gate_decision.ood_fraction,
                        "gate_reasons": "|".join(gate_decision.reasons),
                        "gate_method": "frozen_ledoit_wolf_mahalanobis",
                        "gate_scope": "simulation_release_only_not_field_authorization",
                        "missingness_reason": "none" if satellite_id != "NO_GEOMETRY" else "no_eos_geometry_window",
                        "telemetry_cutoff_ns": int(task["release_time_ns"]),
                        "telemetry_sample_ns": int(task["release_time_ns"] - round(age_ms * 1e6)),
                        "observation_age_ms": age_ms,
                        "window_start_ns": int(contact["window_start_ns"]),
                        "window_end_ns": int(contact["window_end_ns"]),
                        "observation_end_ns": observation_end_ns,
                        "planned_downlink_duration_s": downlink_duration_s,
                        "required_bytes": required_bytes,
                        "storage_available_bytes": storage_available,
                        "nominal_storage_bytes": nominal_storage,
                        "downlink_capacity_bytes": capacity_bytes,
                        "forecast_capacity_ratio": capacity_ratio,
                        "predicted_congestion": float(np.clip(required_bytes / max(capacity_bytes, 1), 0.0, 10.0)),
                        "stabilization_time_s": float(
                            max(satellite.get("stabilization_time_s", 0.0), config.slew_guard_s)
                        ),
                        "network_overlay_source": profile.source,
                        "source_geometry": contact["source_geometry"],
                        "evidence_tier": EVIDENCE_TIER,
                    }
                )
        if completion_predictor is not None:
            probabilities = completion_predictor.predict(pd.DataFrame(task_action_rows))
            for row, probability in zip(task_action_rows, probabilities):
                row["predicted_completion_probability"] = float(probability)
                row["completion_prediction_source"] = "development_fitted_synthetic_completion"
                row["completion_predictor_sha256"] = completion_predictor.sha256
        action_hash = _hash_json(
            [
                {
                    key: row[key]
                    for key in (
                        "candidate_path_id",
                        "path_available",
                        "resource_feasible",
                        "predicted_completion_probability",
                        "window_start_ns",
                        "downlink_capacity_bytes",
                    )
                }
                for row in sorted(task_action_rows, key=lambda item: item["candidate_path_id"])
            ]
        )
        for row in task_action_rows:
            row["common_action_set_sha256"] = action_hash
        rows.extend(task_action_rows)
    return pd.DataFrame(rows)


def _interval_is_free(
    occupied: Mapping[str, list[tuple[int, int]]], resource: str, start: int, end: int
) -> bool:
    return all(end <= old_start or start >= old_end for old_start, old_end in occupied.get(resource, []))


def _reserve_interval(
    occupied: dict[str, list[tuple[int, int]]], resource: str, start: int, end: int
) -> None:
    occupied.setdefault(resource, []).append((int(start), int(end)))


def _schedule_policy(
    task_specs: pd.DataFrame,
    actions: pd.DataFrame,
    policy_arm: str,
) -> tuple[dict[str, str | None], dict[str, tuple[str, str]]]:
    """Schedule one policy without access to any realized future outcome."""

    feasible = actions.loc[actions["resource_feasible"]].copy()
    opportunities = feasible.groupby("logical_task_id")["geometry_contact_id"].nunique()
    ordering = task_specs.copy()
    ordering["feasible_opportunities"] = ordering["logical_task_id"].map(opportunities).fillna(0)
    if policy_arm == POLICY_BASELINE:
        ordering = ordering.sort_values(["deadline_ns", "priority", "logical_task_id"], ascending=[True, False, True])
    elif policy_arm == POLICY_INTEGRATED:
        # Deadline remains the primary scheduling constraint.  Scarcity and
        # priority break deadline ties; the joint action is risk-ranked below.
        ordering = ordering.sort_values(
            ["deadline_ns", "feasible_opportunities", "priority", "logical_task_id"],
            ascending=[True, True, False, True],
        )
    else:
        raise ValueError(f"unknown policy arm: {policy_arm}")

    occupied_satellite: dict[str, list[tuple[int, int]]] = {}
    occupied_gateway: dict[str, list[tuple[int, int]]] = {}
    selected: dict[str, str | None] = {}
    gate_state: dict[str, tuple[str, str]] = {}

    for task in ordering.to_dict(orient="records"):
        logical_id = task["logical_task_id"]
        task_actions = feasible.loc[feasible["logical_task_id"] == logical_id].copy()
        if task_actions.empty:
            selected[logical_id] = None
            gate_state[logical_id] = ("no_action", "no_resource_feasible_action")
            continue
        reactive = task_actions.sort_values(
            ["window_start_ns", "rf_snr_db", "candidate_path_id"],
            ascending=[True, False, True],
        )
        if policy_arm == POLICY_BASELINE:
            candidate_orders: list[tuple[pd.DataFrame, str, str]] = [
                (reactive, "baseline", "edf_then_highest_current_snr")
            ]
        else:
            released = task_actions.loc[task_actions["gate_release"]].sort_values(
                ["predicted_completion_probability", "window_start_ns", "candidate_path_id"],
                ascending=[False, True, True],
            )
            if not released.empty and (
                "completion_prediction_source" not in released
                or not released["completion_prediction_source"].eq(
                    "development_fitted_synthetic_completion"
                ).all()
            ):
                raise ValueError("full-chain risk ranking requires assigned development-fitted predictions")
            candidate_orders = []
            if not released.empty:
                candidate_orders.append(
                    (
                        released,
                        "released",
                        "frozen_data_driven_domain_gate_passed",
                    )
                )
            candidate_orders.append((reactive, "fallback", "gate_failed_or_released_actions_conflicted"))

        choice: dict[str, Any] | None = None
        chosen_state = ("no_action", "all_feasible_actions_conflicted")
        already_seen: set[str] = set()
        for candidate_order, state, reason in candidate_orders:
            for candidate in candidate_order.to_dict(orient="records"):
                action_id = candidate["candidate_path_id"]
                if action_id in already_seen:
                    continue
                already_seen.add(action_id)
                guard_ns = int(float(candidate["stabilization_time_s"]) * 1e9)
                observation_start = int(candidate["window_start_ns"])
                observation_end = int(candidate["observation_end_ns"])
                satellite_start = observation_start - guard_ns
                satellite_end = observation_end + guard_ns
                downlink_start = observation_end
                downlink_end = observation_end + int(candidate["planned_downlink_duration_s"] * 1e9)
                if not _interval_is_free(
                    occupied_satellite,
                    str(candidate["satellite_id"]),
                    satellite_start,
                    satellite_end,
                ):
                    continue
                if not _interval_is_free(
                    occupied_gateway,
                    str(candidate["gateway_id"]),
                    downlink_start,
                    downlink_end,
                ):
                    continue
                choice = candidate
                chosen_state = (state, reason)
                _reserve_interval(
                    occupied_satellite,
                    str(candidate["satellite_id"]),
                    satellite_start,
                    satellite_end,
                )
                _reserve_interval(
                    occupied_gateway,
                    str(candidate["gateway_id"]),
                    downlink_start,
                    downlink_end,
                )
                break
            if choice is not None:
                break
        selected[logical_id] = None if choice is None else str(choice["candidate_path_id"])
        gate_state[logical_id] = chosen_state
    return selected, gate_state


def _reactive_reference(actions: pd.DataFrame, logical_id: str) -> str | None:
    options = actions.loc[
        (actions["logical_task_id"] == logical_id) & actions["resource_feasible"]
    ].sort_values(
        ["window_start_ns", "rf_snr_db", "candidate_path_id"],
        ascending=[True, False, True],
    )
    return None if options.empty else str(options.iloc[0]["candidate_path_id"])


def _realize_outcome(
    *,
    selected: Mapping[str, Any] | None,
    task: Mapping[str, Any],
    seed: int,
    config: EOSFullChainConfig,
) -> dict[str, Any]:
    if selected is None:
        return {
            "action_executed": False,
            "observation_acquired": False,
            "product_valid": False,
            "bytes_delivered": 0,
            "application_complete": False,
            "completion_time_ns": np.nan,
            "met_deadline": False,
            "no_action": True,
            "failure_stage": "scheduling_no_action",
            "realized_rate_mbps": 0.0,
            "realized_outage": False,
            "realized_handover": False,
        }
    action_id = str(selected["candidate_path_id"])
    rng = _rng_for(seed, task["logical_task_id"], action_id, "potential_outcome")
    install_failed = bool(rng.random() < config.install_failure_probability)
    acquired = bool(
        not install_failed and rng.random() >= config.acquisition_failure_probability
    )
    valid = bool(acquired and rng.random() >= config.product_invalid_probability)
    handover = bool(rng.random() < float(selected["forecast_handover_probability"]))
    outage_probability = float(selected["profile_outage_probability"])
    outage = bool(rng.random() < outage_probability)
    rate = float(
        float(selected["profile_mean_rate_mbps"])
        * math.exp(rng.normal(0.0, float(selected["profile_rate_log_sd"])))
    )
    if handover:
        rate *= 0.20
    if outage or install_failed or not valid:
        rate = 0.0
    budget_s = float(selected["planned_downlink_duration_s"])
    delivered = int(min(int(task["required_bytes"]), max(0.0, rate * 1_000_000 / 8.0 * budget_s)))
    complete = bool(valid and delivered >= int(task["required_bytes"]))
    if complete and rate > 0:
        delivery_s = int(task["required_bytes"]) * 8.0 / (rate * 1_000_000)
    else:
        delivery_s = budget_s
    completion_ns = int(selected["observation_end_ns"] + round(delivery_s * 1e9))
    met_deadline = bool(complete and completion_ns <= int(task["deadline_ns"]))
    if install_failed:
        failure_stage = "action_install"
    elif not acquired:
        failure_stage = "observation_acquisition"
    elif not valid:
        failure_stage = "product_validation"
    elif not complete:
        failure_stage = "byte_delivery"
    elif not met_deadline:
        failure_stage = "deadline"
    else:
        failure_stage = "none"
    return {
        "action_executed": not install_failed,
        "observation_acquired": acquired,
        "product_valid": valid,
        "bytes_delivered": delivered,
        "application_complete": bool(complete and met_deadline),
        "completion_time_ns": completion_ns,
        "met_deadline": met_deadline,
        "no_action": False,
        "failure_stage": failure_stage,
        "realized_rate_mbps": rate,
        "realized_outage": outage,
        "realized_handover": handover,
        "install_failed": install_failed,
    }


def _arm_tables(
    *,
    scenario: ParsedEOSScenario,
    task_specs: pd.DataFrame,
    common_actions: pd.DataFrame,
    policy_arm: str,
    seed: int,
    config: EOSFullChainConfig,
    config_sha256: str,
    model_sha256: str,
) -> dict[str, pd.DataFrame]:
    campaign_id = f"{scenario.scenario_id}::seed{seed}::{policy_arm}"
    task_prefix = f"seed{seed}::{policy_arm}"
    selected_by_task, gate_by_task = _schedule_policy(task_specs, common_actions, policy_arm)

    task_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []

    task_id_by_logical: dict[str, str] = {}
    decision_id_by_logical: dict[str, str] = {}
    for task in task_specs.to_dict(orient="records"):
        logical_id = task["logical_task_id"]
        task_id = f"{task_prefix}::task::{logical_id}"
        decision_id = f"{task_prefix}::decision::{logical_id}"
        task_id_by_logical[logical_id] = task_id
        decision_id_by_logical[logical_id] = decision_id
        task_rows.append(
            {
                "campaign_id": campaign_id,
                "task_id": task_id,
                "decision_id": decision_id,
                "release_time_ns": int(task["release_time_ns"]),
                "required_duration_s": float(task["required_duration_s"]),
                "required_bytes": int(task["required_bytes"]),
                "priority": float(task["priority"]),
                "deadline_ns": int(task["deadline_ns"]),
                "task_type": task["task_type"],
                "logical_task_id": logical_id,
                "mission_id": task["mission_id"],
                "target_latitude": task["target_latitude"],
                "target_longitude": task["target_longitude"],
                "simulation_seed": int(seed),
                "workload_source": task["workload_source"],
                "evidence_tier": EVIDENCE_TIER,
            }
        )

    for candidate in common_actions.to_dict(orient="records"):
        logical_id = candidate["logical_task_id"]
        decision_id = decision_id_by_logical[logical_id]
        candidate_rows.append(
            {
                "campaign_id": campaign_id,
                "decision_id": decision_id,
                **{
                    key: candidate[key]
                    for key in common_actions.columns
                    if key not in {"logical_task_id"}
                },
                "logical_task_id": logical_id,
            }
        )
        resource_rows.append(
            {
                "campaign_id": campaign_id,
                "task_id": task_id_by_logical[logical_id],
                "decision_id": decision_id,
                "satellite_id": candidate["satellite_id"],
                "ground_station_id": candidate["gateway_id"],
                "contact_id": candidate["contact_id"],
                "observation_feasible": bool(candidate["observation_feasible"]),
                "storage_available_bytes": int(candidate["storage_available_bytes"]),
                "downlink_capacity_bytes": int(candidate["downlink_capacity_bytes"]),
                "predicted_congestion": float(candidate["predicted_congestion"]),
                "resource_feasible": bool(candidate["resource_feasible"]),
                "nominal_storage_bytes": int(candidate["nominal_storage_bytes"]),
                "geometry_contact_id": candidate["geometry_contact_id"],
                "network_overlay_source": candidate["network_overlay_source"],
                "logical_task_id": logical_id,
                "simulation_seed": int(seed),
                "evidence_tier": EVIDENCE_TIER,
            }
        )

    action_lookup = common_actions.set_index(["logical_task_id", "candidate_path_id"])
    for task in task_specs.to_dict(orient="records"):
        logical_id = task["logical_task_id"]
        selected_id = selected_by_task[logical_id]
        selected = None
        if selected_id is not None:
            selected = action_lookup.loc[(logical_id, selected_id)].to_dict()
            selected["candidate_path_id"] = selected_id
        outcome = _realize_outcome(
            selected=selected,
            task=task,
            seed=seed,
            config=config,
        )
        decision_start = int(task["release_time_ns"])
        decision_latency_ms = (
            config.baseline_decision_ms if policy_arm == POLICY_BASELINE else config.integrated_decision_ms
        )
        decision_end = decision_start + int(decision_latency_ms * 1e6)
        install_failed = bool(outcome.get("install_failed", False))
        install_ack = np.nan if selected is None else decision_end + int(config.install_latency_ms * 1e6)
        late_install = bool(
            selected is not None and int(install_ack) > int(selected["window_start_ns"])
        )
        if late_install:
            outcome.update(
                {
                    "action_executed": False,
                    "observation_acquired": False,
                    "product_valid": False,
                    "bytes_delivered": 0,
                    "application_complete": False,
                    "met_deadline": False,
                    "failure_stage": "late_install",
                }
            )
        gate_state, gate_reason = gate_by_task[logical_id]
        decision_id = decision_id_by_logical[logical_id]
        task_id = task_id_by_logical[logical_id]
        action_rows.append(
            {
                "campaign_id": campaign_id,
                "task_id": task_id,
                "decision_id": decision_id,
                "policy_arm": policy_arm,
                "selected_path_id": selected_id,
                "baseline_path_id": _reactive_reference(common_actions, logical_id),
                "gate_state": gate_state,
                "gate_reason": gate_reason,
                "shift_alarm_state": bool(
                    selected is not None
                    and float(selected["gate_ood_fraction"]) > 0.0
                ),
                "decision_start_ns": decision_start,
                "decision_end_ns": decision_end,
                "install_ack_ns": install_ack,
                "installed_path_id": None if selected is None or install_failed else selected_id,
                "install_failed": install_failed,
                "late_install": late_install,
                "rollback": install_failed,
                "logical_task_id": logical_id,
                "simulation_seed": int(seed),
                "evidence_tier": EVIDENCE_TIER,
            }
        )
        outcome_rows.append(
            {
                "campaign_id": campaign_id,
                "task_id": task_id,
                "decision_id": decision_id,
                **{key: value for key, value in outcome.items() if key != "install_failed"},
                "logical_task_id": logical_id,
                "simulation_seed": int(seed),
                "potential_outcome_key": (
                    None if selected_id is None else f"{seed}|{logical_id}|{selected_id}"
                ),
                "evidence_tier": EVIDENCE_TIER,
            }
        )

    manifest = pd.DataFrame(
        [
            {
                "campaign_id": campaign_id,
                "independent_cluster_id": f"{scenario.scenario_id}::synthetic_seed{seed}",
                "site_id": f"EOS-Bench::{scenario.scenario_id}",
                "collection_date": scenario.creation_time.date().isoformat(),
                "randomized_policy_arm": policy_arm,
                "protocol_version": PROTOCOL_VERSION,
                "model_sha256": model_sha256,
                "config_sha256": config_sha256,
                "evidence_tier": EVIDENCE_TIER,
                "simulation_seed": int(seed),
                "source_scenario_sha256": scenario.source_sha256,
                "field_campaign": False,
            }
        ]
    )
    return {
        "campaign_manifest": manifest,
        "candidate_path_observation": pd.DataFrame(candidate_rows),
        "task": pd.DataFrame(task_rows),
        "decision_action": pd.DataFrame(action_rows),
        "eo_resource_state": pd.DataFrame(resource_rows),
        "outcome": pd.DataFrame(outcome_rows),
    }


def _concat_arm_tables(arms: Iterable[Mapping[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    arm_list = list(arms)
    return {
        table_name: pd.concat([arm[table_name] for arm in arm_list], ignore_index=True)
        for table_name in arm_list[0]
    }


def _exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    smaller = min(improved, regressed)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return float(min(1.0, 2.0 * lower_tail))


def _paired_results(
    ledger: pd.DataFrame, config: EOSFullChainConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = ledger[
        ["simulation_seed", "logical_task_id", "policy_arm", "conservative_on_time_completion"]
    ]
    pivot = selected.pivot(
        index=["simulation_seed", "logical_task_id"],
        columns="policy_arm",
        values="conservative_on_time_completion",
    ).reset_index()
    if {POLICY_BASELINE, POLICY_INTEGRATED}.difference(pivot.columns):
        raise ValueError("paired policy arms are incomplete")
    pivot["paired_difference"] = pivot[POLICY_INTEGRATED] - pivot[POLICY_BASELINE]
    by_seed = (
        pivot.groupby("simulation_seed", as_index=False)
        .agg(
            paired_tasks=("logical_task_id", "size"),
            baseline_completion_rate=(POLICY_BASELINE, "mean"),
            integrated_completion_rate=(POLICY_INTEGRATED, "mean"),
            paired_difference=("paired_difference", "mean"),
            improved_tasks=("paired_difference", lambda value: int((value > 0).sum())),
            regressed_tasks=("paired_difference", lambda value: int((value < 0).sum())),
            tied_tasks=("paired_difference", lambda value: int((value == 0).sum())),
        )
        .sort_values("simulation_seed")
        .reset_index(drop=True)
    )
    rng = np.random.default_rng(config.bootstrap_seed)
    seed_values = by_seed["simulation_seed"].to_numpy()
    bootstrap: list[float] = []
    for _ in range(config.bootstrap_resamples):
        sampled = rng.choice(seed_values, size=len(seed_values), replace=True)
        pieces = [pivot.loc[pivot["simulation_seed"] == seed, "paired_difference"] for seed in sampled]
        bootstrap.append(float(pd.concat(pieces, ignore_index=True).mean()))
    improved = int((pivot["paired_difference"] > 0).sum())
    regressed = int((pivot["paired_difference"] < 0).sum())
    inference = {
        "evidence_tier": EVIDENCE_TIER,
        "paired_tasks": int(len(pivot)),
        "synthetic_replications": int(pivot["simulation_seed"].nunique()),
        "baseline_completion_rate": float(pivot[POLICY_BASELINE].mean()),
        "integrated_completion_rate": float(pivot[POLICY_INTEGRATED].mean()),
        "paired_completion_difference": float(pivot["paired_difference"].mean()),
        "cluster_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "improved_task_pairs": improved,
        "regressed_task_pairs": regressed,
        "tied_task_pairs": int((pivot["paired_difference"] == 0).sum()),
        "conditional_task_pair_mcnemar_p": _exact_mcnemar_p(improved, regressed),
        "conditional_task_pair_mcnemar_scope": (
            "Diagnostic only. The calculation treats task pairs as independent, which is not "
            "a defensible field-inference model because all tasks reuse one generated scenario "
            "and share simulated resources. The seed-cluster interval is the primary controlled-"
            "experiment uncertainty summary and is still not an independent-campaign interval."
        ),
        "replications_with_positive_difference": int((by_seed["paired_difference"] > 0).sum()),
        "replications_with_zero_difference": int((by_seed["paired_difference"] == 0).sum()),
        "replications_with_negative_difference": int((by_seed["paired_difference"] < 0).sum()),
        "inference_unit_note": (
            "The bootstrap resamples stochastic seeds as clusters. Seeds reuse one generated "
            "EOS-Bench scenario and are not independent field campaigns."
        ),
    }
    return by_seed, inference


def _completion_status(
    scenario: ParsedEOSScenario,
    audit: FullChainAudit,
    paired_inference: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "evidence_tier": EVIDENCE_TIER,
        "strict_completion_count": 1,
        "criteria": {
            "continuous_geometry_path_scheduling_actual_completion": {
                "satisfied": bool(audit.passed),
                "scope": "synthetic_simulation_only",
                "reason": (
                    "One task identifier is auditable from an EOS-Bench geometry opportunity "
                    "through scheduling, path installation, realized simulated bytes, deadline, "
                    "and application completion."
                ),
            },
            "independent_multisite_or_multioperator_validation": {
                "satisfied": False,
                "reason": (
                    "The run uses one generated EOS-Bench scenario. Synthetic operator labels and "
                    "random seeds are not independent sites or operator campaigns."
                ),
            },
            "matched_satellite_beam_gateway_handover_rf_observations": {
                "satisfied": False,
                "reason": (
                    "Satellite identifiers and visibility windows come from EOS-Bench, but beam, "
                    "gateway, RF and handover values are a matched stochastic overlay, not measurements."
                ),
            },
            "operational_eo_workload_validation": {
                "satisfied": False,
                "reason": (
                    "EOS-Bench is a generated research benchmark; city missions, releases and deadlines "
                    "do not constitute an operator workload or executed EO campaign."
                ),
            },
            "data_driven_gate_or_distribution_shift_detector": {
                "satisfied": False,
                "reason": (
                    "The experiment exercises a frozen data-driven Mahalanobis domain gate fitted on "
                    "disjoint synthetic reference draws. It is not externally validated on independent "
                    "deployment data, and its simulation release rule is not field authorization."
                ),
            },
            "consistent_outperformance_of_baselines": {
                "satisfied": False,
                "reason": (
                    "Paired synthetic results are reported, but repeated seeds over one generated scenario "
                    "cannot establish consistent superiority across independent conditions."
                ),
                "observed_synthetic_replications_positive": paired_inference[
                    "replications_with_positive_difference"
                ],
                "observed_synthetic_replications_total": paired_inference[
                    "synthetic_replications"
                ],
            },
        },
        "scenario_id": scenario.scenario_id,
        "source_scenario_sha256": scenario.source_sha256,
        "claim_boundary": (
            "This artifact validates implementation continuity and supplies a controlled synthetic "
            "stress test. It does not complete field, independent, or operational validation."
        ),
    }


def run_eos_full_chain_experiment(
    scenario_path: str | Path,
    *,
    config: EOSFullChainConfig | None = None,
    network_profile_csv: str | Path | None = None,
    frozen_predictor: FullChainPredictorBundle | None = None,
    network_profiles: tuple[NetworkPathProfile, ...] | None = None,
    condition_label: str = "declared",
) -> EOSFullChainArtifacts:
    """Run paired baseline/integrated policies and return audited artifacts."""

    config = config or EOSFullChainConfig()
    scenario = parse_eos_bench_scenario(scenario_path)
    if network_profiles is not None and network_profile_csv is not None:
        raise ValueError("supply either network profiles or a profile CSV, not both")
    profiles = network_profiles or load_network_profiles(network_profile_csv)
    predictor = frozen_predictor or prepare_full_chain_predictor(scenario, profiles, config)
    predictor_seeds = {
        int(seed)
        for partition in predictor.manifest["partitions"].values()
        for seed in partition["seeds"]
    }
    if predictor_seeds & (set(config.seeds) | {config.gate_reference_seed}):
        raise ValueError("frozen predictor partitions overlap evaluation or gate-reference seeds")
    if predictor.manifest.get("model_sha256") != predictor.model.sha256:
        raise ValueError("frozen predictor model and provenance hashes disagree")
    if predictor.manifest.get("source_scenario_sha256") != scenario.source_sha256:
        raise ValueError("frozen predictor task-size source differs from the evaluated scenario")
    domain_gate, gate_evidence, gate_sha256 = _fit_simulation_domain_gate(
        profiles, config
    )
    config_payload = {
        "config": asdict(config),
        "network_profiles": [asdict(profile) for profile in profiles],
        "source_scenario_sha256": scenario.source_sha256,
        "condition_label": condition_label,
    }
    config_sha256 = _hash_json(config_payload)
    model_sha256 = _hash_json(
        {
            "prediction": "development_fitted_synthetic_completion",
            "completion_predictor_sha256": predictor.model.sha256,
            "gate": "frozen_ledoit_wolf_mahalanobis",
            "gate_sha256": gate_sha256,
            "gate_scope": "simulation_release_only_not_field_authorization",
            "protocol": PROTOCOL_VERSION,
        }
    )

    arms: list[dict[str, pd.DataFrame]] = []
    common_action_hashes: dict[int, str] = {}
    for seed in config.seeds:
        task_specs = _seed_task_table(scenario, seed, config)
        common_actions = _build_common_actions(
            scenario,
            task_specs,
            profiles,
            seed,
            config,
            domain_gate,
            gate_evidence,
            predictor.model,
        )
        common_action_hashes[int(seed)] = _hash_json(
            common_actions[
                [
                    "logical_task_id",
                    "candidate_path_id",
                    "common_action_set_sha256",
                    "path_available",
                    "resource_feasible",
                ]
            ].to_dict(orient="records")
        )
        for policy_arm in (POLICY_BASELINE, POLICY_INTEGRATED):
            arms.append(
                _arm_tables(
                    scenario=scenario,
                    task_specs=task_specs,
                    common_actions=common_actions,
                    policy_arm=policy_arm,
                    seed=seed,
                    config=config,
                    config_sha256=config_sha256,
                    model_sha256=model_sha256,
                )
            )
    tables = _concat_arm_tables(arms)
    audit = validate_full_chain_bundle(tables)
    audit.raise_for_errors()
    # Every exported row carries an evidence label.  The generic ledger builder
    # obtains that label from the manifest and expects no same-named convenience
    # columns in the child tables, so use a non-mutating projection for its join.
    ledger_tables = {
        name: (
            frame
            if name == "campaign_manifest"
            else frame.drop(columns=["evidence_tier"], errors="ignore")
        )
        for name, frame in tables.items()
    }
    ledger = build_full_chain_ledger(ledger_tables)
    policy_summary = summarize_full_chain_completion(ledger)
    paired_seed_results, paired_inference = _paired_results(ledger, config)
    completion_status = _completion_status(scenario, audit, paired_inference)
    run_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_tier": EVIDENCE_TIER,
        "scenario_path": str(Path(scenario_path).resolve()),
        "scenario_id": scenario.scenario_id,
        "source_scenario_sha256": scenario.source_sha256,
        "config_sha256": config_sha256,
        "model_sha256": model_sha256,
        "seeds": list(config.seeds),
        "condition_label": condition_label,
        "completion_prediction_source": "development_fitted_synthetic_completion",
        "completion_predictor_sha256": predictor.model.sha256,
        "predictor_partition_and_evaluation_seeds_disjoint": True,
        "inactive_legacy_parameters": {
            "gate_z_threshold": {
                "value": config.gate_z_threshold,
                "status": "deprecated_inactive_metadata_not_used_in_domain_gate_or_ranking",
            },
        },
        "domain_gate_reference_scope": "condition_specific_declared_synthetic_reference",
        "network_profile_csv": (
            None if network_profile_csv is None else str(Path(network_profile_csv).resolve())
        ),
        "network_profile_sources": sorted({profile.source for profile in profiles}),
        "common_action_set_sha256_by_seed": common_action_hashes,
        "rows": {name: int(len(frame)) for name, frame in tables.items()},
        "audit_passed": bool(audit.passed),
        "future_outcome_leakage_control": (
            "Policies receive task specifications, EOS windows, current synthetic telemetry, frozen "
            "profile parameters and forecast capacity only. Predictor training, regularization tuning, "
            "calibration and probability-validation draws use four separately generated seed partitions, "
            "all disjoint from evaluation seeds. Validation diagnostics do not select the model or policy. "
            "Potential outcomes use a separate hash-keyed "
            "random stream after selection; the key excludes policy arm for common random numbers."
        ),
        "claim_boundary": "synthetic simulation, not operational or field evidence",
    }
    return EOSFullChainArtifacts(
        tables=tables,
        audit=audit,
        ledger=ledger,
        policy_summary=policy_summary,
        paired_seed_results=paired_seed_results,
        paired_inference=paired_inference,
        completion_status=completion_status,
        run_manifest=run_manifest,
        predictor_tables=predictor.tables,
        predictor_manifest=predictor.manifest,
    )


def write_eos_full_chain_artifacts(
    artifacts: EOSFullChainArtifacts, output_directory: str | Path
) -> Path:
    """Write a fresh, hash-recorded full-chain result bundle.

    Refusing a nonempty destination prevents a prior run from being silently
    mixed with a new configuration. Use a new run directory for every
    experiment revision.
    """

    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; choose a fresh run path: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in artifacts.tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    artifacts.audit.checks.to_csv(destination / "full_chain_audit.csv", index=False)
    artifacts.ledger.to_csv(destination / "task_ledger.csv", index=False)
    artifacts.policy_summary.to_csv(destination / "policy_summary.csv", index=False)
    artifacts.paired_seed_results.to_csv(destination / "paired_seed_results.csv", index=False)
    for name, frame in artifacts.predictor_tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    for name, payload in (
        ("paired_inference.json", artifacts.paired_inference),
        ("completion_status.json", artifacts.completion_status),
        ("predictor_manifest.json", artifacts.predictor_manifest),
    ):
        (destination / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    artifact_hashes = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = dict(artifacts.run_manifest)
    manifest["artifact_sha256_before_manifest"] = artifact_hashes
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def robustness_network_profiles(condition: str) -> tuple[NetworkPathProfile, ...]:
    """Declared matched-rate SNR--reliability contrasts, never result-selected.

    The profile risks are synthetic generating assumptions.  Aligned, neutral
    and reversed relations deliberately permit favorable, tied or adverse
    comparisons with the current-SNR baseline.  Mean rate and rate dispersion
    are identical across the two paths within each robustness condition.
    """

    if condition not in {"snr_aligned", "neutral", "snr_reversed"}:
        raise ValueError(f"unknown full-chain robustness condition: {condition}")
    reliable = dict(jitter_ms=8.0, outage_probability=0.045, handover_probability=0.045)
    fragile = dict(jitter_ms=22.0, outage_probability=0.20, handover_probability=0.25)
    middle = dict(jitter_ms=15.0, outage_probability=0.1225, handover_probability=0.1475)
    states = {
        "snr_aligned": (reliable, fragile),
        "neutral": (middle, middle),
        "snr_reversed": (fragile, reliable),
    }[condition]
    return tuple(
        replace(
            profile, path_profile_id=f"robustness_path_{index}",
            mean_rate_mbps=0.032, rate_log_sd=0.18,
            source=f"declared_{condition}_matched_rate_overlay", **state,
        )
        for index, (profile, state) in enumerate(zip(DEFAULT_NETWORK_PROFILES, states))
    )


def run_eos_full_chain_parameter_revision(
    scenario_path: str | Path,
    output_directory: str | Path,
    *,
    config: EOSFullChainConfig | None = None,
) -> pd.DataFrame:
    """Write the declared case and three non-selected frozen-model stress cases."""

    config = config or EOSFullChainConfig()
    destination = Path(output_directory)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}")
    scenario = parse_eos_bench_scenario(scenario_path)
    predictor = prepare_full_chain_predictor(scenario, DEFAULT_NETWORK_PROFILES, config)
    summary_rows: list[dict[str, Any]] = []
    for condition in ("declared", "snr_aligned", "neutral", "snr_reversed"):
        profiles = DEFAULT_NETWORK_PROFILES if condition == "declared" else robustness_network_profiles(condition)
        artifacts = run_eos_full_chain_experiment(
            scenario_path, config=config, frozen_predictor=predictor,
            network_profiles=profiles, condition_label=condition,
        )
        case_directory = destination if condition == "declared" else destination / "robustness" / condition
        write_eos_full_chain_artifacts(artifacts, case_directory)
        inference = artifacts.paired_inference
        summary_rows.append({
            "condition": condition,
            "baseline_completion_rate": inference["baseline_completion_rate"],
            "integrated_completion_rate": inference["integrated_completion_rate"],
            "paired_completion_difference": inference["paired_completion_difference"],
            "seed_cluster_ci_low": inference["cluster_bootstrap_95_ci"][0],
            "seed_cluster_ci_high": inference["cluster_bootstrap_95_ci"][1],
            "positive_seed_differences": inference["replications_with_positive_difference"],
            "zero_seed_differences": inference["replications_with_zero_difference"],
            "negative_seed_differences": inference["replications_with_negative_difference"],
            "synthetic_replications": inference["synthetic_replications"],
            "model_sha256": predictor.model.sha256,
            "audit_passed": artifacts.audit.passed,
            "evidence_tier": EVIDENCE_TIER,
            "condition_used_for_model_selection": False,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(destination / "robustness_summary.csv", index=False)
    robustness_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "predictor_frozen_before_all_evaluation_conditions": True,
        "domain_gate_reference_scope": (
            "Reference distribution is declared separately for each condition; "
            "not a single fixed gate tested on unseen conditions."
        ),
        "model_sha256": predictor.model.sha256,
        "all_conditions_reported_without_winner_selection": True,
        "conditions": summary["condition"].tolist(),
        "evidence_tier": EVIDENCE_TIER,
        "claim_boundary": (
            "Stress cases reuse generated geometry and seeds, not independent field campaigns. "
            "No condition is selected to promote an empirical superiority claim."
        ),
        "artifact_sha256": {
            str(path.relative_to(destination)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(destination.rglob("*")) if path.is_file()
        },
    }
    (destination / "robustness_manifest.json").write_text(
        json.dumps(robustness_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
