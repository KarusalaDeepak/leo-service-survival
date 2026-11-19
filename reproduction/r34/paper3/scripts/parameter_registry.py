"""Authoritative numerical-parameter registry for the RACS study.

The registry separates quantities that describe the supplied benchmark from
quantities selected by an algorithm.  A value appearing here is not, by itself,
evidence that it is physically correct.  ``status`` and ``selection_rule`` say
what may legitimately be claimed about each value.

Fresh primary evaluations must select score and model hyperparameters using
development groups only.  The legacy values remain available solely to
reproduce the already reported reference experiment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


WEIGHT_KEYS = ("margin", "spread", "congestion", "risk", "relay", "delay")

# Published/reference coefficients.  These are deliberately named LEGACY and
# REFERENCE: they were engineering choices and are not deployment constants.
LEGACY_REFERENCE_WEIGHTS = {
    "margin": 2.8,
    "spread": 0.4,
    "congestion": 2.2,
    "risk": 1.6,
    "relay": 0.15,
    "delay": 0.15,
}


@dataclass(frozen=True)
class MarginConfig:
    """Dimensioned scaling and mixture used to form the margin feature."""

    time_scale_seconds: float = 1800.0
    capacity_scale_gb: float = 2.0
    time_mix: float = 0.55

    def validate(self) -> "MarginConfig":
        if not self.time_scale_seconds > 0:
            raise ValueError("time_scale_seconds must be positive")
        if not self.capacity_scale_gb > 0:
            raise ValueError("capacity_scale_gb must be positive")
        if not 0 <= self.time_mix <= 1:
            raise ValueError("time_mix must lie in [0, 1]")
        return self


@dataclass(frozen=True)
class StressProfile:
    """Controlled benchmark transformation, not a measured LEO environment."""

    deadline_factor: float = 1.0
    bandwidth_factor: float = 1.0
    volume_factor: float = 1.0

    def validate(self) -> "StressProfile":
        if min(self.deadline_factor, self.bandwidth_factor, self.volume_factor) <= 0:
            raise ValueError("stress factors must be positive")
        return self


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    max_features: float
    random_state: int = 20261003


LEGACY_MARGIN_CONFIG = MarginConfig()

# Named profiles reproduce the existing paper.  A separate full-factorial
# generator in run_parameter_robustness.py tests whether conclusions depend on
# these single severities.
STRESS_PROFILES = {
    "base": StressProfile(),
    "tight_deadline": StressProfile(deadline_factor=0.70),
    "reduced_bandwidth": StressProfile(bandwidth_factor=0.65),
    "high_volume": StressProfile(volume_factor=1.35),
    "combined": StressProfile(0.65, 0.60, 1.35),
}

# These bands are reporting bins only.  They do not affect request ordering or
# acceptance.  The robustness runner also reports every individual priority so
# that the bins cannot hide an effect.
REPORTING_PRIORITY_BANDS = ((1, 3), (4, 6), (7, 10))

# The old stress transformation guaranteed at least 60 seconds after release.
# It is retained only for exact reproduction.  Fresh sensitivity runs include
# 1, 30, 60, and 120 seconds and report how many deadlines were clipped.
LEGACY_DEADLINE_FLOOR_SECONDS = 60
DEADLINE_FLOOR_SENSITIVITY_SECONDS = (1, 30, 60, 120)

CALIBRATION_SEED = 20261003
MODEL_SEARCH_BUDGET = 33


def registry_rows() -> list[dict]:
    """Return the machine-readable provenance record used by code and paper."""

    rows = [
        dict(
            parameter="legacy_score_weights",
            value=LEGACY_REFERENCE_WEIGHTS,
            unit="dimensionless",
            category="engineering reference",
            status="reproduction only; not a primary or universal setting",
            selection_rule="none; inherited hand-set coefficients",
            sensitivity="33 normalized candidates with outer fleet-size holdout",
        ),
        dict(
            parameter="margin_time_scale",
            value=LEGACY_MARGIN_CONFIG.time_scale_seconds,
            unit="s",
            category="feature scaling",
            status="legacy reference",
            selection_rule="must be selected on development groups for a new primary run",
            sensitivity=[900, 1800, 3600],
        ),
        dict(
            parameter="margin_capacity_scale",
            value=LEGACY_MARGIN_CONFIG.capacity_scale_gb,
            unit="GB",
            category="feature scaling",
            status="legacy reference",
            selection_rule="must be selected on development groups for a new primary run",
            sensitivity=[1, 2, 4],
        ),
        dict(
            parameter="margin_time_mix",
            value=LEGACY_MARGIN_CONFIG.time_mix,
            unit="fraction",
            category="policy hyperparameter",
            status="legacy reference",
            selection_rule="development-only selection; never evaluation feedback",
            sensitivity=[0, 0.25, 0.5, 0.55, 0.75, 1],
        ),
        dict(
            parameter="stress_profiles",
            value={k: asdict(v) for k, v in STRESS_PROFILES.items()},
            unit="multipliers",
            category="controlled scenario",
            status="not an estimate of field conditions",
            selection_rule="predeclared transformations",
            sensitivity=dict(
                deadline=[1.0, 0.85, 0.70, 0.55],
                bandwidth=[1.0, 0.80, 0.65, 0.50],
                volume=[1.0, 1.20, 1.35, 1.50],
            ),
        ),
        dict(
            parameter="deadline_floor",
            value=LEGACY_DEADLINE_FLOOR_SECONDS,
            unit="s",
            category="scenario safeguard",
            status="legacy reproduction value; not a physical constant",
            selection_rule="disclose clipping count and test alternatives",
            sensitivity=list(DEADLINE_FLOOR_SENSITIVITY_SECONDS),
        ),
        dict(
            parameter="priority_bands",
            value=[list(v) for v in REPORTING_PRIORITY_BANDS],
            unit="priority labels",
            category="reporting convention",
            status="does not affect scheduling",
            selection_rule="predeclared; supplement with per-priority completion",
            sensitivity="alternative equal-count bins and unbinned priorities",
        ),
        dict(
            parameter="score_search_budget",
            value=MODEL_SEARCH_BUDGET,
            unit="candidate configurations",
            category="model-selection budget",
            status="predeclared",
            selection_rule="same budget as random-forest baseline",
            sensitivity="report selected IDs and outer-fold stability",
        ),
        dict(
            parameter="random_forest_search_budget",
            value=MODEL_SEARCH_BUDGET,
            unit="candidate configurations",
            category="model-selection budget",
            status="predeclared",
            selection_rule="same nested groups and primary reward as score family",
            sensitivity="33 deterministic hyperparameter configurations",
        ),
    ]
    return rows


def write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry_rows(), indent=2) + "\n")
