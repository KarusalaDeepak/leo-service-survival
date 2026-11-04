"""Blinded pilot power and precision planning for Objective 4.

The planner intentionally accepts only arm-agnostic aggregate pilot quantities.
It has no API for treatment labels, treatment-specific pilot outcomes, or
candidate-policy estimates.  The resulting requirement is the maximum of a
two-sided power calculation, a confidence-interval precision calculation, and
the protocol's hard floor of ten randomized blocks per arm.

The calculation is a conservative normal approximation for a two-arm,
block-randomized binary endpoint.  Clustering is represented by the usual
design effect ``1 + (m - 1) * ICC``.  Missing outcomes and failed installations
reduce the effective information per planned block multiplicatively; they do
not disappear from the later intention-to-treat denominator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from scipy.stats import norm

from .prospective_release import sign_artifact, verify_signed_artifact


BLINDED_PILOT_PLAN_PURPOSE = "O4_BLINDED_PILOT_POWER_PLAN_V1"
MINIMUM_BLOCKS_PER_ARM = 10
CONFIDENCE_LEVEL = 0.95
ALPHA_TWO_SIDED = 1.0 - CONFIDENCE_LEVEL


@dataclass(frozen=True)
class BlindedPilotInputs:
    """The complete and deliberately label-free pilot-planning input contract."""

    baseline_completion_probability: float
    block_size: int
    intracluster_correlation: float
    missing_outcome_fraction: float
    installation_failure_fraction: float
    minimum_effect_absolute: float
    target_power: float
    target_ci_half_width: float

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.baseline_completion_probability < 1.0:
            raise ValueError("baseline_completion_probability must be in (0, 1)")
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool):
            raise ValueError("block_size must be an integer")
        if self.block_size < 1:
            raise ValueError("block_size must be at least one")
        if not 0.0 <= self.intracluster_correlation < 1.0:
            raise ValueError("intracluster_correlation must be in [0, 1)")
        if not 0.0 <= self.missing_outcome_fraction < 1.0:
            raise ValueError("missing_outcome_fraction must be in [0, 1)")
        if not 0.0 <= self.installation_failure_fraction < 1.0:
            raise ValueError("installation_failure_fraction must be in [0, 1)")
        if not 0.0 < self.minimum_effect_absolute < 1.0:
            raise ValueError("minimum_effect_absolute must be in (0, 1)")
        if (
            self.baseline_completion_probability + self.minimum_effect_absolute
            >= 1.0
        ):
            raise ValueError(
                "baseline completion plus minimum effect must remain below one"
            )
        if not 0.5 < self.target_power < 1.0:
            raise ValueError("target_power must be in (0.5, 1)")
        if not 0.0 < self.target_ci_half_width < 1.0:
            raise ValueError("target_ci_half_width must be in (0, 1)")


def _ceil_blocks(independent_observations_per_arm: float, information_per_block: float) -> int:
    return int(math.ceil(independent_observations_per_arm / information_per_block))


def _selected_driver(power_blocks: int, precision_blocks: int) -> str:
    largest_statistical = max(power_blocks, precision_blocks)
    if largest_statistical < MINIMUM_BLOCKS_PER_ARM:
        return "minimum_blocks_floor"
    if power_blocks > precision_blocks:
        return "power"
    if precision_blocks > power_blocks:
        return "precision"
    return "power_and_precision_tie"


def _unsigned_blinded_pilot_plan(inputs: BlindedPilotInputs) -> dict[str, Any]:
    inputs.validate()

    p0 = float(inputs.baseline_completion_probability)
    delta = float(inputs.minimum_effect_absolute)
    p1 = p0 + delta
    average_probability = (p0 + p1) / 2.0
    z_alpha = float(norm.ppf(1.0 - ALPHA_TWO_SIDED / 2.0))
    z_power = float(norm.ppf(inputs.target_power))
    variance_sum = p0 * (1.0 - p0) + p1 * (1.0 - p1)

    power_numerator = (
        z_alpha * math.sqrt(2.0 * average_probability * (1.0 - average_probability))
        + z_power * math.sqrt(variance_sum)
    ) ** 2
    independent_for_power = power_numerator / (delta**2)
    independent_for_precision = (
        z_alpha**2 * variance_sum / (inputs.target_ci_half_width**2)
    )

    design_effect = 1.0 + (inputs.block_size - 1.0) * inputs.intracluster_correlation
    information_retention = (
        (1.0 - inputs.missing_outcome_fraction)
        * (1.0 - inputs.installation_failure_fraction)
    )
    independent_information_per_planned_block = (
        inputs.block_size * information_retention / design_effect
    )
    if independent_information_per_planned_block <= 0.0:
        raise ValueError("effective information per planned block must be positive")

    power_blocks = _ceil_blocks(
        independent_for_power, independent_information_per_planned_block
    )
    precision_blocks = _ceil_blocks(
        independent_for_precision, independent_information_per_planned_block
    )
    selected_blocks = max(
        power_blocks, precision_blocks, MINIMUM_BLOCKS_PER_ARM
    )

    effective_independent_per_arm = (
        selected_blocks * independent_information_per_planned_block
    )
    expected_ci_half_width = z_alpha * math.sqrt(
        variance_sum / effective_independent_per_arm
    )
    achieved_z = (
        math.sqrt(effective_independent_per_arm) * delta
        - z_alpha * math.sqrt(2.0 * average_probability * (1.0 - average_probability))
    ) / math.sqrt(variance_sum)
    approximate_achieved_power = float(norm.cdf(achieved_z))

    return {
        "schema_version": "1.0.0",
        "artifact_type": "blinded_pilot_power_precision_plan",
        "evidence_status": "BLINDED_AGGREGATE_PLANNING_ONLY",
        "field_trial_completed": False,
        "blinding": {
            "status": "TREATMENT_LABELS_NOT_ACCEPTED",
            "input_scope": "arm-agnostic aggregate pilot quantities only",
            "treatment_specific_estimates_used": False,
        },
        "inputs": asdict(inputs),
        "fixed_assumptions": {
            "endpoint": "binary conservative intention-to-treat completion",
            "allocation_ratio": "1:1",
            "confidence_level": CONFIDENCE_LEVEL,
            "alpha_two_sided": ALPHA_TWO_SIDED,
            "minimum_blocks_per_arm": MINIMUM_BLOCKS_PER_ARM,
            "method": "normal approximation for a two-arm block-randomized difference in proportions",
            "attrition_rule": "missing and installation-failure fractions reduce effective information multiplicatively",
        },
        "calculations": {
            "planning_alternative_completion_probability": p1,
            "design_effect": design_effect,
            "information_retention_fraction": information_retention,
            "independent_information_per_planned_block": independent_information_per_planned_block,
            "independent_observations_per_arm_for_power": independent_for_power,
            "independent_observations_per_arm_for_precision": independent_for_precision,
            "z_two_sided": z_alpha,
            "z_power": z_power,
        },
        "requirements": {
            "power_blocks_per_arm": power_blocks,
            "precision_blocks_per_arm": precision_blocks,
            "minimum_floor_blocks_per_arm": MINIMUM_BLOCKS_PER_ARM,
        },
        "selected_plan": {
            "selection_rule": "maximum of power, precision, and minimum-floor requirements",
            "binding_requirement": _selected_driver(power_blocks, precision_blocks),
            "blocks_per_arm": selected_blocks,
            "total_blocks": 2 * selected_blocks,
            "planned_opportunities_per_arm": selected_blocks * inputs.block_size,
            "total_planned_opportunities": 2 * selected_blocks * inputs.block_size,
            "approximate_achieved_power": approximate_achieved_power,
            "approximate_ci_half_width": expected_ci_half_width,
        },
        "claim_boundary": (
            "This signed report freezes a blinded planning requirement. It is not "
            "field evidence, does not authorize a candidate policy, and must be "
            "replaced only through a documented pre-outcome amendment."
        ),
    }


def build_blinded_pilot_plan(
    inputs: BlindedPilotInputs,
    *,
    signing_key: bytes,
) -> dict[str, Any]:
    """Build a deterministic, purpose-bound, signed planning report."""

    return sign_artifact(
        _unsigned_blinded_pilot_plan(inputs),
        signing_key,
        purpose=BLINDED_PILOT_PLAN_PURPOSE,
    )


def verify_blinded_pilot_plan(
    report: Mapping[str, Any],
    *,
    signing_key: bytes,
) -> None:
    """Verify authentication and recompute the plan from the label-free inputs."""

    verify_signed_artifact(
        report,
        signing_key,
        purpose=BLINDED_PILOT_PLAN_PURPOSE,
    ).raise_for_errors()
    supplied_inputs = report.get("inputs")
    expected_input_fields = set(BlindedPilotInputs.__dataclass_fields__)
    if not isinstance(supplied_inputs, Mapping):
        raise ValueError("signed pilot plan has no blinded input object")
    supplied_fields = set(supplied_inputs)
    if supplied_fields != expected_input_fields:
        raise ValueError(
            "signed pilot plan input contract changed: "
            f"missing={sorted(expected_input_fields - supplied_fields)}; "
            f"extra={sorted(supplied_fields - expected_input_fields)}"
        )
    recomputed = build_blinded_pilot_plan(
        BlindedPilotInputs(**dict(supplied_inputs)), signing_key=signing_key
    )
    if report.get("artifact_sha256") != recomputed.get("artifact_sha256"):
        raise ValueError("signed pilot plan does not match recomputed requirements")

