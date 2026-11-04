"""Focused fail-closed and provenance tests for the frozen evidence gate."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from leo_service_survival.gating import (
    CAMPAIGN_DERIVED_STATISTICAL_BOUNDS,
    GATE_LOGIC_FIXTURE,
    EvidenceGate,
    FrozenMahalanobisDetector,
    GateCriteria,
    GateEvidence,
    evaluate_gate_campaigns,
    campaign_precision_power_plan,
)
from leo_service_survival.parameters import EXTERNAL_VALIDATION_GATE_REQUIREMENTS


def _gate_fixture() -> tuple[EvidenceGate, pd.DataFrame]:
    rng = np.random.default_rng(414)
    development = pd.DataFrame(
        {
            "latency": rng.normal(40.0, 2.0, 1000),
            "loss": rng.beta(1.0, 40.0, 1000),
        }
    )
    detector = FrozenMahalanobisDetector(
        ("latency", "loss"), quantile=0.995
    ).fit(development)
    gate = EvidenceGate(detector, GateCriteria(maximum_ood_fraction=0.25))
    return gate, development.iloc[:100].reset_index(drop=True)


def _passing_evidence(**overrides: object) -> GateEvidence:
    values: dict[str, object] = {
        "schema_passed": True,
        "independent_campaigns": 5,
        "independent_families": 3,
        "paired_opportunity_groups": 5,
        "calibration_gap_upper": 0.03,
        "candidate_minus_baseline_endpoint_effect_lower": -0.01,
        "control_loop_p95_ms": 35.0,
        "endpoint_effect_ci_half_width": 0.03,
        "validation_tier": "confirmatory_release",
        "endpoint_name": "task_completion_rate",
        "evidence_basis": GATE_LOGIC_FIXTURE,
    }
    values.update(overrides)
    return GateEvidence(**values)


@pytest.mark.parametrize("missing", [None, np.nan, np.inf, -np.inf, "missing"])
@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("calibration_gap_upper", "calibration_evidence_unavailable"),
        (
            "candidate_minus_baseline_endpoint_effect_lower",
            "candidate_minus_baseline_endpoint_effect_unavailable",
        ),
        ("control_loop_p95_ms", "control_loop_timing_evidence_unavailable"),
    ],
)
def test_required_numeric_evidence_fails_closed(
    field: str, reason: str, missing: object
) -> None:
    gate, stable = _gate_fixture()
    decision = gate.evaluate(stable, _passing_evidence(**{field: missing}))
    assert not decision.release_candidate
    assert reason in decision.reasons


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, -np.inf, "missing", True])
def test_nonfinite_control_loop_requirement_fails_closed(invalid: object) -> None:
    gate, stable = _gate_fixture()
    gate = EvidenceGate(
        gate.detector,
        GateCriteria(maximum_control_loop_p95_ms=invalid),
    )
    decision = gate.evaluate(stable, _passing_evidence())
    assert not decision.release_candidate
    assert "gate_criteria_invalid" in decision.reasons


@pytest.mark.parametrize("campaigns", [None, np.nan, np.inf, 2.5, "3", True])
def test_campaign_count_must_be_finite_nonnegative_integer(campaigns: object) -> None:
    gate, stable = _gate_fixture()
    decision = gate.evaluate(
        stable, _passing_evidence(independent_campaigns=campaigns)
    )
    assert not decision.release_candidate
    assert "independent_campaign_count_unavailable" in decision.reasons


def test_schema_and_delayed_shift_status_must_be_explicit_booleans() -> None:
    gate, stable = _gate_fixture()
    schema = gate.evaluate(stable, _passing_evidence(schema_passed=np.nan))
    shift = gate.evaluate(stable, _passing_evidence(), delayed_shift_alarm=np.nan)
    assert "schema_or_clock_audit_unavailable" in schema.reasons
    assert "delayed_shift_status_unavailable" in shift.reasons
    assert not schema.release_candidate
    assert not shift.release_candidate


def test_logic_fixture_pass_is_not_statistical_authorization() -> None:
    gate, stable = _gate_fixture()
    decision = gate.evaluate(stable, _passing_evidence())
    assert decision.release_candidate
    assert decision.evidence_basis == GATE_LOGIC_FIXTURE
    assert not decision.statistical_evidence_generated
    assert not decision.statistical_authorization_supported


def test_campaign_derived_bounds_require_inferential_metadata() -> None:
    gate, stable = _gate_fixture()
    incomplete = gate.evaluate(
        stable,
        _passing_evidence(
            evidence_basis=CAMPAIGN_DERIVED_STATISTICAL_BOUNDS,
        ),
    )
    assert not incomplete.release_candidate
    assert "statistical_confidence_level_unavailable" in incomplete.reasons
    assert "multiple_testing_adjustment_unavailable" in incomplete.reasons

    complete = gate.evaluate(
        stable,
        _passing_evidence(
            evidence_basis=CAMPAIGN_DERIVED_STATISTICAL_BOUNDS,
            confidence_level=0.95,
            multiple_testing_adjustment="Holm familywise correction",
        ),
    )
    assert complete.release_candidate
    assert complete.statistical_evidence_generated
    assert complete.statistical_authorization_supported


def test_legacy_safety_names_are_constructor_aliases_only() -> None:
    criteria = GateCriteria(minimum_safety_effect_lower=-0.015)
    evidence = GateEvidence(
        schema_passed=True,
        independent_campaigns=3,
        independent_families=3,
        paired_opportunity_groups=5,
        calibration_gap_upper=0.03,
        safety_effect_lower=-0.01,
        control_loop_p95_ms=35.0,
        endpoint_effect_ci_half_width=0.03,
        validation_tier="confirmatory_release",
    )
    assert criteria.minimum_safety_effect_lower == -0.015
    assert evidence.safety_effect_lower == -0.01
    assert (
        criteria.minimum_candidate_minus_baseline_endpoint_effect_lower == -0.015
    )
    assert evidence.candidate_minus_baseline_endpoint_effect_lower == -0.01
    assert "minimum_safety_effect_lower" not in asdict(criteria)
    assert "safety_effect_lower" not in asdict(evidence)


def test_campaign_family_and_precision_checks_fail_closed() -> None:
    gate, stable = _gate_fixture()
    too_few_campaigns = gate.evaluate(
        stable, _passing_evidence(independent_campaigns=4)
    )
    too_few_families = gate.evaluate(
        stable, _passing_evidence(independent_families=2)
    )
    too_few_paired = gate.evaluate(
        stable, _passing_evidence(paired_opportunity_groups=4)
    )
    imprecise = gate.evaluate(
        stable, _passing_evidence(endpoint_effect_ci_half_width=0.08)
    )
    assert "insufficient_independent_campaigns" in too_few_campaigns.reasons
    assert "insufficient_independent_families" in too_few_families.reasons
    assert "insufficient_paired_opportunity_groups" in too_few_paired.reasons
    assert "endpoint_effect_precision_failed" in imprecise.reasons
    assert not too_few_campaigns.release_candidate
    assert not too_few_families.release_candidate
    assert not too_few_paired.release_candidate
    assert not imprecise.release_candidate


def test_campaign_count_is_a_floor_not_a_power_claim() -> None:
    plan = campaign_precision_power_plan()
    assert plan["planned_campaigns"].ge(5).all()
    assert (plan["planned_campaigns"] >= plan["campaigns_for_precision"]).all()
    assert (plan["planned_campaigns"] >= plan["campaigns_for_power"]).all()
    assert plan["interpretation"].str.contains("planning approximation").all()


def test_external_validation_tier_is_not_deployment_authorization() -> None:
    rng = np.random.default_rng(992)
    development = pd.DataFrame(
        {"latency": rng.normal(40.0, 2.0, 1000), "loss": rng.beta(1.0, 40.0, 1000)}
    )
    detector = FrozenMahalanobisDetector(("latency", "loss"), quantile=0.99).fit(
        development
    )
    profile = EXTERNAL_VALIDATION_GATE_REQUIREMENTS
    gate = EvidenceGate(
        detector,
        GateCriteria(
            minimum_independent_campaigns=profile.minimum_independent_campaigns,
            minimum_independent_families=profile.minimum_independent_families,
            minimum_paired_opportunity_groups=profile.minimum_paired_opportunity_groups,
            maximum_endpoint_effect_ci_half_width=profile.maximum_endpoint_effect_ci_half_width,
            validation_tier=profile.validation_tier,
        ),
    )
    decision = gate.evaluate(
        development.iloc[:100],
        GateEvidence(
            schema_passed=True,
            independent_campaigns=2,
            independent_families=2,
            paired_opportunity_groups=2,
            calibration_gap_upper=0.03,
            candidate_minus_baseline_endpoint_effect_lower=-0.01,
            control_loop_p95_ms=35.0,
            endpoint_effect_ci_half_width=0.08,
            endpoint_name="task_completion_rate",
            evidence_basis=CAMPAIGN_DERIVED_STATISTICAL_BOUNDS,
            confidence_level=0.95,
            multiple_testing_adjustment="Holm familywise correction",
            validation_tier="external_validation",
        ),
    )
    assert decision.release_candidate
    assert decision.statistical_evidence_generated
    assert not decision.statistical_authorization_supported


def test_campaign_output_labels_logic_test_and_does_not_claim_bound_generation() -> None:
    gate, stable = _gate_fixture()
    shifted = pd.DataFrame(
        {"latency": np.full(100, 160.0), "loss": np.full(100, 0.75)}
    )
    detail, summary = evaluate_gate_campaigns(
        gate,
        {"stable": stable, "shift": shifted},
        {"stable": False, "shift": True},
        _passing_evidence(),
        batch_size=50,
    )
    assert set(detail["gate_evidence_basis"]) == {GATE_LOGIC_FIXTURE}
    assert not detail["statistical_bounds_generated_by_this_function"].any()
    assert summary.loc[0, "validation_level"] == "gate_logic_operating_characteristics"
    assert not bool(summary.loc[0, "statistical_bounds_generated_by_this_function"])
    assert "does not construct" in summary.loc[0, "scope"]
