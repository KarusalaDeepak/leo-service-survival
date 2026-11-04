from __future__ import annotations

from leo_service_survival.parameters import (
    CONFIRMATORY_GATE_REQUIREMENTS,
    EXTERNAL_VALIDATION_GATE_REQUIREMENTS,
    PROOF_OF_METHOD_GATE_REQUIREMENTS,
    parameter_registry,
)


def test_parameter_registry_has_unique_complete_rows() -> None:
    registry = parameter_registry()
    assert registry["parameter"].is_unique
    assert not registry.isna().any().any()
    assert {
        "application_requirement",
        "development_selected",
        "numerical_safeguard",
        "controlled_factor",
    }.issubset(set(registry["category"]))


def test_gate_profiles_separate_proof_external_and_confirmatory_evidence() -> None:
    assert PROOF_OF_METHOD_GATE_REQUIREMENTS.minimum_independent_campaigns == 1
    assert EXTERNAL_VALIDATION_GATE_REQUIREMENTS.minimum_independent_campaigns == 2
    assert CONFIRMATORY_GATE_REQUIREMENTS.minimum_independent_campaigns == 5
    assert PROOF_OF_METHOD_GATE_REQUIREMENTS.minimum_independent_families == 1
    assert EXTERNAL_VALIDATION_GATE_REQUIREMENTS.minimum_independent_families == 2
    assert CONFIRMATORY_GATE_REQUIREMENTS.minimum_independent_families == 3
    assert (
        CONFIRMATORY_GATE_REQUIREMENTS.maximum_endpoint_effect_ci_half_width
        < EXTERNAL_VALIDATION_GATE_REQUIREMENTS.maximum_endpoint_effect_ci_half_width
    )
