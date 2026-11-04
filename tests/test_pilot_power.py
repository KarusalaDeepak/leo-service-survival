from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from leo_service_survival.pilot_power import (
    MINIMUM_BLOCKS_PER_ARM,
    BlindedPilotInputs,
    build_blinded_pilot_plan,
    verify_blinded_pilot_plan,
)


ROOT = Path(__file__).resolve().parents[1]
KEY = b"objective-four-blinded-pilot-key-32-bytes-minimum"


def _inputs(**overrides: object) -> BlindedPilotInputs:
    values: dict[str, object] = {
        "baseline_completion_probability": 0.60,
        "block_size": 20,
        "intracluster_correlation": 0.08,
        "missing_outcome_fraction": 0.05,
        "installation_failure_fraction": 0.03,
        "minimum_effect_absolute": 0.08,
        "target_power": 0.80,
        "target_ci_half_width": 0.07,
    }
    values.update(overrides)
    return BlindedPilotInputs(**values)


def test_plan_selects_larger_requirement_and_enforces_floor() -> None:
    report = build_blinded_pilot_plan(_inputs(), signing_key=KEY)
    requirements = report["requirements"]
    selected = report["selected_plan"]
    assert selected["blocks_per_arm"] == max(
        requirements["power_blocks_per_arm"],
        requirements["precision_blocks_per_arm"],
        MINIMUM_BLOCKS_PER_ARM,
    )
    assert selected["total_blocks"] == 2 * selected["blocks_per_arm"]
    assert selected["approximate_achieved_power"] >= 0.80
    assert selected["approximate_ci_half_width"] <= 0.07


def test_precision_and_power_can_each_be_binding() -> None:
    precision = build_blinded_pilot_plan(
        _inputs(target_ci_half_width=0.025), signing_key=KEY
    )
    assert precision["selected_plan"]["binding_requirement"] == "precision"

    power = build_blinded_pilot_plan(
        _inputs(target_ci_half_width=0.25, minimum_effect_absolute=0.04, target_power=0.95),
        signing_key=KEY,
    )
    assert power["selected_plan"]["binding_requirement"] == "power"


def test_floor_is_binding_for_small_requirement() -> None:
    report = build_blinded_pilot_plan(
        _inputs(
            block_size=1000,
            intracluster_correlation=0.0,
            minimum_effect_absolute=0.20,
            target_ci_half_width=0.20,
        ),
        signing_key=KEY,
    )
    assert report["selected_plan"]["binding_requirement"] == "minimum_blocks_floor"
    assert report["selected_plan"]["blocks_per_arm"] == MINIMUM_BLOCKS_PER_ARM


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"baseline_completion_probability": 1.0}, "baseline_completion"),
        ({"block_size": 1.5}, "block_size"),
        ({"intracluster_correlation": -0.01}, "intracluster"),
        ({"missing_outcome_fraction": 1.0}, "missing_outcome"),
        ({"installation_failure_fraction": 1.0}, "installation_failure"),
        ({"minimum_effect_absolute": 0.50}, "remain below one"),
        ({"target_power": 0.50}, "target_power"),
        ({"target_ci_half_width": 0.0}, "target_ci"),
    ],
)
def test_invalid_blinded_aggregates_fail_closed(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_blinded_pilot_plan(_inputs(**overrides), signing_key=KEY)


def test_signed_report_recomputes_and_rejects_tampering_or_extra_inputs() -> None:
    report = build_blinded_pilot_plan(_inputs(), signing_key=KEY)
    verify_blinded_pilot_plan(report, signing_key=KEY)

    tampered = json.loads(json.dumps(report))
    tampered["selected_plan"]["blocks_per_arm"] += 1
    with pytest.raises(ValueError, match="signature"):
        verify_blinded_pilot_plan(tampered, signing_key=KEY)

    relabelled = json.loads(json.dumps(report))
    relabelled["inputs"]["candidate_completion_probability"] = 0.9
    with pytest.raises(ValueError, match="signature"):
        verify_blinded_pilot_plan(relabelled, signing_key=KEY)

    with pytest.raises(TypeError):
        BlindedPilotInputs(
            **{
                **report["inputs"],
                "treatment_label": "candidate",
            }
        )


def test_cli_creates_and_verifies_report_without_label_arguments(tmp_path: Path) -> None:
    key_path = tmp_path / "operator.key"
    key_path.write_bytes(KEY)
    output = tmp_path / "pilot-plan.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "plan_blinded_pilot.py"),
        "--baseline-completion",
        "0.60",
        "--block-size",
        "20",
        "--icc",
        "0.08",
        "--missing-fraction",
        "0.05",
        "--installation-failure-fraction",
        "0.03",
        "--effect-margin",
        "0.08",
        "--target-power",
        "0.80",
        "--target-ci-half-width",
        "0.07",
        "--signing-key",
        str(key_path),
        "--output",
        str(output),
    ]
    created = subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.is_file()
    assert "blocks_per_arm" in created.stdout

    verified = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plan_blinded_pilot.py"),
            "--verify",
            str(output),
            "--signing-key",
            str(key_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"verified": true' in verified.stdout
    assert set(json.loads(output.read_text(encoding="utf-8"))["inputs"]) == {
        "baseline_completion_probability",
        "block_size",
        "intracluster_correlation",
        "missing_outcome_fraction",
        "installation_failure_fraction",
        "minimum_effect_absolute",
        "target_power",
        "target_ci_half_width",
    }

