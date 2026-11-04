#!/usr/bin/env python3
"""Validate Objective 4 testbed and route configuration without route mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.field_preparation import (  # noqa: E402
    ProbeResult,
    SubprocessReadOnlyProbeRunner,
    run_read_only_qualification_probes,
    validate_route_driver_configuration,
    validate_testbed_qualification,
)
from leo_service_survival.prospective_release import (  # noqa: E402
    load_signing_key,
    sign_artifact,
)
from leo_service_survival.prospective_trial import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed-config", type=Path, required=True)
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument(
        "--probe-results",
        type=Path,
        help="Previously captured probe-result JSON; does not execute commands.",
    )
    parser.add_argument(
        "--execute-live-read-only-probes",
        action="store_true",
        help="Execute only configured read-only probes. Never changes a route.",
    )
    parser.add_argument(
        "--acknowledge-live-hardware",
        action="store_true",
        help="Required together with --execute-live-read-only-probes.",
    )
    parser.add_argument(
        "--allow-fake-driver-test",
        action="store_true",
        help="Permit explicitly labelled FAKE_DRIVER_TEST evidence for CI only.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ledger-schema",
        type=Path,
        help="Frozen action-ledger schema to bind into a live qualification.",
    )
    parser.add_argument(
        "--signing-key",
        type=Path,
        help="Operator key. Required to issue LIVE_QUALIFIED evidence.",
    )
    return parser.parse_args()


def _load_probe_results(path: Path) -> tuple[ProbeResult, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # A file supplied to this process is imported evidence, regardless of any
    # origin label embedded in that file.  Only probes executed by this process
    # can be authenticated as live in the signed qualification artifact.
    return tuple(
        ProbeResult(
            probe_id=item["probe_id"],
            passed=item["passed"],
            observed=item["observed"],
            evidence_sha256=item["evidence_sha256"],
            command_sha256=item.get("command_sha256", ""),
            evidence_origin="imported_evidence",
        )
        for item in payload["probe_results"]
    )


def main() -> int:
    args = parse_args()
    if args.execute_live_read_only_probes and not args.acknowledge_live_hardware:
        raise SystemExit(
            "--execute-live-read-only-probes requires --acknowledge-live-hardware"
        )
    if args.execute_live_read_only_probes and args.probe_results:
        raise SystemExit("choose live execution or --probe-results, not both")

    testbed = json.loads(args.testbed_config.read_text(encoding="utf-8"))
    route = json.loads(args.route_config.read_text(encoding="utf-8"))
    probe_results: tuple[ProbeResult, ...] = ()
    if args.execute_live_read_only_probes:
        if testbed.get("evidence_source") != "live_hardware":
            raise SystemExit("live probes require evidence_source=live_hardware")
        probe_results = run_read_only_qualification_probes(
            testbed,
            SubprocessReadOnlyProbeRunner(),
        )
    elif args.probe_results:
        probe_results = _load_probe_results(args.probe_results)

    testbed_report = validate_testbed_qualification(
        testbed,
        probe_results=probe_results,
        allow_fake_driver=args.allow_fake_driver_test,
    )
    route_report = validate_route_driver_configuration(
        route,
        allow_fake_driver=args.allow_fake_driver_test,
    )
    live_qualified = bool(
        args.execute_live_read_only_probes
        and testbed_report.live_eligible
        and route_report.live_eligible
    )
    if live_qualified and (not args.signing_key or not args.ledger_schema or not args.output):
        raise SystemExit(
            "issuing LIVE_QUALIFIED requires --signing-key, --ledger-schema, and --output"
        )
    output = {
        "schema_version": "1.0.0",
        "artifact_type": "live_testbed_qualification",
        "field_trial_status": "NOT_RUN",
        "qualification_status": "LIVE_QUALIFIED" if live_qualified else testbed_report.evidence_status,
        "evidence_source": testbed.get("evidence_source", "LIVE_REQUIRED"),
        "probe_evidence_origin": (
            "AUTHENTICATED_LIVE_EXECUTION"
            if live_qualified
            else "IMPORTED_UNAUTHENTICATED"
            if args.probe_results
            else "NOT_EXECUTED"
        ),
        "testbed_config_sha256": sha256_file(args.testbed_config),
        "route_config_sha256": sha256_file(args.route_config),
        "ledger_schema_sha256": (
            sha256_file(args.ledger_schema) if args.ledger_schema else ""
        ),
        "testbed_evidence_status": testbed_report.evidence_status,
        "testbed_live_eligible": testbed_report.live_eligible,
        "route_evidence_status": route_report.evidence_status,
        "route_live_eligible": route_report.live_eligible,
        "probe_results": [item.__dict__ for item in probe_results],
        "testbed_checks": testbed_report.checks.to_dict(orient="records"),
        "route_checks": route_report.checks.to_dict(orient="records"),
        "passed": testbed_report.passed and route_report.passed,
        "claim_boundary": (
            "Passing FAKE_DRIVER_TEST verifies software wiring only. "
            "Only live_hardware evidence can produce LIVE_QUALIFIED."
        ),
    }
    if live_qualified:
        output = sign_artifact(
            output,
            load_signing_key(args.signing_key),
            purpose="O4_LIVE_TESTBED_QUALIFICATION_V1",
        )
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
