#!/usr/bin/env python3
"""Verify rejected-policy baseline retention and an isolated rollback drill.

This command can mutate routes.  It refuses template configurations, requires
two explicit execution acknowledgements, and permits the injected fault only in
the named isolated test namespace defined by the frozen route configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leo_service_survival.prospective_release import (  # noqa: E402
    RollbackDrillResult,
    load_signing_key,
    run_rejection_branch,
)
from leo_service_survival.prospective_trial import (  # noqa: E402
    ClosedLoopController,
    CommandRouteDriver,
    DecisionContext,
)


class CommandIsolatedRollbackDrillDriver:
    def __init__(self, configuration: dict) -> None:
        self.configuration = configuration

    @staticmethod
    def _expand(command: list[str], **values: str) -> list[str]:
        return [
            token.replace("{namespace_id}", values["namespace_id"])
            .replace("{destination}", values["destination"])
            .replace("{baseline_path_id}", values["baseline_path_id"])
            for token in command
        ]

    def _run(self, name: str, **values: str) -> subprocess.CompletedProcess[str]:
        command = self._expand(self.configuration[name], **values)
        return subprocess.run(command, shell=False, check=False, capture_output=True, text=True, timeout=10.0)

    def run_isolated_rollback_drill(
        self,
        *,
        isolated_namespace_id: str,
        destination: str,
        baseline_path_id: str,
    ) -> RollbackDrillResult:
        if isolated_namespace_id != self.configuration.get("namespace_id"):
            raise ValueError("requested drill namespace differs from frozen configuration")
        values = {
            "namespace_id": isolated_namespace_id,
            "destination": destination,
            "baseline_path_id": baseline_path_id,
        }
        inject = self._run("inject_fault_command", **values)
        rollback = self._run("restore_baseline_command", **values)
        snapshot = self._run("verify_baseline_command", **values)
        packet = self._run("observe_first_packet_command", **values)
        detail = " | ".join(
            f"{name}:rc={result.returncode}:{(result.stdout + result.stderr).strip()}"
            for name, result in (
                ("inject", inject),
                ("rollback", rollback),
                ("snapshot", snapshot),
                ("packet", packet),
            )
        )
        return RollbackDrillResult(
            isolated_namespace_id=isolated_namespace_id,
            fault_injected=inject.returncode == 0,
            rollback_acknowledged=rollback.returncode == 0,
            baseline_verified=snapshot.returncode == 0,
            first_packet_observed=packet.returncode == 0,
            details=detail,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-live-baseline-verification", action="store_true")
    parser.add_argument("--execute-isolated-rollback-drill", action="store_true")
    args = parser.parse_args()
    if not args.execute_live_baseline_verification or not args.execute_isolated_rollback_drill:
        raise RuntimeError("both live baseline and isolated-drill acknowledgements are required")
    route = json.loads(args.route_config.read_text(encoding="utf-8"))
    if route.get("status") != "FROZEN_DEPLOYMENT_CONFIG":
        raise RuntimeError("route configuration is not frozen for deployment")
    drill_config = route.get("isolated_rollback_drill")
    if not isinstance(drill_config, dict) or drill_config.get("isolation_attested") is not True:
        raise RuntimeError("frozen isolated rollback-drill configuration and attestation are required")

    driver = CommandRouteDriver(
        snapshot_command=route["snapshot_command"],
        install_commands=route["install_commands"],
        first_packet_commands=route["first_packet_commands"],
        allow_live_mutation=True,
    )
    controller = ClosedLoopController(
        driver,
        maximum_total_loop_ms=float(route["maximum_total_loop_ms"]),
        first_packet_timeout_ms=float(route["first_packet_timeout_ms"]),
    )
    context = DecisionContext(
        destination=route["destination"],
        baseline_path_id=route["baseline_path_id"],
        proposed_path_id=route["candidate_path_ids"][0],
        gate_state="not_evaluated",
        gate_reason="loaded_from_signed_report",
        guard_state="fallback",
        monitor_state="not_mature",
        canary_state="baseline",
        collect_ms=0.0,
        feature_ms=0.0,
        decision_ms=0.0,
        disseminate_ms=0.0,
    )
    report = run_rejection_branch(
        gate_report=json.loads(args.gate_report.read_text(encoding="utf-8")),
        signing_key=load_signing_key(args.signing_key),
        controller=controller,
        context=context,
        drill_driver=CommandIsolatedRollbackDrillDriver(drill_config),
        isolated_namespace_id=drill_config["namespace_id"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "artifact_sha256": report["artifact_sha256"]}, indent=2))
    return 0 if report["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
