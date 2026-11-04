from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from leo_service_survival.prospective_trial import (
    ClockSample,
    ClosedLoopController,
    CommandRouteDriver,
    DecisionContext,
    FirstPacketObservation,
    InstallAcknowledgement,
    RouteSnapshot,
    append_hash_chained_record,
    audit_decision_lifecycles,
    audit_hash_chain,
    build_reproducibility_manifest,
    load_ledger_schema,
    validate_campaign_register,
    validate_ledger_record,
    validate_trial_configuration,
    verify_reproducibility_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "prospective_trial"
SCHEMA_PATH = PACKAGE / "schema" / "decision_ledger.schema.json"
CONFIG_PATH = PACKAGE / "config" / "gate_canary_rollback.json"
REGISTER_PATH = PACKAGE / "config" / "acquisition_family_register.csv"


class FakeClock:
    def __init__(self, step_ms: float = 1.0) -> None:
        self.wall = 1_800_000_000_000_000_000
        self.mono = 100_000_000
        self.step_ns = int(step_ms * 1_000_000)

    def sample(self) -> ClockSample:
        sample = ClockSample(self.wall, self.mono)
        self.wall += self.step_ns
        self.mono += self.step_ns
        return sample


class FakeRouteDriver:
    def __init__(
        self,
        *,
        state: str = "baseline",
        candidate_ack: bool = True,
        candidate_mismatch: bool = False,
        packet_observed: bool = True,
        candidate_packet_path: str = "candidate",
    ) -> None:
        self.state = state
        self.candidate_ack = candidate_ack
        self.candidate_mismatch = candidate_mismatch
        self.packet_observed = packet_observed
        self.candidate_packet_path = candidate_packet_path
        self.install_calls: list[str] = []
        self._mismatch_pending = False

    def snapshot(self, destination: str) -> RouteSnapshot:
        if self._mismatch_pending:
            self._mismatch_pending = False
            return RouteSnapshot("unexpected_path", f"mismatch for {destination}")
        return RouteSnapshot(self.state, f"state={self.state}; destination={destination}")

    def install(self, path_id: str, destination: str) -> InstallAcknowledgement:
        self.install_calls.append(path_id)
        if path_id == "candidate":
            if not self.candidate_ack:
                return InstallAcknowledgement(False, "injected acknowledgement failure")
            self.state = path_id
            if self.candidate_mismatch:
                self._mismatch_pending = True
            return InstallAcknowledgement(True, "candidate ack")
        self.state = path_id
        return InstallAcknowledgement(True, "baseline ack")

    def observe_first_packet(
        self, path_id: str, destination: str, timeout_ms: float
    ) -> FirstPacketObservation:
        if self.packet_observed or path_id != "candidate":
            observed_path = (
                self.candidate_packet_path if path_id == "candidate" else path_id
            )
            return FirstPacketObservation(True, observed_path, "packet observed")
        return FirstPacketObservation(False, None, "injected packet timeout")


def _context(**changes: object) -> DecisionContext:
    base = DecisionContext(
        destination="198.51.100.10",
        baseline_path_id="baseline",
        proposed_path_id="candidate",
        gate_state="authorized",
        gate_reason="all_frozen_conditions_passed",
        guard_state="pass",
        monitor_state="pass",
        canary_state="candidate",
        collect_ms=2.0,
        feature_ms=2.0,
        decision_ms=2.0,
        disseminate_ms=2.0,
    )
    return replace(base, **changes)


def _schema() -> dict:
    return load_ledger_schema(SCHEMA_PATH)


def _template(name: str) -> dict:
    return json.loads((PACKAGE / "schema" / name).read_text(encoding="utf-8"))


def test_package_configuration_is_explicitly_not_run_and_valid() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    status = json.loads((PACKAGE / "STATUS.json").read_text(encoding="utf-8"))
    report = validate_trial_configuration(config)
    assert report.passed, report.checks.to_dict(orient="records")
    assert status["field_trial_status"] == "NOT_RUN"
    assert status["field_trial_completed"] is False
    assert status["benefit_claim_supported"] is False
    assert status["campaign_count_boundary"]["confirmatory_gate_campaign_floor"] == 5
    assert status["campaign_count_boundary"]["confirmatory_gate_family_floor"] == 3
    assert "not a universal" in status["campaign_count_boundary"]["interpretation"]
    assert status["external_dependencies"]


def test_campaign_register_separates_campaigns_families_and_roles() -> None:
    frame = pd.read_csv(REGISTER_PATH, keep_default_na=False)
    report = validate_campaign_register(frame)
    assert report.passed, report.checks.to_dict(orient="records")
    gate = frame[frame["role"].eq("gate_evidence")]
    assert gate["campaign_id"].nunique() == 5
    assert gate["acquisition_family_id"].nunique() == 3


def test_campaign_family_cannot_cross_roles() -> None:
    frame = pd.read_csv(REGISTER_PATH, keep_default_na=False)
    final_index = frame.index[frame["role"].eq("final_test")][0]
    frame.loc[final_index, "acquisition_family_id"] = "TBD-GATE-F01"
    report = validate_campaign_register(frame)
    check = report.checks.set_index("check")
    assert not bool(check.loc["register.family_role_separation", "passed"])


def test_repeated_pipeline_context_cannot_be_relabelled_as_independent_family() -> None:
    frame = pd.read_csv(REGISTER_PATH, keep_default_na=False)
    gate = frame.index[frame["role"].eq("gate_evidence")]
    first, second = gate[:2]
    for column in (
        "acquisition_pipeline_id",
        "site_id",
        "operator_id",
        "terminal_id",
        "route_id",
        "server_id",
    ):
        frame.loc[second, column] = frame.loc[first, column]
    report = validate_campaign_register(frame)
    check = report.checks.set_index("check")
    assert not bool(check.loc["register.no_family_relabelling", "passed"])


def test_registration_mode_rejects_planned_placeholders() -> None:
    frame = pd.read_csv(REGISTER_PATH, keep_default_na=False)
    report = validate_campaign_register(frame, require_locked=True)
    check = report.checks.set_index("check")
    assert not bool(check.loc["register.all_included_campaigns_locked", "passed"])


def test_schema_templates_and_data_dictionary_match() -> None:
    schema = _schema()
    dictionary = pd.read_csv(
        PACKAGE / "schema" / "decision_ledger_data_dictionary.csv",
        keep_default_na=False,
    )
    assert set(dictionary["field"]) == set(schema["fields"])
    assert not dictionary["field"].duplicated().any()
    assert validate_ledger_record(
        _template("action_record.template.json"), schema
    ).passed
    assert validate_ledger_record(
        _template("outcome_record.template.json"), schema
    ).passed


def test_ledger_rejects_invalid_hash_and_inconsistent_completion() -> None:
    schema = _schema()
    action = _template("action_record.template.json")
    action["model_sha256"] = "not-a-hash"
    action_report = validate_ledger_record(action, schema)
    assert not action_report.passed
    assert not bool(
        action_report.checks.set_index("check").loc["ledger.field_formats", "passed"]
    )

    outcome = _template("outcome_record.template.json")
    outcome["conservative_on_time_completion"] = True
    outcome_report = validate_ledger_record(outcome, schema)
    assert not outcome_report.passed
    assert not bool(
        outcome_report.checks.set_index("check").loc[
            "ledger.conservative_completion_definition", "passed"
        ]
    )


def test_hash_chained_ledger_detects_tampering(tmp_path: Path) -> None:
    schema = _schema()
    ledger = tmp_path / "trial.jsonl"
    append_hash_chained_record(
        ledger, _template("action_record.template.json"), schema
    )
    append_hash_chained_record(
        ledger, _template("outcome_record.template.json"), schema
    )
    assert audit_hash_chain(ledger, schema).passed

    lines = ledger.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["task_id"] = "silently-modified"
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not audit_hash_chain(ledger, schema).passed


def _linked_success_records() -> tuple[dict, dict]:
    action = _template("action_record.template.json")
    action.update(
        {
            "recorded_wall_ns": 100,
            "recorded_monotonic_ns": 100,
            "gate_state": "authorized",
            "gate_reason": "all_frozen_conditions_passed",
            "guard_state": "pass",
            "monitor_state": "pass",
            "canary_state": "candidate",
            "policy_arm": "gated_candidate",
            "available_path_ids": ["baseline", "candidate"],
            "baseline_path_id": "baseline",
            "proposed_path_id": "candidate",
            "intended_path_id": "candidate",
            "pre_fib_path_id": "baseline",
            "post_fib_path_id": "candidate",
            "installed_path_id": "candidate",
            "final_fib_path_id": "candidate",
            "install_requested": True,
            "install_acknowledged": True,
            "install_verified": True,
            "first_packet_observed": True,
            "first_packet_path_id": "candidate",
            "fallback_triggered": False,
            "fallback_reason": "",
            "rollback_triggered": False,
            "rollback_reason": "",
            "rollback_verified": True,
        }
    )
    outcome = _template("outcome_record.template.json")
    outcome.update(
        {
            "recorded_wall_ns": 200,
            "recorded_monotonic_ns": 200,
            "required_bytes": 100,
            "delivered_bytes": 100,
            "deadline_wall_ns": 500,
            "completion_wall_ns": 400,
            "application_acknowledgement_wall_ns": 450,
            "observation_valid": True,
            "application_complete": True,
            "met_deadline": True,
            "no_action": False,
            "controller_failure": False,
            "late_install": False,
            "conservative_on_time_completion": True,
            "failure_stage": "none",
            "missingness_reason": "",
        }
    )
    return action, outcome


def test_lifecycle_audit_links_installed_action_to_mature_outcome(tmp_path: Path) -> None:
    schema = _schema()
    action, outcome = _linked_success_records()
    ledger = tmp_path / "complete.jsonl"
    append_hash_chained_record(ledger, action, schema)
    append_hash_chained_record(ledger, outcome, schema)
    report = audit_decision_lifecycles(
        ledger, schema, require_mature_outcomes=True
    )
    assert report.passed, report.checks.to_dict(orient="records")


def test_lifecycle_audit_detects_unauthorized_candidate_installation(
    tmp_path: Path,
) -> None:
    schema = _schema()
    action, outcome = _linked_success_records()
    action["gate_state"] = "rejected"
    action["gate_reason"] = "tail_bound_failed"
    ledger = tmp_path / "unauthorized.jsonl"
    append_hash_chained_record(ledger, action, schema)
    append_hash_chained_record(ledger, outcome, schema)
    report = audit_decision_lifecycles(ledger, schema)
    check = report.checks.set_index("check")
    assert not bool(
        check.loc["lifecycle.no_unauthorized_candidate_installation", "passed"]
    )


def test_gate_rejection_retains_baseline_and_never_installs_candidate() -> None:
    driver = FakeRouteDriver(state="baseline")
    result = ClosedLoopController(driver, clock=FakeClock()).execute(
        _context(gate_state="rejected", gate_reason="tail_bound_failed")
    )
    assert not result.candidate_authorized
    assert result.intended_path_id == "baseline"
    assert "candidate" not in driver.install_calls
    assert result.final_fib_path_id == "baseline"
    assert result.fallback_triggered
    assert not result.rollback_triggered


def test_candidate_requires_gate_guard_monitor_and_canary_authorization() -> None:
    contexts = (
        _context(gate_state="insufficient_evidence"),
        _context(guard_state="fallback"),
        _context(monitor_state="alarm"),
        _context(canary_state="baseline"),
    )
    for context in contexts:
        driver = FakeRouteDriver(state="baseline")
        result = ClosedLoopController(driver, clock=FakeClock()).execute(context)
        assert not result.candidate_authorized
        assert result.intended_path_id == "baseline"
        assert "candidate" not in driver.install_calls


def test_authorized_candidate_is_verified_by_fib_and_first_packet() -> None:
    driver = FakeRouteDriver(state="baseline")
    result = ClosedLoopController(driver, clock=FakeClock()).execute(_context())
    assert result.candidate_authorized
    assert result.install_acknowledged
    assert result.install_verified
    assert result.first_packet_observed
    assert result.final_fib_path_id == "candidate"
    assert not result.rollback_triggered


def test_install_acknowledgement_failure_fails_closed() -> None:
    driver = FakeRouteDriver(state="baseline", candidate_ack=False)
    result = ClosedLoopController(driver, clock=FakeClock()).execute(_context())
    assert result.rollback_triggered
    assert "install_not_acknowledged" in result.rollback_reason
    assert result.final_fib_path_id == "baseline"


def test_fib_mismatch_triggers_verified_baseline_rollback() -> None:
    driver = FakeRouteDriver(state="baseline", candidate_mismatch=True)
    result = ClosedLoopController(driver, clock=FakeClock()).execute(_context())
    assert result.rollback_triggered
    assert "installed_path_mismatch" in result.rollback_reason
    assert driver.install_calls[-1] == "baseline"
    assert result.final_fib_path_id == "baseline"


def test_first_packet_timeout_triggers_baseline_rollback() -> None:
    driver = FakeRouteDriver(state="baseline", packet_observed=False)
    result = ClosedLoopController(driver, clock=FakeClock()).execute(_context())
    assert result.rollback_triggered
    assert "first_packet_timeout" in result.rollback_reason
    assert result.final_fib_path_id == "baseline"
    assert result.rollback_acknowledged
    assert result.rollback_install_verified
    assert result.rollback_first_packet_observed
    assert result.rollback_first_packet_path_id == "baseline"
    assert result.rollback_verified
    assert result.rollback_total_ms == result.rollback_install_ms + result.rollback_switch_ms
    assert result.action_total_ms == result.total_loop_ms + result.rollback_total_ms


def test_independent_first_packet_path_mismatch_forces_verified_rollback() -> None:
    driver = FakeRouteDriver(
        state="baseline", candidate_packet_path="unexpected_path"
    )
    result = ClosedLoopController(driver, clock=FakeClock()).execute(_context())
    assert result.first_packet_path_id == "unexpected_path"
    assert not result.first_packet_observed
    assert result.rollback_triggered
    assert result.rollback_verified
    assert result.rollback_first_packet_path_id == "baseline"


def test_control_loop_budget_breach_triggers_baseline_rollback() -> None:
    driver = FakeRouteDriver(state="baseline")
    result = ClosedLoopController(
        driver,
        clock=FakeClock(step_ms=60.0),
        maximum_total_loop_ms=100.0,
    ).execute(_context())
    assert result.total_loop_ms > 100.0
    assert result.rollback_triggered
    assert "control_loop_budget_exceeded" in result.rollback_reason
    assert result.final_fib_path_id == "baseline"


def test_command_driver_never_mutates_by_default() -> None:
    driver = CommandRouteDriver(
        snapshot_command=("/usr/bin/printf", "baseline\\n"),
        install_commands={"candidate": ("/usr/bin/true",)},
        first_packet_commands={"candidate": ("/usr/bin/true",)},
    )
    acknowledgement = driver.install("candidate", "198.51.100.10")
    assert not acknowledgement.acknowledged
    assert "disabled" in acknowledgement.raw


def test_command_driver_uses_observed_packet_path_not_requested_path() -> None:
    driver = CommandRouteDriver(
        snapshot_command=("/usr/bin/printf", "baseline\n"),
        install_commands={"candidate": ("/usr/bin/true",)},
        first_packet_commands={
            "candidate": ("/usr/bin/printf", "unexpected-path\n")
        },
    )
    observation = driver.observe_first_packet(
        "candidate", "198.51.100.10", 100.0
    )
    assert not observation.observed
    assert observation.path_id == "unexpected-path"


def test_reproducibility_manifest_roundtrip_and_tamper_detection(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    manifest = build_reproducibility_manifest(
        tmp_path,
        ("a.txt", "b.txt"),
        protocol_status="DRAFT_NOT_REGISTERED",
    )
    assert manifest["field_trial_completed"] is False
    assert verify_reproducibility_manifest(tmp_path, manifest).passed
    (tmp_path / "b.txt").write_text("changed", encoding="utf-8")
    assert not verify_reproducibility_manifest(tmp_path, manifest).passed
