# Prospective Trial Execution Runbook

This runbook is for an approved deployment. The current package status is
`NOT_RUN`; do not execute route changes on a production gateway without the named
operator and institutional permissions.

## 1. Required systems

- dual-access gateway with cellular/5G and LEO interfaces;
- session-preserving MP-QUIC, MPTCP, or anchored tunnel;
- controlled remote anchor and application endpoint;
- route or tunnel control-plane acknowledgement;
- route/FIB resolver independent of controller intent;
- first-packet observation at the gateway and preferably at the anchor;
- synchronized clocks with measured worst-case error;
- packet capture, timestamped loss/timeouts, and durable storage;
- emergency baseline restore outside the experimental process; and
- operational application or EO tasks with declared bytes, deadlines, validity,
  and destination completion acknowledgements.

## 2. Seven-step frozen workflow

The commands below implement the workflow; angle-bracket values must come from the
approved preregistration. A confirmatory run must not use the fake-driver switch.

### Before Step 1: freeze the blinded block-count plan

Use only label-free aggregate pilot quantities. The planner does not accept arm
labels or treatment-specific estimates, applies the declared clustering and
attrition adjustments, and selects the larger power or confidence-interval
requirement:

```bash
PYTHONPATH=src python3 scripts/plan_blinded_pilot.py \
  --baseline-completion <blinded-pooled-completion> --block-size <opportunities> \
  --icc <blinded-icc> --missing-fraction <fraction> \
  --installation-failure-fraction <fraction> --effect-margin <absolute-margin> \
  --target-power <power> --target-ci-half-width <half-width> \
  --signing-key <operator-key> --output <signed-blinded-pilot-plan.json>
```

The signed plan is a sample-size and precision artifact, not gate evidence and not
an authorization result.

### Step 1: qualify the live testbed

Complete the blinded instrumentation pilot, replace every `TBD`, freeze the route
and trial configurations, and execute only the configured read-only probes. The
frozen route file must also contain the approved total-loop and first-packet
budgets plus an attested isolated rollback namespace and its inject, restore,
verify, and first-packet commands:

```bash
# Run from the extracted project or release-package root.
PYTHONPATH=src python3 scripts/qualify_field_testbed.py \
  --testbed-config <approved-testbed.json> \
  --route-config <frozen-route-config.json> \
  --ledger-schema prospective_trial/schema/decision_ledger.schema.json \
  --signing-key <operator-key> \
  --execute-live-read-only-probes --acknowledge-live-hardware \
  --output <signed-live-qualification.json>
```

Imported or fake probe results can test software but cannot produce
`LIVE_QUALIFIED` evidence.

### Step 2: lock role-separated campaigns

The source register must identify development, gate-evidence, canary, and untouched
final-test roles before outcomes are opened. Repeated days from one unchanged setup
remain one acquisition family. `raw_capture_sha256` binds the synchronized telemetry
capture used by live-action verification. A separate `gate_opportunity_sha256`
binds the paired-opportunity CSV used by the gate and is required only for
gate-evidence campaigns; the two hashes must never be substituted for each other.

```bash
PYTHONPATH=src python3 scripts/lock_campaign_registry.py \
  --input <completed-campaign-register.csv> \
  --protocol-sha256 <registered-protocol-sha256> \
  --qualification-sha256 <qualification-artifact-sha256> \
  --lock-timestamp-utc <registered-utc-time> \
  --output <campaign-registry-lock.json>
```

### Step 3: ingest synchronized, loss-preserving telemetry

```bash
PYTHONPATH=src python3 scripts/ingest_synchronized_telemetry.py \
  --input <raw-telemetry.jsonl> \
  --expected-probe-schedule <expected-probe-schedule.csv> \
  --campaign-lock <campaign-registry-lock.json> \
  --output <validated-telemetry.jsonl>
```

The ingester refuses missing scheduled probes disguised as complete cases and binds
network, RF/beam/handover, route, timing, and task records to the locked identities.

### Step 4: freeze and run the raw-derived gate

First build the authenticated trial freeze, then bind one opportunity table to each
registered gate campaign through its separately frozen
`gate_opportunity_sha256`, recompute the declared metrics, and run the gate. The
raw telemetry remains independently bound by `raw_capture_sha256`:

```bash
PYTHONPATH=src python3 scripts/build_trial_freeze_manifest.py \
  --execution-mode confirmatory_live --protocol-id <protocol-id> \
  --freeze-timestamp-utc <registered-utc-time> \
  --testbed-qualification <signed-live-qualification.json> \
  --route-config <frozen-route-config.json> \
  --campaign-registry-lock <campaign-registry-lock.json> \
  --ledger-schema prospective_trial/schema/decision_ledger.schema.json \
  --model <frozen-model-artifact> --trial-config <frozen-trial-config.json> \
  --analysis-plan <frozen-analysis-plan.json> \
  --opportunity-derivation <frozen-derivation-config.json> \
  --signing-key <operator-key> --output <trial-freeze-manifest.json>

PYTHONPATH=src python3 scripts/build_gate_opportunity_manifest.py \
  --trial-freeze-manifest <trial-freeze-manifest.json> \
  --signing-key <operator-key> --execution-mode confirmatory_live \
  --table <campaign-id>=<campaign-opportunities.csv> \
  --output <opportunity-manifest.json>

PYTHONPATH=src python3 scripts/build_locked_gate_evidence.py \
  --trial-freeze-manifest <trial-freeze-manifest.json> \
  --opportunity-manifest <opportunity-manifest.json> \
  --signing-key <operator-key> --execution-mode confirmatory_live \
  --output <locked-gate-evidence.json>

PYTHONPATH=src python3 scripts/run_frozen_evidence_gate.py \
  --trial-config <frozen-trial-config.json> \
  --campaign-register <completed-campaign-register.csv> \
  --gate-evidence <locked-gate-evidence.json> \
  --trial-freeze-manifest <trial-freeze-manifest.json> \
  --opportunity-manifest <opportunity-manifest.json> \
  --signing-key <operator-key> --confirmatory-live --execute-frozen-gate \
  --output <signed-gate-report.json>
```

Hand-authored pooled metrics are rejected in confirmatory mode. Any changed frozen
file, table, role, or hash invalidates the run.

## 3. Failure-injection qualification

Run failure injection on an isolated gateway or network namespace. Verify:

- gate rejection cannot install a candidate;
- missing acknowledgement restores the baseline;
- FIB mismatch restores the baseline;
- first-packet timeout restores the baseline;
- loop-budget breach restores the baseline;
- shift alarm and emergency stop restore the baseline;
- process restart does not break the ledger hash chain;
- clock failure stops candidate action;
- route and packet evidence refer to the same destination and session; and
- no-action and failed-action tasks appear in the endpoint denominator.

The repository unit tests and controlled seven-step rehearsal exercise these branches
with deterministic fake hardware. Deployment qualification must repeat them on the
actual gateway and sign the observed evidence.

## 4. Stage execution

### Pilot and development

Use pilot/development register rows only. Do not relabel these data later. Record
environment, firmware, operator, terminal, route, server, and clock-error changes.

### Gate evidence

Run all candidates concurrently in shadow where technically possible. Keep the
reactive baseline installed. Capture losses and timeouts rather than dropping them.
Do not open final-test data. Generate and sign the gate report.

### Gate branch

- `insufficient_evidence` or `rejected`: execute baseline-retention validation;
  candidate mutation remains forbidden.
- `authorized`: reveal the canary allocation and begin with the frozen exposure
  cap. The operator holds an independent emergency stop.

### Final test

Open only after the branch, code, analysis, and manifest are locked. No retuning is
permitted. Run until the registered information target is reached or a registered
stop occurs.

## 5. Steps 5--7: branch, actuate, and reconcile

If the signed gate rejects or is insufficient, run
`verify_rejection_fallback.py`. It prohibits candidate mutation, resolves the
baseline independently from the controller request, observes the first packet, and
signs the baseline/rollback proof. If the gate authorizes, build the fixed canary
plan, evaluate every registered safety trigger, and run
`execute_canary_rollback.py` whenever the checkpoint stops. Only a passed signed
checkpoint plus an explicit no-retuning attestation can unlock the separately
randomized final-test plan.

The command below illustrates one authorized canary action. Populate every value
from the signed artifacts; never reuse placeholder hashes. The runner refuses live
mutation unless the qualification, freeze, route configuration, campaign role,
telemetry file, assignment, and model all match.

```bash
# Continue from the extracted project or release-package root.
PYTHONPATH=src python3 scripts/run_prospective_action.py \
  --route-config <frozen-route-config.json> \
  --testbed-config <approved-testbed.json> \
  --qualification-artifact <signed-live-qualification.json> \
  --campaign-lock <campaign-registry-lock.json> \
  --trial-freeze-manifest <trial-freeze-manifest.json> \
  --trial-config <frozen-trial-config.json> \
  --schema prospective_trial/schema/decision_ledger.schema.json \
  --ledger <durable-decision-ledger.jsonl> \
  --study-id <study-id> --trial-id <trial-id> --campaign-id <campaign-id> \
  --acquisition-family-id <family-id> --session-id <session-id> \
  --task-id <task-id> --decision-id <decision-id> \
  --model-sha256 <frozen-model-sha256> \
  --raw-telemetry-sha256 <raw-telemetry-sha256> \
  --raw-telemetry-file <raw-telemetry.jsonl> \
  --evidence-tier field_live_randomized \
  --available-path-id cellular --available-path-id leo \
  --proposed-path-id <assigned-path> \
  --gate-report <signed-gate-report.json> --signing-key <operator-key> \
  --assignment-plan <signed-canary-plan.json> --release-phase canary \
  --block-id <assigned-block-id> --guard-state pass --monitor-state pass \
  --telemetry-age-ms <ms> --collect-ms <ms> --feature-ms <ms> \
  --decision-ms <ms> --disseminate-ms <ms> \
  --execute-live-route-change
```

The action event is complete only after FIB verification and first-packet
observation. When the endpoint matures, the controlled destination application
first emits one immutable acknowledgement bound to the audited action:

```bash
PYTHONPATH=src python3 scripts/emit_application_outcome_acknowledgement.py \
  --ledger <durable-decision-ledger.jsonl> \
  --ledger-schema prospective_trial/schema/decision_ledger.schema.json \
  --acknowledgement-schema prospective_trial/schema/application_outcome_acknowledgement.schema.json \
  --decision-id <decision-id> --signing-key <operator-key> \
  --output <signed-application-acknowledgement.json> \
  --acknowledgement-id <unique-ack-id> \
  --acknowledgement-source <controlled-endpoint-id> \
  --required-bytes <bytes> --delivered-bytes <bytes> \
  --deadline-wall-ns <ns> --completion-wall-ns <ns> \
  --observation-valid <true-or-false> --application-complete <true-or-false> \
  --no-action <true-or-false> --controller-failure <true-or-false> \
  --late-install <true-or-false>
```

The emitter does not accept a success flag and does not alter the ledger or trial
status. Transfer the read-only acknowledgement to the ledger host, then append the
linked outcome with:

```bash
PYTHONPATH=src python3 scripts/append_prospective_outcome.py \
  --ledger <durable-decision-ledger.jsonl> \
  --schema prospective_trial/schema/decision_ledger.schema.json \
  --decision-id <decision-id> \
  --outcome-evidence <signed-application-acknowledgement.json> \
  --signing-key <operator-key>
```

The acknowledgement is purpose-bound to the exact study, campaign, session, task,
decision, model, and configuration. The helper recomputes its evidence and body
hashes and derives `met_deadline`, failure stage, and conservative completion from
the timestamps, byte counts, validity, and linked action; these are not CLI success
flags. It refuses a second outcome for the same decision. Never edit an existing
JSONL line; corrections are new authenticated records with a documented reason in
the controlled analysis layer.

After all registered outcomes mature, build the signed audit and run only the
frozen intention-to-treat analysis:

```bash
PYTHONPATH=src python3 scripts/build_closed_loop_audit_report.py \
  --ledger <durable-decision-ledger.jsonl> \
  --schema prospective_trial/schema/decision_ledger.schema.json \
  --campaign-register <completed-campaign-register.csv> \
  --gate-report <signed-gate-report.json> \
  --checkpoint-report <signed-canary-checkpoint.json> \
  --final-test-unlock <signed-final-test-unlock.json> \
  --final-test-plan <signed-final-test-plan.json> \
  --signing-key <operator-key> --output <signed-end-to-end-audit.json>

PYTHONPATH=src python3 scripts/run_final_test_itt_analysis.py \
  --ledger <durable-decision-ledger.jsonl> \
  --schema prospective_trial/schema/decision_ledger.schema.json \
  --campaign-register <completed-campaign-register.csv> \
  --analysis-spec <frozen-final-test-analysis.json> \
  --gate-report <signed-gate-report.json> \
  --final-test-unlock <signed-final-test-unlock.json> \
  --final-test-plan <signed-final-test-plan.json> \
  --end-to-end-audit <signed-end-to-end-audit.json> \
  --signing-key <operator-key> --execute-frozen-analysis \
  --output <signed-final-itt-analysis.json>
```

The final plan signs the analysis-specification hash. A deployment-benefit flag also
requires confirmatory-live gate evidence, authenticated live outcome
acknowledgements, a matching signed audit, and at least the frozen number of
independent randomized blocks per arm; the enforced software floor is ten.

## 6. Daily closeout

1. Stop new candidate assignments.
2. Allow outstanding outcomes to mature or mark them missing by the frozen rule.
3. Verify the ledger hash chain and raw-capture hashes.
4. Reconcile randomized assignment, proposed action, installed action, and first
   packet counts.
5. Confirm the gateway is on the reactive baseline.
6. Copy logs to write-once storage and record the transfer hashes.
7. Sign the campaign manifest row and incident report.

## 7. Live execution boundary

All seven software mechanisms above are implemented and adversarially tested. Their
corresponding field activities below are **`NOT_RUN`**. This is an execution
checklist, not a record of completed live work.

1. Obtain approved cellular/5G and LEO terminals, a dual-access gateway, a controlled
   remote anchor, active service, and required institutional and operator permissions.
2. Collect role-separated development, gate-evidence, and untouched final-test
   acquisition families.
3. Preserve synchronized losses, timeouts, RF/beam/handover observations,
   installed-route evidence, and operational application or EO outcomes.
4. Run the frozen evidence gate and sign its report.
5. If the gate rejects, verify that the reactive baseline remains installed and run
   the registered rollback drill. This is a valid prospective safety result, not a
   candidate-benefit result.
6. If the gate authorizes, execute the predefined canary and then the untouched
   final test without retuning. The execution permits assessment of benefit; a
   benefit claim additionally requires the frozen performance and safety criteria
   to be met.
7. Reconcile proposed and installed routes, installation acknowledgements, first
   post-installation packets, complete controller-loop latency, and final
   application or EO outcomes.

The deterministic rehearsal can be regenerated with
`scripts/run_controlled_seven_step_dry_run.py --acknowledge-fake-driver-test`, but it
cannot supply these live dependencies or observations. The bounded offline
Objective 4 is complete; until the registered protocol is executed, its prospective
field extension remains `NOT_RUN` and unvalidated on hardware. A rejected live gate
can establish safe baseline retention;
only an authorized candidate followed by successful canary and untouched final-test
evidence that satisfies the frozen criteria can support a live-benefit claim.
