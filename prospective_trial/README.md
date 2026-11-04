# Objective 4 Offline Integration Package and Prospective Extension

This directory completes the bounded offline integration and evaluation scope of
Objective 4 and preserves its prospective field-extension protocol. All seven
software mechanisms are implemented and tested through evidence-bound records,
controlled drivers, fail-closed branches, and deterministic rehearsal. The live
field experiment remains **`NOT_RUN`**: no controlled rehearsal, unit test, or
generated record in this directory is evidence that real terminals, independent
campaigns, installed field actions, or operational task outcomes were observed.

Objective 4 is therefore **complete within the declared bounded offline scope**.
Prospective hardware-integrated execution is future deployment validation needed
before any live-routing or deployment-benefit claim.

## What is implemented

| Step | Executable mechanism | Software status | Live status |
|---|---|---|---|
| 1 | Signed qualification of 5G/LEO service, dual-access gateway, independent route observation, emergency restore, clock bounds, session preservation, and approvals | Implemented and tested | `NOT_RUN` |
| 2 | Immutable role-separated campaign registry with independent raw-telemetry and gate-opportunity hashes, non-overlap checks, and distinct campaign/family eligibility | Implemented and tested | `NOT_RUN` |
| 3 | Loss-preserving synchronized ingestion for replies, losses, timeouts, RF/beam/handover observations, route stages, timing, and final outcomes | Implemented and tested | `NOT_RUN` |
| 4 | HMAC-authenticated trial freeze and evidence gate whose confirmatory metrics are recomputed from hash-locked per-campaign opportunity tables | Implemented and tested | `NOT_RUN` |
| 5 | Rejection branch that prohibits candidate mutation and independently verifies baseline FIB, first packet, and rollback | Implemented and tested | `NOT_RUN` |
| 6 | Frozen canary assignment, complete safety checkpoint, verified stopped-canary rollback, signed no-retuning unlock, and separately frozen untouched final test | Implemented and tested | `NOT_RUN` |
| 7 | Hash-chained intended-to-installed-action ledger, full loop and rollback timing, first-packet evidence, final outcome linkage, audit, and conservative intention-to-treat analysis | Implemented and tested | `NOT_RUN` |

The package also includes the preregistration, statistical analysis plan, threshold
provenance, append-only SHA-256 ledger, lifecycle checks, package validator,
blinded clustered power/precision planner, production endpoint-acknowledgement
emitter, and reproducibility-manifest builder. Confirmatory/live execution refuses hand-authored
pooled gate metrics, imported probe claims, changed frozen files, wrong campaign
roles, unauthorized assignments, and unverified route or rollback claims.

## Controlled seven-step rehearsal

Run `scripts/run_controlled_seven_step_dry_run.py` to exercise the whole chain with
deterministic fake drivers. The generated artifacts are in
`controlled_dry_run/` and are labelled **`FAKE_DRIVER_TEST`** throughout. The
rehearsal derives gate metrics from five hash-locked fake campaign tables, verifies
a rejection/rollback branch, exercises an authorized canary/final-test branch, and
audits 20 action--outcome pairs across ten randomized blocks per arm. Its frozen
illustrative analysis is neutral (candidate 0.5, baseline 0.5, risk difference 0.0,
95% interval [-0.4, 0.4] on the risk-proportion scale, or -40 to +40 percentage
points) and sets `benefit_claim_supported=false`. This is a plumbing and analysis-
path check; controlled fixtures can never establish deployment benefit.

## Future hardware-integrated validation

The corresponding actions on approved live hardware have not been run. Hardware,
operator access, role-separated field acquisition families, synchronized field
measurements, installed live actions, and operational application/EO outcomes remain
external dependencies. Consequently, the package completes the stated offline
integration objective but does not establish field validation or candidate
benefit. See `STATUS.json`.

## Future live execution checklist (`NOT_RUN`)

The software for every step exists, but the following real-world executions remain
prospective:

1. Obtain approved 5G and LEO terminals with active service, a dual-access
   gateway, a controlled remote anchor, session-preserving connectivity, and the
   required institutional and operator permissions.
2. Collect role-separated development, gate-evidence, and untouched final-test
   acquisition families.
3. Preserve synchronized loss and timeout records, RF/beam/handover observations,
   proposed and installed actions, and operational application or EO outcomes.
4. Run the frozen evidence gate using only the registered gate-evidence
   campaigns.
5. If rejected, verify that the reactive baseline remains installed and exercise
   the predefined rollback path. This is a scientifically valid prospective
   safety result, but it cannot support candidate benefit.
6. If authorized, run the registered canary with automatic rollback and then,
   only if the canary passes, run the untouched final test. This branch supplies
   the prospective evidence needed to assess benefit; a benefit claim additionally
   requires the final results to satisfy the prespecified performance and safety
   criteria.
7. For either branch, record the intended path, installed path, installation
   acknowledgement, first post-installation packet, the complete control-loop
   latency, rollback events, and final application or EO task outcomes.

Only an authorized candidate followed by successful canary and untouched final-test
evidence can support a live-benefit claim.

## Local verification

```bash
# Run from the extracted project or release-package root.
PYTHONPATH=src python3 scripts/validate_prospective_trial_package.py
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 scripts/run_controlled_seven_step_dry_run.py \
  --acknowledge-fake-driver-test --output /tmp/O4_controlled_dry_run
PYTHONPATH=src python3 scripts/build_prospective_trial_manifest.py \
  --output /tmp/O4_local_package_manifest.json
PYTHONPATH=src python3 scripts/build_prospective_trial_manifest.py \
  --verify /tmp/O4_local_package_manifest.json
```

The normal validation accepts planned campaigns and verifies the explicit separation
between implemented software, fake rehearsal, and live `NOT_RUN` state. At
registration, use
`--require-locked-campaigns`; it must fail until the placeholders are replaced,
campaign roles are locked, and real artifact hashes are recorded.

The complete private working tree contains **223 passing tests**, and the
redistribution-safe extracted package contains **216 passing tests**. The tests
and fake-driver rehearsal verify software behavior only; they do not establish
live hardware integration, operational reliability, or deployment benefit.

Before the randomized field phase, create and authenticate the blinded block-count
plan with `scripts/plan_blinded_pilot.py`. At the controlled application endpoint,
use `scripts/emit_application_outcome_acknowledgement.py` to produce one immutable,
purpose-bound acknowledgement for an already audited live action. The latter records
endpoint facts and is consumed by the ledger append step; it does not self-declare
task success or alter the trial status.

## Evidence interpretation

A prospective negative gate decision can complete evaluation of safe baseline
retention if the installed action, first packet, full timing, and final task outcome
are recorded. It cannot support a candidate-benefit claim. Candidate benefit
requires gate authorization, a successful registered canary, and favorable
untouched final-test evidence.
