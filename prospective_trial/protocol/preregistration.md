# Prospective Closed-Loop Field-Extension Preregistration

This protocol is the future deployment-validation extension to the completed
bounded offline Objective 4; it is not part of the evidence used to mark that
objective complete.

**Protocol:** `O4-PROSPECTIVE-CLOSED-LOOP-V1`  
**Status:** `DRAFT_NOT_REGISTERED`  
**Field trial:** `NOT_RUN`  
**Version date:** 2 November 2026

This document is a protocol awaiting supervisor, operator, and—where required—
institutional approval. It is not retrospective evidence and must not be cited as
a completed field experiment.

## 1. Research question and claim boundary

The experiment asks whether a frozen, evidence-gated cellular–LEO path policy can
be executed safely and traced from telemetry to the installed path, the first
post-installation packet, and the final application or Earth-observation (EO) task
outcome.

Two conclusions are possible:

1. If the gate rejects or lacks evidence, verified retention of the reactive
   baseline can establish that the safety mechanism is enforced prospectively.
   It cannot establish candidate benefit.
2. If the gate authorizes the candidate, a predefined randomized canary may
   estimate benefit and safety. A benefit claim requires the lower confidence
   bound for the prespecified primary effect to clear the frozen practical-gain
   threshold on untouched final-test families.

A negative gate decision or a null trial result is scientifically valid. No
universal superiority claim is planned.

## 2. Primary endpoint and estimand

The primary endpoint is **conservative on-time task completion** on the full
scheduled intention-to-treat denominator. A task succeeds only if:

- the intended action is verified as installed;
- the observation or application output is valid;
- all bytes declared before assignment arrive at the destination;
- the application reports completion; and
- completion occurs by the declared deadline.

No action, failed or late installation, controller failure, missing completion
evidence, incomplete bytes, invalid output, and deadline expiry count as failures.

The primary estimand after gate authorization is the risk difference

\[
P(Y=1\mid\text{gated candidate})-
P(Y=1\mid\text{reactive baseline})
\]

under block-level intention-to-treat assignment. If the gate does not authorize
the candidate, the primary operational estimand becomes the rate of unauthorized
candidate installations; its required value is zero.

## 3. Acquisition roles and independence

Every campaign belongs to exactly one acquisition family and one role:

| Role | May fit or tune? | May decide the gate? | May estimate final effect? |
|---|---:|---:|---:|
| Pilot | No confirmatory reuse | No | No |
| Development | Yes | No | No |
| Gate evidence | No | Yes | No |
| Untouched final test | No | No | Yes |

An acquisition family is defined by a common measurement pipeline and operating
context, including the collection system, route, terminal, provider, server,
site, and collection process. Repeated days from the same setup may be separate
campaigns but remain one acquisition family. A family may not cross development,
gate-evidence, or final-test roles.

The current confirmatory profile requires at least **five eligible gate campaigns
from at least three gate acquisition families**. These are conservative eligibility
floors, not a universal statistical minimum. The blinded pilot power and precision
analysis may require additional campaigns or blocks. At least one further family
is reserved for untouched final testing.

## 4. Study stages

### Stage 0 — instrumentation pilot

Confirm clock error, loss and timeout retention, route/FIB inspection, installation
acknowledgement, first-packet detection, endpoint maturity, missingness, event rate,
intracluster correlation, and loop latency. Pilot outcomes are not reused as gate
or final-test evidence.

### Stage 1 — development freeze

Fit the prediction model, preprocessing, calibration, immediate input-domain
guard, policy candidates, and reactive baseline using development families only.
Freeze the code, feature contract, model, configuration, route driver, endpoint,
and analysis before opening gate evidence.

### Stage 2 — concurrent shadow gate evidence

Probe all concurrently available paths while the reactive baseline remains
installed. Preserve timestamped losses and timeouts in the scheduled denominator.
Compute the prespecified gate bounds by acquisition family. No candidate action is
installed in this stage.

### Stage 3 — gate decision

All frozen conditions in `config/gate_canary_rollback.json` must pass. Missing,
nonfinite, stale, insufficiently independent, imprecise, unsafe, or shifted
evidence fails closed. Record one of:

- `insufficient_evidence` — an eligibility or precision requirement is unmet;
- `rejected` — evidence is sufficient to evaluate but a bound fails; or
- `authorized` — every frozen condition passes.

### Stage 4A — baseline-retention branch

If the gate is insufficient or rejected, keep the baseline installed, run the
controller prospectively, and verify that candidate installation count is zero.
Record the complete loop, action, first packet, and task outcome. This branch tests
safe enforcement; it does not test candidate benefit.

### Stage 4B — randomized canary branch

If authorized, begin with the frozen 10% candidate exposure cap using complete
site-day or session blocks, stratified by site, operator or terminal context, and
time of day. Candidate allocation may increase only at registered checkpoints.
All failures remain in the assigned arm.

### Stage 5 — untouched final test

Open the final-test family only after the gate decision, canary rule, statistical
code, and artifact hashes are frozen. No outcome-driven tuning is permitted.

## 5. Intervention and baseline

The baseline is the predefined reactive policy used in Paper 2, with its exact
thresholds and implementation frozen before the trial. The candidate uses frozen
usable-service probabilities and the frozen evidence gate. Both receive identical
contemporaneously available inputs and feasible paths.

Session continuity must be provided by MP-QUIC, MPTCP, or an anchored tunnel. The
chosen mechanism, version, anchor, route tables, firmware, and terminal software
must be recorded in the campaign register and reproducibility manifest.

## 6. Required action trace

Every decision must connect, with one `decision_id`:

1. telemetry sample and cutoff;
2. candidate availability and proposal;
3. gate, immediate guard, delayed monitor, and canary state;
4. baseline and intended paths;
5. route installation request and acknowledgement;
6. pre- and post-installation FIB or tunnel state;
7. first post-installation packet and observed path;
8. fallback or rollback; and
9. valid output, complete bytes, application completion, and deadline outcome.

The loop decomposition is

\[
L_{collect}+L_{feature}+L_{decision}+L_{disseminate}+
L_{install}+L_{switch}.
\]

Monotonic time is used for duration; synchronized wall time is used for joining
systems. The worst-case clock error is recorded per campaign.

## 7. Gate, canary, and rollback rule

The machine-readable rule is authoritative. Before registration, the supervisor
and operator must approve every threshold and the pilot must confirm precision.
The candidate starts in shadow mode and can be installed only when:

- campaign-level gate state is `authorized`;
- the immediate guard is `pass`;
- the delayed monitor has no active alarm;
- the randomized canary arm is `candidate`; and
- the action has not expired.

Rollback is mandatory for emergency stop, missing acknowledgement, installed-path
mismatch, first-packet timeout, loop-budget breach, shift alarm, endpoint safety
breach, or clock/logging failure. Re-entry requires a new gate decision.

## 8. Registration and freeze checklist

Before changing `protocol_status` to `FROZEN_REGISTERED`:

- obtain all approvals and record signatories and timestamps;
- replace every `TBD` in the included campaign and route configurations;
- run the blinded pilot power and precision analysis;
- freeze the primary endpoint, practical-gain and noninferiority margins;
- freeze randomization blocks, seed commitment, and checkpoint schedule;
- freeze model, code, containers, firmware, preprocessing, feature contract,
  route commands, schema, and analysis code;
- generate and externally timestamp the reproducibility manifest;
- validate the package with `--require-locked-campaigns`; and
- store the registration record where it cannot be silently overwritten.

## 9. Conditions for calling the prospective field extension executed

The prospective field extension is executed only after role-separated field
data have been collected and the registered branch has been completed. The
record must contain verified installed actions, first packets, complete timing,
mature task outcomes, all failures and exclusions, and the untouched-test
analysis. Local code, replay, hardware-in-the-loop smoke tests, ns-3, or
synthetic admission do not by themselves satisfy this field-extension
condition; they are part of the bounded offline evidence used for Objective 4.
