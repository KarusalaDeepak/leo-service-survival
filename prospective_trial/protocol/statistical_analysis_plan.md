# Statistical Analysis Plan

**Protocol:** `O4-PROSPECTIVE-CLOSED-LOOP-V1`  
**Status:** draft; no field data have been opened  
**Authoritative configuration:** `../config/gate_canary_rollback.json`

## 1. Analysis populations

- **Intention to treat (primary):** every scheduled task in its assigned block,
  including no action, failed installation, rollback, controller failure, missing
  completion evidence, and deadline expiry.
- **Installed action (secondary):** tasks for which the assigned action was
  acknowledged and verified in the route table or tunnel state.
- **Per protocol (sensitivity only):** tasks satisfying the frozen telemetry,
  clock, and protocol contract. This population cannot replace the primary result.
- **Shadow gate population:** all paired opportunities in the frozen gate-evidence
  families, including loss and timestamped timeout records.

Pilot and development families are excluded from confirmatory effect estimation.
Final-test families remain inaccessible until freeze.

## 2. Independent unit and randomization

Time bins, packets, and tasks within one block are repeated observations, not
independent samples. The primary randomization and resampling field is therefore
fixed to `block_id`; blocks are prespecified site-day or session units nested within
acquisition family. The family definition is frozen in the acquisition register.
Repeated days collected with one acquisition pipeline do not increase the number of
independent families.

If the canary is authorized, blocks are assigned in a stratified crossover using a
seed committed before test access. Strata are site, operator or terminal context,
and time of day. Carryover is controlled with a frozen washout interval. Assignment
failures remain in the assigned arm.

## 3. Sample size and precision

The pilot estimates baseline completion, block size, intracluster correlation,
missingness, install-failure rate, and endpoint maturity. The calculation must
target both:

1. the registered power for the minimum practically important completion effect;
2. the registered maximum confidence-interval half-width.

The larger requirement governs and must provide at least ten independent randomized
blocks per arm, which is a software claim floor rather than a substitute for the
blinded power calculation. Five gate campaigns and three gate families are
only the conservative confirmatory eligibility floor. Collection continues or the
trial is declared underpowered if the blinded information calculation requires
more; the threshold is not reduced after outcome inspection.

The signed pilot report, code, parameters, seed, and output hash become part of the
freeze manifest. No effect estimate from pilot data is pooled into the final test.

## 4. Gate analysis

Gate evidence is generated from paired decision opportunities and aggregated by
campaign and acquisition family. Every condition must pass:

- schema and clock audit;
- campaign/family and paired-opportunity support;
- feature coverage and input-domain bound;
- task-horizon calibration-gap upper confidence bound;
- candidate-minus-baseline safety lower confidence bound;
- capped latency-tail or control-loop requirement;
- effect-precision requirement; and
- no active shift alarm.

The gate reports insufficiency separately from evidence-based rejection. A logic
fixture or synthetic passing case cannot authorize field release. Gate estimates,
confidence level, multiplicity procedure, and reason codes are archived even when
the candidate is rejected.

## 5. Primary endpoint analysis after authorization

For each randomized block, compute the conservative completion proportion. The
primary effect is the candidate-minus-baseline risk difference. Estimate it with a
prespecified family-aware block analysis and report a two-sided 95% confidence
interval. With few randomized blocks, use exact or permutation-based randomization
inference consistent with the allocation. With adequate blocks, the registered
cluster bootstrap or a generalized linear mixed model may be used; the method is
chosen and frozen after the blinded pilot, not after treatment labels are opened.

Candidate benefit is supported only when the lower 95% confidence bound exceeds
the frozen minimum practical gain, at least the frozen number of independent blocks
exists in each arm, the gate is authenticated `CONFIRMATORY_LIVE`, and every used
outcome has a verified live application acknowledgement. A controlled fixture may
exercise the numerical criterion but can never set the deployment-benefit flag.
Safety noninferiority is assessed first using the frozen −0.02 margin. Failure of
noninferiority stops the hierarchy.

Before outcomes are opened, the signed final-test plan binds the analysis-specification
hash. Final analysis additionally requires a signed end-to-end audit binding the
ledger, schema, campaign register, gate report, final-test unlock, and assignment
plan. Any drift or identity mismatch makes the analysis ineligible.

## 6. Analysis when the gate does not authorize

Do not estimate a causal candidate effect from shadow proposals. Report:

- gate state and every failed bound;
- number of baseline-retention decisions;
- unauthorized candidate installation count and upper confidence bound;
- installation fidelity and first-packet verification;
- loop-latency components;
- fallback and rollback counts; and
- task completion and tail metrics descriptively under the baseline.

This analysis can complete the prospective safe-enforcement evaluation. It cannot
be rewritten as candidate equivalence, superiority, or deployment benefit.

## 7. Secondary endpoints and multiplicity

Secondary endpoints are evaluated in this frozen order:

1. p95 and CVaR task latency;
2. scheduled-interval availability including timeout and loss;
3. installation fidelity and unauthorized-action rate;
4. switching interruption and full loop p95;
5. fallback and rollback frequency;
6. bytes delivered, goodput, and deadline lateness;
7. probe, compute, and memory cost; and
8. priority and site-level fairness where supported.

Holm familywise correction is applied within the declared secondary family. Report
unadjusted effect estimates and confidence intervals plus adjusted decisions. No
secondary result rescues a failed primary endpoint.

## 8. Missing data, timeouts, and failures

The primary endpoint uses no favorable imputation. Missing completion evidence,
untimestamped outcome loss, no action, controller failure, failed or late install,
and incomplete delivery are failures. Timestamped probe timeouts and losses remain
in scheduled-interval denominators. Noncritical missing covariates carry explicit
reason codes.

Report missingness and failure stage by arm and family. Sensitivities include
extreme missing-outcome assignments and removal of campaigns with documented
external outages, without changing the primary result.

## 9. Heterogeneity and sensitivity

Report effects by acquisition family without treating subgroup estimates as
independent replications. Prespecified sensitivities are:

- leave one acquisition family out;
- gate OOD limits 0.05, 0.10, and 0.20;
- calibration-gap limits 0.025, 0.05, and 0.10;
- noninferiority margins −0.01, −0.02, and −0.05;
- switching-cost and first-packet-timeout sensitivity;
- full scheduled denominator versus verified-installed secondary population; and
- clock-error and telemetry-age exclusions.

Unplanned analyses are labelled exploratory.

## 10. Safety monitoring and stopping

Operational rollback is automatic and separate from statistical stopping. The
candidate stops immediately on a prespecified runtime trigger. Accumulating safety
outcomes are examined only at registered checkpoints using the frozen boundary.
No optional stopping is used for benefit. After a stop, all assigned tasks remain
in the intention-to-treat analysis.

## 11. Reporting

Report counts before percentages; denominators; family and campaign counts;
absolute effects with confidence intervals; exact exclusions; gate reason codes;
all planned analyses; and the complete negative as well as positive findings. Keep
field, replay, simulation, and hardware-in-the-loop evidence in separate tables.

The manuscript must state whether the trial followed the baseline-retention or
candidate-canary branch. A field-benefit conclusion requires an authenticated live
gate authorization, verified installed candidate actions, a completed canary, an
untouched final test, authenticated live outcome acknowledgements, the prespecified
independent-block floor, and a matching signed end-to-end audit.
