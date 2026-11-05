# Claim-bounded aggregate results

This is an author-written summary of the aggregate values reported in the R3
manuscript and final 4 November empirical rerun. It contains no external measurement rows, EOS-Bench scenario rows,
or simulator traces. The private evidence tree retains the row-level authorities
and hashes; their omission here is a redistribution decision, not independent
replication.

## Measured prediction and transfer

- WetLinks test support decreases from 657 evaluable episodes at 15 minutes to
  414 at 60 minutes; 407 of 821 outcomes are unknown at 60 minutes.
- Final 60-minute pointwise Brier score is 0.13597 and through-horizon IPCW
  Brier score is 0.076993. Fixed-orientation concordance is 0.577937 and the
  held-out 60-minute calibration slope is 3.534.
- On the Palo Alto stress record, event prevalence is 0.8997, through-horizon
  Brier score is 0.6760, and ECE is 0.6989. The manuscript treats this as
  descriptive transport-shift evidence, not confirmatory external validation.

## Recorded replay and controlled decisions

- In the single-campaign five-second COMMECT replay, candidate and reactive
  success are 63.27% and 57.31%; the paired difference is 5.959 percentage
  points with a 60-second moving-block interval [3.102, 9.143]. The release gate
  retains the reactive baseline because one acquisition campaign is insufficient.
- In 120 controlled scheduling workloads, survival-aware completion is 0.3954
  versus 0.4069 for EDF: -1.146 percentage points. The four-block descriptive
  interval is [-1.799, -0.493], with exact sign-flip p = 0.125; four blocks
  do not support confirmatory independent-campaign inference.
- In the six-arm synthetic EOS mechanism study, the reliability-control minus
  current-SNR contrast is +23.6 points [20.7, 26.7] only under adverse alignment;
  favorable and neutral contrasts are unresolved. This is an oracle-informed
  synthetic mechanism contrast and does not validate the WetLinks model.

## EO scheduling integration

- In the nine-unit combined-stress benchmark, the unweighted mean per-scenario
  completion rate is 60.19% for cross-fitted RACS and 60.56% for the matched full-margin selector; the
  RACS difference is -0.37 percentage points with a 95% interval of
  [-1.11, 0.00] and a two-sided p-value of 1.
- The equal-budget selected random forest has a mean scenario rate of 59.63%; RACS minus that
  comparator is +0.56 points [0.00, 1.30], with Holm-adjusted p = 1.
- In the frozen EOS-Bench transfer audit, all compared selectors complete
  26.79% across 24 source scenarios and 384 validated schedules. Ground
  service, releases, deadlines, and delivery outcomes are simulated.
- The frozen scheduler decision is `RETAIN_MATCHED_SIMPLE_BASELINE`:
  `full_margin` is selected and `cross_fitted_racs` remains in shadow
  evaluation. This is a successful fail-closed integration result, not a claim
  of RACS superiority or operational EO benefit.
- A hash-verified interface audit reruns `full_margin` on a shared 100-task
  scenario, reproduces and validates all 28 selected plans, maps every selected
  observation exactly to Paper 4 geometry, and exposes 56 path candidates. The
  P3 and P4 outcomes remain separate because their workload and service
  overlays differ.

## Prospective readiness

- The portable rehearsal contains five fake gate campaigns in three fake
  families and exercises both rejection/fallback and controlled authorization.
- Its 20 linked fake outcomes yield candidate and baseline completion of 0.50,
  risk difference 0.00, and a 95% interval [-0.40, 0.40]. The benefit flag is
  false by both result and evidence class.
- Field status remains `NOT_RUN`; protocol status remains
  `DRAFT_NOT_REGISTERED`; deployment benefit remains unsupported.

The nine-scenario rates give each scenario equal weight. Pooled diagnostic
request counts, which weight larger scenarios more, are 259/315 for RACS and
261/315 for full margin and must not be substituted for those scenario means.
For definitions, uncertainty methods, and limitations, consult the current
manuscript. This summary must not be cited as a new independent dataset.
