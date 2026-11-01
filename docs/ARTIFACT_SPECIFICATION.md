# Reviewer artifact specification

This document freezes the reviewer-facing evidence chain for the primary
WetLinks comparison. It distinguishes predictions, observed outcomes, score
contributions, calibration fitting, and held-out calibration diagnostics so
that none of them is mistaken for another.

## Primary relational keys

The release uses one held-out set, identified as
`wetlinks-heldout-d4317eff6d0f1d1b`. Its ordered 821 `episode_id` values have
SHA-256 digest
`d4317eff6d0f1d1b497175b0a9fb43a52ecac3a9b51d998f72955df0253866f9`.
The order was captured before model prediction. Every pipeline and horizon
must repeat exactly that row order.

The row-level prediction file has the unique key
`(test_set_id, test_row_index, pipeline, horizon_bins)`. Its required
cardinality is `821 x 12 x 12 = 118,224` rows. The horizon score file has the
unique key `(test_set_id, pipeline, horizon_bins)` and exactly 144 rows.

The twelve frozen pipeline names are:

1. `constant_exponential`
2. `kaplan_meier`
3. `weibull_aft`
4. `cox_ph`
5. `additive_discrete_hazard`
6. `linear_discrete_hazard`
7. `boosted_discrete_hazard`
8. `random_forest_discrete_hazard`
9. `xgboost_discrete_hazard`
10. `calibrated_single_risk`
11. `competing_risk_uncalibrated`
12. `calibrated_competing_risk`

The required horizon grid is every integer from 1 through 12, where one bin is
five minutes.

## Outcome and score convention

At horizon `h`, an event observed at duration `d <= h` is a known failure with
target 0. A record with `d > h` is a known survivor with target 1. A record
censored at `d <= h` is unknown: the exported target is missing, its IPCW
weight is zero, and it is never relabelled as a failure or survivor.

The censoring distribution is the marginal reverse Kaplan--Meier estimate on
the held-out diagnostic scope. Event records are censoring-distribution
censors and administrative-censoring records are censoring-distribution
events. The implementation uses the post-boundary estimate `G(t)=P(C>t)` and
clips it below at 0.001. An observed failure at `d` receives weight `1/G(d)`;
a known survivor at `h` receives `1/G(h)`.

For scoring, survival probabilities are clipped to `[1e-6,1-1e-6]`, matching
the canonical evaluator; the unclipped prediction is retained in a separate
column. For each pipeline and horizon, the pointwise Brier score is the mean over all
821 rows of `IPCW weight x (target - score probability)^2`. Unknown censored
rows therefore contribute zero, while remaining explicitly unknown. The
pointwise log loss divides the sum of IPCW-weighted binary log losses by the
sum of positive IPCW weights. The integrated Brier value through a horizon is
the arithmetic mean of the pointwise Brier values from horizon 1 through that
horizon. These are descriptive held-out estimates, not independent-row
hypothesis tests.

## Calibration fit versus calibration diagnostic

Calibration *fit* occurs only for `calibrated_single_risk` and
`calibrated_competing_risk`, on the disjoint chronological calibration
partition. Each horizon fits logistic regression to the logit of raw survival
using IPCW weights (`C=0.1`, maximum 1,000 iterations, seed 17). Raw survival
is used unchanged if fewer than 25 outcomes are evaluable or only one target
class is present. A cumulative-minimum projection is applied after the twelve
horizon maps. The fit audit exposes support, fitted intercept and slope, and
any fallback reason for both calibrated pipelines at every horizon.

Held-out calibration *diagnostics* do not alter predictions. The free-slope
diagnostic fits
`logit(P(Y=1)) = alpha + beta logit(predicted survival)` with IPCW weights.
The diagnostic clips probabilities to `[1e-5, 1-1e-5]`, starts at
`(alpha,beta)=(0,1)`, uses at most 100 Newton steps, a `1e-8` step tolerance,
a `1e-10` diagonal solve ridge, and a `1e-8` fitted-variance floor. It is not
estimated with fewer than five evaluable rows, one target class, a singular
solve, or a divergent coefficient. Calibration-in-the-large uses the same
procedure with slope fixed to one. ECE uses ten equal-width probability bins
with IPCW weights. Full machine-readable details are in
`configs/calibration_diagnostic_spec.json`.

## Execution and failure semantics

`wetlinks_pipeline_execution_status.csv` joins the declared configuration to
the successful release run. Blank failure fields mean that fitting and
prediction completed. A failed build writes
`wetlinks_pipeline_build_failures.csv` with pipeline, stage, exception type,
and reason, then stops before publishing a partial prediction grid. A
calibrated pipeline is labelled `fitted_at_all_horizons` only if its audit has
exactly horizons 1--12 and no fallback. Missing, duplicate, or extra horizons
fail closed or receive an explicit incomplete status.

## Missing features and imputation

Coverage is counted on numeric-coerced feature values before model imputation.
The encoder then replaces missing or non-finite values with medians learned
only from the training partition and applies training-partition mean/standard
deviation scaling. `rain_latency_interaction` is special: the episode builder
replaces missing rain with zero before multiplying by latency, so its apparent
100% coverage does not imply that rain itself was observed. The coverage table
states this structural-fill rule explicitly.

## Model objects and exactness

No pickle, joblib, ONNX, XGBoost model, or other executable fitted object is
released. Models must be refitted from the cited source data and frozen source
code. This avoids unsafe deserialization and prevents an undocumented binary
from becoming the scientific specification. See `docs/MODEL_OBJECT_POLICY.md`.

Identifiers, pipeline and horizon sets, row counts, row-order digests, and file
hashes are exact checks. Numerical comparisons use only the tolerances declared
in `configs/validation_tolerances.json`; those engineering tolerances are not
confidence intervals or evidence of statistical equivalence.

## Redistribution boundary

The row-level release contains derived WetLinks episodes, outcomes,
predictions, and score inputs; it does not contain the upstream measurement
tables. It is distributed separately under CC BY-SA 4.0 with attribution and a
modification statement in `results/reviewer_prediction_release/README.md`.
That permission does not extend to COMMECT, Palo Alto, SatInView, NEXASPHERE,
EOS-Bench, or any other source whose row-level redistribution terms are absent
or unresolved. Those sources remain excluded. The repository's original
source code also remains under the terms stated in `NOTICE.md`; the data
license does not relicense the code.

