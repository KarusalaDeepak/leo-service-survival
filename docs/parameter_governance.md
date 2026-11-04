# Numerical parameter governance

Paper 4 does not treat every number as a physical constant. The authoritative
machine-readable definitions are in `src/leo_service_survival/parameters.py`;
`configs/parameter_registry.csv` is the submission-facing summary.

## Selection rules

- **Requirements** define the investigated service or control profile. The
  80-ms RTT, 1% loss, two-bin persistence and 100-ms loop budget are declared
  requirements, not universal LEO constants. Their alternatives are evaluated
  as sensitivity conditions.
- **Learned quantities** are estimated only from training/development data.
  The focal hazard learner, every trainable baseline, calibration regularizer,
  scheduling congestion penalty and synthetic endpoint coefficients use
  bounded development-only selection. Test outcomes cannot change them.
- **Numerical safeguards** such as probability clipping prevent undefined
  arithmetic and do not encode physical behavior.
- **Controlled factors** define synthetic mechanism levels. They are reported
  separately and are not presented as empirical operator frequencies.

## Evidence-gate tiers

One campaign supports proof of method only. Two independent campaigns from two
independent acquisition families can support a frozen external-validation test.
The optional conservative confirmatory-release profile uses at least five paired
campaign groups from at least three acquisition families; five yields five held-out
assessments in one leave-one-campaign-out cycle. Five is neither an IEEE rule nor a
claim of statistical sufficiency. The gate also requires paired policy
opportunities, an adequately narrow campaign-block confidence interval,
calibration, domain coverage, shift-monitor status and measured control-loop
latency. The current measured policy gate still has only one qualified campaign,
COMMECT, so it retains the reactive baseline. A prospective precision/power
calculation can require more campaigns.

## Removed hand-set terms

The scheduling score no longer contains the unsupported `-0.002 end_slot`
term; earliest finish is used only as a deterministic tie break. The controlled
EOS study no longer uses the hand-written reliability equation with coefficients
`0.98`, `0.99`, `0.65`, `4`, `0.85`, and `250`. It fits a logistic
technical-completion predictor on disjoint synthetic development draws, selects
regularization internally, calibrates on a second partition, and selects the
congestion penalty on tuning seeds disjoint from reported evaluation seeds.

## Reproducibility outputs

The main runner exports `parameter_registry.csv`,
`wetlinks_focal_hyperparameter_selection.csv`, and
`wetlinks_baseline_hyperparameter_selection.csv`. EOS protocol v3 additionally
exports predictor coefficients, predictor/calibration regularization selection,
and congestion-penalty selection. These artifacts state what was fitted, where
it was selected and whether test data were excluded.
