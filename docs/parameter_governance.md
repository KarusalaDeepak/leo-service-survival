# Numerical parameter governance

Paper 4 does not treat every number as a physical constant. The authoritative
machine-readable definitions are in `src/leo_service_survival/parameters.py`;
`configs/parameter_registry.csv` is the submission-facing summary.

## Selection rules

- **Requirements** define the investigated service or control profile. The
  80-ms RTT, 1% loss, two-bin persistence and 100-ms loop budget are declared
  requirements, not universal LEO constants. This is an author-declared
  evaluation profile, not an operator SLA recovered from the source data.
  Alternatives must be distinguished as executed, non-estimable or unsupported;
  listing a value in the registry does not mean an experiment was performed.
- **Learned quantities** are estimated only from training/development data.
  The focal hazard learner, every trainable baseline, calibration regularizer,
  scheduling congestion penalty and synthetic predictor coefficients use
  bounded development-only selection. Test outcomes cannot select a setting.
  A chosen value is optimal only within its declared search and objective, not
  a universal optimum. The R5 audits preserve the earlier primary empirical fit
  and separately examine search-boundary stability within development data.
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

## Requirements are not fitted physical coefficients

The endpoint answers a conditional question: service is usable under the
declared latency, loss and persistence limits. Selecting those limits to
maximize a model's test score would change the question after seeing its
answer. The reason for the 80-ms/1%/two-bin profile is to define a persistent
QoS endpoint and suppress isolated bursts; it is not evidence that every LEO
application requires those values. Endpoint comparisons therefore report event
prevalence, usable support, model failures and both focal and simple baseline
scores, not just a best-performing threshold.

The gate's two-percentage-point non-inferiority margin declares a tolerated
completion deficit. Its precision, feature-coverage, calibration-error and
controller-delay limits are separate risk and implementation budgets. No
stakeholder cost study or measured deployment SLA has been supplied, so these
remain illustrative policy profiles, not validated operational acceptance
criteria. A decision-stability sweep can show how a result changes with a
budget; it cannot decide the acceptable harm for an application.

The monitor's reference error and betting parameter likewise define the
monitored null and its sensitivity. Its mathematical alarm guarantee is
conditional on that null; synthetic alarm fixtures are not measured field
false-alarm or detection-delay estimates. The input-domain guard has a
development-fitted distance distribution, while its quantile and batch alarm
budget remain explicit choices. Measured held-out distance exceedances are
descriptive within the supplied acquisitions, not proof of general shift
detection.

## Removed hand-set prediction terms and retained synthetic assumptions

The scheduling score no longer contains the unsupported `-0.002 end_slot`
term; earliest finish is used only as a deterministic tie break. The v3
mechanism-ablation experiment replaced the hand-written reliability equation
with coefficients `0.98`, `0.99`, `0.65`, `4`, `0.85`, and `250` by a logistic
technical-completion predictor fitted on disjoint synthetic development draws.
It selects regularization internally, calibrates on a second partition, and
selects its congestion penalty on tuning seeds disjoint from evaluation seeds.

The earlier full-chain v2 experiment was different: it retained that hand-written
prediction equation. The earlier statement that all EOS studies had removed it
was too broad. R5 replaces the active full-chain predictor and retains the old
v2 results only as historical provenance. Prediction coefficients learned from
generated outcomes still depend on the declared generator; they are not
empirically identified geometry, beam, rain or RF coefficients. Alternative
network profiles and alignment conditions probe this dependence without
retuning the predictor on evaluation outcomes.

Paper 1's accepted engineering coefficients remain historical assumptions.
Its optional training-only additive fitter is an implemented estimation path,
not evidence that matched physical labels exist or that those coefficients
were fitted to operator failures. Physical cause claims are not restored by
the new service-level or synthetic fitting.

## Reproducibility outputs

The main runner exports `parameter_registry.csv`,
`wetlinks_focal_hyperparameter_selection.csv`, and
`wetlinks_baseline_hyperparameter_selection.csv`. EOS protocol v3 additionally
exports predictor coefficients, predictor/calibration regularization selection,
and congestion-penalty selection. These artifacts state what was fitted, where
it was selected and whether test data were excluded.

## Executed R5 coverage

- `results/endpoint_parameter_sensitivity_v1/` executes 15 endpoint profiles;
  one-minute aggregation is unsupported. Loss 0.5/1/2%, latency 70/80/90 ms,
  persistence 1/2/3, gap 1/2/3 and horizons 6/9/12 are covered. Ten-minute
  aggregation also has a time-matched persistence/horizon contrast. Identical
  gap-profile labels are explicitly cached, not counted as independent evidence.
- `results/parameter_defensibility_v1/` retains two nested development folds,
  original/expanded searches, 25/50/100 calibration-support limits, probability
  clipping alternatives, hazard-bound activation, feature-only guard sweeps,
  and controlled monitoring/release-budget logic. Wider search is not better on
  the average outer development score and boundary selection remains visible.
- `results/eos_full_chain_parameter_revision_v1/` retains the learned full-chain
  predictor, all four generation roles, wider regularization-range audit and
  four declared profile conditions. The source geometry and simulator assumptions
  are shared; these are not four independent validation campaigns.

The active cause-incidence calculation additionally shares the same bounded
total as survival. Positive raw hazards retain the historical survival formula;
exactly zero raw event mass remains zero. This repairs probability mass under
clipping, not the physical interpretation of the learned model. The original
measured prediction derivative is preserved with its original source snapshot;
it was not relabelled as a rerun of this correction.

Run the three focused scripts `run_endpoint_sensitivity_audit.py`,
`run_parameter_defensibility.py` and `run_eos_full_chain.py` in fresh output
directories with the pinned research environment. Do not overwrite the original
empirical freeze. `build_parameter_revision_evidence.py` exports only an explicit
aggregate allowlist for the local manuscript/reviewer package.
