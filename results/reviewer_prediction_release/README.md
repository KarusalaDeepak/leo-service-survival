# WetLinks prediction and score derivative

This directory is the reviewer-facing derivative for the primary WetLinks
held-out comparison. It contains one prediction for every combination of 821
held-out episodes, 12 declared pipelines, and 12 five-minute horizons, plus
row-level censoring-aware score inputs and compact fit/status audits.

## Attribution and license

The derivative is based on:

D. Laniewski, E. Lanfer, B. Meijerink, R. van Rijswijk-Deij, and N.
Aschenbruck, “WetLinks: A Large-Scale Longitudinal Starlink Dataset with
Contiguous Weather Data,” TMA 2024, DOI: 10.23919/TMA62044.2024.10558998.

The files in this directory are licensed under the [Creative Commons
Attribution-ShareAlike 4.0 International
license](https://creativecommons.org/licenses/by-sa/4.0/) as derived WetLinks
data. The source measurements were modified by five-minute aggregation,
service-episode construction, chronological partitioning, administrative
horizon truncation, model fitting/calibration, prediction, and
censoring-aware scoring. The upstream measurement CSV files are not included.

CC BY-SA applies to this derivative directory, not to the repository's author
source code, which remains governed by `NOTICE.md`.

## Files

- `wetlinks_all_pipeline_horizon_predictions.csv.gz`: 118,224 row-level
  predictions and score contributions; unique key
  `(test_set_id,test_row_index,pipeline,horizon_bins)`.
- `wetlinks_all_pipeline_horizon_scores.csv`: 144 pipeline/horizon summaries,
  including Brier, log loss, ECE, calibration intercept/slope, support, and
  explicit not-estimable reasons.
- `wetlinks_pipeline_execution_status.csv`: all 12 declared configurations and
  observed successful-run status.
- `wetlinks_pipeline_calibration_fit_audit.csv`: both calibrated pipelines at
  all 12 horizons, including support, coefficients, and fallback status.
- `wetlinks_preimputation_feature_coverage.csv`: train/calibration/test feature
  availability measured before model imputation.
- `wetlinks_prediction_alignment_audit.json`: cardinality and ordered row-ID
  digest.
- `wetlinks_prediction_release_manifest.json`: input, code, environment, and
  output hashes.

See `docs/ARTIFACT_SPECIFICATION.md` before interpreting the fields. Unknown
censored outcomes retain a missing target and zero IPCW weight; they are not
silently recoded. The release is for audit of the reported held-out comparison
and does not establish independent operator or deployment validation.

