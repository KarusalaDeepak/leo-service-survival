# Reproducibility guide

This repository is an auditable code and aggregate-results package. It is not a
self-contained data archive. Full reruns require the user to obtain the cited
third-party inputs under their original terms.

## Reference environment

- Python 3.11
- NumPy 1.24.3
- pandas 2.0.3
- SciPy 1.11.1
- scikit-learn 1.3.0
- Matplotlib 3.7.2
- XGBoost 3.1.2
- primary random seed 17
- scheduler tuning seed 1017
- 400 bootstrap repetitions for the manuscript run
- 120 scheduling scenarios for the manuscript run

Create the environment and verify the public package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

## Full measured evaluation

`scripts/run_all.py` requires five input groups:

1. the two WetLinks `analysis_data_*.csv` files;
2. the 12 Palo Alto `ping_metrics_2024-05-*.*` files;
3. the processed concurrent COMMECT table;
4. the released LENS-derived prediction table; and
5. the Paper 1 TLE-derived geometry-window table.

The inputs are intentionally not redistributed here. See
`docs/data_provenance.md` for citations, evidence limits, and license notes.

```bash
PYTHONPATH="$PWD/src" python scripts/run_all.py \
  --output-dir /tmp/leo-service-survival/results \
  --figure-dir /tmp/leo-service-survival/figures \
  --wetlinks-dir /path/to/WetLinks/Preprocessed_Data \
  --palo-alto-dir /path/to/palo_alto_starlink \
  --commect-path /path/to/commect_multiaccess_1s.csv \
  --lens-path /path/to/temporal_forecast_predictions.csv \
  --paper1-windows-path /path/to/regional_starlink_risk_aware_visibility.csv \
  --bootstrap-repetitions 400 \
  --scheduling-scenarios 120
```

Always use fresh output directories. The runner records input hashes, software
versions, arguments, seeds, and declared limitations.

## Focused manuscript audit

The compact audit tables in `results/review_audit/` were derived from saved
prediction rows, EOS candidate/task rows, and the WetLinks source tables. Those
row-level inputs are not included in the public repository. If you have lawful
copies, rebuild the tables with:

```bash
PYTHONPATH="$PWD/src" python scripts/build_manuscript_audits.py \
  --stage-predictions /path/to/wetlinks_calibration_stage_predictions.csv \
  --eos-candidates /path/to/candidate_scores.csv \
  --eos-task-results /path/to/task_results.csv \
  --wetlinks-dir /path/to/WetLinks/Preprocessed_Data \
  --output-dir /tmp/leo-service-survival/review-audit
```

The builder omits the row-level observed-segment table by default. Use
`--include-observed-segments` only when the output will be handled under the
WetLinks attribution and share-alike requirements.

## Complete primary prediction and score derivative

The released reviewer derivative was rebuilt from the final public source, not
reconstructed from selected-horizon tables. It contains all twelve declared
pipelines at all horizons 1--12 on the same 821 ordered held-out episode IDs.
Build it into a fresh directory with:

```bash
PYTHONPATH="$PWD/src" python scripts/build_prediction_release_artifacts.py \
  --wetlinks-dir /path/to/WetLinks/Preprocessed_Data \
  --output-dir /tmp/leo-service-survival/prediction-release
```

The builder refuses partial grids, non-finite probabilities, duplicate row
identities, incomplete calibration audits, and aggregate metrics that differ
from `results/manuscript/wetlinks_model_metrics.csv` beyond the declared
software tolerance. Before the final refit, it repeats the primary runner's
chronological development-only selection for the focal hazard/calibration
settings and the trainable comparator settings; final calibration and held-out
test rows remain excluded from that selection. It exports per-row IPCW weights and Brier/log-loss
contributions, all-pipeline horizon diagnostics, both calibrated-pipeline fit
audits, execution status, pre-imputation feature coverage, row-order digest,
input/code/output hashes, and no model objects.

Because this is a WetLinks-derived row-level artifact, retain the CC BY-SA 4.0
attribution and modification notice in
`results/reviewer_prediction_release/README.md`. Do not use this pathway to
publish row-level COMMECT, Palo Alto, SatInView, NEXASPHERE, or EOS-Bench
material; their redistribution boundaries are different or unresolved.

Run `python scripts/build_result_to_script_manifest.py` after any permitted
result update. It refreshes `results/result_to_script_manifest.csv`, which maps
each public result or figure to one producer and records a portable hash.

## Figure builders

The released aggregate tables are sufficient to regenerate two explanatory
figures without raw measurements:

```bash
python scripts/build_event_censoring_figure.py
python scripts/build_decision_contrasts_figure.py
```

Both write to `figures/` and use portable Matplotlib fonts. The decision
contrast script reads its values directly from the released CSV tables.

## Comparator configuration

`results/manuscript/comparator_configurations.csv` records, for all twelve
declared pipelines, the input features, time representation, censoring and cause
handling, hyperparameters, tuning data, calibration, stopping rule, and software
version. The benchmark compares these complete declared pipelines; it is not a
matched comparison of estimator families under identical post-hoc calibration.

## Episode-separated stopping sensitivity

The focal library default selects its stopping iteration from expanded
person-period rows. `scripts/run_early_stopping_sensitivity.py` provides the
bounded reviewer-requested check: it selects the iteration count on an earlier
chronological episode holdout, refits all declared training episodes with library
early stopping disabled, preserves the calibration and final test partitions, and
writes only aggregate outputs.

```bash
PYTHONPATH="$PWD/src" python scripts/run_early_stopping_sensitivity.py \
  --wetlinks-dir /path/to/WetLinks/Preprocessed_Data \
  --output-dir /tmp/leo-service-survival/early-stopping
```

The frozen aggregate audit is in `results/early_stopping_sensitivity/`. The
chronological rule selected one boosting iteration. Its final-test Brier point
estimate was slightly lower than the automatic-stopping result, but the four-block
interval versus the feature-free reference still included zero and its 60-minute
probabilities were highly compressed. The principal conclusion therefore did not
change.

## EOS mechanism study

The EOS mechanism runner requires the separately obtained EOS-Bench scenario
identified in the manuscript. Because the locally acquired scenario did not
carry a clear redistribution license, this repository includes the experiment
code and aggregate outputs but not the scenario or row-level candidate/task
tables. The results remain controlled synthetic evidence, not field or operator
validation.

The runner also writes `candidate_scores.csv`, `task_results.csv`, and
`action_set_audit.csv` for private diagnosis. These row-level files are
explicitly denied from the public result manifest and ignored by Git. Retain
them locally and publish only their aggregate companions.

## Interpretation boundary

Reproducing a table or passing a test does not turn a trace replay or synthetic
experiment into operational evidence. In particular:

- WetLinks contains service measurements but no serving-satellite, beam,
  gateway, handover, or physical-cause labels.
- the scheduler uses generated task and resource conflicts over measured trace
  outcomes;
- the EOS mechanism experiment uses synthetic scores and outcomes; and
- no result establishes universal controller superiority.
