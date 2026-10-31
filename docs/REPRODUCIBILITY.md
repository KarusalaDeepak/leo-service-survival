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

## Figure builders

The released aggregate tables are sufficient to regenerate two explanatory
figures without raw measurements:

```bash
python scripts/build_event_censoring_figure.py
python scripts/build_decision_contrasts_figure.py
```

Both write to `figures/` and use portable Matplotlib fonts. The decision
contrast script reads its values directly from the released CSV tables.

## EOS mechanism study

The EOS mechanism runner requires the separately obtained EOS-Bench scenario
identified in the manuscript. Because the locally acquired scenario did not
carry a clear redistribution license, this repository includes the experiment
code and aggregate outputs but not the scenario or row-level candidate/task
tables. The results remain controlled synthetic evidence, not field or operator
validation.

## Interpretation boundary

Reproducing a table or passing a test does not turn a trace replay or synthetic
experiment into operational evidence. In particular:

- WetLinks contains service measurements but no serving-satellite, beam,
  gateway, handover, or physical-cause labels.
- the scheduler uses generated task and resource conflicts over measured trace
  outcomes;
- the EOS mechanism experiment uses synthetic scores and outcomes; and
- no result establishes universal controller superiority.
