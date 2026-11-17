# LEO usable-service survival evaluation

## Local R5 parameter revision

The working tree includes the unpublished 5 November R5 parameter revision:
a development-fitted full-chain predictor, rebuilt endpoint sensitivity,
training-only nested search/guard audits, and probability-mass correctness under
hazard clipping. Selected aggregate evidence is in `results/parameter_revision_r5/`.
Full private inputs and row-level experiments remain outside this checkout.
See `docs/parameter_revision_results.txt` for results and limitations. No new Git
release or push has been performed; the published release remains v1.3.1.

The measured prediction derivative retains its original code hashes.
`reproduction/empirical_core_v1_3_1/` is an exact read-only source snapshot, not
another active project. New development uses the top-level `src/` tree.
The local R5 suite passes 270 tests with one expected private-evidence skip.

This repository contains the Python implementation and redistribution-safe
reproducibility artifacts for:

> **Integrated Offline Evaluation of Usable-Service Prediction, Cellular--LEO
> Path Selection, and Earth-Observation Scheduling**

The study asks a bounded network question: when a low-Earth-orbit (LEO) service
is usable now, what is the probability that it remains within specified latency
and packet-loss limits for the duration of a task? It evaluates episode
construction, right censoring, probability calibration, competing risks, and
how those probabilities enter controlled path-selection and scheduling studies.

The code does **not** claim universal controller superiority or operational
field effectiveness. The main empirical result is an evaluation lesson: a
lower pooled probability-error estimate does not by itself establish useful
ranking, reliable long-horizon calibration, or a better decision policy.

## Repository contents

- `src/leo_service_survival/` — endpoint construction, survival models,
  calibration, diagnostics, decision studies, evidence gates, and adapters
- `scripts/` — experiment and validation entry points
- `tests/` — unit and scientific-consistency tests
- `configs/` — frozen manuscript-evaluation configuration, campaign-evidence
  declarations, and the numerical-parameter registry
- `results/manuscript/` — compact aggregate tables from the measured and
  controlled evaluations
- `results/review_audit/` — focused aggregate checks added during the final
  review revision
- `results/early_stopping_sensitivity/` — aggregate episode-separated stopping
  sensitivity requested during final review
- `results/eos_mechanism/` — redistribution-safe aggregate synthetic
  mechanism results
- `results/gate_logic/` — aggregate fail-closed gate stress summaries
- `results/gate_parameter_governance_v1/` — aggregate gate stress and
  prospective campaign-precision checks; these controlled fixtures are not
  operational field evidence
- `results/eos_mechanism_ablation_v3/` — aggregate EOS mechanism study with
  development-only predictor, calibration, and congestion-penalty selection
- `results/reviewer_prediction_release/` — the complete 12-pipeline by
  12-horizon WetLinks held-out prediction/score derivative, distributed under
  a separate CC BY-SA 4.0 data notice
- `aggregate_evidence/` — bounded, redistribution-safe scheduling-transfer,
  scheduler-release, and scheduler-to-path summaries
- `aggregate_evidence/scheduler_diagnostics/P03_DIAGNOSTIC_PROVENANCE_R3.json`
  — pinned RACS code/evidence hashes and independently recomputed scenario-mean
  versus pooled-request denominators for the R3 manuscript
- `prospective_trial/` — frozen schemas, runbooks, gate/canary/rollback rules,
  and a deterministic fake-data rehearsal for future installed-action testing
- `docs/` — source provenance, evidence boundaries, parameter governance, and
  adapter details

Raw third-party datasets, row-level outputs with unresolved redistribution
terms, unpublished manuscript files, superseded runs, and vendored simulator
repositories are intentionally excluded. The one included row-level derivative
is the attributed WetLinks prediction/score audit above; it contains no
upstream measurement table and does not grant permission for any other source.
Private EOS v3 candidate, task, and action-set diagnostic rows are likewise
excluded; their aggregate summaries remain in the public result tree.

This is therefore an **auditable code and aggregate-results package**, not a
self-contained one-command reproduction archive.

## Installation

Python 3.11 is the reference environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Tests

```bash
python -m pytest -q
```

The test suite exercises event/censor boundaries, chronological split
isolation, survival identities, calibration diagnostics, prediction alignment,
decision adapters, fail-closed gating, scheduler/path interfaces, prospective
trial freezing, installed-action ledgers, application acknowledgements,
canary/rollback behavior, and reproducibility manifests. One frozen-evidence
audit is skipped unless the separately held row-level EOS bundle is present;
its aggregate results are included in this repository.

Historical published release status for version `1.3.1`: **234 passed, 1 expected skip**
under the pinned Python 3.11 environment.

## Objective 4 integration boundary

The repository includes the complete bounded offline integration interface and
the executable seven-stage prospective protocol. The included controlled dry
run uses explicitly fake inputs and exercises both authorization and rejection
branches. It verifies software behavior, record binding, baseline retention,
canary planning, rollback, and final-outcome analysis; it is not a live field
trial.

Actual synchronized 5G--LEO hardware execution remains `NOT_RUN`. Operational
benefit requires independently collected campaigns, installed-route
acknowledgements, first post-installation packets, full loop latency, and real
application or EO outcomes. See `prospective_trial/STATUS.json` and
`REDISTRIBUTION_BOUNDARY.md` for the exact boundary.

## Reproduce the primary evaluation

Obtain each dataset from its cited source and retain its original license and
attribution. Then run the evaluation into new, empty output directories:

```bash
python scripts/run_all.py \
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

The runner records seeds, arguments, versions, input hashes, and declared
limitations. Use `--help` for all options. For a quick software check, reduce
the bootstrap repetitions and scheduling scenarios and add
`--skip-sensitivity`.

See [the reproducibility guide](docs/REPRODUCIBILITY.md) for the exact input
roles, focused audit command, figure builders, and the limits of what can be
reproduced without redistributing third-party data.

To rebuild only the reviewer prediction derivative after obtaining WetLinks:

```bash
python scripts/build_prediction_release_artifacts.py \
  --wetlinks-dir /path/to/WetLinks/Preprocessed_Data \
  --output-dir /tmp/leo-service-survival/prediction-release
```

The focused builder fails closed unless all twelve pipelines produce all twelve
horizons and the recomputed aggregate metrics match the canonical table. See
[the artifact specification](docs/ARTIFACT_SPECIFICATION.md) for row keys,
score formulas, calibration diagnostics, failure semantics, tolerances, and
the derivative-data license boundary. It repeats the primary runner's
chronological development-only configuration selection before refitting, so
the released predictions and the manuscript aggregate table describe the same
twelve declared pipelines.

## Data and evidence boundaries

The project keeps four evidence types separate:

1. measured service traces for prediction and calibration;
2. measured-trace replay for bounded decision checks;
3. synthetic simulation for controlled mechanism studies; and
4. packet-level simulator smoke checks.

They must not be combined into a claim of one measured end-to-end deployment.
See [data provenance](docs/data_provenance.md) and [public trace
adapters](docs/public_trace_adapters.md) before obtaining or using any input.

## Reproducibility snapshot

The aggregate tables in `results/` correspond to the final reproducibility run
completed on 4 November 2026. They preserve negative and unresolved findings,
including limited competing-risk support and the absence of a resolved
calibration-attributable scheduler benefit.

## Citation

Use [CITATION.cff](CITATION.cff). This reviewer archive is version `1.3.1`.
The GitHub repository is private; its URL identifies the author-maintained
archive and does not imply anonymous access. Repository visibility and source-code
reuse rights remain separate. No open-source license is asserted.

## License and third-party material

No license has yet been granted for the authors' original source code. Public
visibility permits inspection but does not by itself grant reuse rights. See
[NOTICE.md](NOTICE.md). Third-party datasets are not redistributed and remain
subject to their own licenses and access terms. The derived WetLinks files in
`results/reviewer_prediction_release/` are separately identified as CC BY-SA
4.0 and do not relicense the source code.
