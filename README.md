# LEO usable-service survival evaluation

This repository contains the Python implementation and redistribution-safe
reproducibility artifacts for:

> **Censoring-Aware Evaluation of Usable-Service Duration Predictions for LEO
> Network Decisions**

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
- `configs/` — frozen manuscript-evaluation configuration
- `results/manuscript/` — compact aggregate tables from the measured and
  controlled evaluations
- `results/review_audit/` — focused aggregate checks added during the final
  review revision
- `results/early_stopping_sensitivity/` — aggregate episode-separated stopping
  sensitivity requested during final review
- `results/eos_mechanism/` — redistribution-safe aggregate synthetic
  mechanism results
- `results/gate_logic/` — aggregate fail-closed gate stress summaries
- `docs/` — source provenance, evidence boundaries, and adapter details

Raw third-party datasets, row-level outputs with unresolved redistribution
terms, unpublished manuscript files, superseded runs, and vendored simulator
repositories are intentionally excluded.

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

The public test suite exercises event/censor boundaries, chronological split
isolation, survival identities, calibration diagnostics, prediction alignment,
decision adapters, and fail-closed gating. One frozen-evidence audit is skipped
unless the separately held row-level EOS bundle is present; its aggregate
results are included in this repository.

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

The aggregate tables in `results/` correspond to the manuscript revision
frozen on 30 October 2026. They preserve negative and unresolved findings,
including limited competing-risk support and the absence of a resolved
calibration-attributable scheduler benefit.

## Citation

Use [CITATION.cff](CITATION.cff). The versioned source snapshot is available at
<https://github.com/KarusalaDeepak/leo-service-survival/tree/v1.0.0>.

## License and third-party material

No license has yet been granted for the authors' original source code. Public
visibility permits inspection but does not by itself grant reuse rights. See
[NOTICE.md](NOTICE.md). Third-party datasets are not redistributed and remain
subject to their own licenses and access terms.
