# Reviewer reproduction: R34 causal delivery queue

This code-only bundle contains author-written implementation, the common Paper 3 scheduler dependency, tests, protocols, and assignment-level aggregate outputs. It excludes manuscripts, Git history, credentials, raw measurements, benchmark scenario rows, and episode-level model predictions.

## Setup

Install Python with NumPy and pandas. Set `PAPER3_CODE_ROOT` to the extracted `paper3` directory; then work in `paper4`.

```
export PAPER3_CODE_ROOT=/absolute/path/to/extracted/paper3
cd /absolute/path/to/extracted/paper4
python3 -m unittest discover -s tests -p 'test_r34_causal_chunks.py'
python3 scripts/run_r34_causal_chunks.py --predictions /path/to/wetlinks_all_pipeline_horizon_predictions.csv.gz --instance /path/to/eos_s1_overlay.json --protocol docs/r34_causal_chunk_protocol_2026-11-19.md --output-dir new_native_results
python3 scripts/run_r34_causal_chunks.py --capacity-profile duration_matched --predictions /path/to/wetlinks_all_pipeline_horizon_predictions.csv.gz --instance /path/to/eos_s1_overlay.json --protocol docs/r34_causal_chunk_protocol_2026-11-19.md --output-dir new_duration_results
python3 scripts/summarize_r34_causal_chunks.py new_duration_results
```

The provided duration addendum is at `paper4/docs/r34_duration_matching_addendum_2026-11-19.md`. Outputs require empty destinations; the scripts never overwrite a frozen run.

## Inputs and access

The exact prediction derivative and EOS-Bench-derived overlay must be obtained separately with the original source/data-use permissions and matching hashes from the local execution manifests. The provided aggregate tables do not replace these inputs. EOS-Bench source is at https://github.com/EthanY9Q/EOS-Bench; the actual transformed scenario is not redistributed here. The WetLinks paper is DOI 10.23919/TMA62044.2024.10558998. Ask the authors for the frozen prediction derivative and precise scenario transformation if not already supplied.

Accordingly, this is a reproducible **software** bundle with external input requirements, not an entirely self-contained open-data archive. An editor should receive both the code and an access arrangement for necessary input derivatives before claiming independent reproducibility.

## What to compare

There are two panels, all four policies, all nine load/overhead cells, and 20 fixed assignment seeds. Compare `runs.csv`, `summary.csv`, and `paired_inference.csv`. The completed-priority ceiling is set upstream by the common 59-request acquisition scaffold. The previous R33 +6.48-point recovery statement is withdrawn; see the correction note. There is no independent-field or universal-superiority claim.
