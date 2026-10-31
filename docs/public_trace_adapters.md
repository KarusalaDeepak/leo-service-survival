# Public trace adapters and bounded validation

This project treats public traces as supporting evidence, not as substitutes
for a synchronized operational EO field campaign.  The adapters preserve the
source design, provenance, alignment uncertainty, and missing fields.

## Evidence boundaries

| Source | What is measured | Adapter evidence tier | What is not supplied |
|---|---|---|---|
| NUWiNS Alaska/Hawaii | Concurrent road-test path outcomes in publisher-paired rows | `field_observational` | Satellite, beam, gateway, orbital geometry, policy action, EO outcome |
| NUWiNS mainland | Per-operator road-test outcomes without an exact cross-operator key | `field_observational` | Exact simultaneous pair identity and all fields above |
| LENS | Starlink IRTT/ping and minute dish status at public sites | `field_observational` | Concurrent 5G path, serving satellite, beam, gateway, RF measurements, EO outcome |
| SatInView | Ping plus a post-processed obstruction-map/TLE trajectory match | `field_observational`, with identity explicitly labeled inferred | Direct serving identity, explicit handover telemetry, beam, gateway, RF, site coordinates |
| NEXASPHERE Athens | Seven-day Starlink latency, dish status, and separately sampled throughput | `field_observational`, private aggregate analysis only | Concurrent alternative path, serving satellite, beam, gateway, RF observations, scheduler action, EO outcome |
| Fixed-payload future-capacity benchmark | Task labels replayed over measured NUWiNS throughput | `measured_trace_replay` | Operational EO acquisition/delivery acknowledgement or policy trial |

The adapters never populate nonexistent satellite, beam, gateway, RF, or EO
fields.  Each output row includes a source identifier containing the archive or
CSV SHA-256, its source file, and a source row/record reference.

## NUWiNS adapter

`src/leo_service_survival/nuwins_adapter.py` has separate entry points for the
two released synchronization designs.

- `load_nuwins_pre_aligned_panel` converts one Alaska/Hawaii publisher-paired
  RTT or throughput file to long form.  The two panel kinds cannot be combined
  by the API.  A source row becomes exactly two candidate-path rows.
- `align_nuwins_mainland_pair` first maps each anchor run to the nearest
  candidate run, then performs deterministic nearest-unused-timestamp matching.
  Default tolerances are 5,000 ms for run starts and 250 ms for throughput or
  100 ms for RTT samples.  Rows outside either frozen tolerance are excluded
  from the observation table and retained in the alignment audit with a reason.
  Both source timestamps, both run IDs, the run-start skew, and sample skew are
  retained.

Example:

```bash
python scripts/adapt_public_traces.py nuwins-pre-aligned \
  data/external/nuwins_sigmetrics26/datasets/extracted/alaska_road_test_202406/processed/mptcp/mptcp_trace.alaska.tcp_downlink.starlink_att.csv \
  results/adapted/alaska_starlink_att \
  --campaign-id alaska_202406 --panel-kind downlink_throughput --max-rows 100000
```

## LENS adapter

`src/leo_service_survival/lens_adapter.py` streams bounded records instead of
expanding the full public corpus.

- `load_lens_irtt` incrementally decodes `round_trips` from JSON or JSON.gz.
- `load_lens_ping` accepts an hourly ping archive or a weekly archive whose
  members are nested hourly `.txt.tar.zst` files.  `max_members`, `max_records`,
  UTC bounds, and an optional member glob are applied before materialization.
  Missing ping replies are not invented; observed sequence gaps are recorded.
- `load_lens_dish_status` streams Flux-annotated weekly dish-status CSVs.
- `align_lens_irtt_to_dish` performs a causal backward as-of match, never using
  telemetry sampled after a probe.  Unmatched probes remain explicit.
- `build_lens_connectivity_episodes` run-length encodes received/lost probe
  states and breaks at sequence or timestamp gaps.  These are connectivity
  episodes, not visibility windows.

The default validation run loaded bounded samples from Brisbane, Ghana,
Seattle, and Victoria.  These are four Starlink sites, not four operators.

## SatInView adapter

`src/leo_service_survival/sat_in_view_adapter.py` renames
`Connected_Satellite` to `inferred_satellite_id` and attaches the evidence label
`post_processed_trajectory_obstruction_map_match_not_direct_telemetry`.  A
possible handover is emitted only when consecutive, temporally contiguous rows
change inferred identity.  A change after a trace gap is labeled unresolved,
not a handover.

## NEXASPHERE Athens adapter

`src/leo_service_survival/nexasphere_adapter.py` reads the four measurement
tables independently and filters every table to the frozen metadata interval
`2026-05-11T12:06:16Z` through `2026-05-18T12:07:03Z`, inclusive.  This
excludes an April pilot that is present in the CSV files but outside the
declared seven-day run.  Rows carry the source-file SHA-256 and the restriction
`private_analysis_license_unresolved_no_raw_redistribution`.

The local acquisition record is
`data/external/nexasphere_athens_2026/LOCAL_ACQUISITION_RECORD.md`.  Until the
archive license is resolved, only private aggregate summaries may be used; raw
or row-level records must not be redistributed.  The four tables are not
treated as a synchronized policy trial or operational availability study.

## Bounded validation runner

Run:

```bash
python scripts/run_public_trace_validation.py
```

The command writes the following auditable artifacts to
`results/public_trace_validation/`:

- `campaign_site_operator_coverage.csv`;
- `path_qos_distributions.csv`;
- `lens_telemetry_alignment.csv`;
- `sat_in_view_inference_summary.csv`;
- `nexasphere_athens_private_summary.csv`;
- `cross_campaign_replay_proxy_metrics.csv`;
- `cross_campaign_replay_proxy_uncertainty.csv`;
- `validation_manifest.json`.

The frozen proxy benchmark trains on the bounded Alaska Starlink--AT&T
throughput panel and tests without refitting on Hawaii.  Its endpoint is whether
a 5 MB payload can be served by ten future 0.5-second measured-throughput rows.
The comparators are training-set prevalence and current-state persistence.  A
lagged logistic model is included as a transparent data-driven reference, not
as the proposed journal framework.  Uncertainty is a paired bootstrap over
test `source_run_id` clusters.  Negative `delta_brier` favors the first model,
but a single Alaska-to-Hawaii result cannot establish consistent superiority
across climates, operators, or operational EO workloads.

## Reproducibility checks

The compact fixtures cover paired-panel normalization, mainland fail-closed
matching, streaming IRTT, ping sequence gaps, causal dish alignment,
connectivity episodes, inferred SatInView transitions, and exclusion of the
NEXASPHERE April pilot:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q tests/test_public_data_adapters.py
```

The complete test suite should be run before paper figures or tables are
regenerated.
