# Redistribution boundary

This version 1.3.0 repository is an allowlisted, redistribution-safe reviewer
package. Its Objective 4 integration files were imported from the verified R1
submission archive. It is not a copy of the complete private working tree.

## Included

- Author-written software, tests, configurations, protocol documents, schemas,
  and aggregate summaries.
- The separately identified WetLinks reviewer prediction derivative under CC
  BY-SA 4.0; it does not contain the upstream measurement tables.
- Aggregate RACS benchmark, scheduler-release, and EOS-Bench transfer summaries
  that contain no source scenario rows, task identifiers, or saved schedules.
- Aggregate scheduler-to-path bridge status and hashes; task-level handoff
  tables are excluded.
- A deterministic `FAKE_DRIVER_TEST` rehearsal regenerated with package-relative
  paths and fresh hashes/signatures. It is software evidence, not field evidence.

## Excluded

- All row-level outputs from `results/eos_mechanism_ablation_v3/` and
  `results/racs_eos_transfer_integration_v1/`. They depend on EOS-Bench source
  scenarios for which an explicit redistribution license was not found.
- Raw WetLinks measurements and all row-level datasets or derived row-level
  copies without explicit release clearance, including Palo Alto, COMMECT,
  LENS, OneWeb, NUWiNS, SatInView, NEXASPHERE, FjordLink, SPOT5, and Paper 1
  geometry rows. The licensed WetLinks reviewer derivative described above is
  the only stated exception.
- Private `results/`, `data/`, `upstream/`, `simulation/`, manuscript-run,
  completion-run, archive, cache, and local build directories.
- Local execution manifests that contained machine-specific absolute paths.
- Third-party ns-3, 5G-LENA, and ns-3-NTN source trees and generated traces.

These exclusions preserve the private evidence tree without silently granting
redistribution rights. The manuscript reports bounded aggregate results; the
author-written summary in `aggregate_evidence/AGGREGATE_RESULTS.md` does not
contain measurement rows or permit reconstruction of restricted source data.

## Source acquisition

Obtain third-party inputs directly from their authoritative sources and retain
their licenses, citations, acquisition timestamps, and hashes:

- WetLinks: DOI `10.23919/TMA62044.2024.10558998` (source identifies CC BY-SA
  4.0).
- Palo Alto Starlink record: DOI `10.5281/zenodo.14987305`; verify the record's
  current redistribution terms before use.
- COMMECT: DOI `10.5281/zenodo.14620779`; inspect the downloaded archive and
  current Zenodo terms before redistribution.
- LENS-derived artifact: DOI `10.5281/zenodo.15331299`; verify the full
  provenance and redistribution chain.
- LENS OneWeb Iowa: DOI `10.5281/zenodo.15786332` (source documentation states
  CC BY-SA 4.0).
- NUWiNS, FjordLink, and SPOT5: acquire from the cited project/publisher records
  and retain their stated CC BY obligations and modification notices.
- SatInView and EOS-Bench: obtain from the authors' repositories; do not
  redistribute acquired or derived row-level material unless the authors grant
  explicit permission.
- NEXASPHERE Athens: keep private until its reuse terms are resolved.
- ns-3 and related modules: obtain the pinned upstream sources and comply with
  every component's GPL and notice obligations.

After lawful acquisition, use the code paths documented in `scripts/` and the
manuscript's protocol. A valid reproduction must preserve the manuscript's
development/calibration/test separation and must not relabel repeated rows,
days, or sessions as independent acquisition families.

## Public-release condition

Do not publish this package as reusable open-source software until the authors
select an explicit code license. Do not add a public repository URL to the
manuscript until the repository and this boundary have been reviewed together.
