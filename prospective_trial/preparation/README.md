# Prospective Field Extension, Steps 1–3: Executable Preparation Boundary

This package implements the software controls for the first three prospective
field steps. It does **not** supply hardware, permissions, field campaigns, or
operational outcomes.

## Evidence states

- `NOT_RUN`: a required action or measurement has not occurred.
- `LIVE_REQUIRED`: the contract is implemented, but qualifying live evidence is
  absent.
- `FAKE_DRIVER_TEST`: deterministic continuous-integration evidence; useful for
  failure injection and wiring checks, never a field result.
- `LIVE_QUALIFIED`: emitted only when a non-placeholder `live_hardware`
  configuration, immutable evidence hashes, approved permissions, and all named
  read-only probes pass.

`FAKE_DRIVER_TEST` is intentionally not promotable to `LIVE_QUALIFIED`.

## Step 1: qualify the testbed and real driver

Copy `config/testbed_qualification.template.json` and
`config/route_driver.template.json` into an approved configuration location.
Replace every placeholder, capture permission and component evidence, and set
`evidence_source` to `live_hardware`. The route validator checks a controlled
anchor destination, distinct baseline/candidate paths, absolute no-shell command
arrays, complete install/first-packet commands, positive loop/first-packet
budgets, a separately attested isolated rollback namespace with all four drill
commands, and mutation-disabled defaults. Set `status` to
`FROZEN_DEPLOYMENT_CONFIG` only after these checks and operator approval pass.

The following command validates only; it executes no route change:

```bash
PYTHONPATH=src python3 scripts/qualify_field_testbed.py \
  --testbed-config /approved/testbed.json \
  --route-config /approved/route_driver.json \
  --ledger-schema prospective_trial/schema/decision_ledger.schema.json \
  --signing-key /approved/operator_signing.key \
  --execute-live-read-only-probes --acknowledge-live-hardware \
  --output /approved/signed_live_qualification.json
```

Live read-only probes require both `--execute-live-read-only-probes` and
`--acknowledge-live-hardware`. Imported `--probe-results` may be used for an audit,
but cannot produce `LIVE_QUALIFIED` evidence. Route mutation is not implemented in
this command.

## Step 2: lock role-separated campaigns

Populate `config/campaign_registry_lock.template.csv` only after each raw bundle
and derived manifest is complete. Keep the artifacts separate:
`raw_capture_sha256` binds the loss-preserving synchronized telemetry file, while
`gate_opportunity_sha256` binds the derived paired-opportunity CSV for each
`gate_evidence` row. The latter is empty for roles that do not contribute to the
gate. The lock refuses family reuse across roles, fingerprint reuse, raw-capture
or gate-opportunity reuse, shared role-access groups, cross-role time overlap on
the same terminal/server pair, placeholders, invalid clocks, and missing artifact
hashes. It writes a deterministic row-level and whole-register SHA-256 snapshot
and refuses overwrite.

```bash
PYTHONPATH=src python3 scripts/lock_campaign_registry.py \
  --input /approved/complete_campaign_register.csv \
  --output /approved/locked_campaign_register.json \
  --protocol-sha256 ... \
  --qualification-sha256 ... \
  --lock-timestamp-utc 2026-11-03T00:00:00Z
```

## Step 3: ingest synchronized telemetry without dropping failures

The contract in `schema/synchronized_telemetry.schema.json` preserves:

- exactly one `reply`, `loss`, or `timeout` record for every scheduled probe;
- RF measurements, beam and gateway identifiers, and handover events;
- proposed path, install acknowledgement, independently observed FIB path,
  first packet, final path, and rollback evidence; and
- application or EO task size, deadline, delivered bytes, completion, and
  no-action/failure outcomes.

Each event carries wall and monotonic timestamps, a clock epoch, worst-case clock
error, and clock-evidence hash. Ingestion refuses overwrite, writes a durable
hash chain, checks expected-probe completeness, joins actions to outcomes, and
checks campaign/family identity against the immutable lock.

```bash
PYTHONPATH=src python3 scripts/ingest_synchronized_telemetry.py \
  --input /capture/raw_events.jsonl \
  --expected-probe-schedule /capture/expected_probes.csv \
  --campaign-lock /approved/locked_campaign_register.json \
  --output /durable/campaign/telemetry.jsonl
```

## Deterministic verification

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_field_preparation.py
```

The tests exercise fake component probes, driver validation, leakage rejection,
tamper detection, and a complete reply/loss/timeout plus route/outcome lifecycle.
They assert that all fake evidence remains `FAKE_DRIVER_TEST` and is not live
eligible.
