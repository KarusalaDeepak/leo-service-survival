# Application outcome acknowledgement emitter

## Purpose

`scripts/emit_application_outcome_acknowledgement.py` creates one authenticated
application acknowledgement for one action already present in the prospective
decision ledger. It records endpoint observations only. It does not declare task
success, append an outcome, change the field-trial status, or create missing field
measurements.

The emitter binds the acknowledgement to the action's study, trial, protocol,
campaign, acquisition family, session, task, decision, model hash, and
configuration hash. It records the required and delivered byte counts, deadline,
observation-finalization time, observation validity, application-completion state,
and explicit failure or missingness indicators.

## Authentication and immutability

The command loads the frozen acknowledgement schema and refuses to run if its
version, required evidence fields, or authentication purpose differ from the
supported contract. Live output uses the existing project convention:

- algorithm: `HMAC-SHA256`
- purpose: `O4_APPLICATION_OUTCOME_ACKNOWLEDGEMENT_V1`
- key identifier: the first 16 hexadecimal characters of the signing-key SHA-256
- content binding: SHA-256 of the canonical acknowledgement body

Keep the HMAC key outside the repository and make it available only to the
authorized endpoint or evidence operator. HMAC provides shared-key integrity and
authentication. It does not provide public non-repudiation.

The emitter writes to a temporary file, verifies the complete signed artifact with
the existing outcome-evidence verifier, changes the file to read-only mode, and
atomically links it to the requested destination. It refuses to replace an
existing path. The resulting content hash and HMAC still provide the authoritative
tamper check if an administrator later changes filesystem permissions.

## Command

The output directory must already exist. Every endpoint observation is explicit;
only `acknowledged_wall_ns` may be omitted, in which case the emitter records its
current wall-clock time.

```bash
python3 scripts/emit_application_outcome_acknowledgement.py \
  --ledger /secure/trial/action_outcome_ledger.jsonl \
  --ledger-schema prospective_trial/schema/decision_ledger.schema.json \
  --acknowledgement-schema prospective_trial/schema/application_outcome_acknowledgement.schema.json \
  --decision-id DECISION-0001 \
  --signing-key /secure/keys/application-observer.hmac \
  --output /secure/evidence/DECISION-0001.application-ack.json \
  --acknowledgement-id ACK-DECISION-0001 \
  --acknowledgement-source remote-anchor/application-observer/v1 \
  --required-bytes 1048576 \
  --delivered-bytes 1048576 \
  --deadline-wall-ns 1800000005000000000 \
  --completion-wall-ns 1800000004000000000 \
  --observation-valid true \
  --application-complete true \
  --no-action false \
  --controller-failure false \
  --late-install false
```

Use `--missingness-reason` when `--observation-valid false`. Valid observations
must have an empty missingness reason. A `no_action` acknowledgement must report
zero delivered bytes and an incomplete application. The emitter also rejects
placeholder action identities, all-zero model or configuration hashes, timestamps
that precede the linked action, and contradictory completion fields.

## Ledger consumption

Transfer the immutable acknowledgement to the ledger host without changing its
bytes. Then use the existing appender:

```bash
python3 scripts/append_prospective_outcome.py \
  --ledger /secure/trial/action_outcome_ledger.jsonl \
  --schema prospective_trial/schema/decision_ledger.schema.json \
  --decision-id DECISION-0001 \
  --outcome-evidence /secure/evidence/DECISION-0001.application-ack.json \
  --signing-key /secure/keys/application-observer.hmac
```

The appender independently verifies the acknowledgement signature, hashes,
identity binding, timestamps, and semantic consistency. It derives
`met_deadline`, the conservative completion endpoint, and the failure stage. Those
derived fields are deliberately forbidden in the endpoint acknowledgement so the
observer cannot self-declare the research outcome.

## Claim boundary

An emitted acknowledgement is evidence for one linked application observation.
It is not evidence that the full prospective trial ran, that the evidence gate
authorized a candidate, or that the candidate improved performance. Those claims
require the separately frozen campaign, release, canary, final-test, and analysis
artifacts.
