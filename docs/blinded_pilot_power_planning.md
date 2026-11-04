# Blinded Pilot Power and Precision Planning

`scripts/plan_blinded_pilot.py` freezes the future field-extension randomized-block
requirement without reading treatment assignments or treatment-specific pilot
outcomes. Its creation interface accepts exactly eight aggregate quantities:

- baseline completion probability;
- planned opportunities per randomized block;
- intracluster correlation coefficient (ICC);
- missing-outcome fraction;
- installation-failure fraction;
- minimum practically important absolute completion effect;
- target power; and
- target confidence-interval half-width.

The method uses a two-sided 95% normal approximation for a difference in binary
completion proportions. It applies the block design effect
`1 + (block_size - 1) * ICC` and conservatively reduces information by
`(1 - missing_fraction) * (1 - installation_failure_fraction)`. It calculates
the per-arm block requirement independently for power and precision, then chooses
the maximum of those two values and the fixed minimum of ten blocks per arm.

Create a plan with a signing key held outside the repository:

```bash
PYTHONPATH=src python3 scripts/plan_blinded_pilot.py \
  --baseline-completion 0.60 \
  --block-size 20 \
  --icc 0.08 \
  --missing-fraction 0.05 \
  --installation-failure-fraction 0.03 \
  --effect-margin 0.08 \
  --target-power 0.80 \
  --target-ci-half-width 0.07 \
  --signing-key /approved/operator.key \
  --output /approved/blinded-pilot-plan.json
```

The output contains the inputs, fixed assumptions, intermediate calculations,
the separate power and precision requirements, the selected requirement, a
canonical SHA-256 content hash and a purpose-bound HMAC-SHA256 authentication
record. Creation refuses to overwrite an existing plan.

Verify both authentication and recomputation with:

```bash
PYTHONPATH=src python3 scripts/plan_blinded_pilot.py \
  --verify /approved/blinded-pilot-plan.json \
  --signing-key /approved/operator.key
```

Verification rejects any changed field and any extra input, including a treatment
label or treatment-specific estimate. This is a planning artifact only: it does
not constitute field evidence, authorize a policy, or replace the registered
intention-to-treat analysis.
