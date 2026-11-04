# Paper 3 Scheduling Completion Inside Paper 4

Date: 3 November 2026

## Completion decision

The Earth-observation scheduling component developed in Paper 3 is complete
within the declared controlled offline scope and is now integrated into Paper 4
as a release-controlled subsystem.  Completion does not depend on the proposed
RACS ranking score outperforming every baseline.  It depends on implementing
the scheduling problem, validating generated schedules, comparing like-for-like
action spaces, and making a reproducible decision about which scheduler may be
used by the integrated framework.

## Integrated evidence

The integration uses the authoritative Paper 3 scheduler and evidence rather
than the earlier `racs_like` heuristic:

- nine controlled benchmark units and 6,768 validated schedule evaluations;
- cross-fitted RACS completion of 60.19% under combined stress;
- matched full-margin completion of 60.56%;
- equal-budget selected random-forest completion of 59.63%;
- RACS minus random forest of +0.56 percentage points, with a 95% interval of
  [0, 1.30] points and Holm-adjusted p-value of 1; and
- a frozen 24-scenario EOS-Bench transfer audit with 384 validated schedules,
  where every compared selector obtains 26.79% completion.

These results show that expanded delivery options are useful, but they do not
establish an advantage for the RACS ranking score over simpler matched
selectors.

## Frozen scheduler-release decision

`src/leo_service_survival/scheduler_release.py` evaluates a frozen set of
benchmark, learned-comparator, and transfer checks.  The decision is
`RETAIN_MATCHED_SIMPLE_BASELINE`: the full-margin scheduler is installed, while
cross-fitted RACS remains in shadow evaluation.  The decision fails closed
because the candidate has no positive controlled-benchmark difference, its
confidence interval does not support a positive effect, corrected significance
is absent, and the transfer difference is zero.

The decision can be rebuilt with:

```bash
PYTHONPATH="$PWD/src" python3.11 scripts/build_racs_scheduler_release.py
```

The authoritative outputs are in `results/racs_scheduler_release_v1/`.  The
underlying frozen transfer audit is in
`results/racs_eos_transfer_integration_v1/`.

The subsequent hash-verified interface audit in
`results/scheduler_path_bridge_v1/` reruns the installed `full_margin` policy
on a shared EOS-Bench scenario. It reproduces and independently validates all
28 selected plans, maps all 28 observations exactly to Paper 4 geometry, and
exposes 56 deduplicated path candidates. This proves schedule-to-path
traceability. It does not merge completion outcomes because the scheduling and
path experiments use different declared releases, deadlines, service overlays,
and post-decision outcomes.

## What is complete and what is not

Complete in software and controlled evaluation:

- resource-feasible observation-to-delivery schedule generation;
- matched full-capability baseline comparisons;
- independent validation of saved schedules;
- frozen transfer evaluation without retuning;
- fail-closed scheduler selection; and
- a hash-verified schedule-to-path handoff on shared geometry.

Not claimed:

- superiority of the RACS score;
- operational EO mission validation;
- a synchronized live geometry-to-installed-route-to-task-outcome field trial;
- multi-operator deployment benefit; or
- completed prospective hardware execution.

The last items require external terminals, approvals, synchronized field
telemetry, installed-action records, and real application or EO outcomes.  The
software protocol for collecting that evidence is implemented, but its field
status remains `NOT_RUN`.
