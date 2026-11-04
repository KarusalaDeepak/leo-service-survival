# Frozen scheduler-to-path integration boundary

The Paper 3 scheduling component is now applied in Paper 4 through a
hash-verified interface bridge.  The release decision keeps the matched
`full_margin` selector installed and leaves cross-fitted RACS in shadow mode.
The bridge reruns the authoritative Paper 3 `matched_scheduler`, requires the
rerun to match the frozen transfer schedule, and then maps each selected
observation to Paper 4 path candidates on the same EOS-Bench geometry.

Controlled result for `output_re_s3/Sats3_M100_T7d_cities_08`:

- the authoritative `full_margin` rerun completed 28 of 100 requests;
- all 28 released plans passed Paper 3's independent schedule validator;
- all 28 selected observations mapped to the exact Paper 4 geometry contact;
- the interface exposed 56 deduplicated path candidates, two per plan;
- RACS was not installed or declared superior.

The output is in `results/scheduler_path_bridge_v1/`.  All source artifacts,
schedule outputs, the shared scenario, and the Paper 4 candidate table are
checked by SHA-256 before the ledger is written.

## Scientific boundary

This bridge proves scheduler-release application and schedule-to-path
traceability.  It does not merge the two experiments' completion outcomes.
Paper 3 adds a deterministic simulated ground-service schedule, release time
zero, and a scenario-horizon deadline.  Paper 4 independently generates task
releases, deadlines, network rates, gateways, and post-decision outcomes.
Combining those outcomes would silently change both frozen protocols.

Recommended manuscript wording:

> The frozen scheduler gate retained the matched full-margin selector and kept
> RACS in shadow evaluation.  In a hash-verified interface audit on one
> EOS-Bench scenario, rerunning the authoritative selector reproduced 28
> validated plans; every selected observation mapped exactly to the Paper 4
> path-candidate interface, exposing two synthetic path candidates per plan.
> Because the scheduling and path studies use different declared service and
> workload overlays, this audit establishes software traceability, not a new
> end-to-end completion comparison or operational benefit.
