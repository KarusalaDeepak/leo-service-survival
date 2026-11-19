# R34 correction and interpretation

## What changed

R33's recovery code corrected storage using all assigned original future failures before earlier recovery events. This is a causal-order defect. Its previously reported +6.48-point recovery contrast is excluded from the current manuscript's performance conclusions; it must not be cited as a verified online recovery gain. Older snapshots remain archived for audit, not as competing final versions.

R34 implements a separate causal event queue. Policy-visible contact objects contain forecasts, resource identity, bounds and rate, but never future event times or labels. Jobs enter after the common acquisition/relay scaffold is ready. The engine reveals chunk byte acknowledgements and two-bin alarm confirmations in time order, retains remaining bytes, and permits matched recovery. A contact's assigned episode never resets with chunking or retry. Original fixed observation/relay reservations remain exclusive.

## Units and bottleneck

The common scaffold admits 59/100 requests, with priority fraction 0.726631 and volume fraction 0.5900. At the supplied nominal rate each admitted payload needs only 4.05 seconds. A long reserved contact therefore was not evidence of a long payload transfer. The native-rate panel is retained.

The separate declared constrained-rate panel multiplies every rate by 0.0045, making median nominal transmission 900 seconds at load one. This is a constructed capacity sensitivity, not a measured RF calibration. Both panels execute all 20 assignments × 3 payload levels × 3 overheads × 4 policies = 720 runs each, 1,440 total. All admitted requests complete in every cell. Primary and goodput differences are zero; their assignment bootstrap intervals are [0,0] and Holm p-values are 1. This is a fixed-workload ceiling, not a proof of policy equivalence.

In constrained-rate cells isotonic scoring uses 0.42–1.71% less occupied transmission time than maximum-chunk EDF, but frozen scoring is lower than isotonic in six of nine cells. Resource use is descriptive; there is no established calibrated task-completion advantage. Do not manufacture one by selecting a successful seed or adjusting deadlines after seeing results.

## Scope retained

The queue is not full online acquisition/energy/pointing optimization. It uses a common geometry-derived scaffold, constructed windows, independent hash-assigned service episodes, complete-case selection, ideal checkpoint bytes and declared overheads. Conditional use of episode-start curves is an explicit unvalidated control assumption. Whole-seed inference is conditional on this one workload and trace pool, not independent field campaigns.

## Verification

Targeted tests cover causal noninterference, no lifetime reset, two-bin confirmation and switching delay, byte/storage accounting, deadlines and resource exclusion. Verified native and duration panels reproduce the first-run CSVs after removing the R33 dependency and making simultaneous detection/completion ordering explicit. Historical independent checks are retained separately and are not called operational validation.
