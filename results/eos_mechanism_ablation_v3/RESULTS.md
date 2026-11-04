# EOS mechanism ablation (synthetic simulation)

The fitted WetLinks survival model was **not** applied because the EOS overlay does not provide its 16 transport/weather features, five-minute units, or validated horizon adapter.

All six arms use matched actions, EDF task order, shared resource constraints and common hash-keyed potential outcomes. Results below are controlled synthetic evidence only.

## Completion summary

| condition   | policy_arm                                 |   tasks |   synthetic_replications |   completion_rate |   seed_cluster_bootstrap_95_ci_low |   seed_cluster_bootstrap_95_ci_high |   no_action_rate |   high_snr_path_selection_rate | evidence_tier        |
|:------------|:-------------------------------------------|--------:|-------------------------:|------------------:|-----------------------------------:|------------------------------------:|-----------------:|-------------------------------:|:---------------------|
| adverse     | edf_current_snr                            |    1000 |                       10 |             0.501 |                           0.47     |                            0.529    |            0.181 |                          0.797 | synthetic_simulation |
| adverse     | edf_simple_reliability_heuristic           |    1000 |                       10 |             0.737 |                           0.722    |                            0.750025 |            0.182 |                          0.011 | synthetic_simulation |
| adverse     | synthetic_calibrated_no_congestion_penalty |    1000 |                       10 |             0.746 |                           0.734    |                            0.757    |            0.181 |                          0.009 | synthetic_simulation |
| adverse     | synthetic_calibrated_no_input_screening    |    1000 |                       10 |             0.726 |                           0.709    |                            0.742    |            0.181 |                          0.009 | synthetic_simulation |
| adverse     | synthetic_calibrated_score                 |    1000 |                       10 |             0.725 |                           0.709    |                            0.74     |            0.181 |                          0.009 | synthetic_simulation |
| adverse     | synthetic_uncalibrated_score               |    1000 |                       10 |             0.726 |                           0.712    |                            0.74     |            0.181 |                          0.009 | synthetic_simulation |
| favorable   | edf_current_snr                            |    1000 |                       10 |             0.734 |                           0.709    |                            0.756    |            0.181 |                          0.797 | synthetic_simulation |
| favorable   | edf_simple_reliability_heuristic           |    1000 |                       10 |             0.736 |                           0.709    |                            0.756    |            0.182 |                          0.807 | synthetic_simulation |
| favorable   | synthetic_calibrated_no_congestion_penalty |    1000 |                       10 |             0.729 |                           0.714    |                            0.746    |            0.182 |                          0.808 | synthetic_simulation |
| favorable   | synthetic_calibrated_no_input_screening    |    1000 |                       10 |             0.727 |                           0.705    |                            0.749    |            0.182 |                          0.809 | synthetic_simulation |
| favorable   | synthetic_calibrated_score                 |    1000 |                       10 |             0.727 |                           0.706975 |                            0.748025 |            0.182 |                          0.808 | synthetic_simulation |
| favorable   | synthetic_uncalibrated_score               |    1000 |                       10 |             0.727 |                           0.707    |                            0.747    |            0.182 |                          0.808 | synthetic_simulation |
| neutral     | edf_current_snr                            |    1000 |                       10 |             0.638 |                           0.61     |                            0.662    |            0.181 |                          0.797 | synthetic_simulation |
| neutral     | edf_simple_reliability_heuristic           |    1000 |                       10 |             0.619 |                           0.606    |                            0.629    |            0.182 |                          0.414 | synthetic_simulation |
| neutral     | synthetic_calibrated_no_congestion_penalty |    1000 |                       10 |             0.609 |                           0.584    |                            0.635    |            0.182 |                          0.407 | synthetic_simulation |
| neutral     | synthetic_calibrated_no_input_screening    |    1000 |                       10 |             0.615 |                           0.59     |                            0.638025 |            0.182 |                          0.413 | synthetic_simulation |
| neutral     | synthetic_calibrated_score                 |    1000 |                       10 |             0.609 |                           0.583    |                            0.634    |            0.182 |                          0.407 | synthetic_simulation |
| neutral     | synthetic_uncalibrated_score               |    1000 |                       10 |             0.609 |                           0.583975 |                            0.636    |            0.182 |                          0.407 | synthetic_simulation |

## Synthetic calibration validation

| condition   |   validation_rows |   validation_prevalence |   raw_brier |   calibrated_brier |   raw_log_loss |   calibrated_log_loss |   raw_ece_10bin |   calibrated_ece_10bin | calibrator_sha256                                                | evidence_tier        |
|:------------|------------------:|------------------------:|------------:|-------------------:|---------------:|----------------------:|----------------:|-----------------------:|:-----------------------------------------------------------------|:---------------------|
| favorable   |              3000 |                0.757667 |    0.158314 |           0.158512 |       0.480605 |              0.481046 |      0.01999    |             0.0229209  | f464ac957e714a9483a5b9d238fac05ea0c8c34c9b3c2c73514895132702fbc1 | synthetic_simulation |
| neutral     |              3000 |                0.744667 |    0.190273 |           0.190159 |       0.568439 |              0.568152 |      0.00493039 |             0.00563379 | f739ba3253dde3fd784a856f2c32fd90114d3a80e6c3caaec7d9f45bc0f423d7 | synthetic_simulation |
| adverse     |              3000 |                0.748    |    0.167269 |           0.167451 |       0.505234 |              0.505718 |      0.0104155  |             0.0127074  | a36a7fa78113d34da3ae5572b744de100ff5bb78743461fdb2276fade83c2cc4 | synthetic_simulation |

See `pairwise_inference.csv` for paired differences and seed-cluster intervals. Those intervals reuse one generated scenario and are not field-inference confidence intervals.
Direct calibration, input-screening and congestion-penalty ablations are in `mechanism_contrasts.csv`.
