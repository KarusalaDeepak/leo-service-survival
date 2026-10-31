# EOS mechanism ablation (synthetic simulation)

The fitted WetLinks survival model was **not** applied because the EOS overlay does not provide its 16 transport/weather features, five-minute units, or validated horizon adapter.

All six arms use matched actions, EDF task order, shared resource constraints and common hash-keyed potential outcomes. Results below are controlled synthetic evidence only.

## Completion summary

| condition   | policy_arm                                 |   tasks |   synthetic_replications |   completion_rate |   seed_cluster_bootstrap_95_ci_low |   seed_cluster_bootstrap_95_ci_high |   no_action_rate |   high_snr_path_selection_rate | evidence_tier        |
|:------------|:-------------------------------------------|--------:|-------------------------:|------------------:|-----------------------------------:|------------------------------------:|-----------------:|-------------------------------:|:---------------------|
| adverse     | edf_current_snr                            |    1000 |                       10 |             0.501 |                           0.47     |                               0.529 |            0.181 |                          0.797 | synthetic_simulation |
| adverse     | edf_simple_reliability_heuristic           |    1000 |                       10 |             0.742 |                           0.728    |                               0.754 |            0.182 |                          0.012 | synthetic_simulation |
| adverse     | synthetic_calibrated_no_congestion_penalty |    1000 |                       10 |             0.734 |                           0.716    |                               0.751 |            0.181 |                          0.009 | synthetic_simulation |
| adverse     | synthetic_calibrated_no_input_screening    |    1000 |                       10 |             0.735 |                           0.723    |                               0.749 |            0.181 |                          0.008 | synthetic_simulation |
| adverse     | synthetic_calibrated_score                 |    1000 |                       10 |             0.734 |                           0.722    |                               0.747 |            0.181 |                          0.008 | synthetic_simulation |
| adverse     | synthetic_uncalibrated_score               |    1000 |                       10 |             0.735 |                           0.724    |                               0.747 |            0.181 |                          0.008 | synthetic_simulation |
| favorable   | edf_current_snr                            |    1000 |                       10 |             0.734 |                           0.709    |                               0.756 |            0.181 |                          0.797 | synthetic_simulation |
| favorable   | edf_simple_reliability_heuristic           |    1000 |                       10 |             0.735 |                           0.721    |                               0.749 |            0.182 |                          0.81  | synthetic_simulation |
| favorable   | synthetic_calibrated_no_congestion_penalty |    1000 |                       10 |             0.732 |                           0.715    |                               0.749 |            0.182 |                          0.808 | synthetic_simulation |
| favorable   | synthetic_calibrated_no_input_screening    |    1000 |                       10 |             0.728 |                           0.715    |                               0.742 |            0.182 |                          0.808 | synthetic_simulation |
| favorable   | synthetic_calibrated_score                 |    1000 |                       10 |             0.729 |                           0.716    |                               0.742 |            0.182 |                          0.808 | synthetic_simulation |
| favorable   | synthetic_uncalibrated_score               |    1000 |                       10 |             0.733 |                           0.721    |                               0.745 |            0.182 |                          0.807 | synthetic_simulation |
| neutral     | edf_current_snr                            |    1000 |                       10 |             0.638 |                           0.61     |                               0.662 |            0.181 |                          0.797 | synthetic_simulation |
| neutral     | edf_simple_reliability_heuristic           |    1000 |                       10 |             0.622 |                           0.605    |                               0.639 |            0.181 |                          0.407 | synthetic_simulation |
| neutral     | synthetic_calibrated_no_congestion_penalty |    1000 |                       10 |             0.623 |                           0.61     |                               0.635 |            0.181 |                          0.408 | synthetic_simulation |
| neutral     | synthetic_calibrated_no_input_screening    |    1000 |                       10 |             0.625 |                           0.605    |                               0.642 |            0.181 |                          0.395 | synthetic_simulation |
| neutral     | synthetic_calibrated_score                 |    1000 |                       10 |             0.618 |                           0.598    |                               0.634 |            0.181 |                          0.393 | synthetic_simulation |
| neutral     | synthetic_uncalibrated_score               |    1000 |                       10 |             0.619 |                           0.595975 |                               0.639 |            0.181 |                          0.396 | synthetic_simulation |

## Synthetic calibration validation

| condition   |   validation_rows |   validation_prevalence |   raw_brier |   calibrated_brier |   raw_log_loss |   calibrated_log_loss |   raw_ece_10bin |   calibrated_ece_10bin | calibrator_sha256                                                | evidence_tier        |
|:------------|------------------:|------------------------:|------------:|-------------------:|---------------:|----------------------:|----------------:|-----------------------:|:-----------------------------------------------------------------|:---------------------|
| favorable   |              3000 |                0.757667 |    0.159652 |           0.158855 |       0.486053 |              0.481745 |       0.0406962 |             0.0224128  | 421d4648be49b9c583dbc3d4a137deab09fd1c14ad3fd74c02fca1607cfa571c | synthetic_simulation |
| neutral     |              3000 |                0.744667 |    0.190739 |           0.19052  |       0.569613 |              0.569134 |       0.0209184 |             0.00599034 | 48475ff92ddd04d35b55d09b85b5921b302b9e307f0bb38299eae830bb0975d0 | synthetic_simulation |
| adverse     |              3000 |                0.748    |    0.167818 |           0.167548 |       0.507593 |              0.506346 |       0.025436  |             0.0175739  | 75b43d640f0afba998e9062ebce2460d4501afc9add82f3324ed29f7467bf8c4 | synthetic_simulation |

See `pairwise_inference.csv` for paired differences and seed-cluster intervals. Those intervals reuse one generated scenario and are not field-inference confidence intervals.
Direct calibration, input-screening and congestion-penalty ablations are in `mechanism_contrasts.csv`.
