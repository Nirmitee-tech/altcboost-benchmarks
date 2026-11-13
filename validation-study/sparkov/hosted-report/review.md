# Validation study: sparkov

V6 settings were held fixed. Each baseline family received four prespecified configurations and validation-only selection of its configuration and tree prefix. Search counts and tree caps are matched across baseline families; wall time and model capacity are not matched.

Evidence: Consumed synthetic development evidence. Source type: synthetic card transactions generated with Sparkov.
The evaluated cohort contains 378 positive and 99622 negative card-transaction rows. Counts were measured in the run completed at 2026-11-12T14:31:06.028228+00:00.

The primary comparison uses seed 17 at the 0.1% calibration target. Targets do not guarantee the same false-positive rate in a later time period.

| Model | Positive cases caught | False positives | Recall | Observed FPR | Selected prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| V6 | 253 | 67 | 66.93% | 0.067% | 94 |
| CatBoost_native | 184 | 75 | 48.68% | 0.075% | 99 |
| XGBoost | 246 | 92 | 65.08% | 0.092% | 57 |
| LightGBM_native | 228 | 103 | 60.32% | 0.103% | 100 |

Training repeats use the same data cohorts. Their ranges describe stochastic fitting and validation-prefix selection, not independent replications of the dataset.

| Model | Seeds evaluated | Recall range | FPR range |
| --- | ---: | ---: | ---: |
| V6 | 3 | 66.93% to 66.93% | 0.067% to 0.067% |
| CatBoost_native | 3 | 48.68% to 50.53% | 0.075% to 0.078% |
| XGBoost | 3 | 64.81% to 65.34% | 0.089% to 0.092% |
| LightGBM_native | 3 | 58.73% to 61.90% | 0.079% to 0.103% |

The paired cluster checks below condition on the selected models and fixed calibration cutoffs. They omit uncertainty from retraining, hyperparameter selection and calibration. Intervals are descriptive percentile intervals with no multiple-comparison correction. A minimum of 20 clusters is a prespecified screen, not a validity theorem.

| Reference | Cluster scheme | Clusters | V6 minus reference recall, percentage points | V6 minus reference FPR, percentage points |
| --- | --- | ---: | ---: | ---: |
| CatBoost_native | calendar_week | 29 | +18.2540 [+14.5998, +22.0711] | -0.0080 [-0.0257, +0.0104] |
| CatBoost_native | card | 919 | +18.2540 [+14.1242, +22.1918] | -0.0080 [-0.0253, +0.0090] |
| XGBoost | calendar_week | 29 | +1.8519 [-0.8382, +4.4449] | -0.0251 [-0.0454, -0.0064] |
| XGBoost | card | 919 | +1.8519 [-0.8003, +4.4580] | -0.0251 [-0.0426, -0.0082] |
| LightGBM_native | calendar_week | 29 | +6.6138 [+3.1782, +9.7493] | -0.0361 [-0.0600, -0.0127] |
| LightGBM_native | card | 919 | +6.6138 [+3.5127, +9.7380] | -0.0361 [-0.0567, -0.0154] |

## For whoever builds this

Source: https://github.com/namebrandon/Sparkov_Data_Generation. Licence: CC0-1.0. Source and cohort hashes, allowlists and time bounds are in discovery.json, observed at 2026-11-12T14:23:48.293329+00:00.
The selected configurations were recorded in selection.json before calibration/test scoring. trials.csv records every baseline candidate; models.json records repeat prefixes and durations. All budgets and seeds remain in results.csv. Window checks use one global calibration cutoff. uncertainty.json retains all supported and unsupported paired checks.

Equal baseline configuration counts and tree caps, not equal wall time or model capacity. Small prespecified search, not exhaustive tuning. Same cohorts across seeds; stochastic refits are not independent datasets. Cluster intervals condition on selected models and fixed calibration cutoffs, omit training/calibration uncertainty and multiple-comparison correction, and do not handle arbitrary dependence. Minimum 20 clusters is a screening rule, not a theorem. Row-wise beta intervals in results assume independent rows and do not replace unsupported cluster checks. No superiority, novelty, point-in-time feature availability or production FPR guarantee.

Decision still needed: the research lead must assess V6 against the tuned baselines and the uncertainty limits. Payment-fraud confirmation on an untouched source and a completed novelty review remain separate requirements before a publication claim.
