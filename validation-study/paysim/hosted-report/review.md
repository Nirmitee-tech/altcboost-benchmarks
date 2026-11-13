# Validation study: paysim

V6 settings were held fixed. Each baseline family received four prespecified configurations and validation-only selection of its configuration and tree prefix. Search counts and tree caps are matched across baseline families; wall time and model capacity are not matched.

Evidence: Consumed synthetic development evidence. Source type: synthetic mobile-money transactions.
The evaluated cohort contains 1252 positive and 88214 negative transaction rows. Counts were measured in the run completed at 2026-11-12T14:27:16.571195+00:00.

The primary comparison uses seed 17 at the 0.1% calibration target. Targets do not guarantee the same false-positive rate in a later time period.

| Model | Positive cases caught | False positives | Recall | Observed FPR | Selected prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| V6 | 888 | 84 | 70.93% | 0.095% | 57 |
| CatBoost_native | 848 | 79 | 67.73% | 0.090% | 100 |
| XGBoost | 825 | 76 | 65.89% | 0.086% | 80 |
| LightGBM_native | 678 | 71 | 54.15% | 0.080% | 1 |

Training repeats use the same data cohorts. Their ranges describe stochastic fitting and validation-prefix selection, not independent replications of the dataset.

| Model | Seeds evaluated | Recall range | FPR range |
| --- | ---: | ---: | ---: |
| V6 | 3 | 70.93% to 70.93% | 0.095% to 0.095% |
| CatBoost_native | 3 | 67.25% to 68.29% | 0.088% to 0.096% |
| XGBoost | 3 | 65.89% to 66.77% | 0.083% to 0.086% |
| LightGBM_native | 3 | 53.19% to 55.35% | 0.063% to 0.080% |

The paired cluster checks below condition on the selected models and fixed calibration cutoffs. They omit uncertainty from retraining, hyperparameter selection and calibration. Intervals are descriptive percentile intervals with no multiple-comparison correction. A minimum of 20 clusters is a prespecified screen, not a validity theorem.

| Reference | Cluster scheme | Clusters | V6 minus reference recall, percentage points | V6 minus reference FPR, percentage points |
| --- | --- | ---: | ---: | ---: |
| CatBoost_native | four_step_time_block | 28 | +3.1949 [+2.0569, +4.4343] | +0.0057 [-0.0116, +0.0148] |
| XGBoost | four_step_time_block | 28 | +5.0319 [+3.7094, +6.5978] | +0.0091 [-0.0012, +0.0175] |
| LightGBM_native | four_step_time_block | 28 | +16.7732 [+15.0324, +18.5806] | +0.0147 [+0.0040, +0.0273] |

## For whoever builds this

Source: https://github.com/EdgarLopezPhD/PaySim. Licence: CC-BY-SA-4.0. Source and cohort hashes, allowlists and time bounds are in discovery.json, observed at 2026-11-12T14:23:40.439254+00:00.
The selected configurations were recorded in selection.json before calibration/test scoring. trials.csv records every baseline candidate; models.json records repeat prefixes and durations. All budgets and seeds remain in results.csv. Window checks use one global calibration cutoff. uncertainty.json retains all supported and unsupported paired checks.

Equal baseline configuration counts and tree caps, not equal wall time or model capacity. Small prespecified search, not exhaustive tuning. Same cohorts across seeds; stochastic refits are not independent datasets. Cluster intervals condition on selected models and fixed calibration cutoffs, omit training/calibration uncertainty and multiple-comparison correction, and do not handle arbitrary dependence. Minimum 20 clusters is a screening rule, not a theorem. Row-wise beta intervals in results assume independent rows and do not replace unsupported cluster checks. No superiority, novelty, point-in-time feature availability or production FPR guarantee.

Decision still needed: the research lead must assess V6 against the tuned baselines and the uncertainty limits. Payment-fraud confirmation on an untouched source and a completed novelty review remain separate requirements before a publication claim.
