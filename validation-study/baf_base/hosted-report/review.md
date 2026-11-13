# Validation study: baf_base

V6 settings were held fixed. Each baseline family received four prespecified configurations and validation-only selection of its configuration and tree prefix. Search counts and tree caps are matched across baseline families; wall time and model capacity are not matched.

Evidence: Consumed synthetic development evidence. Source type: synthetic, generated from anonymized real application data.
The evaluated cohort contains 1408 positive and 98592 negative bank-account application rows. Counts were measured in the run completed at 2026-11-12T14:29:05.111097+00:00.

The primary comparison uses seed 17 at the 0.1% calibration target. Targets do not guarantee the same false-positive rate in a later time period.

| Model | Positive cases caught | False positives | Recall | Observed FPR | Selected prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| V6 | 45 | 63 | 3.20% | 0.064% | 0 |
| CatBoost_native | 76 | 103 | 5.40% | 0.104% | 65 |
| XGBoost | 70 | 91 | 4.97% | 0.092% | 70 |
| LightGBM_native | 67 | 104 | 4.76% | 0.105% | 87 |

V6 validation retained only its starting linear model. No residual trees were selected, so this result cannot be attributed to the proposed tree-training mechanism.

Training repeats use the same data cohorts. Their ranges describe stochastic fitting and validation-prefix selection, not independent replications of the dataset.

| Model | Seeds evaluated | Recall range | FPR range |
| --- | ---: | ---: | ---: |
| V6 | 3 | 3.20% to 3.20% | 0.064% to 0.064% |
| CatBoost_native | 3 | 5.40% to 6.18% | 0.098% to 0.115% |
| XGBoost | 3 | 4.97% to 5.82% | 0.092% to 0.101% |
| LightGBM_native | 3 | 3.91% to 4.76% | 0.086% to 0.124% |

The paired cluster checks below condition on the selected models and fixed calibration cutoffs. They omit uncertainty from retraining, hyperparameter selection and calibration. Intervals are descriptive percentile intervals with no multiple-comparison correction. A minimum of 20 clusters is a prespecified screen, not a validity theorem.

| Reference | Cluster scheme | Clusters | V6 minus reference recall, percentage points | V6 minus reference FPR, percentage points |
| --- | --- | ---: | ---: | ---: |
| CatBoost_native | published_time_group | 2 | Unsupported: too few usable clusters | |
| XGBoost | published_time_group | 2 | Unsupported: too few usable clusters | |
| LightGBM_native | published_time_group | 2 | Unsupported: too few usable clusters | |

## For whoever builds this

Source: https://github.com/Feedzai/bank-account-fraud. Licence: CC-BY-NC-SA-4.0. Source and cohort hashes, allowlists and time bounds are in discovery.json, observed at 2026-11-12T14:23:27.323405+00:00.
The selected configurations were recorded in selection.json before calibration/test scoring. trials.csv records every baseline candidate; models.json records repeat prefixes and durations. All budgets and seeds remain in results.csv. Window checks use one global calibration cutoff. uncertainty.json retains all supported and unsupported paired checks.

Equal baseline configuration counts and tree caps, not equal wall time or model capacity. Small prespecified search, not exhaustive tuning. Same cohorts across seeds; stochastic refits are not independent datasets. Cluster intervals condition on selected models and fixed calibration cutoffs, omit training/calibration uncertainty and multiple-comparison correction, and do not handle arbitrary dependence. Minimum 20 clusters is a screening rule, not a theorem. Row-wise beta intervals in results assume independent rows and do not replace unsupported cluster checks. No superiority, novelty, point-in-time feature availability or production FPR guarantee.

Decision still needed: the research lead must assess V6 against the tuned baselines and the uncertainty limits. Payment-fraud confirmation on an untouched source and a completed novelty review remain separate requirements before a publication claim.
