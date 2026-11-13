# Validation study: elliptic

V6 settings were held fixed. Each baseline family received four prespecified configurations and validation-only selection of its configuration and tree prefix. Search counts and tree caps are matched across baseline families; wall time and model capacity are not matched.

Evidence: First measured real Bitcoin AML transfer evaluation; settings frozen before scoring. Source type: real anonymized Bitcoin transactions, known-label subset only.
The evaluated cohort contains 408 positive and 8433 negative known-label Bitcoin transactions. Counts were measured in the run completed at 2026-11-12T14:30:30.698280+00:00.

The primary comparison uses seed 17 at the 0.1% calibration target. Targets do not guarantee the same false-positive rate in a later time period.

| Model | Positive cases caught | False positives | Recall | Observed FPR | Selected prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| V6 | 1 | 3 | 0.25% | 0.036% | 78 |
| CatBoost_native | 167 | 12 | 40.93% | 0.142% | 97 |
| XGBoost | 166 | 8 | 40.69% | 0.095% | 48 |
| LightGBM_native | 164 | 11 | 40.20% | 0.130% | 99 |

This is an AML transfer task on the known-label subset of real Bitcoin data. Unknown labels are excluded rather than counted as normal. It does not establish independent payment-fraud performance. The explicitly forward/backward-neighbour features are excluded; anonymous local-feature availability and provider normalization cannot be independently audited.

Training repeats use the same data cohorts. Their ranges describe stochastic fitting and validation-prefix selection, not independent replications of the dataset.

| Model | Seeds evaluated | Recall range | FPR range |
| --- | ---: | ---: | ---: |
| V6 | 3 | 0.25% to 0.25% | 0.036% to 0.036% |
| CatBoost_native | 3 | 40.44% to 40.93% | 0.107% to 0.142% |
| XGBoost | 3 | 40.69% to 41.18% | 0.095% to 0.225% |
| LightGBM_native | 3 | 40.20% to 40.20% | 0.071% to 0.130% |

The paired cluster checks below condition on the selected models and fixed calibration cutoffs. They omit uncertainty from retraining, hyperparameter selection and calibration. Intervals are descriptive percentile intervals with no multiple-comparison correction. A minimum of 20 clusters is a prespecified screen, not a validity theorem.

| Reference | Cluster scheme | Clusters | V6 minus reference recall, percentage points | V6 minus reference FPR, percentage points |
| --- | --- | ---: | ---: | ---: |
| CatBoost_native | published_time_group | 8 | Unsupported: too few usable clusters | |
| XGBoost | published_time_group | 8 | Unsupported: too few usable clusters | |
| LightGBM_native | published_time_group | 8 | Unsupported: too few usable clusters | |

## For whoever builds this

Source: https://arxiv.org/abs/1908.02591. Licence: CC-BY-NC-ND-4.0. Source and cohort hashes, allowlists and time bounds are in discovery.json, observed at 2026-11-12T14:23:53.959080+00:00.
The selected configurations were recorded in selection.json before calibration/test scoring. trials.csv records every baseline candidate; models.json records repeat prefixes and durations. All budgets and seeds remain in results.csv. Window checks use one global calibration cutoff. uncertainty.json retains all supported and unsupported paired checks.

Equal baseline configuration counts and tree caps, not equal wall time or model capacity. Small prespecified search, not exhaustive tuning. Same cohorts across seeds; stochastic refits are not independent datasets. Cluster intervals condition on selected models and fixed calibration cutoffs, omit training/calibration uncertainty and multiple-comparison correction, and do not handle arbitrary dependence. Minimum 20 clusters is a screening rule, not a theorem. Row-wise beta intervals in results assume independent rows and do not replace unsupported cluster checks. No superiority, novelty, point-in-time feature availability or production FPR guarantee.

Decision still needed: the research lead must assess V6 against the tuned baselines and the uncertainty limits. Payment-fraud confirmation on an untouched source and a completed novelty review remain separate requirements before a publication claim.
