# Frozen V6 against tuned baselines

Each cell gives positive cases caught / false positives for seed 17 at the 0.1% calibration target. Sources are evaluated separately. All configurations were fixed or selected using training and validation data before final scoring.

| Dataset | Evidence | V6 | Native CatBoost | XGBoost | Native LightGBM |
| --- | --- | ---: | ---: | ---: | ---: |
| baf_base | Consumed synthetic development | 45 / 63 | 76 / 103 | 70 / 91 | 67 / 104 |
| paysim | Consumed synthetic development | 888 / 84 | 848 / 79 | 825 / 76 | 678 / 71 |
| sparkov | Consumed synthetic development | 253 / 67 | 184 / 75 | 246 / 92 | 228 / 103 |
| elliptic | Real AML transfer, known labels | 1 / 3 | 167 / 12 | 166 / 8 | 164 / 11 |

The synthetic datasets have informed development. Elliptic is a separately labelled real Bitcoin AML transfer test; it does not replace independent payment-fraud validation. Unknown Elliptic labels and explicitly backward/forward-neighbour features are excluded.

Four configurations and a 100-tree cap were allocated to each baseline family. V6 numerical settings were frozen. Repeats use the same cohorts with different model seeds; paired cluster intervals condition on fitted models and fixed cutoffs. Small searches, coarse time clusters and dependence limit the conclusions. No pooled accuracy, automatic winner or established novelty claim is reported.

## For whoever builds this

Each dataset folder contains its hosted-report/review.md with primary counts, repeat ranges, paired cluster checks and limitations. The full records are in kaggle-output/validation_results. The plan is validation-study/prespecified-plan.json; the combined result file is benchmark-results.csv. IEEE-CIS access remains unconfirmed after a 403 download response; no competition rules were accepted automatically.

Decision still needed: the research lead must decide whether these results justify a manuscript claim after a fuller novelty review and independent payment-fraud evidence.
