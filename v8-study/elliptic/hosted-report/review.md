# Tail-continuation experiment: elliptic

These are development results. Validation selects prefixes and compares methods on sources already used in development. They do not establish independent confirmation or novelty.

| Method | Mean strict recall | Caught at 0.1% | False positives at 0.1% | Selected prefix | Tree proposals | Objective calls | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V8_target | 82.51% | 849 | 2 | 4 | 40 | 121 | 38.69 |
| V8_all_negative | 83.62% | 865 | 4 | 40 | 40 | 121 | 74.64 |
| V8_geometric | 78.86% | 814 | 4 | 40 | 40 | 121 | 47.45 |
| V8_linear | 61.27% | 661 | 4 | 36 | 40 | 121 | 56.50 |
| V8_adaptive_blind | 82.51% | 849 | 2 | 4 | 40 | 121 | 37.76 |
| V8_adaptive_checked | 82.51% | 849 | 2 | 4 | 40 | 121 | 37.89 |
| CatBoost_native | 51.51% | 535 | 4 | 32 | N/A | N/A | 0.83 |
| XGBoost | 83.58% | 868 | 4 | 39 | N/A | N/A | 0.52 |
| LightGBM_native | 39.12% | 471 | 4 | 2 | N/A | N/A | 0.79 |

Mean strict recall averages recall at empirical validation false-positive budgets of 0.05% and 0.1%. Thresholds use validation negative labels, so these counts do not measure future false-positive control.

## For whoever builds this

Each accepted step decreases its current working-tail objective. The fixed target-tail loss may increase when training uses a relaxed fraction.
Previously selected baseline hyperparameters, refitted with the same tree cap; validation reselects a prefix within that cap. Earlier hyperparameter tuning gives baselines additional search effort.
Adaptive validation-only development on consumed sources. No independent confirmation or established novelty. Shared budget counts all fitted tree proposals, including rejected ones. Numerical directional signs are not interval-arithmetic certificates. Nonnegative exact directional derivative rules out descent only along the proposed tree, not all possible trees. Different working fractions define different objectives; no global target-loss monotonicity or future FPR guarantee. Loss evaluation counts are implementation-specific and vary in cost with tail fraction.

Decision still needed: the research lead must determine whether continuation improves beyond fixed and clock schedules. A separate confirmation source and comparison with the closest published optimizers are required before a publication claim.
