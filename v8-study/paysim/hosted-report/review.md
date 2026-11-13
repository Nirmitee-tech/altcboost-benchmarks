# Tail-continuation experiment: paysim

These are development results. Validation selects prefixes and compares methods on sources already used in development. They do not establish independent confirmation or novelty.

| Method | Mean strict recall | Caught at 0.1% | False positives at 0.1% | Selected prefix | Tree proposals | Objective calls | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V8_target | 72.53% | 897 | 93 | 40 | 40 | 121 | 33.56 |
| V8_all_negative | 67.74% | 863 | 87 | 40 | 40 | 121 | 42.99 |
| V8_geometric | 71.90% | 874 | 93 | 40 | 40 | 121 | 34.07 |
| V8_linear | 70.09% | 855 | 93 | 40 | 40 | 121 | 37.41 |
| V8_adaptive_blind | 72.53% | 897 | 93 | 40 | 40 | 121 | 32.65 |
| V8_adaptive_checked | 72.53% | 897 | 93 | 40 | 40 | 121 | 33.50 |
| CatBoost_native | 54.19% | 671 | 90 | 40 | N/A | N/A | 1.73 |
| XGBoost | 72.48% | 888 | 91 | 17 | N/A | N/A | 0.53 |
| LightGBM_native | 55.37% | 735 | 79 | 1 | N/A | N/A | 0.44 |

Mean strict recall averages recall at empirical validation false-positive budgets of 0.05% and 0.1%. Thresholds use validation negative labels, so these counts do not measure future false-positive control.

## For whoever builds this

Each accepted step decreases its current working-tail objective. The fixed target-tail loss may increase when training uses a relaxed fraction.
Previously selected baseline hyperparameters, refitted with the same tree cap; validation reselects a prefix within that cap. Earlier hyperparameter tuning gives baselines additional search effort.
Adaptive validation-only development on consumed sources. No independent confirmation or established novelty. Shared budget counts all fitted tree proposals, including rejected ones. Numerical directional signs are not interval-arithmetic certificates. Nonnegative exact directional derivative rules out descent only along the proposed tree, not all possible trees. Different working fractions define different objectives; no global target-loss monotonicity or future FPR guarantee. Loss evaluation counts are implementation-specific and vary in cost with tail fraction.

Decision still needed: the research lead must determine whether continuation improves beyond fixed and clock schedules. A separate confirmation source and comparison with the closest published optimizers are required before a publication claim.
