# Tail-continuation experiment: sparkov

These are development results. Validation selects prefixes and compares methods on sources already used in development. They do not establish independent confirmation or novelty.

| Method | Mean strict recall | Caught at 0.1% | False positives at 0.1% | Selected prefix | Tree proposals | Objective calls | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V8_target | 71.82% | 494 | 99 | 40 | 40 | 121 | 25.46 |
| V8_all_negative | 61.72% | 420 | 61 | 23 | 40 | 121 | 143.52 |
| V8_geometric | 70.37% | 480 | 98 | 39 | 40 | 121 | 54.95 |
| V8_linear | 66.00% | 464 | 98 | 40 | 40 | 121 | 88.10 |
| V8_adaptive_blind | 71.82% | 494 | 99 | 40 | 40 | 121 | 24.65 |
| V8_adaptive_checked | 71.82% | 494 | 99 | 40 | 40 | 121 | 25.36 |
| CatBoost_native | 38.67% | 282 | 99 | 40 | N/A | N/A | 1.77 |
| XGBoost | 67.61% | 468 | 99 | 40 | N/A | N/A | 0.65 |
| LightGBM_native | 62.17% | 425 | 99 | 40 | N/A | N/A | 0.61 |

Mean strict recall averages recall at empirical validation false-positive budgets of 0.05% and 0.1%. Thresholds use validation negative labels, so these counts do not measure future false-positive control.

## For whoever builds this

Each accepted step decreases its current working-tail objective. The fixed target-tail loss may increase when training uses a relaxed fraction.
Previously selected baseline hyperparameters, refitted with the same tree cap; validation reselects a prefix within that cap. Earlier hyperparameter tuning gives baselines additional search effort.
Adaptive validation-only development on consumed sources. No independent confirmation or established novelty. Shared budget counts all fitted tree proposals, including rejected ones. Numerical directional signs are not interval-arithmetic certificates. Nonnegative exact directional derivative rules out descent only along the proposed tree, not all possible trees. Different working fractions define different objectives; no global target-loss monotonicity or future FPR guarantee. Loss evaluation counts are implementation-specific and vary in cost with tail fraction.

Decision still needed: the research lead must determine whether continuation improves beyond fixed and clock schedules. A separate confirmation source and comparison with the closest published optimizers are required before a publication claim.
