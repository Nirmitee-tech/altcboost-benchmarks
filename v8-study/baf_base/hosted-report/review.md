# Tail-continuation experiment: baf_base

These are development results. Validation selects prefixes and compares methods on sources already used in development. They do not establish independent confirmation or novelty.

| Method | Mean strict recall | Caught at 0.1% | False positives at 0.1% | Selected prefix | Tree proposals | Objective calls | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V8_target | 0.00% | 0 | 0 | 0 | 40 | 41 | 98.66 |
| V8_all_negative | 4.78% | 61 | 98 | 27 | 40 | 121 | 316.46 |
| V8_geometric | 4.38% | 60 | 98 | 23 | 40 | 121 | 153.39 |
| V8_linear | 5.36% | 68 | 98 | 40 | 40 | 121 | 217.29 |
| V8_adaptive_blind | 2.30% | 35 | 98 | 35 | 40 | 296 | 113.91 |
| V8_adaptive_checked | 2.30% | 35 | 98 | 35 | 40 | 236 | 105.07 |
| CatBoost_native | 5.67% | 77 | 98 | 40 | N/A | N/A | 4.34 |
| XGBoost | 5.09% | 68 | 98 | 30 | N/A | N/A | 1.76 |
| LightGBM_native | 3.85% | 52 | 98 | 39 | N/A | N/A | 1.47 |

Mean strict recall averages recall at empirical validation false-positive budgets of 0.05% and 0.1%. Thresholds use validation negative labels, so these counts do not measure future false-positive control.

## For whoever builds this

Each accepted step decreases its current working-tail objective. The fixed target-tail loss may increase when training uses a relaxed fraction.
Previously selected baseline hyperparameters, refitted with the same tree cap; validation reselects a prefix within that cap. Earlier hyperparameter tuning gives baselines additional search effort.
Adaptive validation-only development on consumed sources. No independent confirmation or established novelty. Shared budget counts all fitted tree proposals, including rejected ones. Numerical directional signs are not interval-arithmetic certificates. Nonnegative exact directional derivative rules out descent only along the proposed tree, not all possible trees. Different working fractions define different objectives; no global target-loss monotonicity or future FPR guarantee. Loss evaluation counts are implementation-specific and vary in cost with tail fraction.

Decision still needed: the research lead must determine whether continuation improves beyond fixed and clock schedules. A separate confirmation source and comparison with the closest published optimizers are required before a publication claim.
