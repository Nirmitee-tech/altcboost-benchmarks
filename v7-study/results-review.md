# Prior-revision research results

These are adaptive training/validation experiments. Validation both selects model prefixes and compares the hypotheses. No calibration or final predictions were computed. Existing sources informed this redesign, so these results are not independent confirmation.

The table reports mean recall across empirical validation false-positive budgets of 0.05% and 0.1%. The full three-component validation key breaks ties by false-positive rate and average precision.

| Source | Fixed V6 | Best static control | Diagonal adaptive update | Coupled adaptive update | Highest tested reference |
| --- | ---: | --- | ---: | ---: | --- |
| elliptic | 0.05% | 82.51% (V7_zero) | 3.30% | 68.90% | 84.16% (XGBoost) |
| paysim | 71.60% | 75.00% (V7_zero) | 74.54% | 74.71% | 73.99% (XGBoost) |
| sparkov | 69.37% | 73.20% (V7_clipped) | 72.21% | 70.52% | 68.53% (XGBoost) |
| baf_base | 3.90% | 3.90% (V6_fixed) | 3.90% | 3.90% | 5.85% (CatBoost_native) |

The static controls include zero start, a smooth bounded prior and fixed shrinkage. A positive result against fixed V6 alone cannot establish that the extra optimizer is needed.

- elliptic: coupled-update mean recall differs by -13.61 percentage points from the strongest static control and +65.60 percentage points from the diagonal adaptive control. These are validation differences, not significance claims.
- paysim: coupled-update mean recall differs by -0.29 percentage points from the strongest static control and +0.17 percentage points from the diagonal adaptive control. These are validation differences, not significance claims.
- sparkov: coupled-update mean recall differs by -2.68 percentage points from the strongest static control and -1.68 percentage points from the diagonal adaptive control. These are validation differences, not significance claims.
- baf_base: coupled-update mean recall differs by +0.00 percentage points from the strongest static control and +0.00 percentage points from the diagonal adaptive control. These are validation differences, not significance claims.

## For whoever builds this

The plan is v7-study/prespecified-plan.json. Individual hosted reports include the validation counts, selected prior coefficients and prefixes. Raw row data and future prediction scores are not exported. The preceding Elliptic final-test failure remains unchanged.

Related-work overlap includes fully corrective partial-AUC ensembles and variable-projection boosting. Exact small-block curvature, coefficient correction and empirical descent are not individually claimed as new inventions. A publishable novelty argument requires direct comparison with the closest algorithms and fresh confirmation after model selection.

Decision still needed: the research lead must decide whether the coupled update earns its complexity over the controls. the maintainer must choose whether to continue this mechanism or withdraw it based on the complete evidence.
