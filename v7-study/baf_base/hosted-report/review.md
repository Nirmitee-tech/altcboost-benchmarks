# Prior-revision experiment: baf_base

This is a training/validation mechanism experiment. The same validation labels select model prefixes and compare the hypotheses. These scores are development evidence and do not establish independent confirmation or novelty.

| Method | Mean recall at two strict validation budgets | Caught at 0.1% | False positives at 0.1% | Selected prior weight | Selected prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| V6_fixed | 3.90% | 58 | 98 | 1.0000 | 0 |
| V7_zero | 0.00% | 0 | 0 | 0.0000 | 0 |
| V7_clipped | 3.90% | 58 | 98 | 1.0000 | 0 |
| V7_shrunk | 3.90% | 58 | 98 | 0.2500 | 0 |
| V7_adaptive_diagonal | 3.90% | 58 | 98 | 1.0000 | 0 |
| V7_adaptive_joint | 3.90% | 58 | 98 | 1.0000 | 0 |
| CatBoost_native | 5.85% | 77 | 98 | N/A | 65 |
| XGBoost | 5.31% | 77 | 98 | N/A | 70 |
| LightGBM_native | 4.30% | 62 | 98 | N/A | 87 |

The primary mechanism question is whether the coupled update improves over zero, clipping, fixed shrinkage and diagonal-block controls. A simpler control matching or beating it weakens the claim for the extra mechanism. The 0.05% and 1% validation results remain in validation.csv.

## For whoever builds this

Adaptive development experiment. Fit and score training/validation only; no calibration or final prediction is computed.
Validation chooses prefixes and is also used to compare these hypotheses; scores are optimistic development evidence. Existing datasets informed the redesign. No novelty, general superiority, independent confirmation, future FPR control or global optimizer claim. Zero-start can encounter nonsmooth tail ties. Fully corrective boosting and variable-projection methods are related prior work.

The prior weight is constrained to [0,1]. Every accepted update is checked against the actual empirical training loss. Exact directional curvature concerns a two-coordinate restriction away from tail membership ties; the tree structure still uses a diagonal Newton approximation. Neither this check nor a validation gain proves future low-FPR performance.

Decision still needed: the research lead must assess whether the coupled update earns its complexity over the controls and the closest published methods. A revised method needs a new confirmation protocol before a publication claim.
