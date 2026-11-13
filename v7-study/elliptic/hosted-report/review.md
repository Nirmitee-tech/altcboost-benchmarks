# Prior-revision experiment: elliptic

This is a training/validation mechanism experiment. The same validation labels select model prefixes and compare the hypotheses. These scores are development evidence and do not establish independent confirmation or novelty.

| Method | Mean recall at two strict validation budgets | Caught at 0.1% | False positives at 0.1% | Selected prior weight | Selected prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| V6_fixed | 0.05% | 1 | 4 | 1.0000 | 78 |
| V7_zero | 82.51% | 849 | 2 | 0.0000 | 4 |
| V7_clipped | 66.33% | 692 | 4 | 1.0000 | 54 |
| V7_shrunk | 16.72% | 343 | 4 | 0.2500 | 99 |
| V7_adaptive_diagonal | 3.30% | 37 | 4 | 0.0171 | 3 |
| V7_adaptive_joint | 68.90% | 721 | 4 | 0.0097 | 5 |
| CatBoost_native | 54.37% | 573 | 4 | N/A | 97 |
| XGBoost | 84.16% | 874 | 4 | N/A | 48 |
| LightGBM_native | 75.90% | 852 | 4 | N/A | 99 |

The primary mechanism question is whether the coupled update improves over zero, clipping, fixed shrinkage and diagonal-block controls. A simpler control matching or beating it weakens the claim for the extra mechanism. The 0.05% and 1% validation results remain in validation.csv.

## For whoever builds this

Adaptive development experiment. Fit and score training/validation only; no calibration or final prediction is computed.
Validation chooses prefixes and is also used to compare these hypotheses; scores are optimistic development evidence. Existing datasets informed the redesign. No novelty, general superiority, independent confirmation, future FPR control or global optimizer claim. Zero-start can encounter nonsmooth tail ties. Fully corrective boosting and variable-projection methods are related prior work.

The prior weight is constrained to [0,1]. Every accepted update is checked against the actual empirical training loss. Exact directional curvature concerns a two-coordinate restriction away from tail membership ties; the tree structure still uses a diagonal Newton approximation. Neither this check nor a validation gain proves future low-FPR performance.

Decision still needed: the research lead must assess whether the coupled update earns its complexity over the controls and the closest published methods. A revised method needs a new confirmation protocol before a publication claim.
