# Measured detection at the 0.1% validation false-positive budget

The current adaptive method does not yet meet a general accuracy or publication claim. The table recomputes descriptive performance from saved validation confusion counts. These sources already informed development.

| Source | Method | Caught / fraud cases | Missed | False alerts / legitimate cases | Recall | Precision | Overall accuracy | All-legitimate accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Elliptic | Adaptive candidate | 849 / 1029 | 180 | 2 / 4148 | 82.51% | 99.76% | 96.484% | 80.124% |
| Elliptic | XGBoost | 868 / 1029 | 161 | 4 / 4148 | 84.35% | 99.54% | 96.813% | 80.124% |
| PaySim | Adaptive candidate | 897 / 1192 | 295 | 93 / 93398 | 75.25% | 90.61% | 99.590% | 98.740% |
| PaySim | XGBoost | 888 / 1192 | 304 | 91 / 93398 | 74.50% | 90.70% | 99.582% | 98.740% |
| Sparkov | Adaptive candidate | 494 / 653 | 159 | 99 / 99347 | 75.65% | 83.31% | 99.742% | 99.347% |
| Sparkov | XGBoost | 468 / 653 | 185 | 99 / 99347 | 71.67% | 82.54% | 99.716% | 99.347% |
| BAF Base | Adaptive candidate | 35 / 1129 | 1094 | 98 / 98871 | 3.10% | 26.32% | 98.808% | 98.871% |
| BAF Base | CatBoost | 77 / 1129 | 1052 | 98 / 98871 | 6.82% | 44.00% | 98.850% | 98.871% |

A model that labels every case legitimate catches no fraud. Its accuracy is included to show why overall accuracy cannot carry the fraud-detection claim. Recall measures the share of fraud caught; precision measures the share of alerts that are fraud. Elliptic is a known-label AML transfer task. The other three sources are synthetic.

## For whoever builds this

Reproduce with experiment/accuracy_audit.py. all-operating-points.csv includes every method and all three existing budgets. candidate-and-reference.csv keeps the candidate and the reference chosen using the original selection key. manifest.json contains observation time, source paths, hashes, units and scope.

Validation selects prefixes and thresholds and has informed model redesign. No independent confirmation.
Empirical validation FPR is not a future FPR guarantee. No selection-adjusted or time-dependence-adjusted confidence intervals are inferred from counts.
Aggregate counts cannot reconstruct paired errors, score calibration, ROC-AUC or new threshold choices.
Dataset-specific acceptable recall and precision require an application cost target. No universal accuracy pass threshold is inferred.

Decisions still needed: the research lead must choose and freeze a candidate before confirmation; the application owner must set minimum useful recall and precision at the agreed false-positive budget. Those targets must be fixed before looking at fresh confirmation outcomes.
