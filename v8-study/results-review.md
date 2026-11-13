# Tail-continuation research results

Validation both selects prefixes and compares hypotheses on consumed sources. All predictions in this study are training or validation predictions. The table is development evidence.

| Source | Strict target | All negatives | Geometric schedule | Linear schedule | Adaptive checked | Highest tested reference |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| elliptic | 82.51% | 83.62% | 78.86% | 61.27% | 82.51% | 83.58% (XGBoost) |
| paysim | 72.53% | 67.74% | 71.90% | 70.09% | 72.53% | 72.48% (XGBoost) |
| sparkov | 71.82% | 61.72% | 70.37% | 66.00% | 71.82% | 67.61% (XGBoost) |
| baf_base | 0.00% | 4.78% | 4.38% | 5.36% | 2.30% | 5.67% (CatBoost_native) |

- elliptic: blind and checked accepted target-loss histories and selected prefixes match: True. Objective calls: 121 versus 121. Fit seconds: 37.76 versus 37.89. Counts describe this implementation and do not establish a general speedup.
- paysim: blind and checked accepted target-loss histories and selected prefixes match: True. Objective calls: 121 versus 121. Fit seconds: 32.65 versus 33.50. Counts describe this implementation and do not establish a general speedup.
- sparkov: blind and checked accepted target-loss histories and selected prefixes match: True. Objective calls: 121 versus 121. Fit seconds: 24.65 versus 25.36. Counts describe this implementation and do not establish a general speedup.
- baf_base: blind and checked accepted target-loss histories and selected prefixes match: True. Objective calls: 296 versus 236. Fit seconds: 113.91 versus 105.07. Counts describe this implementation and do not establish a general speedup.

## For whoever builds this

The plan is v8-study/prespecified-plan.json. Each policy has 40 total fitted-tree proposals, including rejected proposals. Baselines use previous hyperparameters and a 40-tree cap, with prefixes reselected on validation. This does not match wall time or earlier tuning effort.
Each accepted step decreases the current working-tail loss. Changing the working fraction can increase the fixed target-tail loss. The directional check rules out improvement only along a particular proposed tree, using floating-point arithmetic.

Decision still needed: the research lead must accept or withdraw the claimed benefit against the strongest simple control, and specify a fresh confirmation source before freezing a publication candidate.
