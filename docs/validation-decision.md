# Keep the candidate experimental

Do not submit the current candidate as a generally superior fraud-detection algorithm. The completed validation study preserves useful synthetic results, but its first real Bitcoin AML transfer test showed a large loss of recall against the tuned references. The next research cost should be controlled diagnosis on training and validation data, followed by confirmation on an untouched source.

The results below were measured on Kaggle on 12 November 2026. They use the prespecified primary seed and the 0.1% calibration false-positive target. Each cell gives positive cases caught / false positives. The first three sources concern synthetic fraud; the fourth concerns illicit Bitcoin activity among known labels. The counting populations are separate.

| Source | Frozen candidate | Native CatBoost | XGBoost | Native LightGBM |
| --- | ---: | ---: | ---: | ---: |
| BAF | 45 / 63 | 76 / 103 | 70 / 91 | 67 / 104 |
| PaySim | 888 / 84 | 848 / 79 | 825 / 76 | 678 / 71 |
| Sparkov | 253 / 67 | 184 / 75 | 246 / 92 | 228 / 103 |
| Elliptic AML transfer | 1 / 3 | 167 / 12 | 166 / 8 | 164 / 11 |

The real transfer cohort contains 408 labelled illicit transactions. Catching one of them is a poor outcome even though the candidate raises fewer false alarms at this operating point. At the 1% calibration target, the candidate catches 77 with 246 false positives, while XGBoost catches 192 with 107. The failure therefore extends beyond a single conservative operating point. The candidate's real-data result repeated across three fitting seeds, on the same data cohort.

The synthetic results require a narrower reading. PaySim supports higher recall with some additional false positives against the stronger references. Its paired temporal checks support a recall difference in this sample, while the false-positive differences against CatBoost and XGBoost remain uncertain under that conditional bootstrap. Sparkov has a favourable primary count comparison, but its recall-difference interval against XGBoost includes zero. The false-positive-difference intervals are negative under the two separately applied clustering schemes. These are conditional checks with no multiple-comparison correction, not a general superiority result.

BAF also remains weak. Validation selected only the candidate's starting linear model, so its reported performance supplies no evidence for the proposed residual-tree mechanism. The earlier synthetic success should not be generalized to all financial classification tasks.

The comparison is stronger than the preceding one: each baseline family received the same number of prespecified configurations and a common tree cap, native category handling was retained for two families, selection used validation data, and the selected pipelines were refit with additional seeds. Training time and capacity were not matched. This remains a small search rather than exhaustive optimization.

The real source is a known-label AML subset. Unknown labels were excluded, and the explicitly backward/forward-neighbour features were omitted. Anonymous local-feature semantics and provider normalization remain a limitation. This test does not establish payment-fraud performance or production readiness. It does supply evidence against the current broad claim that the frozen candidate transfers well beyond the synthetic development sources.

One proposed explanation was checked rather than accepted: a large starting linear score might exceed what the bounded tree corrections can repair. The local training/validation diagnostic found extreme starting scores, but its conservative recall ceiling did not restrict attainable validation recall at the declared targets. The hard correction limit alone therefore does not establish the cause of this failure. Other effects, including restricted tree partitions, prior extrapolation, objective choice, numerical conditioning and time shift, remain hypotheses.

A useful next experiment would isolate the starting model's influence with a constant start, the current linear start and a bounded or damped linear start. That comparison must be selected using training and validation data. It is a proposed diagnostic experiment, not a measured fix. The real transfer interval is now consumed evidence and must not be reused as independent confirmation after changing the method. No algorithm settings were changed in response to these final results.

An empirical report describing these conditions and failures may eventually be useful. A claim of a unique new algorithm or general superiority is not supported by the current record.

## For whoever builds this

Skip this section unless you are implementing or reviewing the work.

The plan is `validation-study/prespecified-plan.json`. Primary and repeat results are in `validation-study/benchmark-results.csv` and the per-source `kaggle-output/validation_results/results.csv` files. Individual reviews contain prefix selections, seed ranges and paired uncertainty checks. The combined review is `validation-study/results-review.md`. Source, cohort, frozen-code and selection-record checks are saved in `validation-study/verification.json` and per-source `submission-verification.json` files.

The frozen V6 source files match the preceding candidate study, and its primary synthetic counts reproduce at every declared calibration target. Each baseline family used four configurations, a 100-tree cap and validation-only prefix selection. Seed 17 is primary; seeds 42 and 73 keep the selected hyperparameters and reselect prefixes on validation. These seeds share the same source cohorts and are not independent datasets.

The real source is `ellipticco/elliptic-data-set`. Its [original paper](https://arxiv.org/abs/1908.02591) defines the AML context. The feature allowlist is the 93 local numeric positions excluding ID and time. Only published classes 1 and 2 enter evaluation. BAF's two final months and Elliptic's eight final time steps fall below the prespecified twenty-cluster screen, so paired cluster intervals are unsupported there. Ordinary row-based intervals assume independent rows and do not resolve this gap.

The additional diagnostic is `experiment/prior_diagnostic.py`, with aggregate output in `validation-study/elliptic-prior-diagnostic.json`. It reconstructs the starting model locally from training data and scores validation only. Source discovery still reads full files for audits. Local library differences mean this is not a bit-for-bit reconstruction of the hosted model.

For prior margins b and any residual bounded by absolute magnitude C, let q be the (floor(alpha times negative count)+1)-th largest negative prior margin. Every positive with b at most q-2C is unrecoverable at that empirical false-positive budget: at least one of those leading negatives must stay below the decision cutoff, while that positive cannot outrank the lowest possible score of all those negatives. The resulting fraction of potentially recoverable positives is an upper bound, not an attainable-recall guarantee. For V6, 100 times 0.08 times 5 gives a conservative C of 40; selected prefixes and backtracking can lower the actual bound. In this local validation diagnostic the ceiling is 1.0 at all three targets, so this necessary condition does not explain the observed failure. The diagnostic is tested against exhaustive bounded corrections on small fixtures. No novel generalization theorem is claimed.

The uncertainty implementation resamples shared clusters at fixed fitted scores and calibration cutoffs. PaySim uses four-step temporal blocks. Sparkov uses calendar weeks and card IDs in separate checks. It does not jointly model both grouping structures or arbitrary temporal dependence. The intervals omit training, tuning and calibration uncertainty and do not adjust for multiple comparisons.

The IEEE-CIS download was retried and returned 403. Its independent payment-fraud validation remains unavailable in this account. No rules were accepted automatically and no mirror was used to bypass access restrictions.

Remaining decisions: the maintainer must decide whether to pursue a narrow empirical paper or continue algorithm development. The research lead must diagnose the real-data failure, perform the closest-method comparison and specify a fresh confirmation protocol. The account owner must enable an independent payment-fraud source if that benchmark is to be used.
