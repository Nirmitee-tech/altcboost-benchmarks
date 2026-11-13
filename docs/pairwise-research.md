# Revised temporal pairwise candidate

Keep this revised loss as an experimental research branch. The completed synthetic development study shows a useful improvement on PaySim and operating-point tradeoffs on Sparkov. BAF remains weak: validation retained only the starting linear model and no residual trees. The method spends more arithmetic on direct fraud-versus-normal comparisons. A broader detection benefit and the novelty of the combined method remain open questions.

The first temporal-tail candidate did not earn its complexity on the completed PaySim and Sparkov development runs. At the 0.1% calibration target it caught fewer fraud cases than the prior ALTCBoost. Removing the negative-tail restriction barely changed those results. Its numerical checks still passed: a correct derivative and a decreasing training loss did not establish useful detection performance. Keep that experiment as a negative result rather than describing it as an improvement.

The revision uses a direct average of logistic losses comparing fraud with the highest-scoring normal transactions. It avoids the first candidate's exponential weighting over positive examples. Historical time groups with larger losses still receive more influence, and the trainer still checks the actual empirical loss before accepting an update. This changes one central objective instead of adding unrelated model components.

A training-and-validation-only PaySim screen justified measuring the revised candidate further. It improved the prior ALTCBoost's validation selection score, remained close to a balanced-logistic control and did not beat XGBoost on that selection score. No final interval was scored in that screen. This was adaptive development, and it does not restore the independence of previously consumed datasets.

The fixed experiment compared the full revision, a pooled-time version, an all-negative version and the balanced-logistic control against the same frozen reference models. All three false-positive targets and every regression are retained in the report. A higher recall with more false positives is a tradeoff. The complete candidate needs to earn its extra complexity over its controls; an improvement from class balancing alone cannot be attributed to temporal tail ranking. PaySim supports further investigation at the 0.1% calibration target. Sparkov does not show a consistent benefit from adding temporal grouping over the pooled-time control. BAF's selected linear-only model provides no evidence for the proposed tree-training mechanism.

Direct pairwise ranking, robust partial-AUC objectives and worst-group training all have prior work. Closest foundations include [When AUC meets DRO](https://proceedings.mlr.press/v162/zhu22g.html), [group DRO and regularization](https://arxiv.org/abs/1911.08731), and [pAUCBoost](https://link.springer.com/article/10.1186/1471-2105-11-314). The proposed research concerns this particular tree-training formulation, its checked empirical descent and its practical detection tradeoffs. The preliminary search cannot establish that the combination is unique. It would be inaccurate to call any of those established ingredients our invention.

## For whoever builds this

Skip this section unless you are implementing or reviewing the method.

The implementation is `experiment/altcboost_v6.py`, which inherits the checked-descent trainer, and `pairwise_tail_objective` in `experiment/tail_objective.py`. The fixed runner is `experiment/pairwise_experiment.py`. Its plan and outputs are under `pairwise-study/`. The validation-only screen is reproducible with `experiment/pairwise_pilot.py`; its saved output is `novelty-study/pairwise-paysim-validation.json`. The first candidate and its withdrawn claims remain under `novelty-study/` and in `docs/temporal-tail-research.md`.

For each training time group g, let P_g be its positive rows and N_g be the k_g = ceil(beta times the group's negative count) highest-scoring negative rows. The revised group loss is

```
l_g = mean over i in N_g and j in P_g:
      softplus((f_i - f_j + m) / T)
R = tau * log(mean(exp(l_g / tau), g))
L = (1 - eta) * R + eta * L_balanced_logistic
```

The full candidate uses beta=0.01, T=1, m=1, tau=0.25 and eta=0.2. Four training groups are defined by training timestamps without labels. Each group must contain both classes. The tree cap is 100, the leaf cap is 17, the learning rate is 0.08 and the seed is 17. The empirical training tail fraction is separate from the calibration false-positive target.

The class-balanced linear starting model is fitted once, followed by additive residual trees. The time grouping and pair comparisons occur during training. Serving evaluates the linear model and the selected trees, without needing a time-group label. This preserves the form of the inference path; it does not establish equal measured latency. There are no alternating linear refits or explicit cross-feature updates in this branch. The inherited project name does not establish those architectural features.

The earlier scorer speed measurement belongs to a preceding configuration. It must not be combined with this branch's detection counts as though they describe the same model. Inference timing for this revised candidate has not been measured in the current experiment.

For a selected positive-negative pair, p_ij = sigmoid((f_i-f_j+m)/T). The negative-row gradient sums p_ij/(pair_count*T) over its positive partners; the positive-row gradient is the negative sum over its negative partners. The diagonal curvature sums p_ij(1-p_ij)/(pair_count*T*T) over the corresponding partners. The outer group mixture uses the same derivative and diagonal-curvature composition as the first candidate. The trainer uses a diagonal approximation to the Hessian; it does not use the full cross-row Hessian.

For any fixed negative subset, the average pairwise loss is convex in the margin vector. Every positive row orders the negative losses in the same way as the negative margins, so selecting the highest-scoring negative subset selects the largest subset loss. The group loss is therefore a maximum of convex functions and is convex. The robust outer log-sum-exp and logistic anchor preserve convexity. At tail membership ties it can be nonsmooth. This concerns optimization over margins, not global optimization over tree structures. Backtracking checks descent on the actual empirical objective, not on future error rates.

Pair evaluation is chunked over positives to bound temporary memory. It still performs work proportional to the number of selected positive-negative pairs. The all-negative control can be substantially more expensive. No linear-time training, novel convergence theorem, calibrated probability or universal robustness claim is made.

Run the fixed study with `.venv/bin/python experiment/pairwise_experiment.py DATASET data OUTPUT_DIR` from the project root. Dataset keys are `baf_base`, `paysim` and `sparkov`. The shared preprocessing is fitted on training rows only. Source and predictor allowlists exclude IDs and specified post-event fields. Exports contain aggregate results, group diagnostics and configuration evidence. The later-period checks use one global calibration threshold and no more windows than actual timestamps support. BAF has two final months; its failed first request for four windows was corrected before that candidate trained.

Unit checks cover finite-difference derivatives, chunk equivalence and agreement with an explicit small pair matrix. Integration checks cover loss descent, the validation-selected model, the export workflow and a poisoned calibration/test fixture proving that the screen does not fit or score those frames after cohort loading. Source discovery still reads full files to audit chronology and source counts. Passing those checks establishes the tested implementation properties, not detection superiority.

Remaining decisions: the maintainer must choose the acceptable recall and false-alarm tradeoff. The research lead must decide whether the full revision survives its simpler controls and complete the novelty comparison. The data owner must enable an untouched real-data evaluation before independent confirmation can be claimed.
