# A temporal-tail candidate for ALTCBoost

Keep this as an experimental branch until the removal experiments show a useful detection tradeoff. It adds training cost for an explicit loss and repeated loss checks. Its potential contribution is a precise way to train small tree ensembles against difficult normal transactions across historical time periods. Novelty and improved detection remain unproven.

The practical question is whether a detector can retain fraud recall under a strict false-alarm budget when the transaction mix changes. The earlier PaySim experiment exposed a gap: the current method caught fewer fraud cases with more false alarms than the tested tree baselines at the 0.1% calibration target. That observation motivates this candidate. It does not establish why the method failed.

The candidate makes three connected changes. First, it ranks fraud above the highest-scoring normal transactions through an explicit tail loss. Second, it gives more influence to training time groups with larger losses, so a large easy period cannot entirely hide a difficult one. Third, it accepts a tree update only after checking that the stated training loss decreased. A class-balanced logistic component keeps information from the remaining normal transactions.

The descent check concerns the empirical training objective. It does not guarantee future recall, a future false-positive rate, convergence of the tree learner, or a globally optimal model. The tail fraction used for training is a separate parameter from the operating threshold selected on calibration data. The score interface does not establish calibrated fraud probabilities.

This proposal is grounded in existing research. [When AUC meets DRO](https://proceedings.mlr.press/v162/zhu22g.html), published in 2022, already connects partial-AUC optimization with robust objectives. [Sagawa and colleagues](https://arxiv.org/abs/1911.08731) study worst-group training and the importance of regularization. [pAUCBoost](https://link.springer.com/article/10.1186/1471-2105-11-314) establishes that low-false-positive boosting itself is an existing direction. This preliminary search rules out describing those ingredients individually as new. It does not settle whether this concrete surrogate, temporal grouping and checked-descent tree implementation are distinct enough to constitute a contribution. A fuller related-work comparison is still required.

The experiment is fixed before measuring this branch. It compares the complete candidate with a pooled-time version, an all-negative version and a class-balanced logistic-loss control. The prior ALTCBoost and the frozen CatBoost and XGBoost configurations remain reference models. All use the same predictor allowlists and cohorts. This isolates useful differences within the candidate more clearly than a single comparison against a library model.

The three synthetic datasets are now development evidence because their previous outcomes motivated this work. Results must be reported at every declared calibration target, including regressions. One global calibration threshold is also applied across up to four later time windows, subject to the source timestamp resolution. A smaller worst-window false-positive rate accompanied by much lower fraud recall is a tradeoff. We will need untouched real-data confirmation and stronger baseline tuning before a publication claim.

The first BAF evaluation request for four test windows was withdrawn. Its final interval has two distinct months, and the job stopped before model training. The corrected evaluation reports those two observed windows. An initial explanation incorrectly said one month; the saved source audit corrected that statement. No row-order split is used to manufacture finer time resolution.

The proposed contribution should be withdrawn if the complete candidate consistently loses to its simpler controls. If it helps only one data regime, the paper must describe and test that regime. Adding more components solely to create an appearance of novelty is outside this proposal.

## For whoever builds this

Skip this section unless you are implementing or reviewing the method.

The implementation is `experiment/tail_objective.py` and `experiment/altcboost_v5.py`. The experiment is `experiment/tail_experiment.py`. The frozen configuration is `novelty-study/prespecified-plan.json`. Full source and cohort evidence remain in the saved discovery files. Regenerate the aggregate training audit with `.venv/bin/python experiment/tail_discovery.py data OUTPUT.json` from the project root. Time groups are defined from training timestamps without labels. A group lacking either class causes the ranking experiment to stop rather than silently changing the grouping.

For group g, let P_g contain positive rows and N_g contain the k_g = ceil(beta times number of negatives) highest-scoring negatives. Define

```
b_g = log(mean(exp(f_i / T), i in N_g))
    + log(mean(exp(-f_j / T), j in P_g)) + m / T
l_g = softplus(b_g)
R = tau * log(mean(exp(l_g / tau), g))
L = (1 - eta) * R + eta * L_balanced_logistic
```

The fixed full candidate uses beta=0.01, T=1, m=1, tau=0.25 and eta=0.2. There are four training time groups. The tree cap is 100, the leaf cap is 17, the learning rate is 0.08 and the seed is 17. All removal experiments share the same class-balanced linear starting model, tree limits and stopping rule. Selection uses validation labels only. No search is performed using final interval outcomes.

Away from a negative-tail membership tie, let d_g = sigmoid(b_g), q_i be the within-class exponential weight and s_i be +1 for a negative and -1 for a positive. Then the per-group gradient is s_i d_g q_i / T and its diagonal curvature is d_g q_i (1 - d_g q_i) / T squared. Let r_g = softmax(l_g / tau). The robust gradient is r_g times the group gradient. Its diagonal curvature is r_g times the group curvature plus r_g(1-r_g) times the squared group gradient divided by tau. The code adds the logistic anchor derivatives. It uses the exact diagonal, not the full Hessian.

The surrogate is convex in the vector of training margins: the log of the sum of the k largest exponentials is a maximum over subset log-sum-exp functions, and is therefore convex. The positive log-sum-exp term is convex. Softplus and the outer log-sum-exp are convex and nondecreasing in their arguments, preserving convexity under these compositions. The logistic anchor is convex. This argument concerns the margin vector; selecting tree structures is still a restricted optimization problem. At negative membership ties, a selected active subset supplies a subgradient and the implementation uses deterministic tie ordering. Numerical second-derivative tests deliberately use points away from these ties.

The ranking expression factors an entropic positive-negative pair comparison into two log-sum-exp terms, avoiding an explicit matrix of positive-negative pairs. The present implementation sorts negatives within each group, so it has sorting cost; no linear-time training claim is made. Tree fitting and repeated backtracking loss evaluations add further cost.

Trees are fit with diagonal Newton responses and regularized leaf updates. Up to twelve backtracking trials test the actual full empirical objective after each candidate update. An update is accepted only if its loss is lower by more than the fixed numerical tolerance. The retained prediction model is a validation-selected prefix of those updates. This is a checked empirical descent property, not an external robustness certificate.

Unit checks compare analytic gradients and diagonal curvature with finite differences, test tail selection and group weighting, exercise extreme scores and confirm the pairwise term's shift invariance. Integration checks exercise accepted loss descent, the selected prediction prefix, refitting, validation isolation and the single global threshold across time windows. Tests establish the stated numerical and workflow properties; they do not establish statistical benefit.

Remaining decisions: the maintainer must choose the acceptable recall and false-alarm tradeoff. The research lead must decide whether the complete candidate survives its simpler controls and finish the related-work comparison. The data owner must provide access to an untouched real dataset before independent confirmation can be claimed.
