# Research proposal: adapt the difficult-case cutoff when training stalls

Keep the tie-aware rejection check as an optimization candidate. Withdraw the claim that the adaptive cutoff improves predictive quality over the strongest simple control. The completed pilot ran six training policies on each of four existing sources, plus three reference models. It establishes a narrower cost benefit on one source. It does not establish publication novelty or independent predictive improvement.

Fraud detection needs to catch fraudulent cases while keeping mistaken alerts on legitimate cases low. Concentrating training on the legitimate cases with the highest risk scores can help that aim. It also creates a problem: many cases can share the same score, especially at the start of a tree ensemble. Choosing an arbitrary subset of those tied cases can suggest an update that makes the actual training loss worse.

The proposed rule starts with the narrow cutoff we care about. It tests whether the next proposed tree can improve that training objective. If the proposal fails, the rule widens the set of legitimate cases used for the next proposal. After an accepted update it tries the narrow cutoff again. The rule therefore responds to the proposed tree's ability to make progress. It does not assume the narrowest objective is easy to optimize from the first step.

A separate check handles ties correctly before trying step sizes. Its purpose is to reject a particular tree direction that cannot improve its current objective. It cannot establish that every other tree would fail. The experiment compares this check with the same adaptive rule using ordinary repeated loss evaluations, so we can separate any change in predictive quality from changes in training work.

Earlier work in this project tried to adjust a starting linear model and a new tree together. Across the four completed validation experiments, that mechanism did not beat the strongest simple starting-model control. The claim that the coupled update earns its extra complexity is withdrawn. Removing the starting linear model substantially improved one validation result, but a zero start stalled on another source. Those observations motivated the present optimization question. They do not establish a general explanation of all earlier failures.

The candidate contribution is the specific combination of a proposed-tree-driven cutoff rule and a tie-aware rejection check for this temporal ranking loss. The broader ingredients already exist. Partial-AUC optimization, robust ranking, continuation methods and directional derivatives are established research. Calling the combination new requires a closer algorithm comparison, in addition to a useful empirical result.

The completed comparison does not support the combined mechanism as a stronger predictor. On three sources it produces the same selected validation result as the strict control because no fallback is needed. On the fourth it recovers some detection from a stalled start, but an ordinary schedule performs better. The rejection check avoids some unsuccessful loss evaluations there. A single observed runtime reduction does not establish a general training speedup.

## For whoever builds this

Skip this section unless you are implementing or reviewing the method.

The authoritative implementation is in `experiment/altcboost_v8.py` and `experiment/tail_direction.py`. `experiment/v8_research.py` fits training rows and scores the original validation rows. It never receives calibration or final frames for prediction. Source discovery still reads those source partitions for schema, counts and file verification. The frozen pilot plan is `v8-study/prespecified-plan.json`; it was saved before these full-source V8 runs were submitted on 13 November 2026.

For temporal group g, positive set P_g, negative set N_g and working fraction beta, let k_g = ceil(beta |N_g|), at least one. Write ell(z) = log(1 + exp(z)), temperature T > 0, margin m = 1, and score vector f. The group loss is

```math
L_g(f;\beta)=\max_{S\subseteq N_g,\ |S|=k_g}
\frac{1}{|P_g|k_g}\sum_{i\in P_g}\sum_{j\in S}
\ell\left(\frac{f_j-f_i+m}{T}\right).
```

Every positive case orders negative cases by the same negative score. The maximizing subset is therefore the k_g highest scoring negatives. This identity concerns the mean pairwise logistic surrogate. It does not turn that surrogate into the discontinuous partial AUC.

Let A(f) be the globally class-balanced logistic loss, eta = 0.2, tau = 0.25, and G = 4 training time groups. The full loss is

```math
F_\beta(f)=\eta A(f)+(1-\eta)\tau\log\left(
\frac1G\sum_g\exp\left(L_g(f;\beta)/\tau\right)\right).
```

For fixed beta, every fixed-subset loss is convex in the finite score vector. Their maximum is convex. Log-sum-exp is convex and coordinatewise increasing, so its composition with the group losses is convex. The anchor is also convex. This argument concerns score space. Greedy tree structure selection is still a restricted search and has no global optimum guarantee here.

At the tail cutoff q_g, let B_g be the negative cases tied at that score, r_g the number needed from that tie, and C_g the particular tied cases chosen by the stable sort. Let p_g be the softmax weight of group g in the robust loss, and let g_active be the implemented gradient of that chosen face. For a proposed score direction d, the right directional derivative is

```math
F'_\beta(f;d)=g_{active}^{\mathsf T}d+
(1-\eta)\sum_g p_g c_g
\left[\sum_{\text{largest }r_g\text{ values in }B_g} d_j
-\sum_{j\in C_g}d_j\right],
\qquad
c_g=\frac{1}{k_g T|P_g|}\sum_{i\in P_g}
\sigma\left(\frac{q_g-f_i+m}{T}\right).
```

To obtain this expression, take the maximum directional derivative over the active maximizing subsets. Only members of the boundary tie can exchange. All have the same pair-loss slope, so the maximizing exchange takes the largest r_g directions inside that tie. The correction is nonnegative and vanishes when the boundary has no unselected ties. The group robust loss is differentiable in its group-loss arguments, so its chain rule supplies p_g. This is an application of standard maximum-function directional calculus, not a claim to have invented that calculus.

For a finite convex function phi(t) = F_beta(f + t d), phi(t) >= phi(0) + t phi'_+(0) for t >= 0. Therefore a nonnegative exact right derivative excludes improvement along this direction for every positive step. The code computes a floating-point sign and checks actual losses for accepted steps. It does not provide interval arithmetic bounds on rounding error. A negative derivative allows sufficiently small descent in exact arithmetic, but the finite twelve-step backtracking budget and acceptance tolerance may still reject the proposal.

An executable counterexample is `tests/test_tail_direction_unit.py::test_arbitrary_active_face_can_falsely_look_like_descent`. With two negatives, one positive, zero scores, beta = 0.5, no anchor and direction (-1, 2, 0), a stable active-face gradient reports negative slope. The true right derivative is positive because the second negative immediately becomes the highest scoring negative. Six finite-difference cases also cover tied and untied scores with three anchor weights.

The six policies are `target`, `all_negative`, `geometric`, `linear`, `adaptive_blind` and `adaptive_checked`. Each receives 40 fitted tree proposals. Rejected proposals count. Adaptive policies try beta values 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 and 1 until a proposal is accepted or the budget is exhausted. They reset to 0.01 after acceptance. Clock schedules use proposal index, rather than accepted-tree count. All models start from zero. Trees have at most 17 leaves, at least 40 rows per leaf, leaf outputs clipped to 5, initial step size 0.08 and seed 17 plus proposal index.

An accepted step must decrease its current working-beta loss by more than 1e-12. **The fixed beta = 0.01 loss can increase after an accepted relaxed-beta step.** The implementation records both losses to prevent that distinction from disappearing in a report. The common proposal budget matches fitted-tree attempts; it does not match wall time, accepted ensemble size or pairwise arithmetic. Objective-call counts include derivative-state construction and loss checks, whose cost varies with beta.

Native CatBoost, XGBoost and LightGBM reuse hyperparameters selected in the earlier study, refit with a 40-tree cap, and reselect a prefix on the original validation labels. Earlier hyperparameter search gives those references extra search effort. All policies use the existing source hashes, row cohorts and training-only preprocessing. Validation mean recall at empirical false-positive budgets 0.05% and 0.1% is the primary development measure. Lower mean false-positive rate and higher average precision break ties. The validation cutoff uses validation negative labels, so no deployment threshold or future false-positive guarantee follows.

The saved metric tables also contain binomial intervals from the shared metric helper. Those intervals do not correct for choosing prefixes and cutoffs on the same validation labels, or for temporal dependence. They must not be used as uncertainty bounds for a selected-method superiority claim. The reports deliberately compare descriptive validation counts without claiming significance.

Elliptic measures transfer to known-label Bitcoin AML transactions, rather than payment fraud. Its unknown labels are excluded. Three other sources are synthetic. All four have informed development. The old Elliptic final-test failure remains recorded and is not rerun to advertise a rescue. Dataset licenses and feature availability limits remain those in `docs/frozen-validation-study.md` and the discovery manifests.

The discovery claims are saved before fitting in the plan: whether adaptation exceeds fixed and clock schedules, whether checked and blind accepted trajectories match, and whether BAF stalling is recovered. `v8_report.py` reruns aggregate comparisons from saved validation and model metadata. No raw source rows, customer identifiers or per-row prediction exports are needed.

Completed results, measured on 13 November 2026, use the original validation partition of each named source. Each percentage is mean recall across the empirical validation false-positive budgets of 0.05% and 0.1%. These are development comparisons, without a significance claim.

| Source | Adaptive checked | Strongest simple control | Highest tested native reference |
| --- | ---: | ---: | ---: |
| Elliptic, known-label AML transactions | 82.51% | 83.62%, all negatives | 83.58%, XGBoost |
| PaySim, synthetic transactions | 72.53% | 72.53%, strict tail | 72.48%, XGBoost |
| Sparkov, synthetic card transactions | 71.82% | 71.82%, strict tail | 67.61%, XGBoost |
| BAF Base, synthetic applications | 2.30% | 5.36%, linear schedule | 5.67%, CatBoost |

On BAF Base, blind adaptation used 296 objective calls and the checked version used 236, a 20.27% reduction in this implementation. Fit times were 113.91 and 105.07 seconds respectively. Their accepted target-loss histories and primary validation metrics match. Full source-row prediction equality was not checked; matching aggregate results alone do not prove it. On the other three sources, both versions used 121 objective calls. All current Python candidates train more slowly than the tested native reference implementations in these runs. Older compiled-model inference timings cannot be transferred to these models.

Reproduce the tables with `PYTHONPATH=experiment .venv/bin/python -c 'from v8_report import write_v8_summary; write_v8_summary("v8-study")'`. The derived cost and control comparisons are saved in `v8-study/conclusion.json`. Rerun source, cohort, budget, metric and executed-code checks with `PYTHONPATH=experiment .venv/bin/python experiment/verify_v8.py v8-study`; results are in `v8-study/verification.json`. Private notebook readbacks also match their submitted image and source snapshots. Each hosted run passed nine unit and ten integration test gates. The broader final local suites passed 126 unit and 64 integration tests; `v8-study/local-test-results.json` records the local run separately from the frozen hosted gates.

The confirmation access recheck is saved in `v8-study/confirmation-access.json`. IEEE-CIS file listing succeeded, but downloading the training archive returned 403. No rows from that archive were opened. Access and a fresh confirmation protocol remain unresolved.

Closest checked primary sources, reviewed on 13 November 2026:

| Work | Established overlap | Difference to investigate, not proof of novelty |
| --- | --- | --- |
| [Narasimhan and Agarwal, ICML 2013](https://proceedings.mlr.press/v28/narasimhan13.html) | Structural SVM and cutting planes optimize partial AUC in a specified false-positive range. | Our restricted tree proposals and adaptive working fraction need comparison with their constraint optimization. |
| [Zhu et al., ICML 2022](https://proceedings.mlr.press/v162/zhu22g.html) | CVaR and KL-based robust formulations provide nonsmooth and smooth partial-AUC surrogates and optimization methods. | A hard tail or robust ranking loss alone is not new; our question is the continuation and rejection policy. |
| [Yao, Lin and Yang, 2022](https://arxiv.org/abs/2203.01505) | Moreau-envelope smoothing and stochastic block updates address nonsmooth partial-AUC optimization. | Compare their smoothing and cost tradeoff with widening a working tail; no superiority has been measured. |
| [Hillman and Hocking, JMLR 2023](https://www.jmlr.org/papers/v24/21-0751.html) | Sort-based ROC learning uses efficiently computed directional derivatives for the AUM loss. | Our loss and subset-boundary derivative differ, but directional ROC optimization is already established. |
| [On the Theories Behind Hard Negative Sampling for Recommendation, WWW 2023](https://doi.org/10.1145/3543507.3583223) | Hard-negative sampling connects to one-way partial AUC, with controllable hardness as a practical guideline. | Varying negative hardness is already established; the proposed-tree failure trigger and retry policy require a separate novelty argument. |
| [Nock and Mansour, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1cf760a547822e2b8276881ad45f0fe9-Abstract-Conference.html) | A formal boosting framework extends to very general losses using zeroth-order information. | Our practical tree scheduling experiment does not establish a comparable general boosting theorem. |
| [Jiang et al., 2025](https://arxiv.org/abs/2512.01213) | Instance-wise minimax reformulations combine learned thresholds and smoothing to address partial-AUC approximation and scaling. | Our pairwise loss still has pairwise arithmetic cost. No scaling or approximation advantage over this work has been established. |
| [Chowdhary, Newman and Verma, 2026](https://arxiv.org/abs/2603.23658) | Variable-projection boosting combines linear-weight correction and second-order weak learning for smooth separable approximators. | This overlaps the withdrawn coupled-weight direction; the present study uses nonsmooth tails and zero-start trees. |

The arXiv open API was unavailable during this sweep. Primary publisher and arXiv abstract pages were used instead. This is a focused overlap check, not an exhaustive novelty search. Author implementations of the closest optimizers have not yet been run in this study.

The adaptive-quality claim is withdrawn using all four completed runs. A proposed next experiment is to optimize each tree's leaf weights against every active boundary-tie choice, keeping the requested objective fixed. That could test whether a better direction can be found inside the same tree structure. This extension has not been implemented or benchmarked. Its convex subproblem would not by itself establish that the greedy tree structure is optimal or that the method is novel.

Decisions still needed: the research lead must choose the next mechanism and which published optimizer to reproduce. the maintainer must choose the intended publication venue and arrange access to a fresh confirmation source after a candidate is frozen. Neither decision is evidence that the current method is publishable.
