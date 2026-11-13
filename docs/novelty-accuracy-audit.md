# Novelty and measured detection audit

The current candidate is measurable, but its detection results do not support a general accuracy improvement or a publication novelty claim. Continue it as research. The additional cost of this audit was recomputing saved aggregate results, checking the submitted experiment evidence and reviewing related papers. No new models were trained on source data and no final interval was scored.

The clearest failure is the synthetic bank-application benchmark. The candidate has 98.808% overall accuracy, yet it catches only 35 of 1,129 fraud cases at the chosen false-positive budget. Calling every case legitimate would have 98.871% accuracy while catching no fraud. Overall accuracy would therefore give the wrong impression of this model's usefulness.

The table below uses a single operating budget: at most 0.1% of legitimate validation cases receive false alerts. Recall is the share of fraud caught. Precision is the share of alerts that are fraud. The reference is chosen using the original two-budget selection rule; it is not newly chosen to flatter this table.

| Validation source | Fraud caught / total fraud | Recall | Precision | False alerts / legitimate cases | Reference recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elliptic, known-label Bitcoin AML transactions | 849 / 1,029 | 82.51% | 99.76% | 2 / 4,148 | 84.35%, XGBoost |
| PaySim, synthetic transactions | 897 / 1,192 | 75.25% | 90.61% | 93 / 93,398 | 74.50%, XGBoost |
| Sparkov, synthetic card transactions | 494 / 653 | 75.65% | 83.31% | 99 / 99,347 | 71.67%, XGBoost |
| BAF Base, synthetic applications | 35 / 1,129 | 3.10% | 26.32% | 98 / 98,871 | 6.82%, CatBoost |

These are results from the existing validation cohorts, audited on 13 November 2026. Those labels select model prefixes and thresholds and already informed the redesign. They are development evidence. PaySim and Sparkov show encouraging point differences against the tested native reference. Elliptic is weaker, and BAF is a substantial failure. No significance claim or future false-positive control follows from this table. The three synthetic sources also cannot establish performance on real payment fraud.

The added adaptive rule does not beat the strongest simple training policy on any of these four sources under the original selection rule. Its tie check saves some unsuccessful loss evaluations on BAF, but that narrower optimization result does not establish better detection. Claims that it improves accuracy generally, reduces future false positives or trains faster generally remain withdrawn or unproven.

Novelty is also narrower than the method's name might suggest. Focusing on high-scoring legitimate cases, robust partial-AUC objectives, changing training difficulty and using directional derivatives all have prior work. The specific failure-triggered cutoff rule and exact boundary-tie check remain candidate contributions. This focused search did not establish that their combination is an original, publishable algorithm.

I recommend a prospective research target of at least two percentage points more recall than the strongest tuned reference at the 0.1% false-positive budget, with no more than one percentage point degradation on any other predeclared confirmation source. These are proposed research targets, not measured achievements or universal business requirements. The improvement must also survive removing the proposed new component; otherwise the component has not earned the claim.

Before a confirmation run, freeze the method, preprocessing, reference tuning budget, operating thresholds, sources, primary metric and acceptable margins. Select thresholds using an earlier calibration interval and evaluate a later untouched interval. Report recall, precision, all four confusion counts, average precision, training time and inference time separately. Compare paired outcomes with uncertainty that respects the source's time or entity dependence. If the data cannot support that inference, report descriptive differences and keep the superiority claim open.

## For whoever builds this

Skip this section unless implementing or reviewing the audit.

The measured candidate is `V8_adaptive_checked`, with seed 17 and 40 fitted-tree proposals, including rejected proposals. Native references use the previously selected hyperparameters with a 40-tree cap and validation prefix selection. Tree structure capacity, prior search effort and wall time are not matched. The highest native reference is chosen with the original lexicographic key: mean recall at empirical FPR budgets 0.0005 and 0.001, then lower mean FPR and higher average precision. The body table reports recall at 0.001 alone. Its values therefore differ from the previous mean-recall table without contradicting it.

`experiment/accuracy_audit.py` recomputes accuracy, balanced accuracy, recall, precision, F1 and false alerts per 10,000 legitimate cases from saved TP, FP, FN and TN. It checks saved rates, population consistency, count validity and duplicate operating points. It does not recover prediction scores or paired error overlap from aggregate counts. It cannot calculate new ROC-AUC values, score calibration, new cutoffs or valid selection-adjusted confidence intervals from those aggregates.

Reproduce with:

```sh
PYTHONPATH=experiment .venv/bin/python experiment/verify_v8.py v8-study
PYTHONPATH=experiment .venv/bin/python experiment/accuracy_audit.py v8-study accuracy-novelty-audit
```

`accuracy-novelty-audit/manifest.json` records the audit observation time, dataset kind, counting unit, exact source paths, CSV hashes and cohort hashes. `all-operating-points.csv` preserves every tested method at the three original budgets. `candidate-and-reference.csv` contains the body table's candidate/reference comparisons. `accuracy-review.md` is the generated numerical review. No raw feature rows or identifiers are exported.

The full-source verifier was rerun during this audit. All four source and training/validation cohort hashes, executed embedded files, proposal budgets, population counts and saved recall/FPR values passed. Those checks establish consistency with the submitted experiments. They do not prove that labels are correct, that provider features were available at decision time, or that a consumed validation score will generalize.

The novelty review used the current implementation, the earlier derivation and primary sources accessed on 13 November 2026:

| Component or claim | Closest checked work | Assessment |
| --- | --- | --- |
| Partial AUC through negative tails and robust objectives | [Zhu et al., ICML 2022](https://proceedings.mlr.press/v162/zhu22g.html), which studies CVaR and KL-based partial-AUC formulations. | Established ingredients. The exact temporal tree implementation still needs a direct algorithm comparison. |
| Hard-negative ranking and adjustable hardness | [Shi et al., WWW 2023](https://arxiv.org/abs/2302.03472), including its author-hosted [full paper](https://hexiangnan.github.io/papers/www23-ns-rec.pdf). | Links hard-negative BPR to one-way partial AUC and recommends controllable hardness. Changing hardness alone cannot carry novelty. |
| Adaptive hardness as a broad claim | [Wu et al., August 2026](https://arxiv.org/abs/2608.16587), SAHC-NS. | Adjusts hardness using candidate-pool and embedding structure in recommendation. Its trigger differs from our failed-tree trigger, but adaptive hardness is already occupied as a broad idea. This comparison used the abstract. |
| Directional derivatives for nonsmooth ROC learning, including coincident scores | [Hillman and Hocking, JMLR 2023](https://www.jmlr.org/papers/volume24/21-0751/21-0751.pdf), Sections 3.2 to 3.4 and 4. | Computes directional derivatives for AUM, discusses undefined gradients and a coincident-prediction example. Our tail loss and full-tree direction differ. A broad claim to introduce directional ROC optimization is unsupported. |
| Smoothing and learned thresholds for partial AUC | [Jiang et al., 2025](https://arxiv.org/abs/2512.01213). | Reports instance-wise minimax formulations and scaling results. Our pairwise implementation has not established an advantage over that solver. This comparison used the abstract. |
| Continuation and curriculum as an optimization principle | [Deep Learning, Chapter 8](https://www.deeplearningbook.org/contents/optimization.html). | Established. The failure-triggered rule must be assessed as a specific policy, rather than claiming continuation itself. |
| Our exact boundary-tie correction and per-tree non-descent rejection | `experiment/tail_direction.py`, derivation in `docs/tail-continuation-research.md`. | An application of maximum-function directional calculus. The exact implementation may differ from these papers; mathematical derivation and passing tests alone do not establish priority. Published optimizer baselines have not been run head to head. |

The arXiv API endpoint was inaccessible again, so publisher pages, author pages and verified arXiv abstract pages were used. Several broad search queries returned irrelevant results. They provide no evidence of novelty. This is a focused overlap audit, not an exhaustive priority search or an assertion that the exact method has already been published.

The proposed research gate needs an independent source, valid paired uncertainty and the predeclared margins above. A nominal 95% paired interval with a lower bound above zero would support a positive primary difference only if the sampling assumptions are justified and selection occurs before that confirmation data is seen. Repeating seeds on the same data measures training variation; it does not create independent datasets. Small or dependent cohorts may support only descriptive reporting. Existing binomial intervals in the saved validation CSVs do not adjust for model/cutoff selection or time dependence and must not be used to certify superiority.

No fresh confirmation was performed in this audit. Earlier IEEE-CIS archive access returned 403. No inference-speed result from an older compiled model has been assigned to the current candidate.

Decisions still needed: the research lead must adopt or revise the proposed two-point improvement and one-point degradation margins before new outcomes are inspected, then freeze a candidate and its closest published comparisons. The application owner must set the absolute minimum useful recall and precision using alert capacity and missed-fraud cost. the maintainer must resolve access to a fresh confirmation source. Current measured performance and novelty do not pass those confirmation gates.
