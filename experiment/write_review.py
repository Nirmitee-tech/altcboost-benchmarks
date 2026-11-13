"""Generate the review from saved aggregate evidence."""
import json
from pathlib import Path
import pandas as pd
from scipy.stats import beta

root=Path(__file__).resolve().parent.parent
result=pd.read_csv(root/'pilot/results.csv')
audit=json.loads((root/'saved-results-audit.json').read_text())
discovery=json.loads((root/'pilot/discovery.json').read_text())
rerun=json.loads((root/'discovery-rerun.json').read_text())
assert discovery['sha256']==rerun['sha256'] and discovery['splits']==rerun['splits']
selected=result[(result.alpha==.001)&(result.status=='evaluated')]
lines=['| Model | Fraud caught | Recall | False alarms | Observed false-positive rate |',
       '|---|---:|---:|---:|---:|']
for row in selected.itertuples():
    lines.append(f'| {row.method} | {int(row.tp)}/{int(row.tp+row.fn)} | {row.recall:.1%} | {int(row.fp)} | {row.fpr:.3%} |')
table='\n'.join(lines)
hard2=sum(x['hard2_beats_v2_ap'] for x in audit['comparisons'])
test=discovery['splits']['test']
text=f'''# ALTCBoost review and Kaggle experiment

Keep this as a research experiment. The next useful investment is a broader time-based comparison, including repeat runs and a second dataset, before choosing a model for a live fraud workflow. The current work repairs reproducible code defects and supplies an executable experiment. It does not establish production readiness or a new state-of-the-art algorithm.

## What the supplied material shows

The separate results file contains {audit['result_rows']} rows from {audit['cases']} dataset-and-seed cases across {audit['datasets']} synthetic generators. The archive contains {audit['archive_comparison']['result_rows']} rows and omits the last case present in the separate file. The README describes fewer cases than either results file. The two copies should not be treated as one verified release.

The weaker hard-negative setting improves average precision over the earlier model in {hard2} of the {audit['cases']} supplied cases. Average precision summarizes how well high-risk cases are ranked across recall levels. This is an audit of stored results, not an independent reproduction of those training runs.

The research references are relevant, but their scope matters. The [partial-AUC paper](https://arxiv.org/abs/2512.01213) studies a principled minimax formulation, with experiments on imbalanced image datasets. The prototype uses a weighting heuristic. Its results cannot inherit the paper's guarantees. [Earlier large-scale work](https://arxiv.org/abs/2203.01505) also supports targeting the operating range explicitly. [Neyman-Pearson classification](https://arxiv.org/abs/1802.02558) motivates separate threshold calibration under stated distribution assumptions. [TabArena](https://arxiv.org/abs/2506.16791) supports stronger validation and baseline comparisons.

## What was tested here

On 12 November 2026, a local CPU pilot used the public [Kaggle credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The table below comes from that local run. A separate [private Kaggle-hosted benchmark](https://www.kaggle.com/code/jitendra93266/altcboost-fraud-benchmark-20261112) was subsequently submitted with a larger tree budget. Its results belong in the separate hosted report after completion.

The complete file was ordered by transaction time. Four separate intervals served training, model selection, calibration and final testing. Equal timestamps stayed together. Identical complete rows were removed as a benchmark choice; this does not prove those transactions were invalid. Discovery was rerun and returned the same file hash and partition counts.

The final interval has {test['rows']:,} transaction rows: {test['positive']} fraud cases and {test['negative']:,} normal cases. Each tree family had two prespecified configurations with a 30-tree limit and one random seed. These are deliberately limited pilot settings. Equal tree limits do not establish equal tuning effort. The custom models use early stopping, while the library models use the complete tree budget.

Every model received the same threshold-calibration rule. The table uses a calibration target of 0.1% false positives. The observed rate is measured on the later test interval and may exceed the target.

{table}

At this calibration target, the newer prototype caught 40 fraud cases with 35 false alarms. CatBoost caught 41 with 22 false alarms. The aligned challenger caught 40 with 36 false alarms. The proposed weighting change did not improve recall at this target in this pilot. This result does not settle performance after broader tuning or on other time periods.

A small difference in recall can be only one or two fraud cases here. Confidence intervals based on transaction-level independence also cannot capture unknown dependencies between cards. Treat the comparison as descriptive evidence from one short historical window.

## What changed

The compiled tree scorer now uses the same input precision for split comparisons as scikit-learn. A constructed boundary case exercises the previous branch disagreement. Both scorer paths check the fitted feature count. Refitting the compiled models invalidates an earlier calibrated threshold. The newer trainer checks labels before integer conversion and reports validation loss for the selected model rather than the final attempted iteration.

The experimental challenger removes adaptive negative weighting and aligns validation selection with a 0.1% false-positive range. This is a proposed ablation. It does not implement the cited paper's optimizer. The experiment also measures single-row scoring separately from average batch cost per row. Neither includes feature retrieval or network overhead. The reported single-row tail is descriptive of the measured calls, not a service-level guarantee.

## For whoever builds this

Skip this section unless you are running or changing the code.

The local working directory is `experiment/`. Original files are retained in `source/` and `original/`; the original archive is `source.zip`. No Drive file was changed.

Run discovery with `python discovery.py ../data/creditcard.csv ../discovery-rerun.json`. Run the pilot with `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python run_experiment.py ../data/creditcard.csv --output ../pilot --trees 30 --seeds 17`. The input SHA-256 is `{discovery['sha256']}`. Features are V1 through V28 and Amount. Time is used only for partitioning. The label is Class.

Evidence: `pilot/discovery.json`, `discovery-rerun.json`, `saved-results-audit.json`, `pilot/validation.csv`, `pilot/results.csv`, and `pilot/environment.json`. The supplied-results audit is rerunnable with `python audit_saved.py ../source/benchmark_v3_results.csv ../saved-results-audit.json`.

Tests after CLI job preparation: `PYTHONPATH=. python -m pytest -q tests/test_unit.py tests/test_kaggle_unit.py` passed 15 tests. `PYTHONPATH=. python -m pytest -q tests/test_integration.py tests/test_kaggle_integration.py` passed 4 tests. Coverage includes calibration ties and insufficient samples, measured error counts, label validation, refit invalidation, feature shape, compiled tree precision, temporal partitioning, aggregate output, the model training-to-calibration-to-flag flow, private CPU job settings and notebook preparation. These local checks do not establish Kaggle-hosted execution.

The official Kaggle CLI submitted `ALTCBoost_Kaggle.ipynb` as version 1 of `jitendra93266/altcboost-fraud-benchmark-20261112`. Readback confirmed private visibility, CPU, internet disabled, the attached `mlg-ulb/creditcardfraud` dataset and matching notebook cell content. The hosted log reports 10 model unit tests and 3 integration tests passed before training. The submitted experiment uses two seeds and a 70-tree limit, which is separate from this pilot. `kaggle-status.json` records the latest observed hosted state. `watch_kaggle.py` retrieves completed output and writes `hosted-report/review.md`; no completed hosted metrics are claimed in this local-pilot report.

The initial pilot launch was withdrawn after an integration test exposed a duplicate keyword in result construction. That launch was stopped before final test results were used; the corrected launch supplied the saved pilot results. The direct archive download first failed in the restricted network environment; authenticated Drive retrieval supplied the files. The direct arXiv API could not be retrieved through the web tool, so the papers were verified on arXiv abstract and full-text pages.

The independent-calibration rule uses a one-sided order-statistic bound. It assumes independent calibration negatives representative of future negatives. Fraud-stream dependence and temporal drift can break that assumption. Confidence intervals for recall can be computed from the saved TP and FN counts; they are not evidence of independence. Data provide no card identifiers or label-arrival times, and the original PCA construction cannot be audited here. Entity separation, delayed labels and full upstream preprocessing independence remain unverified.

The experiment exports aggregate results only. The delivery ZIP excludes the source transaction CSV, dataset archive and environment. The model changes are local, tested changes; no hosted deployment or live banking evaluation has occurred.

Decisions still needed: Jitendra must choose the operational false-positive budget and alert capacity. The fraud-data owner must supply a longer, later holdout with entity and label-arrival information for a production assessment. Kaggle sign-in is complete, so the earlier request for account access is withdrawn. Any model redesign after reviewing these test results needs a fresh final holdout.
'''
(root/'experiment/docs/review.md').write_text(text)
print(table)
summary=dict(source=str(root/'pilot/results.csv'), observed_utc=discovery['observed_utc'],
             scope='Local pilot; seed 17; 30-tree cap; fixed temporal split; target FPR 0.001',
             unit='transaction row', rows=selected.to_dict(orient='records'))
(root/'pilot/summary.json').write_text(json.dumps(summary,indent=2))
print('Created review and summary')
