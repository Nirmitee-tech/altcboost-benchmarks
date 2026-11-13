"""Separate primary, stochastic-repeat and conditional uncertainty evidence."""
import json
from pathlib import Path
import pandas as pd

FAMILIES=['V6','CatBoost_native','XGBoost','LightGBM_native']

def primary_rows(rows,alpha=.001):
    selected=rows[(rows.seed==17)&((rows.alpha-alpha).abs()<1e-12)]
    if selected.method.duplicated().any():raise ValueError('Duplicate primary method rows')
    return selected

def write_validation_report(source,destination):
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    rows=pd.read_csv(source/'results.csv');manifest=json.loads((source/'manifest.json').read_text())
    discovery=json.loads((source/'discovery.json').read_text());models=json.loads((source/'models.json').read_text())
    intervals=json.loads((source/'uncertainty.json').read_text());test=discovery['partitions']['test']
    lines=[f"# Validation study: {manifest['dataset']}",'',
        'V6 settings were held fixed. Each baseline family received four prespecified configurations and validation-only selection of its configuration and tree prefix. Search counts and tree caps are matched across baseline families; wall time and model capacity are not matched.','',
        f"Evidence: {manifest['holdout_status']}. Source type: {manifest['dataset_kind']}.",
        f"The evaluated cohort contains {test['sampled_positive']} positive and {test['sampled_negative']} negative {manifest['unit']}s. Counts were measured in the run completed at {manifest['completed_utc']}.",'',
        'The primary comparison uses seed 17 at the 0.1% calibration target. Targets do not guarantee the same false-positive rate in a later time period.','',
        '| Model | Positive cases caught | False positives | Recall | Observed FPR | Selected prefix |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    primary=primary_rows(rows)
    for family in FAMILIES:
        found=primary[primary.method==family]
        if len(found)!=1 or found.iloc[0].status!='evaluated':lines.append(f'| {family} | Unsupported calibration sample | | | | |');continue
        row=found.iloc[0]
        lines.append(f"| {family} | {int(row.tp)} | {int(row.fp)} | {row.recall:.2%} | {row.fpr:.3%} | {int(row.selected_prefix)} |")
    if any(m['family']=='V6' and m['seed']==17 and m['prefix']==0 for m in models):
        lines.extend(['','V6 validation retained only its starting linear model. No residual trees were selected, so this result cannot be attributed to the proposed tree-training mechanism.'])
    if manifest['dataset']=='elliptic':
        lines.extend(['','This is an AML transfer task on the known-label subset of real Bitcoin data. Unknown labels are excluded rather than counted as normal. It does not establish independent payment-fraud performance. The explicitly forward/backward-neighbour features are excluded; anonymous local-feature availability and provider normalization cannot be independently audited.'])
    lines.extend(['','Training repeats use the same data cohorts. Their ranges describe stochastic fitting and validation-prefix selection, not independent replications of the dataset.','',
        '| Model | Seeds evaluated | Recall range | FPR range |','| --- | ---: | ---: | ---: |'])
    supported=rows[(rows.status=='evaluated')&((rows.alpha-.001).abs()<1e-12)]
    for family in FAMILIES:
        part=supported[supported.method==family]
        if len(part):lines.append(f"| {family} | {part.seed.nunique()} | {part.recall.min():.2%} to {part.recall.max():.2%} | {part.fpr.min():.3%} to {part.fpr.max():.3%} |")
    lines.extend(['','The paired cluster checks below condition on the selected models and fixed calibration cutoffs. They omit uncertainty from retraining, hyperparameter selection and calibration. Intervals are descriptive percentile intervals with no multiple-comparison correction. A minimum of 20 clusters is a prespecified screen, not a validity theorem.','',
        '| Reference | Cluster scheme | Clusters | V6 minus reference recall, percentage points | V6 minus reference FPR, percentage points |',
        '| --- | --- | ---: | ---: | ---: |'])
    for item in intervals:
        if abs(item['alpha']-.001)>1e-12:continue
        base=f"| {item['reference']} | {item['cluster_scheme']} | {item['cluster_count']} |"
        if item['status']!='evaluated':lines.append(base+' Unsupported: too few usable clusters | |');continue
        def cell(point,interval):return f'{point*100:+.4f} [{interval[0]*100:+.4f}, {interval[1]*100:+.4f}]'
        lines.append(base+' '+cell(item['observed_delta_recall'],item['delta_recall_interval_95'])+' | '+cell(item['observed_delta_fpr'],item['delta_fpr_interval_95'])+' |')
    lines.extend(['','## For whoever builds this','',
        f"Source: {manifest['source_url']}. Licence: {manifest['licence']}. Source and cohort hashes, allowlists and time bounds are in discovery.json, observed at {discovery['observed_utc']}.",
        'The selected configurations were recorded in selection.json before calibration/test scoring. trials.csv records every baseline candidate; models.json records repeat prefixes and durations. All budgets and seeds remain in results.csv. Window checks use one global calibration cutoff. uncertainty.json retains all supported and unsupported paired checks.',
        '',manifest['limitations'],'',
        'Decision still needed: the research lead must assess V6 against the tuned baselines and the uncertainty limits. Payment-fraud confirmation on an untouched source and a completed novelty review remain separate requirements before a publication claim.'])
    target=destination/'review.md';target.write_text('\n'.join(lines)+'\n');return target

def write_validation_summary(root,keys=('baf_base','paysim','sparkov','elliptic')):
    root=Path(root);lines=['# Frozen V6 against tuned baselines','',
        'Each cell gives positive cases caught / false positives for seed 17 at the 0.1% calibration target. Sources are evaluated separately. All configurations were fixed or selected using training and validation data before final scoring.','',
        '| Dataset | Evidence | V6 | Native CatBoost | XGBoost | Native LightGBM |',
        '| --- | --- | ---: | ---: | ---: | ---: |'];combined=[]
    for key in keys:
        study=root/key;state=json.loads((study/'kaggle-status.json').read_text())
        if state['state']!='COMPLETE':raise ValueError(f'{key} is not complete')
        paths=list((study/'kaggle-output').rglob('validation_results/results.csv'))
        if len(paths)!=1:raise ValueError('Expected exactly one completed result file')
        rows=pd.read_csv(paths[0]);manifest=json.loads(paths[0].with_name('manifest.json').read_text())
        if manifest['dataset']!=key or not rows.dataset.eq(key).all():raise ValueError('Dataset identity mismatch')
        combined.append(rows);primary=primary_rows(rows);cells=[key,'Real AML transfer, known labels' if key=='elliptic' else 'Consumed synthetic development']
        for family in FAMILIES:
            found=primary[primary.method==family]
            cells.append(f'{int(found.iloc[0].tp)} / {int(found.iloc[0].fp)}' if len(found)==1 and found.iloc[0].status=='evaluated' else 'Unsupported')
        lines.append('| '+' | '.join(cells)+' |')
    lines.extend(['','The synthetic datasets have informed development. Elliptic is a separately labelled real Bitcoin AML transfer test; it does not replace independent payment-fraud validation. Unknown Elliptic labels and explicitly backward/forward-neighbour features are excluded.','',
        'Four configurations and a 100-tree cap were allocated to each baseline family. V6 numerical settings were frozen. Repeats use the same cohorts with different model seeds; paired cluster intervals condition on fitted models and fixed cutoffs. Small searches, coarse time clusters and dependence limit the conclusions. No pooled accuracy, automatic winner or established novelty claim is reported.','',
        '## For whoever builds this','',
        'Each dataset folder contains its hosted-report/review.md with primary counts, repeat ranges, paired cluster checks and limitations. The full records are in kaggle-output/validation_results. The plan is validation-study/prespecified-plan.json; the combined result file is benchmark-results.csv. IEEE-CIS access remains unconfirmed after a 403 download response; no competition rules were accepted automatically.','',
        'Decision still needed: the research lead must decide whether these results justify a manuscript claim after a fuller novelty review and independent payment-fraud evidence.'])
    target=root/'results-review.md';target.write_text('\n'.join(lines)+'\n')
    pd.concat(combined,ignore_index=True).to_csv(root/'benchmark-results.csv',index=False)
    return target
