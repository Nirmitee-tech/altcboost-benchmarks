"""Describe the fixed candidate experiment, including failed candidate comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from multi_summary import completed_rows

def observed_comparison(candidate,reference):
    tp=int(candidate['tp'])-int(reference['tp']);fp=int(candidate['fp'])-int(reference['fp'])
    if tp==0 and fp==0:return 'same counts'
    if tp>=0 and fp<=0:return 'more or equal fraud caught with fewer or equal false alarms'
    if tp<=0 and fp>=0:return 'fewer or equal fraud caught with more or equal false alarms'
    return 'recall and false alarms trade off'

def prefix_note(dataset,models,method):
    chosen=next((model for model in models if model['method']==method),None)
    if chosen is not None and chosen.get('selected_iteration')==0:
        return f'{dataset}: validation selected only the starting linear model. No residual trees were retained, so this result cannot be attributed to the proposed tree-training mechanism.'
    return None

def write_tail_summary(root,keys=('baf_base','paysim','sparkov'),generation=5):
    root=Path(root);rows,evidence=completed_rows(root,keys)
    lines=['# Temporal-tail candidate results','',
        'These completed Kaggle experiments evaluate a fixed research candidate and its simpler controls. All three sources are synthetic and have already informed development. They provide no independent final confirmation or established novelty claim.','',
        'Each cell reports fraud caught / false positives at the globally calibrated 0.1% target. Actual test rates can differ from that target. Comparisons below describe observed counts, not statistical significance.','',
        '| Dataset | Complete candidate | Pooled-time tail | All-negative temporal | Balanced logistic control | Prior ALTCBoost | CatBoost | XGBoost |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    prefix=f'V{generation}_'
    methods=[prefix+'temporal_tail',prefix+'pooled_tail',prefix+'temporal_all_negatives',prefix+'balanced_logistic','ALTCBoost_tuned','CatBoost','XGBoost']
    at_budget=rows[np.isclose(rows.alpha,.001)&(rows.status=='evaluated')]
    for key in keys:
        sample=at_budget[at_budget.dataset==key];cells=[key]
        for method in methods:
            found=sample[sample.method==method]
            cells.append(f'{int(found.iloc[0].tp)} / {int(found.iloc[0].fp)}' if len(found)==1 else 'Unavailable')
        lines.append('| '+' | '.join(cells)+' |')
    for item in evidence:
        models=json.loads((root/item['dataset']/'kaggle-output/multi_results/models.json').read_text())
        note=prefix_note(item['dataset'],models,methods[0])
        if note:lines.extend(['',note])
    lines.extend(['','The main research question is whether the complete candidate earns its extra complexity over the controls.',''])
    for key in keys:
        sample=at_budget[at_budget.dataset==key].set_index('method')
        if methods[0] not in sample.index:continue
        for method in methods[1:]:
            if method in sample.index:
                outcome=observed_comparison(sample.loc[methods[0]],sample.loc[method])
                lines.append(f'- {key}, complete candidate versus {method}: {outcome}.')
    design='docs/pairwise-research.md' if generation==6 else 'docs/temporal-tail-research.md'
    study='pairwise-study' if generation==6 else 'novelty-study'
    development='The revision followed V5 development failures and a PaySim validation-only screen. These are consumed sources in an adaptive research process.' if generation==6 else 'These datasets had already informed development before this fixed candidate run.'
    lines.extend(['','The 0.05% and 1% targets remain in the combined CSV and each dataset report. A favourable operating point alone does not establish a general advantage. Failure against a simpler control is evidence against attributing gains to the added mechanism.','',
        '## For whoever builds this','',
        f'The loss, derivatives, assumptions and related-work boundaries are documented in {design}. The plan is {study}/prespecified-plan.json. Configurations were fixed before this run; its final outcomes did not select hyperparameters. {development} Per-period results use one global calibration cutoff, not independently optimized period cutoffs. Up to four windows are used; a source with only one final timestamp group cannot support a within-test temporal stability assessment. The baseline configurations are frozen reference configurations; native categorical handling and equal-budget baseline tuning remain untested.',''])
    for item in evidence:
        path=root/item['dataset']/'kaggle-output/multi_results'
        windows=pd.read_csv(path/'window-results.csv')
        subset=windows[np.isclose(windows.alpha,.001)&(windows.status=='evaluated')]
        lines.append(f"### {item['dataset']} time-window checks")
        lines.extend(['','| Method | Lowest window recall | Highest window FPR |','| --- | ---: | ---: |'])
        for method in methods:
            parts=subset[subset.method==method]
            if len(parts):lines.append(f"| {method} | {parts.recall.min():.2%} | {parts.fpr.max():.3%} |")
        lines.append('')
        lines.append(f"Source observation: {item['discovery']['observed_utc']}; Kaggle run: {item['kernel']}. Full row and cohort counts, source hashes and time boundaries are in discovery.json. No independent-row confidence interval here accounts for customer or time dependence.")
        models=json.loads((path/'models.json').read_text())
        for model in models:
            loss=model.get('training_loss')
            if loss is not None:
                if not np.all(np.diff(loss)<0):raise ValueError('A recorded accepted step did not reduce empirical loss')
                lines.append(f"- {model['method']}: {len(loss)-1} accepted descending training updates; selected prefix {model['selected_iteration']}; fit duration {model['fit_s']:.2f} seconds.")
    lines.extend(['','Decision still needed: Jitendra and the research lead must decide whether the complete candidate survives its simpler controls. Stronger tuned baselines, further related-work review and untouched real-data evidence are required before treating this as a publication-ready contribution.'])
    target=root/'results-review.md';target.write_text('\n'.join(lines)+'\n')
    rows.to_csv(root/'benchmark-results.csv',index=False)
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--generation',type=int,choices=[5,6],default=5)
    args=p.parse_args();print(write_tail_summary(args.root,generation=args.generation))
