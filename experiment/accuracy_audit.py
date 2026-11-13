"""Recompute descriptive validation performance from aggregate confusion counts."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from v7_report import best_record

COUNTS=['tp','fp','fn','tn']
NATIVE=['CatBoost_native','XGBoost','LightGBM_native']

def audit_counts(frame):
    frame=frame.copy();values=frame[COUNTS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any():
        raise ValueError('Confusion counts must be finite nonnegative integers')
    positives=frame.tp+frame.fn;negatives=frame.fp+frame.tn;total=positives+negatives
    if (positives<=0).any() or (negatives<=0).any():raise ValueError('Both classes required')
    frame['recall_recomputed']=frame.tp/positives;frame['fpr_recomputed']=frame.fp/negatives
    frame['precision_recomputed']=np.divide(frame.tp,frame.tp+frame.fp,out=np.zeros(len(frame)),where=(frame.tp+frame.fp).to_numpy()>0)
    for name,computed in [('recall','recall_recomputed'),('fpr','fpr_recomputed'),('precision','precision_recomputed')]:
        if not np.allclose(frame[name],frame[computed],rtol=0,atol=1e-12):raise ValueError(f'Saved {name} differs from counts')
    frame['accuracy']=(frame.tp+frame.tn)/total
    frame['all_legitimate_accuracy']=negatives/total
    frame['accuracy_lift_percentage_points']=100*(frame.accuracy-frame.all_legitimate_accuracy)
    frame['fraud_cases']=positives;frame['legitimate_cases']=negatives;frame['cases']=total
    frame['balanced_accuracy']=(frame.recall_recomputed+1-frame.fpr_recomputed)/2
    frame['false_alerts_per_10000_legitimate']=10000*frame.fpr_recomputed
    frame['within_empirical_budget']=frame.fpr_recomputed<=frame.alpha+1e-12
    frame['f1']=np.divide(2*frame.tp,2*frame.tp+frame.fp+frame.fn)
    if frame.duplicated(['dataset','method','alpha']).any():raise ValueError('Duplicate operating point')
    for _,group in frame.groupby('dataset'):
        if group.fraud_cases.nunique()!=1 or group.legitimate_cases.nunique()!=1:raise ValueError('Methods use different validation populations')
    return frame

def run(study,output):
    study=Path(study);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    keys=json.loads((study/'prespecified-plan.json').read_text())['reference_cohorts']
    frames=[];sources=[]
    for key in keys:
        folder=study/key
        if json.loads((folder/'kaggle-status.json').read_text())['state']!='COMPLETE':raise ValueError('Research run incomplete')
        paths=list((folder/'kaggle-output').rglob('research_results/validation.csv'))
        if len(paths)!=1:raise ValueError('Expected one validation result')
        frame=pd.read_csv(paths[0]);discovery=json.loads(paths[0].with_name('discovery.json').read_text())
        val=discovery['partitions']['validation']
        if not frame.dataset.eq(key).all() or not (frame.tp+frame.fn).eq(val['sampled_positive']).all() or not (frame.fp+frame.tn).eq(val['sampled_negative']).all():raise ValueError('Discovery population differs from counts')
        frames.append(frame);sources.append(dict(dataset=key,path=str(paths[0]),sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),unit=discovery['unit'],kind=discovery['dataset_kind'],cohort_sha256=val['cohort_sha256']))
    audited=audit_counts(pd.concat(frames,ignore_index=True))
    audited.to_csv(output/'all-operating-points.csv',index=False)
    primary=audited[np.isclose(audited.alpha,.001,rtol=0,atol=1e-12)]
    chosen=[]
    for key,group in primary.groupby('dataset',sort=False):
        reference=best_record(group,NATIVE)
        names=['V8_adaptive_checked']+([reference['method']] if reference else [])
        chosen.extend(group[group.method.isin(names)].to_dict('records'))
    table=pd.DataFrame(chosen);table.to_csv(output/'candidate-and-reference.csv',index=False)
    manifest=dict(observed_utc=datetime.now(timezone.utc).isoformat(),scope='Original consumed validation cohorts only; no new model fit or final scoring',
        operating_budget=.001,reference_selection='Native reference selected by original mean strict recall key, then mean FPR and AP; no new reference tuning',
        sources=sources,limitations=['Validation selects prefixes and thresholds and has informed model redesign. No independent confirmation.',
            'Empirical validation FPR is not a future FPR guarantee. No selection-adjusted or time-dependence-adjusted confidence intervals are inferred from counts.',
            'Aggregate counts cannot reconstruct paired errors, score calibration, ROC-AUC or new threshold choices.',
            'Dataset-specific acceptable recall and precision require an application cost target. No universal accuracy pass threshold is inferred.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    lines=['# Measured detection at the 0.1% validation false-positive budget','',
        'The current adaptive method does not yet meet a general accuracy or publication claim. The table recomputes descriptive performance from saved validation confusion counts. These sources already informed development.','',
        '| Source | Method | Caught / fraud cases | Missed | False alerts / legitimate cases | Recall | Precision | Overall accuracy | All-legitimate accuracy |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in chosen:
        dataset={'elliptic':'Elliptic','paysim':'PaySim','sparkov':'Sparkov','baf_base':'BAF Base'}.get(r['dataset'],r['dataset'])
        method={'V8_adaptive_checked':'Adaptive candidate','CatBoost_native':'CatBoost','LightGBM_native':'LightGBM'}.get(r['method'],r['method'])
        lines.append(f"| {dataset} | {method} | {int(r['tp'])} / {int(r['fraud_cases'])} | {int(r['fn'])} | {int(r['fp'])} / {int(r['legitimate_cases'])} | {r['recall_recomputed']:.2%} | {r['precision_recomputed']:.2%} | {r['accuracy']:.3%} | {r['all_legitimate_accuracy']:.3%} |")
    lines.extend(['','A model that labels every case legitimate catches no fraud. Its accuracy is included to show why overall accuracy cannot carry the fraud-detection claim. Recall measures the share of fraud caught; precision measures the share of alerts that are fraud. Elliptic is a known-label AML transfer task. The other three sources are synthetic.','',
        '## For whoever builds this','',
        'Reproduce with experiment/accuracy_audit.py. all-operating-points.csv includes every method and all three existing budgets. candidate-and-reference.csv keeps the candidate and the reference chosen using the original selection key. manifest.json contains observation time, source paths, hashes, units and scope.','',*manifest['limitations'],'',
        'Decisions still needed: the research lead must choose and freeze a candidate before confirmation; the application owner must set minimum useful recall and precision at the agreed false-positive budget. Those targets must be fixed before looking at fresh confirmation outcomes.'])
    (output/'accuracy-review.md').write_text('\n'.join(lines)+'\n');return table,manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('output');args=p.parse_args();table,_=run(args.study,args.output)
    print(table[['dataset','method','tp','fn','fp','fraud_cases','legitimate_cases','recall_recomputed','precision_recomputed','accuracy','all_legitimate_accuracy']].to_string(index=False))
