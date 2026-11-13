"""Summarize completed hosted runs without pooling repeated holdouts."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

METHODS={'Logistic','XGBoost','LightGBM','CatBoost','ALTCBoostV2',
         'ALTCBoostV3_original','ALTCBoostV3_aligned'}

def summarize(frame, alpha=.001, expected_seeds=(17,42)):
    selected=frame[np.isclose(frame['alpha'],alpha)].copy()
    if set(selected['seed'])!=set(expected_seeds):
        raise ValueError('Hosted results do not contain all planned seeds')
    if selected.duplicated(['method','seed']).any():
        raise ValueError('Duplicate method and seed results')
    for seed in expected_seeds:
        if set(selected[selected.seed==seed].method)!=METHODS:
            raise ValueError('Hosted results do not contain all planned model families')
    if set(selected.status)!={'evaluated'}:
        raise ValueError('Some operating points were not evaluated')
    selected['test_positive_count']=selected.tp+selected.fn
    selected['test_negative_count']=selected.fp+selected.tn
    if selected.test_positive_count.nunique()!=1 or selected.test_negative_count.nunique()!=1:
        raise ValueError('Holdout sizes differ across repeated runs')
    if not np.allclose(selected.recall,selected.tp/selected.test_positive_count):
        raise ValueError('Recall disagrees with saved counts')
    if not np.allclose(selected.fpr,selected.fp/selected.test_negative_count):
        raise ValueError('False-positive rate disagrees with saved counts')
    summary=[]
    for method, group in selected.groupby('method',sort=False):
        summary.append(dict(method=method,recall_min=float(group.recall.min()),
            recall_max=float(group.recall.max()),tp_min=int(group.tp.min()),tp_max=int(group.tp.max()),
            fp_min=int(group.fp.min()),fp_max=int(group.fp.max()),
            fpr_min=float(group.fpr.min()),fpr_max=float(group.fpr.max()),
            meets_empirical_budget_each_seed=bool((group.fpr<=alpha).all()),
            scoring_p50_us_median=float(group.single_p50_us.median()),
            fit_seconds_median=float(group.fit_s.median())))
    return summary,selected

def write_report(results,discovery,output):
    results,discovery,output=map(Path,[results,discovery,output])
    frame=pd.read_csv(results); provenance=json.loads(discovery.read_text())
    summary,selected=summarize(frame)
    test=provenance['splits']['test']
    output.mkdir(parents=True,exist_ok=True)
    selected.to_csv(output/'operating-point-0.001.csv',index=False)
    evidence=dict(source=str(results.resolve()),discovery_source=str(discovery.resolve()),
        observed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Kaggle-hosted CPU run; fixed temporal holdout; two training seeds; calibration target FPR 0.001',
        unit='transaction row; repeated seeds use the same holdout',models=summary)
    (output/'summary.json').write_text(json.dumps(evidence,indent=2))
    table=['| Model | Fraud caught per seed | False alarms per seed | Recall range |',
           '|---|---:|---:|---:|']
    for row in summary:
        table.append(f"| {row['method']} | {row['tp_min']}-{row['tp_max']} | {row['fp_min']}-{row['fp_max']} | {row['recall_min']:.1%}-{row['recall_max']:.1%} |")
    report=f'''# Kaggle-hosted benchmark results

Use these results to decide what to test on a fresh time period. They do not establish a production choice. The comparison used a private CPU notebook, two training seeds and a 70-tree limit.

The calibration target was a 0.1% false-positive rate. The later test interval had {test['positive']} fraud cases and {test['negative']:,} normal transaction rows. Every model used the same calibration rule. The table shows the range across seeds on that one holdout. Counts are not added across seeds.

{chr(10).join(table)}

Changes in versions, tree budget and hardware mean that latency and quality can differ from the earlier local pilot. Model scoring excludes feature retrieval and network overhead. The data provide no card identities or label-arrival times, so entity independence and realistic label delays remain unverified. A calibration target does not guarantee the same false-positive rate on future traffic.

## For whoever builds this

Skip this section unless you are running or changing the code.

The source is `{results.resolve()}`. Dataset provenance is in `{discovery.resolve()}`. Summary calculations are saved in `summary.json` and `operating-point-0.001.csv` beside this report. Regenerate with `python hosted_summary.py RESULTS_CSV DISCOVERY_JSON OUTPUT_DIRECTORY`.

The notebook runs `tests/test_unit.py` and `tests/test_integration.py` before training. Hosted logs contain the actual pass counts. The separately tested CLI packaging code runs locally and is not part of hosted model execution.

Decisions still needed: Jitendra must choose alert capacity and the operational false-positive budget. The fraud-data owner must provide a later holdout with identities and label-arrival information before a deployment assessment. Any redesign based on this test interval needs another final holdout.
'''
    (output/'review.md').write_text(report)
    return evidence

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results');p.add_argument('discovery');p.add_argument('output')
    args=p.parse_args();print(json.dumps(write_report(args.results,args.discovery,args.output),indent=2))
