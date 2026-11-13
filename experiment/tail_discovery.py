"""Rerunnable training-group audit; exports aggregate counts only."""
import argparse
import json
from pathlib import Path
import numpy as np
from multi_data import SPECS,load_cohort,find_sources
from tail_objective import temporal_groups

def audit(key,frames,discovery):
    train=frames[0];groups,edges=temporal_groups(train['__time'])
    stats=[]
    for group in np.unique(groups):
        rows=groups==group;positive=int(train.loc[rows,SPECS[key]['label']].sum())
        stats.append(dict(group=int(group),rows=int(rows.sum()),positive=positive,negative=int(rows.sum())-positive))
    if any(not row['positive'] or not row['negative'] for row in stats):
        raise ValueError('Proposed training groups require both classes')
    return dict(dataset=key,observed_utc=discovery['observed_utc'],source_files=discovery['files'],
        train_cohort=discovery['partitions']['train'],time_edges=edges.tolist(),groups=stats,
        hypothesis='Each label-blind training time group contains both classes',verified=True,
        scope='Sampled training rows only; no identifier values or row exports')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');args=p.parse_args()
    records=[]
    for key in SPECS:
        frames,discovery=load_cohort(key,find_sources(key,args.root))
        records.append(audit(key,frames,discovery))
    Path(args.output).write_text(json.dumps(records,indent=2))
