"""Validation-only screening of the revised loss; never score final intervals."""
import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
import numpy as np
from altcboost_v6 import ALTCBoostV6
from multi_data import load_cohort,find_sources,feature_names,SPECS
from multi_benchmark import preprocessing,frozen_models
from tail_objective import temporal_groups
from detection_objective import empirical_cutoff,selection_key
from evaluation import metrics

def screening_models(trees,groups):
    variants=[('V6_temporal_tail',ALTCBoostV6(n_estimators=trees),groups),
              ('V6_balanced_logistic',ALTCBoostV6(n_estimators=trees,anchor_weight=1.),np.zeros(len(groups),dtype=int))]
    variants.extend((name,model,None) for name,config,model in frozen_models() if name in {'ALTCBoost_tuned','CatBoost','XGBoost'})
    return variants

def run(key,root,output,trees=100):
    frames,discovery=load_cohort(key,find_sources(key,root))
    names=feature_names(key);label=SPECS[key]['label'];processor=preprocessing(key)
    x=processor.fit_transform(frames[0][names]);xv=processor.transform(frames[1][names])
    y=frames[0][label].to_numpy();yv=frames[1][label].to_numpy();groups,_=temporal_groups(frames[0]['__time'])
    variants=screening_models(trees,groups)
    rows=[]
    for name,model,g in variants:
        with threadpool_limits(limits=2):
            if g is not None:model.fit(x,y,xv,yv,g)
            elif name.startswith('ALTCBoost'):model.fit(x,y,xv,yv)
            else:model.fit(x,y)
        score=model.decision_function(xv) if hasattr(model,'decision_function') else model.predict_proba(xv)[:,1]
        row=dict(method=name,selection_key=selection_key(yv,score),selected_iteration=getattr(model,'best_iteration_',None),metrics=[])
        for alpha in [.0005,.001,.01]:
            cutoff=empirical_cutoff(yv,score,alpha)
            row['metrics'].append(dict(alpha=alpha,**metrics(yv,score,cutoff,alpha)))
        rows.append(row);print(name,row['selection_key'],flush=True)
        Path(output).write_text(json.dumps(dict(dataset=key,scope='Validation-only adaptive development screen. No final interval scores computed.',train_cohort=discovery['partitions']['train'],validation_cohort=discovery['partitions']['validation'],rows=rows),indent=2))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SPECS);p.add_argument('root');p.add_argument('output');args=p.parse_args()
    run(args.dataset,args.root,args.output)
