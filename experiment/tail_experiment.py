"""Fixed ablations of the V5 temporal-tail candidate on consumed development data."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from altcboost_v5 import ALTCBoostV5
from evaluation import calibrate,metrics
from multi_benchmark import preprocessing,frozen_models
from multi_data import SPECS,feature_names,load_cohort,find_sources
from tail_objective import temporal_groups,evaluation_windows
from tail_discovery import audit

VARIANTS={
 'V5_temporal_tail':dict(tail_fraction=.01,anchor_weight=.2,group_count=4),
 'V5_pooled_tail':dict(tail_fraction=.01,anchor_weight=.2,group_count=1),
 'V5_temporal_all_negatives':dict(tail_fraction=1.,anchor_weight=.2,group_count=4),
 'V5_balanced_logistic':dict(tail_fraction=1.,anchor_weight=1.,group_count=1)}

def candidate_models(trees=100):
    for name,config in VARIANTS.items():
        parameters={k:v for k,v in config.items() if k!='group_count'}
        yield name,dict(config,trees=trees,leaves=17,temperature=1.,group_temperature=.25),ALTCBoostV5(n_estimators=trees,**parameters)
    for name,config,model in frozen_models():
        if name in {'ALTCBoost_tuned','CatBoost','XGBoost'}:yield name,config,model

def run(key,paths,output,train_cap=150000,eval_cap=100000,models=None,trees=100):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frames,discovery=load_cohort(key,paths,train_cap,eval_cap)
    discovery['holdout_policy']='Consumed development evaluation; no independent confirmation claim'
    (output/'discovery.json').write_text(json.dumps(discovery,indent=2))
    (output/'training-groups.json').write_text(json.dumps(audit(key,frames,discovery),indent=2))
    names=feature_names(key);label=SPECS[key]['label'];processor=preprocessing(key)
    matrices=[processor.fit_transform(frames[0][names])]
    matrices.extend(processor.transform(frame[names]) for frame in frames[1:])
    matrices=[np.ascontiguousarray(x,dtype=np.float64) for x in matrices]
    if any(not np.isfinite(x).all() for x in matrices):raise ValueError('Nonfinite transformed predictors')
    labels=[frame[label].to_numpy(dtype=int) for frame in frames]
    x,xv,xc,xt=matrices;y,yv,yc,yt=labels
    train_groups,edges=temporal_groups(frames[0]['__time'])
    windows,window_edges=evaluation_windows(frames[3]['__time'])
    rows=[];window_rows=[];settings=[]
    for method,config,model in (candidate_models(trees) if models is None else models):
        start=time.perf_counter()
        with threadpool_limits(limits=2):
            if isinstance(model,ALTCBoostV5):
                groups=train_groups if config.get('group_count',4)==4 else np.zeros(len(y),dtype=int)
                model.fit(x,y,xv,yv,groups)
            elif method.startswith('ALTCBoost'):model.fit(x,y,xv,yv)
            else:model.fit(x,y)
        duration=time.perf_counter()-start
        if isinstance(model,ALTCBoostV5):
            pc=model.decision_function(xc);pt=model.decision_function(xt)
            score_type='raw margin; not a calibrated fraud probability'
        else:
            pc=np.asarray(model.predict_proba(xc)[:,1],dtype=float)
            pt=np.asarray(model.predict_proba(xt)[:,1],dtype=float);score_type='model score'
        settings.append(dict(method=method,config=config,fit_s=duration,score_type=score_type,
            selected_iteration=getattr(model,'best_iteration_',None),
            training_loss=getattr(model,'training_loss_',None),
            step_diagnostics=getattr(model,'step_diagnostics_',None)))
        for alpha in [.0005,.001,.01]:
            try:threshold,details=calibrate(yc,pc,alpha)
            except ValueError as error:
                if 'Insufficient calibration negatives' not in str(error):raise
                rows.append(dict(dataset=key,method=method,alpha=alpha,status='unsupported',reason=str(error)));continue
            result=dict(dataset=key,method=method,status='evaluated',threshold=threshold,fit_s=duration,
                **details,**metrics(yt,pt,threshold,alpha))
            result.update(test_positive=int((yt==1).sum()),test_negative=int((yt==0).sum()))
            rows.append(result)
            for window in np.unique(windows):
                mask=windows==window
                window_result=dict(dataset=key,method=method,alpha=alpha,window=int(window),rows=int(mask.sum()))
                if set(np.unique(yt[mask]))=={0,1}:
                    window_result.update(status='evaluated',threshold=threshold,**metrics(yt[mask],pt[mask],threshold,alpha))
                else:window_result.update(status='unsupported',reason='Time window lacks both classes')
                window_rows.append(window_result)
            print(key,method,alpha,'TP',result['tp'],'FP',result['fp'],flush=True)
        pd.DataFrame(rows).to_csv(output/'results.csv',index=False)
        pd.DataFrame(window_rows).to_csv(output/'window-results.csv',index=False)
        (output/'models.json').write_text(json.dumps(settings,indent=2))
    manifest=dict(dataset=key,dataset_kind=SPECS[key]['kind'],source_url=SPECS[key]['source'],
        licence=SPECS[key]['licence'],unit=SPECS[key]['unit'],source_ref=SPECS[key]['ref'],
        scope='Fixed candidate and removal experiments; all dataset holdouts are consumed development evidence',
        model_count=len(settings),train_cap=train_cap,eval_cap=eval_cap,budgets=[.0005,.001,.01],
        train_time_edges=edges.tolist(),test_window_edges=window_edges.tolist(),test_window_count=int(len(np.unique(windows))),
        limitations='No novelty, statistical superiority or future FPR guarantee is established. These are capped synthetic development cohorts. Baselines retain earlier frozen settings and common one-hot preprocessing; no matched tuning or native categorical study. Only V5 checks descent of its explicit empirical surrogate. Score outputs are not calibrated risk probabilities. Window intervals assume independent rows and do not adjust for customer or time dependence.',
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','catboost']})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return rows,window_rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SPECS);p.add_argument('root');p.add_argument('output');args=p.parse_args()
    run(args.dataset,find_sources(args.dataset,args.root),args.output)
