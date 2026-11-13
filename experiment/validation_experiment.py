"""Frozen V6 versus equally sized baseline searches, repeats and paired cluster checks."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from validation_data import SPECS,Inputs,load_data,sources
from tuned_models import GRIDS,build_model,fit_model,select_prefix,raw_score,tune_baselines
from tail_objective import temporal_groups,evaluation_windows
from evaluation import calibrate,metrics
from cluster_uncertainty import cluster_schemes,paired_cluster_interval

def run(key,paths,output,train_cap=150000,eval_cap=100000,seeds=(17,42,73),trees=100,replicates=2000):
    if not seeds or seeds[0]!=17 or len(set(seeds))!=len(seeds):raise ValueError('Unique seeds starting at primary seed 17 are required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frames,discovery=load_data(key,paths,train_cap,eval_cap)
    (output/'discovery.json').write_text(json.dumps(discovery,indent=2))
    label=SPECS[key]['label'];train,validation,calibration,test=frames
    inputs=Inputs(key,train);y=train[label].to_numpy(dtype=int);yv=validation[label].to_numpy(dtype=int)
    groups,_=temporal_groups(train['__time'])
    with threadpool_limits(limits=2):
        chosen,trials=tune_baselines(inputs,train,validation,label,groups,seed=17,trees=trees)
        candidate=build_model('V6',{},y,inputs.cats,17,trees)
        start=time.perf_counter()
        fit_model('V6',candidate,inputs.transform(train,'V6'),y,inputs.transform(validation,'V6'),yv,groups)
        duration=time.perf_counter()-start
        prefix,validation_key=select_prefix('V6',candidate,inputs.transform(validation,'V6'),yv)
    chosen['V6']=dict(model=candidate,config={},prefix=prefix,key=validation_key,
        record=dict(family='V6',config='Frozen V6 settings; no hyperparameter search',prefix=prefix,validation_key=list(validation_key),fit_s=duration,selection_s=0.))
    selection=dict(primary_seed=17,scope='Training and validation only; frozen before calibration/test scoring',
        selected={family:value['record'] for family,value in chosen.items()},
        training_cohort=discovery['partitions']['train']['cohort_sha256'],validation_cohort=discovery['partitions']['validation']['cohort_sha256'])
    text=json.dumps(selection,indent=2);(output/'selection.json').write_text(text)
    pd.DataFrame(trials).to_csv(output/'trials.csv',index=False)
    # Future labels are used for evaluation only after the selection record is written.
    yc=calibration[label].to_numpy(dtype=int);yt=test[label].to_numpy(dtype=int)
    windows,_=evaluation_windows(test['__time'])
    rows=[];window_rows=[];model_records=[];primary_flags={};uncertainty=[]
    for seed in seeds:
        for family in ['V6',*GRIDS]:
            selection_record=chosen[family];config=selection_record['config']
            if seed==17:
                model=selection_record['model'];prefix=selection_record['prefix']
                fit_s=selection_record['record']['fit_s'];selection_s=selection_record['record']['selection_s']
            else:
                model=build_model(family,config,y,inputs.cats,seed,trees)
                with threadpool_limits(limits=2):
                    start=time.perf_counter();fit_model(family,model,inputs.transform(train,family),y,inputs.transform(validation,family),yv,groups)
                    fit_s=time.perf_counter()-start
                    start=time.perf_counter();prefix,_=select_prefix(family,model,inputs.transform(validation,family),yv)
                    selection_s=time.perf_counter()-start
            pc=raw_score(family,model,inputs.transform(calibration,family),prefix)
            pt=raw_score(family,model,inputs.transform(test,family),prefix)
            if not np.isfinite(pc).all() or not np.isfinite(pt).all():raise ValueError('Nonfinite evaluation scores')
            loss=getattr(model,'training_loss_',None)
            model_records.append(dict(family=family,seed=seed,config=config,prefix=int(prefix),fit_s=fit_s,
                prefix_selection_s=selection_s,training_loss=loss))
            for alpha in [.0005,.001,.01]:
                try:cutoff,details=calibrate(yc,pc,alpha)
                except ValueError as error:
                    if 'Insufficient calibration negatives' not in str(error):raise
                    rows.append(dict(dataset=key,method=family,seed=seed,alpha=alpha,status='unsupported',reason=str(error)));continue
                row=dict(dataset=key,method=family,seed=seed,status='evaluated',threshold=cutoff,
                    selected_prefix=int(prefix),fit_s=fit_s,**details,**metrics(yt,pt,cutoff,alpha))
                row.update(test_positive=int((yt==1).sum()),test_negative=int((yt==0).sum()))
                rows.append(row)
                if seed==17:primary_flags[(family,alpha)]=pt>=cutoff
                for window in np.unique(windows):
                    mask=windows==window;part=dict(dataset=key,method=family,seed=seed,alpha=alpha,window=int(window),rows=int(mask.sum()))
                    if set(np.unique(yt[mask]))=={0,1}:part.update(status='evaluated',**metrics(yt[mask],pt[mask],cutoff,alpha))
                    else:part.update(status='unsupported',reason='Window lacks both classes')
                    window_rows.append(part)
                print(key,seed,family,alpha,'TP',row['tp'],'FP',row['fp'],flush=True)
            pd.DataFrame(rows).to_csv(output/'results.csv',index=False)
            pd.DataFrame(window_rows).to_csv(output/'window-results.csv',index=False)
            (output/'models.json').write_text(json.dumps(model_records,indent=2))
    for alpha in [.0005,.001,.01]:
        for family in GRIDS:
            if ('V6',alpha) not in primary_flags or (family,alpha) not in primary_flags:continue
            for scheme,clusters in cluster_schemes(key,test).items():
                result=paired_cluster_interval(yt,primary_flags[('V6',alpha)],primary_flags[(family,alpha)],clusters,replicates=replicates)
                uncertainty.append(dict(dataset=key,reference=family,alpha=alpha,cluster_scheme=scheme,**result))
    (output/'uncertainty.json').write_text(json.dumps(uncertainty,indent=2))
    spec=SPECS[key]
    manifest=dict(dataset=key,dataset_kind=spec['kind'],unit=spec['unit'],source_url=spec['source'],licence=spec['licence'],
        completed_utc=datetime.now(timezone.utc).isoformat(),holdout_status=discovery['holdout_status'],
        frozen_v6=True,baseline_trials_per_family={family:len(grid) for family,grid in GRIDS.items()},trees=trees,seeds=list(seeds),
        selection_sha256=hashlib.sha256(text.encode()).hexdigest(),
        preprocessing='Training-fitted numeric imputation and one-hot inputs for V6/XGBoost; native categories and missing-value handling for CatBoost and LightGBM',
        selection='Baseline configuration search on seed 17 validation only; all baseline prefixes scored with same selection_key as V6. Repeat seeds freeze selected hyperparameters and reselect prefix on validation.',
        limitations='Equal baseline configuration counts and tree caps, not equal wall time or model capacity. Small prespecified search, not exhaustive tuning. Same cohorts across seeds; stochastic refits are not independent datasets. Cluster intervals condition on selected models and fixed calibration cutoffs, omit training/calibration uncertainty and multiple-comparison correction, and do not handle arbitrary dependence. Minimum 20 clusters is a screening rule, not a theorem. Row-wise beta intervals in results assume independent rows and do not replace unsupported cluster checks. No superiority, novelty, point-in-time feature availability or production FPR guarantee.',
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','scipy','numba','catboost','xgboost','lightgbm']})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return rows,selection

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SPECS);p.add_argument('root');p.add_argument('output');args=p.parse_args()
    run(args.dataset,sources(args.dataset,args.root),args.output)
