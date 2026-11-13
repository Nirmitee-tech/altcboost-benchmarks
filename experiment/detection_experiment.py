"""Matched-count development search; all final ULB results are exploratory."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from discovery import FEATURES,load_data
from detection_objective import selection_key
from evaluation import calibrate,metrics

def candidate_models(trees=100):
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from altcboost_v3 import ALTCBoostV3
    from altcboost_v4 import ALTCBoostV4
    for clip in [6.,None]:
        for linear in [True,False]:
            config=dict(feature_clip=clip,linear_prior=linear)
            yield 'ALTCBoostV4',config,ALTCBoostV4(n_estimators=trees,**config)
    for strength in [0.,2.5]:
        for leaves in [9,17]:
            config=dict(hard_negative_strength=strength,max_leaf_nodes=leaves)
            yield 'ALTCBoostV3',config,ALTCBoostV3(n_estimators=trees,use_cross=False,
                min_samples_leaf=40,low_fpr_validation=.001,random_state=17,**config)
    for depth,weight in [(3,1.),(5,1.),(6,5.),(7,10.)]:
        config=dict(depth=depth,scale_pos_weight=weight)
        yield 'CatBoost',config,CatBoostClassifier(iterations=trees,learning_rate=.06,
            random_seed=17,thread_count=2,verbose=False,allow_writing_files=False,**config)
        yield 'XGBoost',config,XGBClassifier(n_estimators=trees,max_depth=depth,
            scale_pos_weight=weight,learning_rate=.06,n_jobs=2,tree_method='hist',random_state=17)

class Blend:
    def __init__(self,first,second,weight):
        self.first,self.second,self.weight=first,second,weight
    def predict_proba(self,x):
        return self.weight*self.first.predict_proba(x)+(1-self.weight)*self.second.predict_proba(x)

def run(csv,output,trees=100,training_limit=None,include_final=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frame,parts,provenance=load_data(csv)
    provenance['holdout_policy']='ULB is development-only because earlier final results informed this work. Independent IEEE-CIS validation is a separate pending gate.'
    (output/'discovery.json').write_text(json.dumps(provenance,indent=2))
    arrays=[(frame.iloc[i][FEATURES].to_numpy(),frame.iloc[i].Class.to_numpy()) for i in parts]
    (xt,yt),(xv,yv),(xc,yc),(xe,ye)=arrays
    if training_limit is not None:xt,yt=xt[:training_limit],yt[:training_limit]
    trials=[];best={}
    for family,config,model in candidate_models(trees):
        started=time.perf_counter()
        with threadpool_limits(limits=2):
            if family.startswith('ALTC'):model.fit(xt,yt,xv,yv)
            else:model.fit(xt,yt)
        duration=time.perf_counter()-started
        key=selection_key(yv,model.predict_proba(xv)[:,1])
        trial=dict(family=family,config=json.dumps(config),validation_recall_key=key[0],
                   validation_negative_fpr_key=key[1],validation_ap=key[2],fit_s=duration)
        if family.startswith('ALTC'):
            trial['selected_iteration']=model.best_iteration_
            trial['attempted_iterations']=len(model.steps_)
            prior=model._prior(model._features(xv)) if family=='ALTCBoostV4' else model.linear_.decision_function(model._features(xv))
            trial['prior_validation_key']=json.dumps(selection_key(yv,prior))
        trials.append(trial)
        if family not in best or key>best[family][0]:best[family]=(key,config,model)
        pd.DataFrame(trials).to_csv(output/'trials.csv',index=False)
        print('Development trial',family,config,'key',key,flush=True)
    # Blends are an additional, separately labelled search after individual-family selection.
    for weight in [.25,.5,.75]:
        model=Blend(best['ALTCBoostV4'][2],best['CatBoost'][2],weight)
        key=selection_key(yv,model.predict_proba(xv)[:,1]);family='ALTCV4_CatBoost_blend'
        config=dict(altc_weight=weight,first=best['ALTCBoostV4'][1],second=best['CatBoost'][1])
        trials.append(dict(family=family,config=json.dumps(config),validation_recall_key=key[0],
            validation_negative_fpr_key=key[1],validation_ap=key[2],fit_s=0.))
        if family not in best or key>best[family][0]:best[family]=(key,config,model)
    pd.DataFrame(trials).to_csv(output/'trials.csv',index=False)
    # Freeze selection before accessing calibration or the previously consumed final interval.
    plan={name:dict(config=config,validation_key=key) for name,(key,config,_) in best.items()}
    (output/'selected.json').write_text(json.dumps(plan,indent=2))
    champion=max(best,key=lambda name:best[name][0])
    (output/'champion.json').write_text(json.dumps(dict(method=champion,**plan[champion]),indent=2))
    rows=[]
    if include_final:
        for family,(_,config,model) in best.items():
            pc=model.predict_proba(xc)[:,1];pe=model.predict_proba(xe)[:,1]
            for alpha in [.0005,.001]:
                try:cut,details=calibrate(yc,pc,alpha)
                except ValueError as error:
                    if 'Insufficient calibration negatives' not in str(error):raise
                    rows.append(dict(method=family,alpha=alpha,status='unsupported',reason=str(error)));continue
                row=dict(method=family,config=json.dumps(config),status='exploratory',threshold=cut,
                         **details,**metrics(ye,pe,cut,alpha))
                rows.append(row);print('Exploratory result',family,alpha,'TP',row['tp'],'FP',row['fp'],flush=True)
        pd.DataFrame(rows).to_csv(output/'results.csv',index=False)
    manifest=dict(scope='ULB development search; no independent performance claim',unit='transaction row',
        objective='Mean validation fraud recall at empirical FPR 0.0005 and 0.001; ties favour fewer false positives then AP',
        trials_per_single_family=4,additional_blend_trials=3,max_trees=trees,training_limit=training_limit,
        budgets=[.0005,.001],independent_evaluation='pending IEEE-CIS access and discovery',
        chosen_before_final_scoring=champion,
        hypotheses=['Unclipped features improve low-FPR ranking','Regularized Newton leaf sums improve rare-fraud corrections',
                    'A linear prior outperforms a constant prior','A separately labelled ensemble improves the detection tradeoff'],
        limitations='Equal configuration counts and tree caps do not establish equal wall-clock tuning effort. ALTC uses early stopping; library baselines use the complete tree budget. Speed preservation is unverified for V4.',
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','xgboost','catboost']})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return rows,plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--output',default='quality_results')
    p.add_argument('--trees',type=int,default=100);p.add_argument('--training-limit',type=int)
    args=p.parse_args();run(args.csv,args.output,args.trees,args.training_limit)
