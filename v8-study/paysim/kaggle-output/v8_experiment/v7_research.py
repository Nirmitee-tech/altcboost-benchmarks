"""Mechanism experiments on training/validation only; no later-interval scoring."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from altcboost_v6 import ALTCBoostV6
from altcboost_v7 import ALTCBoostV7
from validation_data import Inputs,SPECS,load_data,sources
from tuned_models import build_model,fit_model,raw_score,select_prefix
from tail_objective import temporal_groups
from detection_objective import selection_key,empirical_cutoff
from evaluation import metrics

MODES=['zero','clipped','shrunk','adaptive_diagonal','adaptive_joint']

def fit_modes(key,train,validation,references=None,trees=100,candidate_factory=None,reference_policy='frozen'):
    if reference_policy not in {'frozen','budget_prefix'}:raise ValueError('Unknown reference policy')
    inputs=Inputs(key,train);label=SPECS[key]['label']
    y=train[label].to_numpy(dtype=int);yv=validation[label].to_numpy(dtype=int)
    x=inputs.transform(train,'V6');xv=inputs.transform(validation,'V6');groups,_=temporal_groups(train['__time'])
    candidates=[('V6_fixed',ALTCBoostV6(n_estimators=trees))]
    candidates.extend(('V7_'+mode,ALTCBoostV7(prior_mode=mode,n_estimators=trees)) for mode in MODES)
    if candidate_factory is not None:candidates=list(candidate_factory(trees))
    rows=[];models=[]
    def record(name,score,metadata):
        key_value=selection_key(yv,score)
        for alpha in [.0005,.001,.01]:
            cutoff=empirical_cutoff(yv,score,alpha)
            rows.append(dict(dataset=key,method=name,alpha=alpha,
                validation_mean_recall=key_value[0],validation_mean_fpr=-key_value[1],
                threshold=cutoff,**metrics(yv,score,cutoff,alpha),**metadata))
        print(key,name,'validation key',key_value,flush=True)
    for name,model in candidates:
        with threadpool_limits(limits=2):
            start=time.perf_counter();model.fit(x,y,xv,yv,groups);duration=time.perf_counter()-start
        score=model.decision_function(xv)
        weight=model.prior_weights_[model.best_iteration_] if hasattr(model,'prior_weights_') else 1.
        metadata=dict(fit_s=duration,selected_prefix=int(model.best_iteration_),selected_prior_weight=float(weight),
            weak_learner_fits=getattr(model,'weak_learner_fits_',None),objective_evaluations=getattr(model,'objective_evaluations_',None))
        record(name,score,metadata)
        models.append(dict(method=name,**metadata,training_loss=model.training_loss_,
            prior_weights=getattr(model,'prior_weights_',None),step_diagnostics=model.step_diagnostics_,
            attempt_diagnostics=getattr(model,'attempt_diagnostics_',None)))
    for family,reference in (references or {}).items():
        model=build_model(family,reference['config'],y,inputs.cats,17,trees)
        with threadpool_limits(limits=2):
            start=time.perf_counter();fit_model(family,model,inputs.transform(train,family),y,inputs.transform(validation,family),yv,groups)
            duration=time.perf_counter()-start
        prefix=int(reference['prefix']) if reference_policy=='frozen' else select_prefix(family,model,inputs.transform(validation,family),yv)[0]
        score=raw_score(family,model,inputs.transform(validation,family),prefix)
        checked=selection_key(yv,score)
        if reference_policy=='frozen' and not np.allclose(checked,reference['validation_key'],rtol=0,atol=1e-12):raise ValueError('Frozen reference validation scores did not reproduce')
        record(family,score,dict(fit_s=duration,selected_prefix=prefix,selected_prior_weight=None))
    return rows,models

def run(key,paths,output,references=None,trees=100,train_cap=150000,eval_cap=100000,expected_cohorts=None,expected_files=None,candidate_factory=None,reference_policy='frozen'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frames,discovery=load_data(key,paths,train_cap,eval_cap)
    if expected_cohorts:
        for name,expected in expected_cohorts.items():
            if discovery['partitions'][name]['cohort_sha256']!=expected:raise ValueError('Research cohort differs from frozen reference')
    if expected_files and {item['name']:item['sha256'] for item in discovery['files']}!=expected_files:
        raise ValueError('Source file hash differs from frozen reference')
    discovery['holdout_status']='Consumed source; this experiment fits and scores training/validation only'
    rows,models=fit_modes(key,frames[0],frames[1],references,trees,candidate_factory,reference_policy)
    pd.DataFrame(rows).to_csv(output/'validation.csv',index=False)
    (output/'models.json').write_text(json.dumps(models,indent=2))
    (output/'discovery.json').write_text(json.dumps(discovery,indent=2))
    manifest=dict(dataset=key,scope='Adaptive development experiment. Fit and score training/validation only; no calibration or final prediction is computed.',
        source_url=SPECS[key]['source'],unit=SPECS[key]['unit'],dataset_kind=SPECS[key]['kind'],
        modes=['V6_fixed',*['V7_'+mode for mode in MODES]],tree_cap=trees,seed=17,
        prior_controls=dict(clipped_mapping='4*tanh(raw_prior/4)',shrunk_coefficient=.25),
        candidate='Box-constrained prior/tree block updates using exact directional curvature of the temporal negative-tail objective',
        limitations='Validation chooses prefixes and is also used to compare these hypotheses; scores are optimistic development evidence. Existing datasets informed the redesign. No novelty, general superiority, independent confirmation, future FPR control or global optimizer claim. Zero-start can encounter nonsmooth tail ties. Fully corrective boosting and variable-projection methods are related prior work.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return rows,models

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SPECS);p.add_argument('root');p.add_argument('output');p.add_argument('--references')
    args=p.parse_args();references=json.loads(Path(args.references).read_text()) if args.references else None
    run(args.dataset,sources(args.dataset,args.root),args.output,references)
