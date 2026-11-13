"""Four prespecified configurations per baseline; validation-only prefix selection."""
import time
import numpy as np
from altcboost_v6 import ALTCBoostV6
from detection_objective import selection_key

GRIDS={
 'CatBoost_native':[dict(depth=4,positive_weight=1.),dict(depth=6,positive_weight=5.),dict(depth=6,positive_weight='train_ratio'),dict(depth=8,positive_weight=10.)],
 'XGBoost':[dict(depth=4,positive_weight=1.),dict(depth=7,positive_weight=10.),dict(depth=5,positive_weight='train_ratio'),dict(depth=9,positive_weight=5.)],
 'LightGBM_native':[dict(leaves=15,positive_weight=1.),dict(leaves=31,positive_weight=10.),dict(leaves=31,positive_weight='train_ratio'),dict(leaves=63,positive_weight=5.)]}

def positive_weight(config,y):
    y=np.asarray(y)
    return float(np.sum(y==0)/np.sum(y==1)) if config['positive_weight']=='train_ratio' else float(config['positive_weight'])

def build_model(family,config,y,categories,seed=17,trees=100):
    if family=='V6':return ALTCBoostV6(n_estimators=trees,random_state=seed)
    weight=positive_weight(config,y)
    if family=='CatBoost_native':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=trees,depth=config['depth'],scale_pos_weight=weight,
            learning_rate=.06,random_seed=seed,cat_features=categories,thread_count=2,
            bootstrap_type='Bayesian',verbose=False,allow_writing_files=False)
    if family=='XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=trees,max_depth=config['depth'],scale_pos_weight=weight,
            learning_rate=.06,random_state=seed,n_jobs=2,tree_method='hist',subsample=.9,colsample_bytree=.9)
    if family=='LightGBM_native':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=trees,num_leaves=config['leaves'],scale_pos_weight=weight,
            learning_rate=.06,random_state=seed,n_jobs=2,verbosity=-1,subsample=.9,subsample_freq=1,
            colsample_bytree=.9,deterministic=True,force_col_wise=True)
    raise ValueError('Unknown model family')

def fit_model(family,model,x,y,xv,yv,groups):
    if family=='V6':model.fit(x,y,xv,yv,groups)
    else:model.fit(x,y)
    return model

def raw_score(family,model,data,prefix):
    if family not in {'V6',*GRIDS}:raise ValueError('Unknown model family')
    if family=='V6':
        if prefix!=model.best_iteration_:raise ValueError('V6 prefix must match its frozen validation selection')
        return np.asarray(model.decision_function(data),dtype=float)
    if prefix==0:return np.zeros(len(data),dtype=float)
    if family=='CatBoost_native':return np.asarray(model.predict(data,prediction_type='RawFormulaVal',ntree_end=prefix,thread_count=2),dtype=float)
    if family=='XGBoost':
        from xgboost import DMatrix
        return np.asarray(model.get_booster().predict(DMatrix(data,nthread=2),output_margin=True,iteration_range=(0,prefix)),dtype=float)
    if family=='LightGBM_native':return np.asarray(model.booster_.predict(data,raw_score=True,num_iteration=prefix,num_threads=2),dtype=float)
    raise ValueError('Unknown model family')

def staged_scores(family,model,data):
    if family=='CatBoost_native':
        # Use reference prefix calls so staged summation cannot change tied-score ranks.
        for i in range(1,model.tree_count_+1):yield i,raw_score(family,model,data,i)
    elif family=='XGBoost':
        from xgboost import DMatrix
        matrix=DMatrix(data,nthread=2);booster=model.get_booster()
        for i in range(1,booster.num_boosted_rounds()+1):
            yield i,np.asarray(booster.predict(matrix,output_margin=True,iteration_range=(0,i)),dtype=float)
    elif family=='LightGBM_native':
        # Direct prefix prediction avoids introducing a separate summation implementation.
        for i in range(1,model.booster_.current_iteration()+1):yield i,raw_score(family,model,data,i)
    else:raise ValueError('Unsupported staged model family')

def select_prefix(family,model,data,y):
    if family=='V6':return int(model.best_iteration_),selection_key(y,raw_score(family,model,data,model.best_iteration_))
    best=selection_key(y,np.zeros(len(y)));prefix=0
    for count,score in staged_scores(family,model,data):
        key=selection_key(y,score)
        if key>best:best=key;prefix=count
    checked=selection_key(y,raw_score(family,model,data,prefix))
    if not np.allclose(checked,best,rtol=0,atol=1e-12):
        raise ValueError('Staged validation selection differs from direct prefix prediction')
    return prefix,checked

def tune_baselines(inputs,train,validation,label,groups,seed=17,trees=100):
    """Only training and validation frames enter this selection function."""
    y=train[label].to_numpy(dtype=int);yv=validation[label].to_numpy(dtype=int)
    chosen={};trials=[]
    for family,grid in GRIDS.items():
        x=inputs.transform(train,family);xv=inputs.transform(validation,family);best=None
        for trial,config in enumerate(grid):
            model=build_model(family,config,y,inputs.cats,seed,trees)
            start=time.perf_counter();fit_model(family,model,x,y,xv,yv,groups);fit_s=time.perf_counter()-start
            start=time.perf_counter();prefix,key=select_prefix(family,model,xv,yv);selection_s=time.perf_counter()-start
            record=dict(family=family,trial=trial,config=config,resolved_positive_weight=positive_weight(config,y),
                prefix=prefix,validation_key=list(key),fit_s=fit_s,selection_s=selection_s)
            trials.append(record)
            if best is None or key>best['key']:best=dict(model=model,config=config,prefix=prefix,key=key,record=record)
        chosen[family]=best
        print('Validation selected',family,best['config'],'prefix',best['prefix'],flush=True)
    return chosen,trials
