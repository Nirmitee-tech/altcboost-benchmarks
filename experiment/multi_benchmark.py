"""Frozen-configuration evaluation across separate synthetic fraud sources."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from evaluation import calibrate,metrics
from multi_data import SPECS,PARTS,feature_names,find_sources,load_cohort

def preprocessing(key):
    spec=SPECS[key]
    numeric=spec['numeric']+(['hour_of_day','day_of_week'] if key=='sparkov' else [])
    transformer=ColumnTransformer([
        ('numeric',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),numeric),
        ('category',Pipeline([
            ('missing',SimpleImputer(strategy='constant',fill_value='missing')),
            ('encode',OneHotEncoder(handle_unknown='infrequent_if_exist',max_categories=32,
                                   min_frequency=100,sparse_output=False,dtype=np.float64))]),spec['categorical'])
    ],remainder='drop',sparse_threshold=0.)
    return Pipeline([('prepare',transformer),('remove_train_constants',VarianceThreshold())])

def frozen_models():
    from altcboost_v3 import ALTCBoostV3
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    yield 'ALTCBoost_incumbent',dict(trees=70,leaves=9,hard_negative_strength=2.5),ALTCBoostV3(
        n_estimators=70,max_leaf_nodes=9,min_samples_leaf=40,use_cross=False,random_state=17)
    yield 'ALTCBoost_tuned',dict(trees=100,leaves=17,hard_negative_strength=2.5,validation_max_fpr=.001),ALTCBoostV3(
        n_estimators=100,max_leaf_nodes=17,min_samples_leaf=40,use_cross=False,
        low_fpr_validation=.001,hard_negative_strength=2.5,random_state=17)
    yield 'CatBoost',dict(trees=100,depth=6,positive_weight=5.),CatBoostClassifier(
        iterations=100,depth=6,scale_pos_weight=5.,learning_rate=.06,random_seed=17,
        thread_count=2,verbose=False,allow_writing_files=False)
    yield 'XGBoost',dict(trees=100,depth=7,positive_weight=10.),XGBClassifier(
        n_estimators=100,max_depth=7,scale_pos_weight=10.,learning_rate=.06,
        tree_method='hist',random_state=17,n_jobs=2)
    yield 'Logistic',dict(max_iter=500),make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))

def run(key,paths,output,train_cap=150000,eval_cap=100000,models=None,chunksize=100000):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frames,discovery=load_cohort(key,paths,train_cap,eval_cap,chunksize=chunksize)
    (output/'discovery.json').write_text(json.dumps(discovery,indent=2))
    names=feature_names(key);label=SPECS[key]['label']
    processor=preprocessing(key)
    matrices=[processor.fit_transform(frames[0][names])]
    matrices.extend(processor.transform(frame[names]) for frame in frames[1:])
    matrices=[np.ascontiguousarray(matrix,dtype=np.float64) for matrix in matrices]
    if any(not np.isfinite(matrix).all() for matrix in matrices):raise ValueError('Preprocessing produced nonfinite features')
    labels=[frame[label].to_numpy(dtype=np.int64) for frame in frames]
    xtrain,xval,xcal,xtest=matrices;ytrain,yval,ycal,ytest=labels
    (output/'preprocessing.json').write_text(json.dumps(dict(fitted_on='sampled training rows only',
        predictor_allowlist=names,encoded_features=xtrain.shape[1],
        common_pipeline='training-median numeric imputation plus indicators; bounded one-hot categories; drop training constants',
        native_categorical_baselines=False),indent=2))
    rows=[];configuration=[]
    for method,config,model in (frozen_models() if models is None else models):
        started=time.perf_counter()
        with threadpool_limits(limits=2):
            if method.startswith('ALTCBoost'):model.fit(xtrain,ytrain,xval,yval)
            else:model.fit(xtrain,ytrain)
        duration=time.perf_counter()-started
        pc=np.asarray(model.predict_proba(xcal)[:,1],dtype=np.float64)
        pe=np.asarray(model.predict_proba(xtest)[:,1],dtype=np.float64)
        configuration.append(dict(method=method,config=config,fit_s=duration,
            selected_iteration=getattr(model,'best_iteration_',None)))
        for alpha in [.0005,.001,.01]:
            try:threshold,details=calibrate(ycal,pc,alpha)
            except ValueError as error:
                if 'Insufficient calibration negatives' not in str(error):raise
                rows.append(dict(dataset=key,method=method,alpha=alpha,status='unsupported',reason=str(error)))
                continue
            row=dict(dataset=key,dataset_kind=SPECS[key]['kind'],unit=SPECS[key]['unit'],
                method=method,status='evaluated',fit_s=duration,threshold=threshold,
                **details,**metrics(ytest,pe,threshold,alpha))
            row['test_positive']=int((ytest==1).sum());row['test_negative']=int((ytest==0).sum())
            rows.append(row)
            print(key,method,alpha,'TP',row['tp'],'FP',row['fp'],'recall',round(row['recall'],4),flush=True)
        pd.DataFrame(rows).to_csv(output/'results.csv',index=False)
        (output/'models.json').write_text(json.dumps(configuration,indent=2))
    manifest=dict(dataset=key,source_ref=SPECS[key]['ref'],dataset_kind=SPECS[key]['kind'],
        source_url=SPECS[key]['source'],licence=SPECS[key]['licence'],unit=SPECS[key]['unit'],
        scope='New synthetic-dataset evaluation using frozen ULB-selected hyperparameters; no per-dataset model search',
        common_cohort=True,common_preprocessing=True,train_cap=train_cap,eval_cap=eval_cap,
        budgets=[.0005,.001,.01],model_count=len(configuration),
        limitations='Capped label-blind cohorts, not full-data training. Same transformed features for all models; native categorical optimizations not assessed. ALTC uses validation early stopping; other models use fixed tree counts. Sampling seeds are not independent datasets. Temporal and entity dependence can violate calibration assumptions. No production or real-banking validation claim.',
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','numba','xgboost','catboost']})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return rows,discovery

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=SPECS);parser.add_argument('root')
    parser.add_argument('output');parser.add_argument('--train-cap',type=int,default=150000)
    parser.add_argument('--eval-cap',type=int,default=100000);args=parser.parse_args()
    run(args.dataset,find_sources(args.dataset,args.root),args.output,args.train_cap,args.eval_cap)
