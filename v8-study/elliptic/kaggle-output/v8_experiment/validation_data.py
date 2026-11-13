"""Shared development cohorts and a separately labelled real Bitcoin AML transfer test."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from multi_data import SPECS as ORIGINAL,PARTS,load_cohort,find_sources,feature_names,file_sha256,priorities
from multi_benchmark import preprocessing
from tail_objective import temporal_groups
from cluster_uncertainty import cluster_schemes

SPECS=dict(ORIGINAL)
SPECS['elliptic']=dict(ref='ellipticco/elliptic-data-set',files=['elliptic_txs_features.csv','elliptic_txs_classes.csv'],
    numeric=[f'f_{i:03d}' for i in range(2,95)],categorical=[],label='is_illicit',
    kind='real anonymized Bitcoin transactions, known-label subset only',unit='known-label Bitcoin transaction',
    licence='CC-BY-NC-ND-4.0',source='https://arxiv.org/abs/1908.02591',
    excluded_reason='Transaction IDs, time and the final 72 backward/forward-neighbour features are excluded. Unknown labels are excluded, not treated as normal. Individual local-feature availability cannot be audited from anonymized columns.')

def predictors(key):return feature_names(key) if key!='elliptic' else SPECS[key]['numeric']

def processor(key):
    if key!='elliptic':return preprocessing(key)
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),VarianceThreshold())

def sources(key,root):
    if key!='elliptic':return find_sources(key,root)
    result=[]
    for name in SPECS[key]['files']:
        matches=list(Path(root).rglob(name))
        if len(matches)!=1:raise ValueError(f'Expected exactly one {name}')
        result.append(matches[0])
    return result

def load_elliptic(paths,train_cap=150000,eval_cap=100000,chunksize=50000):
    if train_cap<1 or eval_cap<1:raise ValueError('Positive cohort caps required')
    feature_path,label_path=map(Path,paths)
    width=pd.read_csv(feature_path,header=None,nrows=1).shape[1]
    if width!=167:raise ValueError('Expected ID plus the published 166 Elliptic features')
    labels=pd.read_csv(label_path,dtype={'txId':str,'class':str})
    if list(labels.columns)!=['txId','class'] or labels.txId.isna().any() or labels.txId.duplicated().any():
        raise ValueError('Expected unique present transaction IDs in labels')
    if not set(labels['class'].unique()).issubset({'1','2','unknown'}):raise ValueError('Unexpected Elliptic label state')
    chunks=[]
    for frame in pd.read_csv(feature_path,header=None,usecols=list(range(95)),dtype={0:str},chunksize=chunksize):
        frame.columns=['txId','__time',*SPECS['elliptic']['numeric']]
        chunks.append(frame)
    full=pd.concat(chunks,ignore_index=True)
    if full.txId.isna().any() or full.txId.duplicated().any():raise ValueError('Expected unique feature transaction IDs')
    if set(full.txId)!=set(labels.txId):raise ValueError('Feature and label transaction IDs differ')
    full=full.merge(labels,on='txId',validate='one_to_one',sort=False)
    full['__row']=np.arange(len(full),dtype=np.int64)
    times=pd.to_numeric(full['__time'],errors='raise').to_numpy(dtype=float)
    if not np.isfinite(times).all() or not np.equal(times,np.floor(times)).all():raise ValueError('Expected finite integer time steps')
    full['__time']=times.astype(np.int64)
    numeric=full[SPECS['elliptic']['numeric']].to_numpy(dtype=float)
    if np.isinf(numeric).any():raise ValueError('Infinite local features are unsupported')
    unique=np.unique(times)
    if len(unique)<8:raise ValueError('Insufficient time groups')
    edges=np.array([unique[int(len(unique)*q)] for q in [.55,.70,.85]])
    label_counts={key:int(value) for key,value in full['class'].value_counts().items()}
    known=full[full['class']!='unknown'].copy();known['is_illicit']=(known['class']=='1').astype(int)
    known['__priority']=priorities(known['__row'],20261112)
    ids=np.searchsorted(edges,known['__time'],side='right')
    frames=[];partitions={}
    for i,name in enumerate(PARTS):
        source=known[ids==i];cap=train_cap if i==0 else eval_cap
        selected=source.nsmallest(cap,['__priority','__row']).sort_values('__row').reset_index(drop=True)
        if set(selected.is_illicit.unique())!={0,1}:raise ValueError(f'Both known classes required in {name}')
        frames.append(selected)
        partitions[name]=dict(source_rows=len(source),source_positive=int(source.is_illicit.sum()),
            sampled_rows=len(selected),sampled_positive=int(selected.is_illicit.sum()),sampled_negative=int((selected.is_illicit==0).sum()),
            time_min=int(selected.__time.min()),time_max=int(selected.__time.max()),
            cohort_sha256=hashlib.sha256(selected['__row'].to_numpy(dtype=np.int64).tobytes()).hexdigest())
    for a,b in zip(frames,frames[1:]):
        if a.__time.max()>=b.__time.min():raise ValueError('Temporal partitions overlap')
    spec=SPECS['elliptic']
    report=dict(dataset='elliptic',dataset_ref=spec['ref'],dataset_kind=spec['kind'],unit=spec['unit'],
        source_url=spec['source'],licence=spec['licence'],observed_utc=datetime.now(timezone.utc).isoformat(),
        files=[dict(name=p.name,sha256=file_sha256(p)) for p in [feature_path,label_path]],source_rows=len(full),
        source_label_counts=label_counts,known_label_rows=len(known),unknown_label_rows=label_counts.get('unknown',0),
        predictor_allowlist=predictors('elliptic'),raw_feature_columns_read=list(range(95)),
        time_edges=edges.tolist(),time_unit='published time step',partitions=partitions,
        sampling=dict(seed=20261112,train_cap=train_cap,eval_cap=eval_cap,method='row-only stable hash priorities'),
        excluded_reason=spec['excluded_reason'],source_missing_predictor_values={c:int(full[c].isna().sum()) for c in predictors('elliptic')},
        limitations='Real AML transfer task, not independent payment-fraud confirmation. Evaluation is conditional on known labels. Local anonymous feature semantics and provider normalization cannot establish point-in-time availability. Graph edges and forward-neighbour features are not used. No raw row export.',
        hypotheses=['IDs join one-to-one without missing or duplicate keys','All four known-label partitions contain both classes','Source timestamp groups do not cross partitions','Only the published local feature positions excluding time enter predictors'])
    return frames,report

def load_data(key,paths,train_cap=150000,eval_cap=100000,chunksize=50000):
    frames,report=load_elliptic(paths,train_cap,eval_cap,chunksize) if key=='elliptic' else load_cohort(key,paths,train_cap,eval_cap,chunksize=chunksize)
    groups,edges=temporal_groups(frames[0]['__time']);y=frames[0][SPECS[key]['label']].to_numpy()
    report['training_groups']=[dict(group=int(g),rows=int((groups==g).sum()),positive=int(((groups==g)&(y==1)).sum()),negative=int(((groups==g)&(y==0)).sum())) for g in np.unique(groups)]
    if any(not g['positive'] or not g['negative'] for g in report['training_groups']):raise ValueError('A training group lacks one class')
    report['training_time_edges']=edges.tolist()
    report['uncertainty_cluster_counts']={name:len(np.unique(values)) for name,values in cluster_schemes(key,frames[3]).items()}
    report['holdout_status']='First measured real Bitcoin AML transfer evaluation; settings frozen before scoring' if key=='elliptic' else 'Consumed synthetic development evidence'
    return frames,report

class Inputs:
    def __init__(self,key,train):
        self.key=key;self.names=predictors(key);self.cats=SPECS[key]['categorical']
        self.processor=processor(key).fit(train[self.names])
        self.categories={c:sorted(train[c].dropna().astype(str).unique().tolist()) for c in self.cats}
    def transform(self,frame,family):
        if family not in {'V6','XGBoost','CatBoost_native','LightGBM_native'}:raise ValueError('Unknown model family')
        data=frame[self.names].copy()
        if family in {'V6','XGBoost'}:
            return np.ascontiguousarray(self.processor.transform(data),dtype=float)
        for column in self.cats:
            values=data[column].where(data[column].notna(),'missing').astype(str)
            if family=='CatBoost_native':data[column]=values
            elif family=='LightGBM_native':
                values=values.where(values.isin(self.categories[column]),np.nan)
                data[column]=pd.Categorical(values,categories=self.categories[column])
            else:raise ValueError('Unknown model family')
        return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SPECS);p.add_argument('root');p.add_argument('output');args=p.parse_args()
    _,report=load_data(args.dataset,sources(args.dataset,args.root));Path(args.output).write_text(json.dumps(report,indent=2))
