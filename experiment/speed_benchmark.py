"""Compare public and lower-overhead scorers with identical model settings."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from discovery import FEATURES,load_data
from evaluation import calibrate,metrics
from altcboost_v2 import _score

def checked_input(value,n_features):
    x=np.asarray(value,dtype=np.float64,order='C')
    if x.ndim!=2 or x.shape[1]!=n_features or not np.isfinite(x).all():
        raise ValueError('Expected finite numeric rows with the fitted feature count')
    return x

class Adapter:
    def __init__(self,name,family,n_features,predict):
        self.name,self.family,self.n_features,self.predict=name,family,n_features,predict
    def __call__(self,x):
        # This validates shape and finiteness; it does not establish feature correctness.
        return self.predict(checked_input(x,self.n_features))

class BoundaryGuard:
    def __init__(self,predict,reference,thresholds,tolerance=1e-12):
        self.predict,self.reference=predict,reference
        self.thresholds=list(thresholds.values());self.tolerance=tolerance
        self.fallback_rows=0
    def __call__(self,x):
        scores=self.predict(x)
        close=np.zeros(len(scores),dtype=bool)
        for threshold in self.thresholds:
            close|=np.abs(scores-threshold)<=self.tolerance
        if close.any():
            scores[close]=self.reference(x[close])
            self.fallback_rows+=int(close.sum())
        return scores

class CompiledCatBoost:
    def __init__(self,model,n_features,directory):
        compiler=shutil.which('c++') or shutil.which('g++')
        if not compiler:
            raise RuntimeError('A C++ compiler is required for this scorer')
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        source=directory/'catboost_model.cpp'
        model.save_model(str(source),format='cpp')
        if model.get_cat_feature_indices():
            raise ValueError('This bridge supports numeric features only')
        bridge=directory/'bridge.cpp'
        bridge.write_text('''#include "catboost_model.cpp"
#include <cmath>
#include <cstddef>
extern "C" void score_rows(const double* data, size_t rows, size_t columns, double* output) {
    std::vector<float> features(columns);
    for (size_t row=0;row<rows;++row) {
        for (size_t col=0;col<columns;++col) features[col]=static_cast<float>(data[row*columns+col]);
        double margin=ApplyCatboostModel(features);
        output[row]=1.0/(1.0+std::exp(-margin));
    }
}
''')
        library=directory/('scorer.dylib' if platform.system()=='Darwin' else 'scorer.so')
        started=time.perf_counter()
        subprocess.run([compiler,'-std=c++17','-O3','-shared','-fPIC',str(bridge),'-o',str(library)],
                       check=True,capture_output=True,text=True)
        self.build_seconds=time.perf_counter()-started
        self.library=ctypes.CDLL(str(library.resolve()))
        self.function=self.library.score_rows
        pointer=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.function.argtypes=[pointer,ctypes.c_size_t,ctypes.c_size_t,pointer]
        self.function.restype=None
        self.n_features=n_features
    def __call__(self,x):
        out=np.empty(len(x),dtype=np.float64)
        self.function(x,len(x),self.n_features,out)
        return out

def make_adapters(models,n_features,directory):
    cat=models['CatBoost'];xgb=models['XGBoost'];alt=models['ALTCBoostV3']
    compiled=CompiledCatBoost(cat,n_features,directory)
    booster=xgb.get_booster()
    booster.set_param({'nthread':1})
    alt_args=alt._args()
    paths=[
        Adapter('CatBoost_python','CatBoost',n_features,
                lambda x:cat.predict_proba(x,thread_count=1)[:,1]),
        Adapter('CatBoost_cpp','CatBoost',n_features,compiled),
        Adapter('XGBoost_python','XGBoost',n_features,lambda x:xgb.predict_proba(x)[:,1]),
        Adapter('XGBoost_inplace','XGBoost',n_features,
                lambda x:booster.inplace_predict(x,validate_features=False)),
        Adapter('ALTCBoost_python','ALTCBoostV3',n_features,lambda x:alt.predict_proba(x)[:,1]),
        Adapter('ALTCBoost_numba','ALTCBoostV3',n_features,
                lambda x:expit(np.clip(_score(x,*alt_args),-35.,35.))),
    ]
    return paths,dict(catboost_cpp_build_seconds=compiled.build_seconds,
                      compiler=shutil.which('c++') or shutil.which('g++'))

def check_parity(reference,candidate,thresholds,atol=1e-6):
    if not thresholds:
        raise ValueError('At least one calibrated threshold is required')
    a,b=np.asarray(reference,dtype=np.float64),np.asarray(candidate,dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(b).all():
        raise ValueError('Scorer output shape or finiteness differs')
    error=float(np.max(np.abs(a-b))) if len(a) else 0.
    disagreements={str(alpha):int(np.count_nonzero((a>=cut)!=(b>=cut)))
                   for alpha,cut in thresholds.items()}
    return dict(max_probability_error=error,flag_disagreements=disagreements,
                accepted=bool(error<=atol and not any(disagreements.values())))

def measure(paths,x,single_calls=1000,rounds=5,seed=20261112):
    if not len(x) or single_calls<1 or rounds<1:
        raise ValueError('Timing requires rows, calls and rounds')
    rng=np.random.default_rng(seed)
    singles=[x[int(i):int(i)+1] for i in rng.integers(len(x),size=single_calls)]
    batch_sizes=sorted({min(128,len(x)),min(4096,len(x))}-{1})
    samples={(p.name,size):[] for p in paths for size in [1,*batch_sizes]}
    for path in paths:
        for _ in range(10):path(singles[0])
        for size in batch_sizes:path(x[:size])
    with threadpool_limits(limits=1):
        for round_number in range(rounds):
            for index in rng.permutation(len(paths)):
                path=paths[int(index)]
                for value in singles:
                    started=time.perf_counter_ns();path(value)
                    samples[path.name,1].append((time.perf_counter_ns()-started)/1000.)
                for size in batch_sizes:
                    for _ in range(20):
                        started=time.perf_counter_ns();path(x[:size])
                        samples[path.name,size].append((time.perf_counter_ns()-started)/1000.)
    rows=[]
    for path in paths:
        for size in [1,*batch_sizes]:
            values=np.asarray(samples[path.name,size])
            rows.append(dict(method=path.family,path=path.name,batch_size=size,calls=len(values),
                p50_call_us=float(np.median(values)),p99_call_us=float(np.quantile(values,.99)),
                p50_us_per_row=float(np.median(values)/size)))
    return rows

def run(csv,output,trees=70,single_calls=1000,rounds=5,training_limit=None):
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier
    from altcboost_v3 import ALTCBoostV3
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frame,parts,discovery=load_data(csv)
    (output/'discovery.json').write_text(json.dumps(discovery,indent=2))
    arrays=[(frame.iloc[i][FEATURES].to_numpy(dtype=float),frame.iloc[i].Class.to_numpy()) for i in parts]
    (xt,yt),(xv,yv),(xc,yc),(xe,ye)=arrays
    if training_limit is not None:
        xt,yt=xt[:training_limit],yt[:training_limit]
    models=dict(
        CatBoost=CatBoostClassifier(iterations=trees,depth=5,learning_rate=.06,thread_count=2,
            verbose=False,allow_writing_files=False,random_seed=17),
        XGBoost=XGBClassifier(n_estimators=trees,max_depth=5,learning_rate=.06,n_jobs=2,
            tree_method='hist',random_state=17),
        ALTCBoostV3=ALTCBoostV3(n_estimators=trees,random_state=17,min_samples_leaf=40,use_cross=False))
    fits={}
    for family,model in models.items():
        started=time.perf_counter()
        with threadpool_limits(limits=2):
            if family=='ALTCBoostV3':model.fit(xt,yt,xv,yv)
            else:model.fit(xt,yt)
        fits[family]=time.perf_counter()-started
        if family=='XGBoost':model.set_params(n_jobs=1)
        print('Trained',family,'seconds',round(fits[family],3),flush=True)
    paths,build=make_adapters(models,len(FEATURES),output/'compiled')
    cold={}
    for path in paths:
        started=time.perf_counter_ns();path(xe[:1]);cold[path.name]=(time.perf_counter_ns()-started)/1000.
    refs={};thresholds={};quality=[]
    for family,model in models.items():
        pc=model.predict_proba(xc)[:,1];pe=model.predict_proba(xe)[:,1]
        refs[family]=np.r_[pc,pe]
        thresholds[family]={}
        for alpha in [.0001,.001,.01]:
            try:
                cut,details=calibrate(yc,pc,alpha)
            except ValueError as error:
                if 'Insufficient calibration negatives' not in str(error):raise
                quality.append(dict(method=family,alpha=alpha,status='unsupported',reason=str(error)))
                continue
            thresholds[family][alpha]=cut
            quality.append(dict(method=family,status='evaluated',**details,**metrics(ye,pe,cut,alpha)))
    combined=np.vstack([xc,xe]);parity=[];accepted=[]
    raw_cpp=next(path for path in paths if path.name=='CatBoost_cpp')
    public_cat=next(path for path in paths if path.name=='CatBoost_python')
    guarded=BoundaryGuard(raw_cpp.predict,public_cat.predict,thresholds['CatBoost'])
    guarded_path=Adapter('CatBoost_cpp_guarded','CatBoost',len(FEATURES),guarded)
    started=time.perf_counter_ns();guarded_path(xe[:1])
    cold[guarded_path.name]=(time.perf_counter_ns()-started)/1000.
    paths.append(guarded_path)
    for path in paths:
        record=dict(path=path.name,method=path.family,
            **check_parity(refs[path.family],path(combined),thresholds[path.family]))
        parity.append(record)
        if record['accepted']:accepted.append(path)
        print('Parity',path.name,record,flush=True)
    guarded_equivalence_fallback_rows=guarded.fallback_rows
    pd.DataFrame(quality).to_csv(output/'quality.csv',index=False)
    (output/'parity.json').write_text(json.dumps(parity,indent=2))
    timing=measure(accepted,xe,single_calls,rounds)
    pd.DataFrame(timing).to_csv(output/'timing.csv',index=False)
    manifest=dict(source='https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
        scope='Fixed seed 17 and configurations selected before this experiment; same temporal holdout; inference study only',
        unit='one scoring call; batch cost per row reported separately',fit_seconds=fits,build=build,
        first_call_us=cold,trees_limit=trees,training_limit=training_limit,
        cpp_guard_tolerance=guarded.tolerance,
        cpp_guard_fallback_rows_during_first_call_and_equivalence=guarded_equivalence_fallback_rows,
        single_calls_per_round=single_calls,rounds=rounds,
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','numba','xgboost','catboost']},
        platform=platform.platform(),python=platform.python_version(),
        limitations='Finite shape checks, ALTC feature scaling and compiled CatBoost boundary-guard costs are included. Feature retrieval, network, concurrency, service queues and process startup are excluded. First calls are path-specific and order-dependent. p99 is descriptive. Numerical equivalence was checked on observed rows, not guaranteed for arbitrary future input.',
        hypotheses=['Alternative scorers reproduce reference probabilities within 1e-6',
            'Alternative scorers preserve flags on calibration and test rows at all three fixed thresholds',
            'ALTC single-call speed advantage persists against compiled CatBoost and native XGBoost'],
        input_sha256=discovery['sha256'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(timing).to_string(index=False),flush=True)
    return timing,parity

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('csv');parser.add_argument('--output',default='speed-results')
    parser.add_argument('--trees',type=int,default=70);parser.add_argument('--calls',type=int,default=1000)
    parser.add_argument('--rounds',type=int,default=5);parser.add_argument('--training-limit',type=int)
    args=parser.parse_args();run(args.csv,args.output,args.trees,args.calls,args.rounds,args.training_limit)
