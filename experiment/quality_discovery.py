"""Aggregate, train-only checks for proposed detection-model changes."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from discovery import FEATURES,load_data

def audit_previous_xgb(csv,output):
    from xgboost import XGBClassifier
    from evaluation import calibrate,metrics
    frame,parts,source=load_data(csv)
    arrays=[(frame.iloc[i][FEATURES].to_numpy(),frame.iloc[i].Class.to_numpy()) for i in parts]
    (xt,yt),_,(xc,yc),(xe,ye)=arrays
    with threadpool_limits(limits=2):
        model=XGBClassifier(n_estimators=70,max_depth=5,learning_rate=.06,n_jobs=2,
                            tree_method='hist',random_state=17).fit(xt,yt)
    pc=model.predict_proba(xc)[:,1];pe=model.predict_proba(xe)[:,1]
    rows=[]
    for alpha in [.0001,.001,.01]:
        cutoff,details=calibrate(yc,pc,alpha)
        legacy=pe>=cutoff
        rows.append(dict(alpha=alpha,legacy_tp=int((legacy&(ye==1)).sum()),
            legacy_fp=int((legacy&(ye==0)).sum()),
            legacy_calibration_fp=int(((pc>=cutoff)&(yc==0)).sum()),
            strict_calibration_fp=int(((pc.astype(np.float64)>=cutoff)&(yc==0)).sum()),
            strict=metrics(ye,pe,cutoff,alpha)))
    report=dict(scope='Historical XGBoost configuration audit; already consumed ULB interval',
        unit='transaction row',observed_utc=source['observed_utc'],source_sha256=source['sha256'],
        prediction_dtype=str(pe.dtype),rows=rows)
    Path(output).write_text(json.dumps(report,indent=2))
    return report

def update_retention(y,probability):
    y,p=np.asarray(y),np.asarray(probability)
    gradient=y-p;hessian=np.maximum(p*(1-p),.002)
    response=gradient/hessian
    implied=hessian*np.clip(response,-15,15)
    return response,implied/np.where(np.abs(gradient)>1e-15,gradient,1.)

def discover(csv,output):
    frame,parts,source=load_data(csv)
    train=frame.iloc[parts[0]];x=train[FEATURES].to_numpy();y=train.Class.to_numpy()
    scaler=StandardScaler().fit(x);z=scaler.transform(x)
    clipped=np.clip(z,-6,6)
    with threadpool_limits(limits=2):
        prior=LogisticRegression(C=1.,max_iter=500).fit(clipped,y)
    p=expit(prior.decision_function(clipped));response,retention=update_retention(y,p)
    source['hypotheses']={
        'feature_clip_affects_fraud_rows':{
            'query':'Any absolute train-standardized feature above six; count by binary label',
            'fraud_rows_affected':int(((np.abs(z)>6).any(axis=1)&(y==1)).sum()),
            'normal_rows_affected':int(((np.abs(z)>6).any(axis=1)&(y==0)).sum())},
        'response_clip_weakens_fraud_updates':{
            'query':'V3 initial logistic model; compare gradient with hessian times clipped response',
            'fraud_rows_clipped':int(((np.abs(response)>15)&(y==1)).sum()),
            'fraud_gradient_retention_median':float(np.median(retention[y==1])),
            'fraud_gradient_retention_min':float(np.min(retention[y==1]))},
        'unseen_final_data_available':{
            'verified':False,'reason':'The ULB test interval has already informed earlier comparisons. It is development evidence now.'}}
    source['scope']='ULB full-file metadata and train-only model diagnostics; no row export'
    Path(output).write_text(json.dumps(source,indent=2))
    print(json.dumps(source['hypotheses'],indent=2))
    return source

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('csv');parser.add_argument('output');args=parser.parse_args()
    discover(args.csv,args.output)
