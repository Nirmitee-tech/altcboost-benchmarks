"""Train/validation-only bound on what a bounded residual ensemble can recover."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from validation_data import load_data,sources,Inputs,SPECS

def recall_ceiling(y,prior,correction_bound,alpha):
    y=np.asarray(y);prior=np.asarray(prior,dtype=float)
    if (y.ndim!=1 or prior.shape!=y.shape or set(np.unique(y))!={0,1} or not np.isfinite(prior).all() or
        not np.isfinite(correction_bound) or correction_bound<0 or not 0<alpha<1):
        raise ValueError('Finite matching binary data, a nonnegative bound and valid budget are required')
    negatives=np.sort(prior[y==0])[::-1];allowed=int(np.floor(alpha*len(negatives)))
    boundary=float(negatives[allowed]);recoverable=prior[y==1]>boundary-2*correction_bound
    return dict(alpha=alpha,negative_count=len(negatives),positive_count=int((y==1).sum()),allowed_false_positives=allowed,
        prior_negative_order_statistic=boundary,absolute_correction_bound=float(correction_bound),
        empirical_recall_upper_bound=float(recoverable.mean()),potentially_recoverable_positives=int(recoverable.sum()),
        scope='Necessary condition for ranking under an absolute per-event residual bound on these observed validation rows. Not an attainable-recall guarantee, population bound or novelty claim.')

def run(key,root,output):
    frames,discovery=load_data(key,sources(key,root));train,validation=frames[:2]
    inputs=Inputs(key,train);x=inputs.transform(train,'V6');xv=inputs.transform(validation,'V6')
    y=train[SPECS[key]['label']].to_numpy(dtype=int);yv=validation[SPECS[key]['label']].to_numpy(dtype=int)
    with threadpool_limits(limits=2):
        scaler=StandardScaler().fit(x)
        model=LogisticRegression(class_weight='balanced',max_iter=500,random_state=17).fit(scaler.transform(x),y)
        score=model.decision_function(scaler.transform(xv))
    report=dict(dataset=key,scope='Local reconstruction of the frozen starting model on training/validation only. Source discovery reads all files; no calibration or final score is computed.',
        observed_utc=discovery['observed_utc'],train_cohort=discovery['partitions']['train'],validation_cohort=discovery['partitions']['validation'],
        bound_derivation='100 trees * maximum learning rate 0.08 * maximum absolute leaf step 5 = 40 per-event correction. Backtracking or prefix selection can only reduce this upper bound.',
        validation_prior_quantiles={str(label):np.quantile(score[yv==label],[0,.5,.95,.99,1]).tolist() for label in [0,1]},
        ceilings=[recall_ceiling(yv,score,40.,alpha) for alpha in [.0005,.001,.01]],
        limitations='This local diagnostic does not reproduce every hosted implementation detail or identify the sole cause of transfer failure. The bound applies conditional on the fitted starting model and a hard residual-magnitude limit; it is an architecture limitation on observed rows, not a new generalization theorem.')
    Path(output).write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SPECS);p.add_argument('root');p.add_argument('output');args=p.parse_args()
    print(json.dumps(run(args.dataset,args.root,args.output),indent=2))
