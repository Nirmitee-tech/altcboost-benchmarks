"""Paired cluster bootstrap conditional on fitted models and fixed cutoffs."""
import numpy as np

def cluster_schemes(key,frame):
    time=frame['__time'].to_numpy(dtype=np.int64)
    if key=='sparkov':
        return {'calendar_week':time//(7*24*3600*10**9),'card':frame['cc_num'].to_numpy(dtype=str)}
    if key=='paysim':return {'four_step_time_block':time//4}
    return {'published_time_group':time}

def paired_cluster_interval(y,candidate_flags,reference_flags,clusters,replicates=2000,seed=20261112,min_clusters=20):
    y=np.asarray(y);a=np.asarray(candidate_flags,dtype=bool);b=np.asarray(reference_flags,dtype=bool);groups=np.asarray(clusters)
    if y.ndim!=1 or a.shape!=y.shape or b.shape!=y.shape or groups.shape!=y.shape or set(np.unique(y))!={0,1}:
        raise ValueError('Matching vectors and both binary classes required')
    if replicates<1 or min_clusters<2:raise ValueError('Invalid bootstrap settings')
    _,index=np.unique(groups,return_inverse=True);count=int(index.max()+1)
    result=dict(cluster_count=count,replicates_requested=replicates,min_clusters=min_clusters,
        observed_delta_recall=float(a[y==1].mean()-b[y==1].mean()),
        observed_delta_fpr=float(a[y==0].mean()-b[y==0].mean()),
        scope='Conditional on selected models and calibration cutoffs. Resamples clusters, not training or calibration data. Percentile intervals; no multiplicity correction or general dependence guarantee.')
    if count<min_clusters:return dict(result,status='unsupported',reason='Too few observed clusters for the prespecified screen')
    pos=np.bincount(index,weights=(y==1),minlength=count);neg=np.bincount(index,weights=(y==0),minlength=count)
    dp=np.bincount(index,weights=(a.astype(int)-b.astype(int))*(y==1),minlength=count)
    dn=np.bincount(index,weights=(a.astype(int)-b.astype(int))*(y==0),minlength=count)
    rng=np.random.default_rng(seed);recall=[];fpr=[]
    for _ in range(replicates):
        draw=rng.integers(0,count,size=count);p=pos[draw].sum();n=neg[draw].sum()
        if p and n:recall.append(dp[draw].sum()/p);fpr.append(dn[draw].sum()/n)
    if not recall:return dict(result,status='unsupported',reason='No replicate contained both classes')
    return dict(result,status='evaluated',valid_replicates=len(recall),
        delta_recall_interval_95=np.quantile(recall,[.025,.975]).tolist(),
        delta_fpr_interval_95=np.quantile(fpr,[.025,.975]).tolist())
