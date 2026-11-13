"""Temporal negative-tail ranking surrogate with exact diagonal derivatives.

This combines established entropic ranking and group-robust ideas. It is not
an exact partial-AUC objective or a bound on future false-positive rates.
"""
import numpy as np
from scipy.special import expit,logsumexp

def temporal_groups(times,count=4):
    times=np.asarray(times)
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or count<1:
        raise ValueError('Finite training times and a positive group count are required')
    unique=np.unique(times)
    if len(unique)<count:raise ValueError('Too few distinct training times')
    edges=unique[[int(len(unique)*i/count) for i in range(1,count)]]
    return np.searchsorted(edges,times,side='right'),edges

def evaluation_windows(times):
    times=np.asarray(times)
    # Coarse source timestamps cannot support invented sub-periods.
    return temporal_groups(times,count=min(4,len(np.unique(times))))

def tail_objective(y,score,groups,tail_fraction=.01,temperature=1.,
                   group_temperature=.25,anchor_weight=.2,margin=1.):
    y=np.asarray(y);score=np.asarray(score,dtype=float);groups=np.asarray(groups)
    if (y.ndim!=1 or score.shape!=y.shape or groups.shape!=y.shape or
        set(np.unique(y))!={0,1} or not np.isfinite(score).all() or
        not np.isfinite(groups).all()):
        raise ValueError('Finite matching scores, groups and both binary classes are required')
    if not (0<tail_fraction<=1 and temperature>0 and group_temperature>0 and 0<=anchor_weight<=1):
        raise ValueError('Invalid objective settings')
    positive=y==1;negative=~positive
    weights=np.where(positive,.5/positive.sum(),.5/negative.sum())
    probability=expit(score)
    anchor=float(np.sum(weights*np.logaddexp(0,np.where(positive,-score,score))))
    gradient=anchor_weight*weights*(probability-y)
    hessian=anchor_weight*weights*probability*(1-probability)
    if anchor_weight==1:return anchor,gradient,hessian,dict(group_losses=[],group_weights=[])
    blocks=[]
    for group in np.unique(groups):
        pos=np.flatnonzero((groups==group)&positive)
        neg=np.flatnonzero((groups==group)&negative)
        if not len(pos) or not len(neg):raise ValueError('Both classes are required in every training group')
        count=max(1,int(np.ceil(tail_fraction*len(neg))))
        # The loss is nonsmooth at a negative-tail membership tie.
        neg=neg[np.argsort(-score[neg],kind='stable')[:count]]
        a=score[neg]/temperature;b=-score[pos]/temperature
        la=logsumexp(a);lb=logsumexp(b)
        z=la-np.log(len(neg))+lb-np.log(len(pos))+margin/temperature
        loss=float(np.logaddexp(0,z));d=expit(z)
        qn=np.exp(a-la);qp=np.exp(b-lb)
        gn=d*qn/temperature;gp=-d*qp/temperature
        hn=d*qn*(1-d*qn)/temperature**2
        hp=d*qp*(1-d*qp)/temperature**2
        blocks.append((loss,neg,pos,gn,gp,hn,hp))
    losses=np.array([block[0] for block in blocks])
    scaled=losses/group_temperature
    robust=float(group_temperature*(logsumexp(scaled)-np.log(len(blocks))))
    block_weights=np.exp(scaled-logsumexp(scaled))
    for r,(_,neg,pos,gn,gp,hn,hp) in zip(block_weights,blocks):
        for index,g,h in [(neg,gn,hn),(pos,gp,hp)]:
            gradient[index]+=(1-anchor_weight)*r*g
            hessian[index]+=(1-anchor_weight)*(r*h+r*(1-r)*g*g/group_temperature)
    value=anchor_weight*anchor+(1-anchor_weight)*robust
    return float(value),gradient,hessian,dict(group_losses=losses.tolist(),group_weights=block_weights.tolist())

def pairwise_tail_objective(y,score,groups,tail_fraction=.01,temperature=1.,
                            group_temperature=.25,anchor_weight=.2,margin=1.,chunk_size=64):
    """Exact mean logistic pair loss over the selected negative tail, in chunks.

    This is a partial-AUC surrogate, not the discontinuous partial AUC itself.
    Chunking bounds memory; it does not remove the pairwise arithmetic cost.
    """
    anchor,ag,ah,_=tail_objective(y,score,groups,tail_fraction,temperature,
                                 group_temperature,1.,margin)
    if not 0<=anchor_weight<=1 or chunk_size<1:raise ValueError('Invalid pairwise objective settings')
    if anchor_weight==1:return anchor,ag,ah,dict(group_losses=[],group_weights=[])
    y=np.asarray(y);score=np.asarray(score,dtype=float);groups=np.asarray(groups)
    gradient=anchor_weight*ag;hessian=anchor_weight*ah;blocks=[]
    for group in np.unique(groups):
        pos=np.flatnonzero((groups==group)&(y==1));neg=np.flatnonzero((groups==group)&(y==0))
        if not len(pos) or not len(neg):raise ValueError('Both classes are required in every training group')
        count=max(1,int(np.ceil(tail_fraction*len(neg))))
        neg=neg[np.argsort(-score[neg],kind='stable')[:count]]
        gn=np.zeros(len(neg));hn=np.zeros(len(neg));gp=np.zeros(len(pos));hp=np.zeros(len(pos));total=0.
        pairs=len(pos)*len(neg)
        for start in range(0,len(pos),chunk_size):
            stop=min(start+chunk_size,len(pos))
            difference=(score[neg][None,:]-score[pos[start:stop]][:,None]+margin)/temperature
            probability=expit(difference);curvature=probability*(1-probability)
            total+=float(np.logaddexp(0,difference).sum())/pairs
            gn+=probability.sum(axis=0)/(pairs*temperature)
            gp[start:stop]=-probability.sum(axis=1)/(pairs*temperature)
            hn+=curvature.sum(axis=0)/(pairs*temperature**2)
            hp[start:stop]=curvature.sum(axis=1)/(pairs*temperature**2)
        blocks.append((total,neg,pos,gn,gp,hn,hp))
    losses=np.array([block[0] for block in blocks]);scaled=losses/group_temperature
    robust=float(group_temperature*(logsumexp(scaled)-np.log(len(blocks))))
    block_weights=np.exp(scaled-logsumexp(scaled))
    for r,(_,neg,pos,gn,gp,hn,hp) in zip(block_weights,blocks):
        for index,g,h in [(neg,gn,hn),(pos,gp,hp)]:
            gradient[index]+=(1-anchor_weight)*r*g
            hessian[index]+=(1-anchor_weight)*(r*h+r*(1-r)*g*g/group_temperature)
    return anchor_weight*anchor+(1-anchor_weight)*robust,gradient,hessian,dict(
        group_losses=losses.tolist(),group_weights=block_weights.tolist())
