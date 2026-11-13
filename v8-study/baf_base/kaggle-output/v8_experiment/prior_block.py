"""Exact small directional Hessian for a temporal negative-tail pair loss."""
import numpy as np
from scipy.special import expit
from tail_objective import pairwise_tail_objective

def bounded_prior(score,limit=4.):
    if limit<=0:raise ValueError('Positive prior bound required')
    return limit*np.tanh(np.asarray(score,dtype=float)/limit)

def directional_derivatives(y,score,groups,directions,tail_fraction=.01,temperature=1.,
                            group_temperature=.25,anchor_weight=.2,chunk_size=64):
    value,row_gradient,_,detail=pairwise_tail_objective(y,score,groups,tail_fraction,
        temperature,group_temperature,anchor_weight,chunk_size=chunk_size)
    y=np.asarray(y);score=np.asarray(score,dtype=float);groups=np.asarray(groups)
    directions=np.asarray(directions,dtype=float)
    if directions.ndim!=2 or directions.shape[0]!=len(y) or not np.isfinite(directions).all():
        raise ValueError('Finite directions with one row per example required')
    gradient=directions.T@row_gradient
    weights=np.where(y==1,.5/np.sum(y==1),.5/np.sum(y==0))
    p=expit(score);hessian=anchor_weight*(directions.T@(directions*(weights*p*(1-p))[:,None]))
    if anchor_weight==1:return value,gradient,hessian
    block_gradients=[];block_hessians=[]
    for group in np.unique(groups):
        pos=np.flatnonzero((groups==group)&(y==1));neg=np.flatnonzero((groups==group)&(y==0))
        count=max(1,int(np.ceil(tail_fraction*len(neg))))
        neg=neg[np.argsort(-score[neg],kind='stable')[:count]]
        pairs=len(pos)*len(neg);g=np.zeros(directions.shape[1]);h=np.zeros((len(g),len(g)))
        for start in range(0,len(pos),chunk_size):
            selected=pos[start:start+chunk_size]
            delta=(score[neg][None,:]-score[selected][:,None]+1.)/temperature
            probability=expit(delta).ravel();curvature=probability*(1-probability)
            difference=((directions[neg][None,:,:]-directions[selected][:,None,:])/temperature).reshape(-1,len(g))
            g+=difference.T@probability/pairs
            h+=difference.T@(difference*curvature[:,None])/pairs
        block_gradients.append(g);block_hessians.append(h)
    r=np.asarray(detail['group_weights']);bg=np.asarray(block_gradients);mean=r@bg
    robust_hessian=sum(weight*h for weight,h in zip(r,block_hessians))
    robust_hessian+=(bg.T@(bg*r[:,None])-np.outer(mean,mean))/group_temperature
    hessian+=(1-anchor_weight)*robust_hessian
    return value,gradient,(hessian+hessian.T)/2

def block_proposals(gradient,hessian,prior_weight,learning_rate=.08,damping=.001,diagonal=False):
    g=np.asarray(gradient,dtype=float);h=np.asarray(hessian,dtype=float)
    if g.shape!=(2,) or h.shape!=(2,2) or not np.isfinite(g).all() or not np.isfinite(h).all():
        raise ValueError('Finite two-coordinate derivatives required')
    if not 0<=prior_weight<=1 or learning_rate<=0 or damping<=0:raise ValueError('Invalid block constraints')
    scale=np.maximum(np.sqrt(np.maximum(np.diag(h),0)),1e-6)
    normalized=h/np.outer(scale,scale)
    if diagonal:normalized=np.diag(np.diag(normalized))
    step=-np.linalg.solve(normalized+damping*np.eye(2),g/scale)/scale
    joint=np.array([np.clip(step[0],-prior_weight,1-prior_weight),np.clip(step[1],0,learning_rate)])
    prior=np.array([np.clip(-g[0]/max(h[0,0]*(1+damping),1e-12),-prior_weight,1-prior_weight),0.])
    return [('tree_only',np.array([0.,learning_rate])),('prior_only',prior),('joint',joint)]
