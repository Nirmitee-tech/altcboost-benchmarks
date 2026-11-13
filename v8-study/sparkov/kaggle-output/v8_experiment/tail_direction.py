"""One-sided tail derivative, including the active-face change at negative ties."""
import numpy as np
from scipy.special import expit
from tail_objective import pairwise_tail_objective

class TailState:
    def __init__(self,y,score,groups,tail_fraction=.01,temperature=1.,group_temperature=.25,anchor_weight=.2):
        self.y=np.array(y,copy=True);self.score=np.array(score,dtype=float,copy=True);self.groups=np.array(groups,copy=True)
        self.fraction=tail_fraction;self.temperature=temperature;self.anchor=anchor_weight
        self.value,self.gradient,self.hessian,self.detail=pairwise_tail_objective(self.y,self.score,self.groups,
            tail_fraction,temperature,group_temperature,anchor_weight)
    def directional_derivative(self,direction):
        direction=np.asarray(direction,dtype=float)
        if direction.shape!=self.score.shape or not np.isfinite(direction).all():raise ValueError('Finite matching direction required')
        derivative=float(self.gradient@direction)
        if self.anchor==1:return derivative
        for weight,group in zip(self.detail['group_weights'],np.unique(self.groups)):
            positive=np.flatnonzero((self.groups==group)&(self.y==1))
            negative=np.flatnonzero((self.groups==group)&(self.y==0))
            count=max(1,int(np.ceil(self.fraction*len(negative))))
            selected=negative[np.argsort(-self.score[negative],kind='stable')[:count]]
            boundary=self.score[selected[-1]]
            tied=negative[self.score[negative]==boundary]
            selected_ties=selected[self.score[selected]==boundary];needed=len(selected_ties)
            if len(tied)<=needed:continue
            maximum=float(np.sort(direction[tied])[-needed:].sum())
            current=float(direction[selected_ties].sum())
            coefficient=float(expit((boundary-self.score[positive]+1.)/self.temperature).mean()/(count*self.temperature))
            derivative+=(1-self.anchor)*weight*coefficient*(maximum-current)
        return float(derivative)
