"""Prior-free tail continuation selected by feasible descent of a proposed tree."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from altcboost_v6 import ALTCBoostV6
from tail_objective import pairwise_tail_objective
from tail_direction import TailState
from detection_objective import selection_key

class ALTCBoostV8(ALTCBoostV6):
    MODES={'target','all_negative','geometric','linear','adaptive_blind','adaptive_checked'}
    def __init__(self,continuation='adaptive_checked',**kwargs):
        super().__init__(**kwargs)
        if continuation not in self.MODES:raise ValueError('Unknown continuation policy')
        self.continuation=continuation
    def _loss(self,y,score,groups,fraction):
        self.objective_evaluations_+=1
        return pairwise_tail_objective(y,score,groups,fraction,self.temperature,self.group_temperature,self.anchor_weight)[0]
    def fractions(self,attempt):
        if self.continuation=='target':return [self.tail_fraction]
        if self.continuation=='all_negative':return [1.]
        progress=attempt/max(1,self.n_estimators-1)
        if self.continuation=='geometric':return [float(self.tail_fraction**progress)]
        if self.continuation=='linear':return [float(1-progress*(1-self.tail_fraction))]
        grid=[self.tail_fraction]
        while grid[-1]<1:grid.append(min(1.,2*grid[-1]))
        return grid
    def fit(self,x,y,x_val,y_val,groups=None):
        x=np.asarray(x,dtype=float);x_val=np.asarray(x_val,dtype=float);y=np.asarray(y);y_val=np.asarray(y_val)
        if (x.ndim!=2 or x_val.ndim!=2 or x.shape[1]!=x_val.shape[1] or
            y.ndim!=1 or y_val.ndim!=1 or len(y)!=len(x) or len(y_val)!=len(x_val) or
            set(np.unique(y))!={0,1} or set(np.unique(y_val))!={0,1} or
            not np.isfinite(x).all() or not np.isfinite(x_val).all()):raise ValueError('Finite matching binary data required')
        groups=np.zeros(len(y),dtype=int) if groups is None else np.asarray(groups)
        self.n_features_=x.shape[1];self.scaler_=StandardScaler().fit(x);xt=self._features(x);xv=self._features(x_val)
        self.steps_=[];self.prior_weights_=[0.];self.best_iteration_=0;self.objective_evaluations_=0
        self.step_diagnostics_=[];self.attempt_diagnostics_=[];self.validation_history_=[]
        score=np.zeros(len(y));validation=np.zeros(len(y_val))
        self.training_loss_=[self._loss(y,score,groups,self.tail_fraction)];best=selection_key(y_val,validation)
        attempt=0
        while attempt<self.n_estimators:
            for fraction in self.fractions(attempt):
                if attempt>=self.n_estimators:break
                state=TailState(y,score,groups,fraction,self.temperature,self.group_temperature,self.anchor_weight)
                self.objective_evaluations_+=1
                gradient=state.gradient*len(y);hessian=np.maximum(state.hessian*len(y),1e-6)
                tree=DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state+attempt)
                tree.fit(xt,np.clip(-gradient/hessian,-50,50),sample_weight=hessian)
                leaves=tree.apply(xt);num=np.bincount(leaves,weights=-gradient)
                den=np.bincount(leaves,weights=hessian)+self.leaf_l2
                values=np.clip(num/np.maximum(den,1e-12),-self.max_leaf_step,self.max_leaf_step)
                tree.tree_.value[:,0,0]=0.
                for leaf in np.unique(leaves):tree.tree_.value[leaf,0,0]=values[leaf]
                direction=tree.predict(xt);checked=self.continuation!='adaptive_blind'
                derivative=state.directional_derivative(direction) if checked else None
                attempt+=1;record=dict(attempt=attempt,fraction=float(fraction),directional_derivative=derivative,accepted=False)
                if checked and derivative>=0:
                    record['reason']='non_descent_direction' if derivative>1e-10 else 'near_stationary_direction'
                    self.attempt_diagnostics_.append(record);continue
                accepted=False
                for backtracks in range(12):
                    rate=self.learning_rate*2.**(-backtracks)
                    candidate=score+rate*direction;loss=self._loss(y,candidate,groups,fraction)
                    if loss<state.value-1e-12:accepted=True;break
                if not accepted:
                    record['reason']='no_accepted_step_within_backtracking_budget';self.attempt_diagnostics_.append(record);continue
                before_target=self.training_loss_[-1];score=candidate;validation+=rate*tree.predict(xv)
                self.steps_.append((tree,rate));self.prior_weights_.append(0.)
                target_loss=self._loss(y,score,groups,self.tail_fraction);self.training_loss_.append(target_loss)
                record.update(accepted=True,working_loss_before=state.value,working_loss_after=loss,rate=rate,
                    backtracks=backtracks,target_loss_before=before_target,target_loss_after=target_loss)
                self.step_diagnostics_.append(record);self.attempt_diagnostics_.append(record)
                key=selection_key(y_val,validation);self.validation_history_.append(key)
                if key>best:best=key;self.best_iteration_=len(self.steps_)
                break
        self.validation_key_=best;self.weak_learner_fits_=attempt
        return self
    def decision_function(self,x):
        z=self._features(x);score=np.zeros(len(z))
        for tree,rate in self.steps_[:self.best_iteration_]:score+=rate*tree.predict(z)
        return score
