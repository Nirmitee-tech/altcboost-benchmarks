"""Research candidate: a revisable linear prior and a small exact-curvature block."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from altcboost_v6 import ALTCBoostV6
from detection_objective import selection_key
from prior_block import directional_derivatives,block_proposals,bounded_prior

class ALTCBoostV7(ALTCBoostV6):
    MODES={'fixed','zero','clipped','shrunk','adaptive_diagonal','adaptive_joint'}
    def __init__(self,prior_mode='adaptive_joint',block_damping=.001,**kwargs):
        super().__init__(**kwargs)
        if prior_mode not in self.MODES or block_damping<=0:raise ValueError('Invalid prior mode or damping')
        self.prior_mode=prior_mode;self.block_damping=block_damping
    def _prior(self,z):
        if self.linear_ is None:return np.zeros(len(z))
        score=self.linear_.decision_function(z)
        return bounded_prior(score) if self.prior_mode=='clipped' else score
    def fit(self,x,y,x_val,y_val,groups=None):
        x=np.asarray(x,dtype=float);x_val=np.asarray(x_val,dtype=float);y=np.asarray(y);y_val=np.asarray(y_val)
        if (x.ndim!=2 or x_val.ndim!=2 or x.shape[1]!=x_val.shape[1] or
            y.ndim!=1 or y_val.ndim!=1 or len(y)!=len(x) or len(y_val)!=len(x_val) or
            set(np.unique(y))!={0,1} or set(np.unique(y_val))!={0,1} or
            not np.isfinite(x).all() or not np.isfinite(x_val).all()):
            raise ValueError('Finite matching matrices and both binary classes required')
        groups=np.zeros(len(y),dtype=int) if groups is None else np.asarray(groups)
        self.n_features_=x.shape[1];self.scaler_=StandardScaler().fit(x)
        xt=self._features(x);xv=self._features(x_val)
        self.linear_=None if self.prior_mode=='zero' else LogisticRegression(
            class_weight='balanced',max_iter=500,random_state=self.random_state).fit(xt,y)
        prior=self._prior(xt);prior_val=self._prior(xv)
        weight=.25 if self.prior_mode=='shrunk' else (0. if self.prior_mode=='zero' else 1.)
        score=weight*prior;validation=weight*prior_val
        self.steps_=[];self.prior_weights_=[weight];self.best_iteration_=0
        self.training_loss_=[self.objective(y,score,groups)[0]];self.validation_history_=[];self.step_diagnostics_=[]
        best=selection_key(y_val,validation);patience=0;validation_directions=[]
        adaptive=self.prior_mode.startswith('adaptive')
        for iteration in range(self.n_estimators):
            value,gradient,hessian,_=self.objective(y,score,groups)
            gradient=gradient*len(y);hessian=np.maximum(hessian*len(y),1e-6)
            tree=DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state+iteration)
            tree.fit(xt,np.clip(-gradient/hessian,-50,50),sample_weight=hessian)
            leaves=tree.apply(xt);numerator=np.bincount(leaves,weights=-gradient)
            denominator=np.bincount(leaves,weights=hessian)+self.leaf_l2
            leaf_values=np.clip(numerator/np.maximum(denominator,1e-12),-self.max_leaf_step,self.max_leaf_step)
            tree.tree_.value[:,0,0]=0.
            for leaf in np.unique(leaves):tree.tree_.value[leaf,0,0]=leaf_values[leaf]
            direction=tree.predict(xt);directions=np.column_stack([prior,direction])
            if adaptive:
                _,g,h=directional_derivatives(y,score,groups,directions,self.tail_fraction,
                    self.temperature,self.group_temperature,self.anchor_weight)
                proposals=block_proposals(g,h,weight,self.learning_rate,self.block_damping,
                    diagonal=self.prior_mode=='adaptive_diagonal')
            else:proposals=[('tree_only',np.array([0.,self.learning_rate]))]
            accepted=None
            for kind,proposal in proposals:
                if np.all(proposal==0):continue
                for backtracks in range(12):
                    step=proposal*2.**(-backtracks);candidate=score+directions@step
                    loss=self.objective(y,candidate,groups)[0]
                    if loss<value-1e-12:
                        if accepted is None or loss<accepted['loss']:
                            accepted=dict(kind=kind,step=step,score=candidate,loss=loss,backtracks=backtracks)
                        break
            if accepted is None:break
            da,rate=accepted['step'];weight=float(np.clip(weight+da,0,1))
            score=accepted['score']
            self.steps_.append((tree,float(rate)));self.prior_weights_.append(weight)
            validation_directions.append(tree.predict(xv))
            # Rebuild the selected representation when the prior coefficient changes.
            validation=weight*prior_val
            for (_,coefficient),prediction in zip(self.steps_,validation_directions):validation+=coefficient*prediction
            self.training_loss_.append(accepted['loss'])
            self.step_diagnostics_.append(dict(kind=accepted['kind'],prior_weight=weight,prior_delta=float(da),tree_rate=float(rate),backtracks=accepted['backtracks']))
            key=selection_key(y_val,validation);self.validation_history_.append(key)
            if key>best:best=key;self.best_iteration_=len(self.steps_);patience=0
            else:patience+=1
            if patience>=self.early_stopping_rounds:break
        self.validation_key_=best
        return self
    def decision_function(self,x):
        z=self._features(x);score=self.prior_weights_[self.best_iteration_]*self._prior(z)
        for tree,rate in self.steps_[:self.best_iteration_]:
            if rate:score+=rate*tree.predict(z)
        return score
