"""Experimental temporal-tail tree boosting with checked empirical descent."""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from detection_objective import selection_key
from tail_objective import tail_objective

class ALTCBoostV5:
    def __init__(self,n_estimators=100,learning_rate=.08,max_leaf_nodes=17,
                 min_samples_leaf=40,leaf_l2=1.,max_leaf_step=5.,tail_fraction=.01,
                 temperature=1.,group_temperature=.25,anchor_weight=.2,
                 early_stopping_rounds=16,random_state=17):
        if (n_estimators<1 or learning_rate<=0 or leaf_l2<0 or max_leaf_step<=0 or
            max_leaf_nodes<2 or min_samples_leaf<1 or early_stopping_rounds<1):
            raise ValueError('Invalid tree or training settings')
        if not (0<tail_fraction<=1 and temperature>0 and group_temperature>0 and 0<=anchor_weight<=1):
            raise ValueError('Invalid objective settings')
        self.n_estimators=n_estimators;self.learning_rate=learning_rate
        self.max_leaf_nodes=max_leaf_nodes;self.min_samples_leaf=min_samples_leaf
        self.leaf_l2=leaf_l2;self.max_leaf_step=max_leaf_step;self.tail_fraction=tail_fraction
        self.temperature=temperature;self.group_temperature=group_temperature
        self.anchor_weight=anchor_weight;self.early_stopping_rounds=early_stopping_rounds
        self.random_state=random_state

    def objective(self,y,score,groups):
        return tail_objective(y,score,groups,self.tail_fraction,self.temperature,
                              self.group_temperature,self.anchor_weight)

    def _features(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=self.n_features_ or not np.isfinite(x).all():
            raise ValueError('Expected finite rows with the fitted feature count')
        return self.scaler_.transform(x)

    def fit(self,x,y,x_val,y_val,groups=None):
        x=np.asarray(x,dtype=float);x_val=np.asarray(x_val,dtype=float)
        y=np.asarray(y);y_val=np.asarray(y_val)
        if (x.ndim!=2 or x_val.ndim!=2 or x.shape[1]!=x_val.shape[1] or
            y.ndim!=1 or y_val.ndim!=1 or len(y)!=len(x) or len(y_val)!=len(x_val) or
            set(np.unique(y))!={0,1} or set(np.unique(y_val))!={0,1} or
            not np.isfinite(x).all() or not np.isfinite(x_val).all()):
            raise ValueError('Finite matching matrices and both binary classes are required')
        groups=np.zeros(len(y),dtype=int) if groups is None else np.asarray(groups)
        self.n_features_=x.shape[1];self.scaler_=StandardScaler().fit(x)
        xt=self._features(x);xv=self._features(x_val)
        self.linear_=LogisticRegression(class_weight='balanced',max_iter=500,
            random_state=self.random_state).fit(xt,y)
        score=self.linear_.decision_function(xt);validation=self.linear_.decision_function(xv)
        value,_,_,_=self.objective(y,score,groups)
        self.training_loss_=[value];self.steps_=[];self.best_iteration_=0
        self.validation_history_=[];self.step_diagnostics_=[]
        best=selection_key(y_val,validation);patience=0
        for iteration in range(self.n_estimators):
            value,gradient,hessian,detail=self.objective(y,score,groups)
            # Scaling preserves the objective direction and sets a per-row L2 scale.
            gradient=gradient*len(y);hessian=np.maximum(hessian*len(y),1e-6)
            tree=DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,random_state=self.random_state+iteration)
            tree.fit(xt,np.clip(-gradient/hessian,-50,50),sample_weight=hessian)
            leaves=tree.apply(xt)
            numerator=np.bincount(leaves,weights=-gradient)
            denominator=np.bincount(leaves,weights=hessian)+self.leaf_l2
            values=np.clip(numerator/np.maximum(denominator,1e-12),-self.max_leaf_step,self.max_leaf_step)
            tree.tree_.value[:,0,0]=0.
            for leaf in np.unique(leaves):tree.tree_.value[leaf,0,0]=values[leaf]
            direction=tree.predict(xt);accepted=False
            for backtracks in range(12):
                rate=self.learning_rate*2.**(-backtracks)
                candidate=score+rate*direction
                candidate_loss=self.objective(y,candidate,groups)[0]
                if candidate_loss<value-1e-12:
                    accepted=True;break
            if not accepted:break
            self.steps_.append((tree,rate));score=candidate
            validation+=rate*tree.predict(xv)
            self.training_loss_.append(candidate_loss)
            self.step_diagnostics_.append(dict(rate=rate,backtracks=backtracks,**detail))
            key=selection_key(y_val,validation);self.validation_history_.append(key)
            if key>best:best=key;self.best_iteration_=len(self.steps_);patience=0
            else:patience+=1
            if patience>=self.early_stopping_rounds:break
        self.validation_key_=best
        return self

    def decision_function(self,x):
        z=self._features(x);score=self.linear_.decision_function(z)
        for tree,rate in self.steps_[:self.best_iteration_]:score+=rate*tree.predict(z)
        return score

    def predict_proba(self,x):
        # A sigmoid score interface does not establish calibrated fraud probabilities.
        positive=expit(self.decision_function(x))
        return np.column_stack([1-positive,positive])
