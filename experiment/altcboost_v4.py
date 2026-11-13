"""Detection-focused research branch with regularized Newton leaf updates.

The learner supports a linear or constant prior and optional feature clipping.
It does not use the V3 hard-negative heuristic or explicit pair interactions.
Prediction uses the reference tree implementation until detection gains are established.
"""
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from detection_objective import selection_key

def leaf_updates(leaf_ids,gradient,hessian,regularization,max_step):
    if regularization<0 or max_step<=0:
        raise ValueError('Invalid leaf regularization or step limit')
    ids=np.asarray(leaf_ids,dtype=int)
    numerator=np.bincount(ids,weights=gradient)
    denominator=np.bincount(ids,weights=hessian)+regularization
    return np.clip(numerator/np.maximum(denominator,1e-12),-max_step,max_step)

class ALTCBoostV4:
    def __init__(self,n_estimators=100,learning_rate=.08,max_leaf_nodes=9,
                 min_samples_leaf=40,leaf_l2=1.,max_leaf_step=5.,linear_prior=True,
                 feature_clip=6.,linear_C=1.,early_stopping_rounds=16,random_state=17):
        self.n_estimators=n_estimators;self.learning_rate=learning_rate
        self.max_leaf_nodes=max_leaf_nodes;self.min_samples_leaf=min_samples_leaf
        self.leaf_l2=leaf_l2;self.max_leaf_step=max_leaf_step
        self.linear_prior=linear_prior;self.feature_clip=feature_clip
        self.linear_C=linear_C;self.early_stopping_rounds=early_stopping_rounds
        self.random_state=random_state
        if n_estimators<1 or learning_rate<=0 or leaf_l2<0 or max_leaf_step<=0:
            raise ValueError('Invalid training settings')
        if feature_clip is not None and feature_clip<=0:
            raise ValueError('feature_clip must be positive or None')
    def _features(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=self.n_features_ or not np.isfinite(x).all():
            raise ValueError('Expected finite rows with the fitted feature count')
        z=self.scaler_.transform(x)
        return np.clip(z,-self.feature_clip,self.feature_clip) if self.feature_clip is not None else z
    def _prior(self,z):
        if self.linear_ is not None:return self.linear_.decision_function(z)
        return np.full(len(z),self.intercept_,dtype=float)
    def fit(self,x,y,x_val,y_val):
        x=np.asarray(x,dtype=float);y=np.asarray(y);x_val=np.asarray(x_val,dtype=float);y_val=np.asarray(y_val)
        if (x.ndim!=2 or x_val.ndim!=2 or x.shape[1]!=x_val.shape[1] or
            y.ndim!=1 or y_val.ndim!=1 or len(x)!=len(y) or len(x_val)!=len(y_val) or
            set(np.unique(y))!={0,1} or set(np.unique(y_val))!={0,1} or
            not np.isfinite(x).all() or not np.isfinite(x_val).all()):
            raise ValueError('Finite matching matrices and both binary classes are required')
        self.n_features_=x.shape[1];self.scaler_=StandardScaler().fit(x)
        xt,xv=self._features(x),self._features(x_val)
        self.linear_=LogisticRegression(C=self.linear_C,max_iter=500).fit(xt,y) if self.linear_prior else None
        self.intercept_=float(logit(np.mean(y)))
        margin=self._prior(xt);validation=self._prior(xv)
        self.steps_=[];self.best_iteration_=0
        best=selection_key(y_val,validation);patience=0
        self.initial_validation_key_=best
        self.validation_history_=[]
        for iteration in range(self.n_estimators):
            p=expit(margin)
            gradient=y-p;hessian=np.maximum(p*(1-p),1e-6)
            # Structure uses the Newton response; leaf values are then recomputed from sums.
            response=gradient/hessian
            tree=DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,random_state=self.random_state+iteration)
            tree.fit(xt,response,sample_weight=hessian)
            leaves=tree.apply(xt)
            values=leaf_updates(leaves,gradient,hessian,self.leaf_l2,self.max_leaf_step)
            tree.tree_.value[:,0,0]=0.
            for leaf in np.unique(leaves):tree.tree_.value[leaf,0,0]=values[leaf]
            margin+=self.learning_rate*tree.predict(xt)
            validation+=self.learning_rate*tree.predict(xv)
            self.steps_.append(tree)
            key=selection_key(y_val,validation)
            self.validation_history_.append(key)
            if key>best:
                best=key;self.best_iteration_=iteration+1;patience=0
            else:patience+=1
            if patience>=self.early_stopping_rounds:break
        self.validation_key_=best
        return self
    def decision_function(self,x):
        z=self._features(x);margin=self._prior(z)
        for tree in self.steps_[:self.best_iteration_]:
            margin+=self.learning_rate*tree.predict(z)
        return margin
    def predict_proba(self,x):
        positive=expit(self.decision_function(x))
        return np.column_stack([1-positive,positive])
