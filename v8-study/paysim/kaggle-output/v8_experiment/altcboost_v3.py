"""ALTCBoost V3: a *research prototype* of hard-negative-aware linear/tree/cross boosting.

Research ingredients motivated by partial ROC-AUC and Neyman-Pearson literature:
    - Explicit emphasis on the current high-scoring negative examples in boosted
      Newton updates (adaptive weights; *not* exact partial AUC minimization).
    - Validation model selection trades average precision and standardized
      partial ROC-AUC at a small false-positive rate.
    - Independent calibration via the inherited V2 method.
    - Compact compiled Numba inference and exact decision early exit from V2.

This is an experimental combination of known ideas, NOT a claim of a novel
published algorithm or superiority to strong established baselines.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from altcboost_v2 import ALTCBoostV2


class ALTCBoostV3(ALTCBoostV2):
    def __init__(self, *, hard_negative_strength=2.5,
                 hard_negative_fraction=0.08,
                 hard_negative_temperature=0.35,
                 hard_negative_warmup=12,
                 low_fpr_validation=0.05,
                 validation_ap_weight=0.5,
                 n_estimators=90, learning_rate=0.08, max_leaf_nodes=9,
                 min_samples_leaf=35, linear_C=1.0, pair_penalty=10.0,
                 pair_rate=0.35, early_stopping_rounds=16, random_state=42,
                 use_cross=True):
        super().__init__(n_estimators=n_estimators,
            learning_rate=learning_rate,max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,linear_C=linear_C,
            pair_penalty=pair_penalty,pair_rate=pair_rate,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,use_cross=use_cross)
        if not (0 < hard_negative_fraction <= 0.5):
            raise ValueError('hard_negative_fraction must be in (0, .5]')
        if hard_negative_strength < 0 or hard_negative_temperature <= 0:
            raise ValueError('invalid hard negative weighting')
        if not 0 < low_fpr_validation <= 1:
            raise ValueError('low_fpr_validation must be in (0,1]')
        if not 0 <= validation_ap_weight <= 1:
            raise ValueError('validation_ap_weight must be in [0,1]')
        self.hard_negative_strength=float(hard_negative_strength)
        self.hard_negative_fraction=float(hard_negative_fraction)
        self.hard_negative_temperature=float(hard_negative_temperature)
        self.hard_negative_warmup=int(hard_negative_warmup)
        self.low_fpr_validation=float(low_fpr_validation)
        self.validation_ap_weight=float(validation_ap_weight)

    def _validation_objective(self, y, logits):
        # standardized partial ROC-AUC by sklearn; the max_fpr is ONLY used
        # for model selection, NOT a guarantee of future operating-point FPR.
        ap=average_precision_score(y, logits)
        if self.low_fpr_validation < 1:
            pauc=roc_auc_score(y, logits, max_fpr=self.low_fpr_validation)
        else:
            pauc=roc_auc_score(y,logits)
        return self.validation_ap_weight*ap+(1-self.validation_ap_weight)*pauc

    def _weights(self,y,F,iteration):
        w=np.ones_like(F)
        if self.hard_negative_strength<=0 or iteration<self.hard_negative_warmup:
            return w
        negatives=(y==0)
        # Train-only order statistic. Never uses validation, calibration, or test.
        cutoff=np.quantile(F[negatives],1-self.hard_negative_fraction)
        softness=expit(np.clip((F[negatives]-cutoff)/self.hard_negative_temperature,-30,30))
        w[negatives]+=self.hard_negative_strength*softness
        # Average weight normalization does not change Newton ratios, but avoids
        # exploding regularization interactions during training.
        return w / w.mean()

    def fit(self,X,y,X_val=None,y_val=None):
        X=np.asarray(X,dtype=np.float64)
        y=np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or set(np.unique(y)) != {0,1}:
            raise ValueError('y must contain binary labels 0 and 1, one per row')
        if X.ndim!=2 or not np.isfinite(X).all():
            raise ValueError('X must be a finite numeric 2D matrix')
        if X_val is None:
            X,X_val,y,y_val=train_test_split(X,y,test_size=.2,stratify=y,random_state=self.random_state)
        else:
            X_val=np.asarray(X_val,dtype=np.float64)
            y_val=np.asarray(y_val)
        if (X_val.ndim != 2 or X_val.shape[1] != X.shape[1] or
                y_val.ndim != 1 or len(y_val) != len(X_val)):
            raise ValueError('Validation rows and feature count must match')
        if not np.isfinite(X_val).all() or set(np.unique(y))!={0,1} or set(np.unique(y_val))!={0,1}:
            raise ValueError('both classes required and all features must be finite')
        # A threshold calibrated for a previous fit does not apply to the new model.
        for name in list(vars(self)):
            if name == 'decision_threshold_' or name.startswith('calibration_'):
                delattr(self, name)
        self.scaler_=StandardScaler().fit(X)
        Xt=self._features(X); Xv=self._features(X_val)
        self.linear_=LogisticRegression(C=self.linear_C,solver='lbfgs',max_iter=500,
                                       random_state=self.random_state).fit(Xt,y)
        F=self.linear_.decision_function(Xt).astype(np.float64)
        Fv=self.linear_.decision_function(Xv).astype(np.float64)
        self.steps_=[]
        self.pairs_=np.array([(i,j) for i in range(X.shape[1])
                              for j in range(i+1,X.shape[1])],dtype=np.int64).reshape(-1,2)
        if self.pairs_.size and self.use_cross:
            ii,jj=self.pairs_[:,0],self.pairs_[:,1]
            raw=Xt[:,ii]*Xt[:,jj]
            self.cross_means_=raw.mean(axis=0)
            self.cross_stds_=np.maximum(raw.std(axis=0),.05)
            Ct=self._cross_matrix(Xt);Cv=self._cross_matrix(Xv)
        else:
            self.cross_means_=np.array([]);self.cross_stds_=np.array([])
            Ct=Cv=None
        best=self._validation_objective(y_val,Fv)
        best_i=0
        patience=0
        self.validation_history_=[]
        for t in range(self.n_estimators):
            p=expit(np.clip(F,-25,25))
            weights=self._weights(y,F,t)
            # Diagonal Newton approximation for a weighted logistic loss.
            grad=(y-p)*weights
            hess=np.maximum(p*(1-p),0.002)*weights
            response=np.clip(grad/hess,-15,15)
            tree=DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state+t)
            tree.fit(Xt,response,sample_weight=hess)
            F+=self.learning_rate*np.clip(tree.predict(Xt),-5,5)
            Fv+=self.learning_rate*np.clip(tree.predict(Xv),-5,5)
            cross_info=None
            if Ct is not None and Ct.shape[1]:
                p=expit(np.clip(F,-25,25))
                weights=self._weights(y,F,t)
                grad=(y-p)*weights
                hess=np.maximum(p*(1-p),0.002)*weights
                num=Ct.T@grad
                den=(Ct**2).T@hess+self.pair_penalty
                gain=num*num/den
                j=int(np.argmax(gain))
                coefficient=np.clip(num[j]/den[j],-3,3)*self.learning_rate*self.pair_rate
                F+=coefficient*Ct[:,j]
                Fv+=coefficient*Cv[:,j]
                cross_info=(j,float(coefficient))
            self.steps_.append((tree,cross_info))
            current=self._validation_objective(y_val,Fv)
            self.validation_history_.append(current)
            if current > best+0.00001:
                best=current;best_i=t+1;patience=0
            else:
                patience+=1
            if patience>=self.early_stopping_rounds:break
        self.best_iteration_=best_i
        self.validation_objective_=best
        self.compile()
        self.validation_logloss_=self._loss(y_val,self.decision_function(X_val))
        return self
