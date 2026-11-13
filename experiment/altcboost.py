"""ALTCBoost — experimental Alternating Linear, Tree and Cross boosting classifier.

Research prototype for numeric tabular binary classification. Not claimed to be
novel relative to all published literature, or universally better than XGBoost.

Algorithm:
  1. Fit a regularized logistic (global linear) prior.
  2. Correct logits using 2nd-order (Newton/IRLS) regression trees.
  3. After every tree, greedily choose one explicit pairwise interaction using
     second-order gain, update the logit margin with an L2-regularized step.
  4. Use a held-out validation set for early stopping.

No XGBoost used in the proposed model itself.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy.special import expit, logit
from sklearn.metrics import log_loss


class ALTCBoostClassifier:
    def __init__(self, n_estimators=90, learning_rate=0.08, max_leaf_nodes=9,
                 min_samples_leaf=35, linear_C=1.0, pair_penalty=10.0,
                 pair_rate=0.35, early_stopping_rounds=16, random_state=42,
                 use_cross=True):
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.max_leaf_nodes=max_leaf_nodes
        self.min_samples_leaf=min_samples_leaf
        self.linear_C=linear_C
        self.pair_penalty=pair_penalty
        self.pair_rate=pair_rate
        self.early_stopping_rounds=early_stopping_rounds
        self.random_state=random_state
        self.use_cross=use_cross

    def _features(self, X):
        X=np.asarray(X,dtype=float)
        if X.ndim != 2: raise ValueError('X must be a 2D numeric array')
        return np.clip(self.scaler_.transform(X),-6,6)

    def _cross_matrix(self,X):
        i,j=self.pairs_[:,0],self.pairs_[:,1]
        C=X[:,i]*X[:,j]
        # subtract train-time means and divide train-time standard deviations
        return np.clip((C-self.cross_means_)/self.cross_stds_,-10,10)

    @staticmethod
    def _loss(y, logits):
        return float(np.mean(np.logaddexp(0, logits)-y*logits))

    def fit(self, X, y, X_val=None, y_val=None):
        X=np.asarray(X,dtype=float)
        y=np.asarray(y,dtype=int)
        if not np.all(np.isfinite(X)): raise ValueError('Only finite numerical X supported in prototype.')
        if X_val is None:
            X,X_val,y,y_val=train_test_split(X,y,test_size=0.22,stratify=y,random_state=self.random_state)
        else:
            X_val=np.asarray(X_val,dtype=float)
            y_val=np.asarray(y_val,dtype=int)
        if np.unique(y).size!=2: raise ValueError('binary y with both classes required')
        self.scaler_=StandardScaler().fit(X)
        Xt=self._features(X); Xv=self._features(X_val)
        self.linear_=LogisticRegression(C=self.linear_C, solver='lbfgs',max_iter=500,random_state=self.random_state).fit(Xt,y)
        F=self.linear_.decision_function(Xt).astype(float)
        Fv=self.linear_.decision_function(Xv).astype(float)
        self.steps_=[]
        self.best_iteration_=0
        best_loss=self._loss(y_val,Fv)
        best_idx=0
        patience=0
        self.pairs_=np.array([(i,j) for i in range(X.shape[1]) for j in range(i+1,X.shape[1])],dtype=int).reshape(-1,2)
        if self.pairs_.size and self.use_cross:
            i,j=self.pairs_[:,0],self.pairs_[:,1]
            Ct0=Xt[:,i]*Xt[:,j]
            self.cross_means_=Ct0.mean(axis=0)
            self.cross_stds_=np.maximum(Ct0.std(axis=0),.05)
            Ct=self._cross_matrix(Xt); Cv=self._cross_matrix(Xv)
        else:
            self.cross_means_=np.array([])
            self.cross_stds_=np.array([])
            Ct=Cv=None
        for t in range(self.n_estimators):
            p=expit(np.clip(F,-25,25))
            g=y-p
            h=np.maximum(p*(1-p),0.002)
            newton=np.clip(g/h,-15,15)
            tree=DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,
                 min_samples_leaf=self.min_samples_leaf,random_state=self.random_state+t)
            tree.fit(Xt,newton,sample_weight=h)
            dtr=np.clip(tree.predict(Xt),-5,5)*self.learning_rate
            dval=np.clip(tree.predict(Xv),-5,5)*self.learning_rate
            F+=dtr; Fv+=dval
            cross_info=None
            if Ct is not None and Ct.shape[1]:
                p=expit(np.clip(F,-25,25))
                g=y-p; h=np.maximum(p*(1-p),0.002)
                # one-coordinate Newton gain, optimized on the current residual
                numerator=Ct.T@g
                denominator=(Ct**2).T@h+self.pair_penalty
                scores=numerator*numerator/denominator
                j=int(np.argmax(scores))
                coeff=np.clip(numerator[j]/denominator[j],-3,3)*self.learning_rate*self.pair_rate
                F+=coeff*Ct[:,j]
                Fv+=coeff*Cv[:,j]
                cross_info=(j, float(coeff))
            self.steps_.append((tree,cross_info))
            val_loss=self._loss(y_val,Fv)
            if val_loss<best_loss-1e-5:
                best_loss=val_loss
                best_idx=t+1
                patience=0
            else:
                patience+=1
            if patience >= self.early_stopping_rounds:
                break
        self.best_iteration_=best_idx
        self.validation_logloss_=best_loss
        return self

    def decision_function(self, X):
        Xt=self._features(X)
        F=self.linear_.decision_function(Xt).astype(float)
        Ct=self._cross_matrix(Xt) if self.use_cross and self.pairs_.size else None
        for tree, cross_info in self.steps_[:self.best_iteration_]:
            F+=self.learning_rate*np.clip(tree.predict(Xt),-5,5)
            if cross_info is not None:
                j,coeff=cross_info
                F+=coeff*Ct[:,j]
        return F

    def predict_proba(self,X):
        p=expit(np.clip(self.decision_function(X),-35,35))
        return np.column_stack((1-p,p))

    def predict(self,X):
        return (self.predict_proba(X)[:,1]>.5).astype(int)
