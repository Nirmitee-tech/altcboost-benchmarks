"""ALTCBoost 2.0 research prototype: exact compiled inference + FPR operating point.

Fit uses the original ALTCBoost algorithm (same training and ranking). Compiles
trained coefficients and tree structure into Numba CPU kernels for single-event
inference; builds optional safe branch-and-bound classification using tree leaf
bounds. Calibrates an empirical operating threshold from *separate* labeled
calibration data. No claim of population FPR guarantee or universal superiority.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit, logit
from scipy.stats import beta
from numba import njit
from altcboost import ALTCBoostClassifier


@njit(cache=True)
def _score(X, means, scales, coef, intercept, ci, cj, cm, cs, cc,
           offsets, left, right, feat, threshold, vals, lr):
    out = np.empty(X.shape[0], dtype=np.float64)
    for row in range(X.shape[0]):
        z = np.empty(X.shape[1], dtype=np.float64)
        m = intercept
        for j in range(X.shape[1]):
            v = (X[row,j] - means[j]) / scales[j]
            z[j] = min(6.0, max(-6.0, v))
            m += coef[j] * z[j]
        for j in range(len(ci)):
            c = (z[ci[j]] * z[cj[j]] - cm[j]) / cs[j]
            m += cc[j] * min(10.0, max(-10.0, c))
        for t in range(len(offsets)-1):
            base = offsets[t]
            node = base
            while left[node] != -1:
                # sklearn converts tree inputs to float32 before comparing splits.
                if np.float32(z[feat[node]]) <= threshold[node]:
                    node = base + left[node]
                else:
                    node = base + right[node]
            m += lr * min(5.0,max(-5.0,vals[node]))
        out[row] = m
    return out


@njit(cache=True)
def _classify_early(X, decision_logit, means, scales, coef, intercept,
                    ci, cj, cm, cs, cc, offsets, left, right, feat,
                    threshold, vals, lr, suffix_min, suffix_max):
    out = np.zeros(X.shape[0], dtype=np.uint8)
    trees_evaluated = np.zeros(X.shape[0],dtype=np.int32)
    for row in range(X.shape[0]):
        z = np.empty(X.shape[1], dtype=np.float64)
        m = intercept
        for j in range(X.shape[1]):
            v = (X[row,j] - means[j]) / scales[j]
            z[j] = min(6.0,max(-6.0,v))
            m += coef[j]*z[j]
        for j in range(len(ci)):
            c = (z[ci[j]]*z[cj[j]]-cm[j])/cs[j]
            m += cc[j]*min(10.0,max(-10.0,c))
        for t in range(len(offsets)-1):
            if m+suffix_max[t] < decision_logit:
                break
            if m+suffix_min[t] >= decision_logit:
                out[row]=1
                break
            base=offsets[t]
            node=base
            while left[node]!=-1:
                if np.float32(z[feat[node]]) <= threshold[node]:
                    node=base+left[node]
                else:
                    node=base+right[node]
            m += lr*min(5.0,max(-5.0,vals[node]))
            trees_evaluated[row]+=1
        else:
            out[row]=int(m>=decision_logit)
    return out,trees_evaluated


class ALTCBoostV2(ALTCBoostClassifier):
    """Numerically compatible accelerated prediction of an ALTCBoost model."""
    def fit(self, X, y, X_val=None, y_val=None):
        super().fit(X,y,X_val,y_val)
        for name in list(vars(self)):
            if name == 'decision_threshold_' or name.startswith('calibration_'):
                delattr(self, name)
        return self.compile()

    def compile(self):
        self._means=np.asarray(self.scaler_.mean_,dtype=np.float64)
        self._scales=np.asarray(self.scaler_.scale_,dtype=np.float64)
        self._coef=np.asarray(self.linear_.coef_[0],dtype=np.float64)
        self._intercept=float(self.linear_.intercept_[0])
        # Collapse repeated pairwise updates into a single coefficient / unique pair.
        agg={}
        for tree,cross in self.steps_[:self.best_iteration_]:
            if cross is not None:
                idx, delta=cross
                agg[idx]=agg.get(idx,0.)+delta
        ids=np.array(list(agg),dtype=np.int32)
        self._ci=np.asarray(self.pairs_[ids,0] if len(ids) else [],dtype=np.int32)
        self._cj=np.asarray(self.pairs_[ids,1] if len(ids) else [],dtype=np.int32)
        self._cm=np.asarray(self.cross_means_[ids] if len(ids) else [],dtype=np.float64)
        self._cs=np.asarray(self.cross_stds_[ids] if len(ids) else [],dtype=np.float64)
        self._cc=np.asarray(list(agg.values()),dtype=np.float64)
        offsets=[0]; left=[];right=[]; feat=[];thr=[];vals=[]
        mins=[];maxs=[]
        for tree,_ in self.steps_[:self.best_iteration_]:
            nodes=tree.tree_
            l=nodes.children_left.astype(np.int32)
            r=nodes.children_right.astype(np.int32)
            v=nodes.value[:,0,0].astype(np.float64)
            left.extend(l);right.extend(r)
            feat.extend(nodes.feature.astype(np.int32))
            thr.extend(nodes.threshold.astype(np.float64));vals.extend(v)
            offsets.append(offsets[-1]+len(v))
            leafvals=np.clip(v[l==-1],-5.,5.)*self.learning_rate
            mins.append(leafvals.min());maxs.append(leafvals.max())
        self._offsets=np.asarray(offsets,dtype=np.int32)
        self._left=np.asarray(left,dtype=np.int32)
        self._right=np.asarray(right,dtype=np.int32)
        self._feat=np.asarray(feat,dtype=np.int32)
        self._thr=np.asarray(thr,dtype=np.float64)
        self._vals=np.asarray(vals,dtype=np.float64)
        self._suffix_min=np.r_[np.cumsum(mins[::-1])[::-1],0.].astype(np.float64)
        self._suffix_max=np.r_[np.cumsum(maxs[::-1])[::-1],0.].astype(np.float64)
        return self

    def _args(self):
        return (self._means,self._scales,self._coef,self._intercept,
                self._ci,self._cj,self._cm,self._cs,self._cc,self._offsets,
                self._left,self._right,self._feat,self._thr,self._vals,float(self.learning_rate))

    def decision_function(self,X):
        X=np.asarray(X,dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self._means) or not np.isfinite(X).all():
            raise ValueError('Input must be finite with the fitted feature count')
        return _score(X,*self._args())

    def predict_proba(self,X):
        p=expit(np.clip(self.decision_function(X),-35.,35.))
        return np.column_stack((1.-p,p))

    def fit_threshold(self, X_cal, y_cal, max_fpr=0.01, confidence=None):
        """Conservative empirical FPR cap, not a future false-positive guarantee.

        Use a calibration set distinct from train, early-stop selection, and test.
        """
        if not 0.<=max_fpr<1.:
            raise ValueError('max_fpr must be between 0 and 1')
        labels=np.asarray(y_cal,dtype=np.int32)
        neg=np.sort(self.predict_proba(X_cal)[labels==0,1])[::-1]
        if len(neg)==0: raise ValueError('No negative calibration examples')
        allow=int(np.floor(max_fpr*len(neg)))
        if confidence is not None:
            if not 0. < confidence < 1.:
                raise ValueError('confidence must be in (0,1)')
            # One-sided order-statistic tolerance bound using Beta tails.
            # Requires calibration negatives representative of future cases.
            permitted=[k for k in range(allow+1)
                       if float(beta.ppf(confidence,k+1,len(neg)-k)) <= max_fpr]
            if not permitted:
                raise ValueError('Not enough negative calibration samples for a'
                                 ' statistically supported FPR bound at this confidence')
            allow=max(permitted)
        if allow>=len(neg):
            raise ValueError('max_fpr too large')
        self.decision_threshold_=float(np.nextafter(neg[allow],np.inf))
        self.calibration_n_negative_=len(neg)
        self.calibration_max_fpr_=max_fpr
        self.calibration_confidence_=confidence
        self.calibration_allowed_false_positives_=allow
        return self

    def predict_flags(self,X,threshold=None):
        if threshold is None:
            threshold=getattr(self,'decision_threshold_',None)
        if threshold is None:
            raise ValueError('fit_threshold or pass a threshold before classification')
        if not 0.<threshold<1.: raise ValueError('threshold should be within (0,1)')
        X=np.asarray(X,dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self._means) or not np.isfinite(X).all():
            raise ValueError('Input must be finite with the fitted feature count')
        return _classify_early(X,float(logit(threshold)),*self._args(),
                               self._suffix_min,self._suffix_max)
