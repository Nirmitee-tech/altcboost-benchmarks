"""Shared calibration and held-out metrics for all benchmark families."""
import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

def calibrate(y, score, alpha=.001, confidence=.95):
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if y.ndim != 1 or score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError('Expected equally sized finite one-dimensional arrays')
    if not set(np.unique(y)).issubset({0, 1}) or not 0 < alpha < 1 or not 0 < confidence < 1:
        raise ValueError('Invalid labels or calibration settings')
    neg = np.sort(score[y == 0])[::-1]
    if not len(neg):
        raise ValueError('Calibration requires negative examples')
    # This bound assumes independent calibration negatives with the future negative distribution.
    k = np.arange(int(np.floor(alpha * len(neg))) + 1)
    upper = beta.ppf(confidence, k + 1, len(neg) - k)
    feasible = k[upper <= alpha]
    if not len(feasible):
        raise ValueError('Insufficient calibration negatives for requested confidence')
    allowed = int(feasible[-1])
    return float(np.nextafter(neg[allowed], np.inf)), dict(
        negative_count=len(neg), allowed=allowed,
        tolerance_bound=float(upper[allowed]), alpha=alpha, confidence=confidence)

def metrics(y, score, threshold, alpha=.001):
    # A float64 nextafter cutoff can round back to a tie in float32 comparisons.
    y, score = np.asarray(y), np.asarray(score,dtype=np.float64)
    flags = score >= threshold
    pos = y == 1; neg = y == 0
    tp = int((flags & pos).sum()); fp = int((flags & neg).sum())
    p, n = int(pos.sum()), int(neg.sum())
    if not p or not n:
        raise ValueError('Test metrics require both classes')
    return dict(ap=float(average_precision_score(y, score)),
                pauc=float(roc_auc_score(y, score, max_fpr=alpha)),
                tp=tp, fp=fp, fn=p-tp, tn=n-fp, recall=tp/p, fpr=fp/n,
                recall_lower_95=float(beta.ppf(.025,tp,p-tp+1)) if tp else 0.,
                recall_upper_95=float(beta.ppf(.975,tp+1,p-tp)) if tp < p else 1.,
                fpr_upper_95=float(beta.ppf(.95, fp+1, n-fp)) if fp < n else 1.,
                precision=tp/max(tp+fp, 1), alerts=tp+fp)
