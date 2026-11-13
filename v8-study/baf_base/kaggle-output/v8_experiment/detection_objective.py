"""Development selection at fixed empirical false-positive budgets."""
import numpy as np
from sklearn.metrics import average_precision_score

def empirical_cutoff(y,score,alpha):
    y,score=np.asarray(y),np.asarray(score,dtype=float)
    if y.ndim!=1 or score.shape!=y.shape or not np.isfinite(score).all():
        raise ValueError('Expected equally sized finite label and score arrays')
    if set(np.unique(y))!={0,1} or not 0<alpha<1:
        raise ValueError('Both binary classes and a valid budget are required')
    negative=np.sort(score[y==0])[::-1]
    return float(np.nextafter(negative[int(np.floor(alpha*len(negative)))],np.inf))

def selection_key(y,score,budgets=(.0005,.001)):
    # Float64 comparisons preserve a nextafter cutoff above tied float32 scores.
    y=np.asarray(y);score=np.asarray(score,dtype=np.float64)
    recall=[];false_positive=[]
    for alpha in budgets:
        flags=score>=empirical_cutoff(y,score,alpha)
        recall.append(float(flags[y==1].mean()))
        false_positive.append(float(flags[y==0].mean()))
    # This is a validation ranking rule, not a constraint on future traffic.
    return (float(np.mean(recall)),-float(np.mean(false_positive)),
            float(average_precision_score(y,score)))
