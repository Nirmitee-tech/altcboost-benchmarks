import itertools
import numpy as np
import pytest
from prior_diagnostic import recall_ceiling
from detection_objective import empirical_cutoff

@pytest.mark.parametrize('alpha',[.2,.4])
def test_recall_ceiling_holds_for_exhaustive_bounded_corrections(alpha):
    y=np.array([0,0,0,1,1]);prior=np.array([1.,2.,3.,1.2,2.4]);bound=.5
    ceiling=recall_ceiling(y,prior,bound,alpha)['empirical_recall_upper_bound']
    for correction in itertools.product([-bound,0,bound],repeat=len(y)):
        score=prior+correction;threshold=empirical_cutoff(y,score,alpha)
        recall=float((score[y==1]>=threshold).mean())
        assert recall<=ceiling

def test_extreme_wrong_prior_cannot_be_repaired_by_small_residuals():
    result=recall_ceiling([0,0,0,1,1],[200,201,202,0,1],40,.1)
    assert result['empirical_recall_upper_bound']==0.

def test_relaxing_the_correction_limit_cannot_lower_the_ceiling():
    y=[0,0,1,1];score=[10,20,2,4]
    values=[recall_ceiling(y,score,bound,.1)['empirical_recall_upper_bound'] for bound in [0,2,10]]
    assert values==sorted(values)
