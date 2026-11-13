import numpy as np
import pytest
from altcboost_v4 import leaf_updates
from detection_objective import empirical_cutoff,selection_key
from quality_discovery import update_retention
from evaluation import calibrate,metrics

def test_leaf_value_uses_summed_gradient_and_regularized_curvature():
    result=leaf_updates([1,1,1],[.99,.99,-.01],[.01,.01,.01],1.,5.)
    assert result[1]==pytest.approx(1.97/1.03)

def test_leaf_step_is_bounded_after_aggregation():
    result=leaf_updates([1,1],[.99,.99],[.001,.001],0.,5.)
    assert result[1]==5.

def test_discovery_detects_clipped_fraud_gradient():
    response,retention=update_retention([1,0],[.0001,.0001])
    assert response[0]>15
    assert retention[0]==pytest.approx(.03/.9999)
    assert retention[1]==1.

def test_tied_negative_scores_do_not_exceed_empirical_budget():
    y=np.array([0]*1000+[1,1]);score=np.r_[np.full(1000,.1),[.9,.05]]
    cutoff=empirical_cutoff(y,score,.001)
    assert (score[y==0]>=cutoff).sum()==0
    assert selection_key(y,score)[0]==.5

def test_recall_at_the_budget_beats_an_all_negative_solution():
    y=np.array([0]*1000+[1,1])
    good=np.r_[np.full(1000,.1),[.9,.05]]
    assert selection_key(y,good)>selection_key(y,np.zeros(len(y)))

def test_fractional_labels_cannot_be_used_for_threshold_selection():
    with pytest.raises(ValueError,match='binary classes'):
        empirical_cutoff([0,.9],[.1,.9],.001)

def test_float32_ties_cannot_undo_a_conservative_threshold():
    y=np.array([0]*10000+[1,1]);scores=np.full(len(y),.0013034,dtype=np.float32)
    cutoff,_=calibrate(y,scores,.001)
    result=metrics(y,scores,cutoff)
    assert result['fp']==0 and result['tp']==0
    assert selection_key(y,scores)[0]==0.
