import numpy as np
import pytest
from tail_objective import tail_objective,temporal_groups,evaluation_windows

@pytest.mark.parametrize('anchor',[0.,.2,1.])
@pytest.mark.parametrize('tail',[.5,1.])
def test_analytic_gradient_and_diagonal_hessian_match_finite_differences(anchor,tail):
    y=np.tile([0,0,0,1,1],2);score=np.array([-.7,.3,1.2,-.2,.9, .2,-.5,1.8,.4,-.8])
    groups=np.repeat([0,1],5)
    value,g,h,_=tail_objective(y,score,groups,tail_fraction=tail,anchor_weight=anchor)
    for index in range(len(y)):
        step=np.zeros(len(y));step[index]=1e-4
        plus=tail_objective(y,score+step,groups,tail_fraction=tail,anchor_weight=anchor)[0]
        minus=tail_objective(y,score-step,groups,tail_fraction=tail,anchor_weight=anchor)[0]
        assert g[index]==pytest.approx((plus-minus)/2e-4,abs=2e-8)
        assert h[index]==pytest.approx((plus-2*value+minus)/1e-8,abs=2e-6)

def test_temporal_groups_keep_equal_times_together_without_labels():
    times=np.repeat(np.arange(8),[5,2,3,4,5,1,2,3])
    groups,edges=temporal_groups(times)
    assert edges.tolist()==[2,4,6]
    assert all(len(np.unique(groups[times==t]))==1 for t in np.unique(times))

def test_harder_time_group_gets_more_weight():
    y=np.tile([0,0,1,1],2);groups=np.repeat([0,1],4)
    score=np.array([-3,-2,3,2, 3,2,-3,-2])
    _,_,_,detail=tail_objective(y,score,groups)
    assert detail['group_weights'][1]>detail['group_weights'][0]

def test_pairwise_objective_is_invariant_to_shared_score_shift():
    y=np.array([0,0,1,1]);score=np.array([-.3,.7,.5,1.]);groups=np.zeros(4)
    first=tail_objective(y,score,groups,anchor_weight=0.)
    second=tail_objective(y,score+10,groups,anchor_weight=0.)
    assert first[0]==pytest.approx(second[0])
    assert np.allclose(first[1],second[1]) and first[1].sum()==pytest.approx(0.)

def test_non_tail_negatives_have_zero_pairwise_gradient():
    y=np.array([0,0,0,1]);score=np.array([-4,-1,2,0.])
    _,g,_,_=tail_objective(y,score,np.zeros(4),tail_fraction=.1,anchor_weight=0.)
    assert g[0]==0 and g[1]==0 and g[2]>0 and g[3]<0

def test_each_ranking_group_requires_both_classes():
    with pytest.raises(ValueError,match='every training group'):
        tail_objective([0,0,1,1],[0,1,2,3],[0,0,1,1])

def test_extreme_scores_remain_finite():
    result=tail_objective([0,0,1,1],[-1000,1000,-1000,1000],[0]*4)
    assert np.isfinite(result[0]) and np.isfinite(result[1]).all() and np.isfinite(result[2]).all()

def test_coarse_evaluation_times_are_not_split_into_artificial_windows():
    groups,edges=evaluation_windows([7,7,7,7])
    assert groups.tolist()==[0,0,0,0] and edges.tolist()==[]
    groups,edges=evaluation_windows([6,6,7,7])
    assert groups.tolist()==[0,0,1,1] and edges.tolist()==[7]
