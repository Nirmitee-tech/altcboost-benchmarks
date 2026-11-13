import numpy as np
import pytest
from prior_block import directional_derivatives,block_proposals
from tail_objective import pairwise_tail_objective

@pytest.mark.parametrize('anchor',[0.,.2,1.])
def test_full_directional_curvature_matches_numeric_mixed_derivatives(anchor):
    y=np.tile([0,0,0,1,1],2);groups=np.repeat([0,1],5)
    score=np.array([-.7,.3,1.2,-.2,.9,.2,-.5,1.8,.4,-.8])
    directions=np.random.default_rng(12).normal(size=(len(y),2));eps=1e-4
    value,g,h=directional_derivatives(y,score,groups,directions,tail_fraction=.5,anchor_weight=anchor)
    loss=lambda delta:pairwise_tail_objective(y,score+directions@delta,groups,tail_fraction=.5,anchor_weight=anchor)[0]
    for i in range(2):
        step=np.eye(2)[i]*eps
        assert g[i]==pytest.approx((loss(step)-loss(-step))/(2*eps),abs=2e-8)
        assert h[i,i]==pytest.approx((loss(step)-2*value+loss(-step))/eps**2,abs=2e-6)
    first=np.array([eps,0]);second=np.array([0,eps])
    mixed=(loss(first+second)-loss(first-second)-loss(-first+second)+loss(-first-second))/(4*eps**2)
    assert h[0,1]==pytest.approx(mixed,abs=2e-6)
    assert np.linalg.eigvalsh(h).min()>=-1e-10

def test_proposals_respect_prior_and_tree_boxes_even_at_boundaries():
    for weight in [0.,.4,1.]:
        proposals=block_proposals([10.,-3.],[[2.,.3],[.3,1.]],weight)
        for _,step in proposals:
            assert 0<=weight+step[0]<=1 and 0<=step[1]<=.08

def test_cross_curvature_changes_joint_proposal():
    g=[.05,-.02];h=[[2.,1.2],[1.2,1.]]
    full=block_proposals(g,h,.5)[-1][1]
    diagonal=block_proposals(g,h,.5,diagonal=True)[-1][1]
    assert not np.allclose(full,diagonal)

def test_bounded_prior_is_monotone_and_cannot_extrapolate_without_limit():
    from prior_block import bounded_prior
    score=np.linspace(-1000,1000,10001);mapped=bounded_prior(score)
    assert np.all(np.diff(mapped)>=0) and np.max(np.abs(mapped))<=4
    assert bounded_prior([0])[0]==0
