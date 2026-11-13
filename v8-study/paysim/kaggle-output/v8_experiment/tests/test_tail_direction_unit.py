import numpy as np
import pytest
from tail_direction import TailState
from tail_objective import pairwise_tail_objective

@pytest.mark.parametrize('tied',[False,True])
@pytest.mark.parametrize('anchor',[0.,.2,1.])
def test_directional_test_matches_right_difference_including_tail_ties(tied,anchor):
    y=np.tile([0,0,0,1,1],2);groups=np.repeat([0,1],5)
    score=np.zeros(len(y)) if tied else np.array([-.7,.3,1.2,-.2,.9,.2,-.5,1.8,.4,-.8])
    direction=np.random.default_rng(8).normal(size=len(y));fraction=.5;eps=1e-7
    state=TailState(y,score,groups,tail_fraction=fraction,anchor_weight=anchor)
    numeric=(pairwise_tail_objective(y,score+eps*direction,groups,tail_fraction=fraction,anchor_weight=anchor)[0]-state.value)/eps
    assert state.directional_derivative(direction)==pytest.approx(numeric,abs=2e-6)

def test_arbitrary_active_face_can_falsely_look_like_descent():
    state=TailState([0,0,1],[0,0,0],[0,0,0],tail_fraction=.5,anchor_weight=0.)
    direction=np.array([-1.,2.,0.])
    assert state.gradient@direction<0
    assert state.directional_derivative(direction)>0
    later=pairwise_tail_objective([0,0,1],.01*direction,[0,0,0],tail_fraction=.5,anchor_weight=0.)[0]
    assert later>state.value
