import numpy as np
import pytest
from tail_objective import pairwise_tail_objective

@pytest.mark.parametrize('anchor',[0.,.2,1.])
def test_direct_pairwise_gradient_and_curvature_match_numeric_checks(anchor):
    y=np.tile([0,0,0,1,1],2);groups=np.repeat([0,1],5)
    score=np.array([-.7,.3,1.2,-.2,.9,.2,-.5,1.8,.4,-.8])
    value,g,h,_=pairwise_tail_objective(y,score,groups,tail_fraction=.5,anchor_weight=anchor)
    for i in range(len(y)):
        step=np.zeros(len(y));step[i]=1e-4
        plus=pairwise_tail_objective(y,score+step,groups,tail_fraction=.5,anchor_weight=anchor)[0]
        minus=pairwise_tail_objective(y,score-step,groups,tail_fraction=.5,anchor_weight=anchor)[0]
        assert g[i]==pytest.approx((plus-minus)/2e-4,abs=2e-8)
        assert h[i]==pytest.approx((plus-2*value+minus)/1e-8,abs=2e-6)

def test_chunking_preserves_direct_pairwise_loss_and_derivatives():
    rng=np.random.default_rng(17);y=np.tile([0]*20+[1]*7,3);score=rng.normal(size=len(y));groups=np.repeat(np.arange(3),27)
    first=pairwise_tail_objective(y,score,groups,tail_fraction=.3,chunk_size=1)
    second=pairwise_tail_objective(y,score,groups,tail_fraction=.3,chunk_size=64)
    assert first[0]==pytest.approx(second[0])
    assert np.allclose(first[1],second[1]) and np.allclose(first[2],second[2])

def test_pair_loss_equals_explicit_small_matrix():
    y=np.array([0,0,0,1,1]);score=np.array([-.5,.2,1.5,-.2,.7])
    value,*_=pairwise_tail_objective(y,score,np.zeros(len(y)),tail_fraction=.5,anchor_weight=0.)
    expected=np.logaddexp(0,score[[1,2]][None,:]-score[[3,4]][:,None]+1).mean()
    assert value==pytest.approx(expected)

def test_validation_screen_contains_candidate_control_and_frozen_references():
    from pairwise_pilot import screening_models
    models=screening_models(2,np.repeat(np.arange(4),10))
    assert [name for name,_,_ in models]==['V6_temporal_tail','V6_balanced_logistic','ALTCBoost_tuned','CatBoost','XGBoost']
    assert models[0][1].n_estimators==2 and models[1][1].anchor_weight==1.

def test_pairwise_removal_experiments_are_labelled_separately():
    from pairwise_experiment import candidate_models
    models=list(candidate_models(trees=2))
    assert len(models)==7 and models[0][0]=='V6_temporal_tail'
    assert all(model.n_estimators==2 for name,config,model in models[:4])
