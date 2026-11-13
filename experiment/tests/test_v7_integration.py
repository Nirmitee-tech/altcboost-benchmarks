import numpy as np
import pytest
from altcboost_v6 import ALTCBoostV6
from altcboost_v7 import ALTCBoostV7
from detection_objective import selection_key
from test_tail_integration import example

@pytest.mark.parametrize('mode',['fixed','zero','clipped','shrunk','adaptive_diagonal','adaptive_joint'])
def test_prior_modes_keep_descent_and_predict_the_selected_weighted_prefix(mode):
    x,y,groups=example();model=ALTCBoostV7(prior_mode=mode,n_estimators=7,min_samples_leaf=15)
    model.fit(x[:1000],y[:1000],x[1000:1300],y[1000:1300],groups)
    assert np.all(np.diff(model.training_loss_)<0)
    assert all(0<=weight<=1 for weight in model.prior_weights_)
    z=model.scaler_.transform(x[1300:]);expected=model.prior_weights_[model.best_iteration_]*model._prior(z)
    for tree,rate in model.steps_[:model.best_iteration_]:expected+=rate*tree.predict(z)
    assert np.allclose(expected,model.decision_function(x[1300:]),rtol=0,atol=1e-12)
    assert selection_key(y[1000:1300],model.decision_function(x[1000:1300]))==model.validation_key_
    assert np.allclose(model.predict_proba(x[1300:]).sum(axis=1),1.)

def test_fixed_prior_control_reproduces_v6():
    x,y,groups=example()
    models=[ALTCBoostV6(n_estimators=7,min_samples_leaf=15),ALTCBoostV7(prior_mode='fixed',n_estimators=7,min_samples_leaf=15)]
    for model in models:model.fit(x[:1000],y[:1000],x[1000:1300],y[1000:1300],groups)
    assert models[0].best_iteration_==models[1].best_iteration_
    assert np.allclose(models[0].decision_function(x[1300:]),models[1].decision_function(x[1300:]),rtol=0,atol=1e-12)

def test_refit_resets_prior_coefficient_history():
    x,y,groups=example();model=ALTCBoostV7(n_estimators=3,min_samples_leaf=15)
    model.fit(x[:1000],y[:1000],x[1000:1300],y[1000:1300],groups)
    model.n_estimators=1;model.fit(x[:1000],y[:1000],x[1000:1300],y[1000:1300],groups)
    assert len(model.prior_weights_)==len(model.steps_)+1<=2
