import numpy as np
import pytest
from altcboost_v8 import ALTCBoostV8
from test_tail_integration import example
from detection_objective import selection_key

@pytest.mark.parametrize('mode',sorted(ALTCBoostV8.MODES))
def test_continuation_uses_a_shared_weak_learner_budget_and_checked_working_loss(mode):
    x,y,groups=example();model=ALTCBoostV8(continuation=mode,n_estimators=12,min_samples_leaf=15)
    model.fit(x[:1000],y[:1000],x[1000:1300],y[1000:1300],groups)
    assert model.weak_learner_fits_==12 and len(model.steps_)<=12
    for step in model.step_diagnostics_:assert step['working_loss_after']<step['working_loss_before']
    assert model.validation_key_==selection_key(y[1000:1300],model.decision_function(x[1000:1300]))
    assert np.max(np.abs(model.decision_function(x[1300:])))<=12*.08*5+1e-12

def test_direction_gate_preserves_adaptive_predictions_while_avoiding_futile_checks():
    x,y,groups=example();models=[]
    for mode in ['adaptive_blind','adaptive_checked']:
        model=ALTCBoostV8(continuation=mode,n_estimators=20,min_samples_leaf=15)
        model.fit(x[:1000],y[:1000],x[1000:1300],y[1000:1300],groups);models.append(model)
    assert np.allclose(models[0].decision_function(x[1300:]),models[1].decision_function(x[1300:]),rtol=0,atol=1e-12)
    assert models[1].objective_evaluations_<=models[0].objective_evaluations_
