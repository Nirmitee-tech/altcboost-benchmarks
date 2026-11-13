import numpy as np
import pytest
from altcboost_v8 import ALTCBoostV8
from v8_research import candidates
from tail_direction import TailState

def test_policy_grid_and_clock_endpoints_are_fixed():
    models=dict(candidates(40));assert len(models)==6
    assert models['V8_adaptive_checked'].fractions(0)==[.01,.02,.04,.08,.16,.32,.64,1.]
    for key in ['V8_geometric','V8_linear']:
        assert models[key].fractions(0)==[1.]
        assert models[key].fractions(39)==pytest.approx([.01])
    assert all(model.n_estimators==40 for model in models.values())

def test_derivative_rejects_nonfinite_or_misaligned_directions():
    state=TailState([0,0,1],[0,0,0],[0,0,0])
    for direction in [[1,2],[1,np.nan,0]]:
        with pytest.raises(ValueError,match='Finite matching'):state.directional_derivative(direction)
    with pytest.raises(ValueError,match='Unknown'):ALTCBoostV8(continuation='unregistered')
