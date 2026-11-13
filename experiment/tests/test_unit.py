import numpy as np
import pytest
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from altcboost import ALTCBoostClassifier
from altcboost_v2 import ALTCBoostV2
from altcboost_v3 import ALTCBoostV3
from evaluation import calibrate, metrics

def test_calibration_ties_and_confidence():
    score = np.ones(40000) * .2
    cut, info = calibrate(np.zeros(len(score)), score, .0001)
    assert not (score >= cut).any()
    assert info['tolerance_bound'] <= .0001

def test_calibration_rejects_small_holdout():
    with pytest.raises(ValueError, match='Insufficient'):
        calibrate(np.zeros(100), np.linspace(0,1,100), .0001)

def test_metrics_count_actual_errors():
    result = metrics([0,0,1,1], [.1,.8,.6,.9], .7)
    assert (result['fp'], result['tp'], result['fn'], result['tn']) == (1,1,1,1)

def fitted(model_type=ALTCBoostV3):
    x,y = make_classification(n_samples=400,n_features=4,n_informative=3,n_redundant=0,random_state=4)
    a,b,c,d = train_test_split(x,y,test_size=.3,random_state=1,stratify=y)
    model = model_type(n_estimators=5,min_samples_leaf=10).fit(a,c,b,d)
    return model,a,c,b,d

def test_validation_loss_matches_selected_model():
    m,_,_,x,y = fitted()
    assert m.validation_logloss_ == pytest.approx(m._loss(y,m.decision_function(x)))

@pytest.mark.parametrize('model_type',[ALTCBoostV2,ALTCBoostV3])
def test_refit_invalidates_threshold(model_type):
    m,x,y,xv,yv = fitted(model_type)
    m.fit_threshold(xv,yv)
    m.fit(x,y,xv,yv)
    with pytest.raises(ValueError,match='fit_threshold'):
        m.predict_flags(xv)

@pytest.mark.parametrize('bad', [np.full(400,.9), np.array([0.,1.9]*200)])
def test_fractional_labels_are_rejected_before_cast(bad):
    with pytest.raises(ValueError,match='binary labels'):
        ALTCBoostV3().fit(np.ones((400,4)),bad)

def test_prediction_rejects_feature_mismatch():
    m,*_ = fitted()
    with pytest.raises(ValueError,match='feature count'):
        m.predict_proba(np.ones((2,3)))

def test_compiled_tree_boundary_matches_sklearn():
    m,_,_,_,_ = fitted()
    # Construct a split where float32 rounding changes the branch.
    tree = DecisionTreeRegressor(max_depth=1).fit([[0,0,0,0],[2,0,0,0]],[-2,2])
    m.steps_=[(tree,None)]; m.best_iteration_=1; m.compile()
    x = m._means.copy().reshape(1,-1)
    x[0,0] += m._scales[0] * (1 + 1e-8)
    expected = ALTCBoostClassifier.decision_function(m,x)
    assert np.allclose(m.decision_function(x),expected,rtol=0,atol=1e-12)
    m.decision_threshold_=float(expit(expected[0]-.01))
    flags,_=m.predict_flags(x)
    assert flags[0] == 1
