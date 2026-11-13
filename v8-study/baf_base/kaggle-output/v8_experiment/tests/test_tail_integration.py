import numpy as np
import pytest
from sklearn.datasets import make_classification
from altcboost_v5 import ALTCBoostV5

def example():
    x,y=make_classification(n_samples=1600,n_features=6,n_redundant=0,weights=[.9,.1],random_state=72)
    return x,y,np.arange(1000)%4

def test_fit_accepts_only_descending_steps_and_returns_selected_model():
    x,y,groups=example()
    model=ALTCBoostV5(n_estimators=8,min_samples_leaf=15).fit(x[:1000],y[:1000],x[1000:1300],y[1000:1300],groups)
    assert len(model.training_loss_)>1 and np.all(np.diff(model.training_loss_)<0)
    assert 0<=model.best_iteration_<=len(model.steps_)<=8
    z=model.scaler_.transform(x[1300:]);expected=model.linear_.decision_function(z)
    for tree,rate in model.steps_[:model.best_iteration_]:expected+=rate*tree.predict(z)
    assert np.allclose(expected,model.decision_function(x[1300:]))
    assert np.allclose(model.predict_proba(x[1300:]).sum(axis=1),1.)
    with pytest.raises(ValueError,match='feature count'):model.predict_proba(np.ones((2,5)))

def test_validation_labels_cannot_change_first_training_update():
    x,y,groups=example()
    models=[ALTCBoostV5(n_estimators=1,min_samples_leaf=15).fit(x[:1000],y[:1000],x[1000:1300],v,groups)
            for v in [y[1000:1300],1-y[1000:1300]]]
    assert models[0].training_loss_==models[1].training_loss_
    assert np.array_equal(models[0].steps_[0][0].tree_.value,models[1].steps_[0][0].tree_.value)

def test_refit_clears_previous_steps():
    x,y,groups=example();model=ALTCBoostV5(n_estimators=3,min_samples_leaf=15)
    model.fit(x[:1000],y[:1000],x[1000:1300],y[1000:1300],groups)
    model.n_estimators=1;model.fit(x[:1000],y[:1000],x[1000:1300],y[1000:1300],groups)
    assert len(model.steps_)==1 and len(model.training_loss_)==2
