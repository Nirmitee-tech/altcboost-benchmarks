import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from altcboost_v4 import ALTCBoostV4
from detection_experiment import run
from discovery import FEATURES
from quality_discovery import discover

def test_new_trainer_returns_selected_model_and_preserves_input_validation():
    import pytest
    x,y=make_classification(n_samples=1000,n_features=6,n_redundant=0,random_state=4)
    for prior in [True,False]:
        model=ALTCBoostV4(n_estimators=5,min_samples_leaf=15,linear_prior=prior,feature_clip=None)
        model.fit(x[:600],y[:600],x[600:800],y[600:800])
        p=model.predict_proba(x[800:])
        assert p.shape==(200,2) and np.allclose(p.sum(axis=1),1.)
        assert 0<=model.best_iteration_<=5
        with pytest.raises(ValueError,match='feature count'):model.predict_proba(np.ones((3,5)))

def test_development_search_exports_selection_and_marks_consumed_holdout(tmp_path):
    rng=np.random.default_rng(11);n=24000
    frame=pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES)
    frame['Time']=np.arange(n);frame['Class']=(rng.random(n)<.04).astype(int)
    csv=tmp_path/'synthetic.csv';frame.to_csv(csv,index=False)
    rows,plan=run(csv,tmp_path/'out',trees=2,training_limit=1000)
    assert set(plan)=={'ALTCBoostV4','ALTCBoostV3','CatBoost','XGBoost','ALTCV4_CatBoost_blend'}
    assert {r['status'] for r in rows}=={'unsupported','exploratory'}
    trials=pd.read_csv(tmp_path/'out/trials.csv')
    assert trials[~trials.family.str.contains('blend')].groupby('family').size().eq(4).all()
    assert 'development-only' in json.loads((tmp_path/'out/discovery.json').read_text())['holdout_policy']
    report=discover(csv,tmp_path/'discovery.json')
    assert report['hypotheses']['unseen_final_data_available']['verified'] is False

def test_selection_does_not_depend_on_consumed_final_labels(tmp_path):
    rng=np.random.default_rng(12);n=12000
    frame=pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES)
    frame['Time']=np.arange(n);frame['Class']=(rng.random(n)<.04).astype(int)
    original=tmp_path/'first.csv';changed=tmp_path/'second.csv';frame.to_csv(original,index=False)
    _,first=run(original,tmp_path/'first',trees=1,training_limit=500,include_final=False)
    frame.loc[frame.Time>=int(.85*n),'Class']=1-frame.loc[frame.Time>=int(.85*n),'Class']
    frame.to_csv(changed,index=False)
    _,second=run(changed,tmp_path/'second',trees=1,training_limit=500,include_final=False)
    assert first==second
    assert json.loads((tmp_path/'first/champion.json').read_text())==json.loads((tmp_path/'second/champion.json').read_text())
