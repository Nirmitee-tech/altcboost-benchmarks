import json
import numpy as np
import pandas as pd
from discovery import FEATURES, load_data
from run_experiment import run

def fixture_csv(tmp_path):
    rng=np.random.default_rng(42); n=24000
    data=pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES)
    data['Time']=np.arange(n)//2
    data['Class']=(rng.random(n)<.04).astype(int)
    path=tmp_path/'synthetic.csv'; data.to_csv(path,index=False)
    return path

def test_temporal_partitions_keep_equal_timestamps_together(tmp_path):
    frame,parts,_=load_data(fixture_csv(tmp_path))
    assert len(set(np.concatenate(parts))) == len(frame)
    for a,b in zip(parts,parts[1:]):
        assert frame.iloc[a].Time.max() < frame.iloc[b].Time.min()

def test_user_receives_heldout_results_and_unsupported_budget(tmp_path):
    output=tmp_path/'out'
    rows=run(fixture_csv(tmp_path),output,seeds=(17,),trees=2,full=False)
    assert len(rows)==3
    assert next(r for r in rows if r['alpha']==.0001)['status']=='unsupported'
    assert next(r for r in rows if r['alpha']==.01)['status']=='evaluated'
    assert (output/'results.csv').exists()
    assert json.loads((output/'discovery.json').read_text())['unit'].startswith('transaction row')

def test_model_training_calibration_and_flags_on_disjoint_rows():
    from sklearn.datasets import make_classification
    from altcboost_v3 import ALTCBoostV3
    x,y=make_classification(n_samples=1600,n_features=6,random_state=12)
    model=ALTCBoostV3(n_estimators=8,min_samples_leaf=15)
    model.fit(x[:700],y[:700],x[700:1000],y[700:1000])
    model.fit_threshold(x[1000:1300],y[1000:1300],max_fpr=.05)
    flags,trees=model.predict_flags(x[1300:])
    assert np.array_equal(flags,model.predict_proba(x[1300:])[:,1]>=model.decision_threshold_)
    assert (trees<=model.best_iteration_).all()
    model.fit(x[:700],y[:700],x[700:1000],y[700:1000])
    model.fit_threshold(x[1000:1300],y[1000:1300],max_fpr=.05)
    assert model.predict_flags(x[1300:])[0].shape==(300,)
