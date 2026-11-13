import json
import numpy as np
import pandas as pd
from altcboost_v5 import ALTCBoostV5
from tail_experiment import run
from test_multi_unit import fixture_csv

def test_candidate_runner_exports_descent_and_fixed_cutoff_time_windows(tmp_path):
    path,_=fixture_csv(tmp_path,40000)
    model=ALTCBoostV5(n_estimators=3,min_samples_leaf=15)
    rows,windows=run('paysim',[path],tmp_path/'out',train_cap=5000,eval_cap=8000,
        models=[('V5_temporal_tail',dict(group_count=4),model)])
    evaluated=[row for row in rows if row['status']=='evaluated']
    assert evaluated
    for row in evaluated:
        parts=[part for part in windows if part['alpha']==row['alpha'] and part['status']=='evaluated']
        assert len(parts)==4
        assert sum(part['tp'] for part in parts)==row['tp']
        assert sum(part['fp'] for part in parts)==row['fp']
        assert all(part['threshold']==row['threshold'] for part in parts)
    manifest=json.loads((tmp_path/'out/manifest.json').read_text())
    assert 'consumed development' in manifest['scope']
    assert np.all(np.diff(model.training_loss_)<0)
    assert 'synthetic0' not in ''.join(p.read_text() for p in (tmp_path/'out').iterdir())

def test_candidate_summary_includes_controls_and_descent_evidence(tmp_path):
    from tail_summary import write_tail_summary
    path,_=fixture_csv(tmp_path,40000)
    study=tmp_path/'study/paysim';output=study/'kaggle-output/multi_results'
    run('paysim',[path],output,train_cap=5000,eval_cap=8000,models=[
        ('V5_temporal_tail',dict(group_count=4),ALTCBoostV5(n_estimators=2,min_samples_leaf=15)),
        ('V5_balanced_logistic',dict(group_count=1),ALTCBoostV5(n_estimators=2,min_samples_leaf=15,anchor_weight=1.))])
    (study/'kaggle-status.json').write_text(json.dumps(dict(state='COMPLETE',kernel='synthetic/run/1')))
    report=write_tail_summary(tmp_path/'study',keys=['paysim'])
    text=report.read_text()
    assert 'Balanced logistic control' in text and 'descending training updates' in text
    assert 'no independent final confirmation' in text

def test_baf_coarse_final_months_keep_two_observed_windows(tmp_path):
    from multi_data import SPECS
    spec=SPECS['baf_base'];row=np.arange(16000)
    frame=pd.DataFrame({column:(row%19).astype(float) for column in spec['numeric']})
    for column in spec['categorical']:frame[column]='synthetic'
    frame['month']=row//2000;frame['fraud_bool']=(row%13==0).astype(int)
    path=tmp_path/'Base.csv';frame.to_csv(path,index=False)
    rows,windows=run('baf_base',[path],tmp_path/'out',500,1000,models=[
        ('V5_temporal_tail',dict(group_count=4),ALTCBoostV5(n_estimators=2,min_samples_leaf=10))])
    assert set(part['window'] for part in windows)=={0,1}
    assert json.loads((tmp_path/'out/manifest.json').read_text())['test_window_count']==2
