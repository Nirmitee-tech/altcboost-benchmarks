import numpy as np
from altcboost_v6 import ALTCBoostV6
from test_tail_integration import example

def test_direct_pairwise_trainer_checks_descent_and_prediction_prefix():
    x,y,groups=example();model=ALTCBoostV6(n_estimators=5,min_samples_leaf=15)
    model.fit(x[:1000],y[:1000],x[1000:1300],y[1000:1300],groups)
    assert len(model.training_loss_)>1 and np.all(np.diff(model.training_loss_)<0)
    assert model.predict_proba(x[1300:]).shape==(300,2)

def test_validation_screen_does_not_fit_or_score_future_frames(tmp_path,monkeypatch):
    import json
    import pairwise_pilot
    from multi_data import load_cohort
    from test_multi_unit import fixture_csv
    path,_=fixture_csv(tmp_path)
    frames,discovery=load_cohort('paysim',[path])
    class Poison:
        def __getitem__(self,key):raise AssertionError('Future rows accessed')
    monkeypatch.setattr(pairwise_pilot,'find_sources',lambda *args:[path])
    monkeypatch.setattr(pairwise_pilot,'load_cohort',lambda *args:([frames[0],frames[1],Poison(),Poison()],discovery))
    output=tmp_path/'screen.json'
    rows=pairwise_pilot.run('paysim',tmp_path,output,trees=2)
    assert len(rows)==5 and 'No final interval scores' in json.loads(output.read_text())['scope']

def test_pairwise_runner_exports_revised_scope_and_generation_summary(tmp_path):
    import json
    from pairwise_experiment import run
    from test_multi_unit import fixture_csv
    from tail_summary import write_tail_summary
    path,_=fixture_csv(tmp_path,40000)
    study=tmp_path/'study/paysim'
    run('paysim',[path],study/'kaggle-output/multi_results',5000,8000,
        models=[('V6_temporal_tail',dict(group_count=4),ALTCBoostV6(n_estimators=2,min_samples_leaf=15))])
    (study/'kaggle-status.json').write_text(json.dumps(dict(state='COMPLETE',kernel='synthetic/run/1')))
    report=write_tail_summary(tmp_path/'study',keys=['paysim'],generation=6)
    assert 'V6_temporal_tail' in report.read_text()
    assert 'docs/pairwise-research.md' in report.read_text()
    assert 'pairwise-study/prespecified-plan.json' in report.read_text()
    model_file=study/'kaggle-output/multi_results/models.json'
    settings=json.loads(model_file.read_text());settings[0]['selected_iteration']=0
    model_file.write_text(json.dumps(settings))
    report=write_tail_summary(tmp_path/'study',keys=['paysim'],generation=6)
    assert 'No residual trees were retained' in report.read_text()
    manifest=json.loads((study/'kaggle-output/multi_results/manifest.json').read_text())
    assert 'Revised candidate after V5' in manifest['scope']
