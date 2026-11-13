import json
from pathlib import Path
import numpy as np
from v7_research import run,fit_modes
from validation_data import load_data
from test_multi_unit import fixture_csv
from watch_kaggle import tick

def test_research_runner_and_report_preserve_validation_scope(tmp_path):
    path,_=fixture_csv(tmp_path,8000);source=tmp_path/'kaggle-output/research_results'
    rows,models=run('paysim',[path],source,trees=3,train_cap=1000,eval_cap=2000)
    assert len(rows)==18 and len(models)==6
    assert all(np.all(np.diff(model['training_loss'])<0) for model in models)
    record=tick(tmp_path,'synthetic/run/1',runner=lambda args:'KernelWorkerStatus.COMPLETE' if args[1]=='status' else '',kind='v7')
    text=Path(record['report']).read_text();assert 'development evidence' in text
    assert 'V7_adaptive_joint' in text and 'V7_clipped' in text
    assert 'synthetic0' not in ''.join(path.read_text() for path in source.iterdir())
    import shutil
    from v7_report import write_v7_summary
    collected=tmp_path/'collected';shutil.copytree(tmp_path/'kaggle-output',collected/'paysim/kaggle-output')
    shutil.copy(tmp_path/'kaggle-status.json',collected/'paysim/kaggle-status.json')
    combined=write_v7_summary(collected,keys=['paysim'])
    assert 'Best static control' in combined.read_text() and 'not independent confirmation' in combined.read_text()

def test_mechanism_fit_never_receives_calibration_or_final_frames(tmp_path,monkeypatch):
    import v7_research
    path,_=fixture_csv(tmp_path,8000);frames,discovery=load_data('paysim',[path],1000,2000)
    class Poison:
        def __getitem__(self,key):raise AssertionError('Future frame used for fitting or scoring')
    monkeypatch.setattr(v7_research,'load_data',lambda *args:([frames[0],frames[1],Poison(),Poison()],discovery))
    run('paysim',[path],tmp_path/'output',trees=2)

def test_source_mismatch_stops_before_mechanism_fitting(tmp_path,monkeypatch):
    import pytest
    import v7_research
    path,_=fixture_csv(tmp_path,8000)
    monkeypatch.setattr(v7_research,'fit_modes',lambda *args:(_ for _ in ()).throw(AssertionError('Fitting should not begin')))
    for guards,message in [({'expected_cohorts':{'train':'wrong source'}},'cohort differs'),
                           ({'expected_files':{path.name:'wrong hash'}},'file hash differs')]:
        with pytest.raises(ValueError,match=message):
            run('paysim',[path],tmp_path/'output',trees=2,**guards)
