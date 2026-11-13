import json
from pathlib import Path
import numpy as np
from v8_research import run
from validation_data import load_data
from test_multi_unit import fixture_csv
from watch_kaggle import tick

def test_v8_workflow_exports_checked_working_descent_and_validation_report(tmp_path):
    path,_=fixture_csv(tmp_path,8000);source=tmp_path/'kaggle-output/research_results'
    rows,models=run('paysim',[path],source,trees=3,train_cap=1000,eval_cap=2000)
    assert len(rows)==18 and len(models)==6
    assert all(m['weak_learner_fits']==3 for m in models)
    assert all(s['working_loss_after']<s['working_loss_before'] for m in models for s in m['step_diagnostics'])
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['weak_learner_proposal_budget']==3 and 'may increase' in manifest['descent_scope']
    record=tick(tmp_path,'synthetic/run/1',runner=lambda args:'KernelWorkerStatus.COMPLETE' if args[1]=='status' else '',kind='v8')
    text=Path(record['report']).read_text();assert 'V8_adaptive_checked' in text and 'development results' in text
    assert 'synthetic0' not in ''.join(p.read_text() for p in source.iterdir())
    import shutil
    from v8_report import write_v8_summary
    collected=tmp_path/'collected';shutil.copytree(tmp_path/'kaggle-output',collected/'paysim/kaggle-output')
    shutil.copy(tmp_path/'kaggle-status.json',collected/'paysim/kaggle-status.json')
    assert 'Objective calls' in write_v8_summary(collected,keys=['paysim']).read_text()

def test_v8_workflow_cannot_use_later_frames(tmp_path,monkeypatch):
    import v7_research
    path,_=fixture_csv(tmp_path,8000);frames,discovery=load_data('paysim',[path],1000,2000)
    class Poison:
        def __getitem__(self,key):raise AssertionError('Future frame reached')
    monkeypatch.setattr(v7_research,'load_data',lambda *args:([frames[0],frames[1],Poison(),Poison()],discovery))
    run('paysim',[path],tmp_path/'output',trees=2)

def test_baseline_prefix_is_reselected_inside_proposal_cap(tmp_path):
    from v7_research import fit_modes
    from tuned_models import GRIDS
    path,_=fixture_csv(tmp_path,8000);frames,_=load_data('paysim',[path],1000,2000)
    references={'XGBoost':dict(config=GRIDS['XGBoost'][0],prefix=99,validation_key=[-1,-1,-1])}
    rows,_=fit_modes('paysim',*frames[:2],references,trees=2,candidate_factory=lambda n:[],reference_policy='budget_prefix')
    assert len(rows)==3 and all(0<=r['selected_prefix']<=2 for r in rows)
