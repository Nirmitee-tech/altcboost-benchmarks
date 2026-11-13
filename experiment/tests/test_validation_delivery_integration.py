import json
import os
import subprocess
import sys
from pathlib import Path
from validation_job import prepare_validation_job
from validation_experiment import run
from validation_report import write_validation_report,write_validation_summary
from watch_kaggle import tick
from test_multi_unit import fixture_csv

def test_installed_notebook_sources_run_their_unit_and_integration_gates(tmp_path,monkeypatch):
    prepare_validation_job('exampleuser','paysim',tmp_path,'pinned-image',dict(scope='synthetic test'))
    book=json.loads((tmp_path/'validation.ipynb').read_text())
    install=''.join(book['cells'][1]['source']).replace('/kaggle/working/validation_experiment',str(tmp_path/'installed'))
    monkeypatch.chdir(tmp_path);exec(compile(install,'<embedded>','exec'),{})
    environment=dict(os.environ,PYTHONPATH=str(tmp_path/'installed')+os.pathsep+str(tmp_path/'installed/tests'))
    for name in ['test_validation_unit.py','test_validation_integration.py']:
        result=subprocess.run([sys.executable,'-m','pytest','-q','tests/'+name],cwd=tmp_path/'installed',env=environment,text=True,capture_output=True)
        assert result.returncode==0,result.stdout+result.stderr

def test_completed_watcher_writes_seed_aware_review_and_combined_summary(tmp_path):
    path,_=fixture_csv(tmp_path,40000);study=tmp_path/'paysim';source=study/'kaggle-output/validation_results'
    run('paysim',[path],source,1000,8000,seeds=(17,42),trees=2,replicates=20)
    calls=[]
    def runner(args):
        calls.append(args);return 'KernelWorkerStatus.COMPLETE' if args[1]=='status' else ''
    record=tick(study,'synthetic/run/1',runner=runner,kind='validation')
    text=Path(record['report']).read_text()
    assert 'primary comparison uses seed 17' in text and 'Training repeats use the same data cohorts' in text
    assert 'condition on the selected models' in text.lower() and calls[1][1]=='output'
    target=write_validation_summary(tmp_path,keys=['paysim'])
    assert 'Native LightGBM' in target.read_text() and 'Consumed synthetic' in target.read_text()
