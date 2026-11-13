import json
import os
import subprocess
import sys
from v7_job import prepare_v7_job

def test_packaged_new_mechanism_passes_its_gates_in_isolation(tmp_path,monkeypatch):
    plan=dict(reference_cohorts={'paysim':dict(training_cohort='synthetic',validation_cohort='synthetic')})
    prepare_v7_job('exampleuser','paysim',tmp_path,'pinned-image',plan,{})
    book=json.loads((tmp_path/'research.ipynb').read_text())
    install=''.join(book['cells'][1]['source']).replace('/kaggle/working/v7_experiment',str(tmp_path/'installed'))
    monkeypatch.chdir(tmp_path);exec(compile(install,'<embedded>','exec'),{})
    env=dict(os.environ,PYTHONPATH=str(tmp_path/'installed')+os.pathsep+str(tmp_path/'installed/tests'))
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_prior_block_unit.py','tests/test_v7_research_unit.py',
        'tests/test_v7_integration.py','tests/test_v7_research_integration.py'],cwd=tmp_path/'installed',env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
