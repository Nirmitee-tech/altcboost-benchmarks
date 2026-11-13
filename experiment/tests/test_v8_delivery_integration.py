import json
import os
import subprocess
import sys
from v8_job import prepare_v8_job

def test_v8_embedded_tests_pass_in_isolated_install(tmp_path,monkeypatch):
    plan=dict(reference_cohorts={'paysim':dict(training_cohort='synthetic',validation_cohort='synthetic',source_files={})})
    prepare_v8_job('exampleuser','paysim',tmp_path,'pinned-image',plan,{})
    book=json.loads((tmp_path/'research.ipynb').read_text())
    install=''.join(book['cells'][1]['source']).replace('/kaggle/working/v8_experiment',str(tmp_path/'installed'))
    monkeypatch.chdir(tmp_path);exec(compile(install,'<embedded>','exec'),{})
    env=dict(os.environ,PYTHONPATH=str(tmp_path/'installed')+os.pathsep+str(tmp_path/'installed/tests'))
    result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_tail_direction_unit.py','tests/test_v8_research_unit.py',
        'tests/test_v8_integration.py','tests/test_v8_research_integration.py'],cwd=tmp_path/'installed',env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
