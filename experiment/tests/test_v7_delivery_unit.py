import json
from v7_job import prepare_v7_job

def test_research_package_cannot_call_final_scoring_runner(tmp_path):
    plan=dict(reference_cohorts={'paysim':dict(training_cohort='synthetic',validation_cohort='synthetic')})
    settings=prepare_v7_job('exampleuser','paysim',tmp_path,'pinned-image',plan,{})
    assert settings['is_private']=='true' and settings['enable_internet']=='false'
    book=json.loads((tmp_path/'research.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
    assert 'from v7_research import run' in code and 'from validation_experiment import run' not in code
    for cell in book['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<test>','exec')
