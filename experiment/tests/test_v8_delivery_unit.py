import json
from v8_job import prepare_v8_job

def test_v8_job_stays_private_and_has_validation_only_main(tmp_path):
    plan=dict(reference_cohorts={'paysim':dict(training_cohort='synthetic',validation_cohort='synthetic',source_files={})})
    settings=prepare_v8_job('exampleuser','paysim',tmp_path,'pinned-image',plan,{})
    assert settings['is_private']=='true' and settings['docker_image']=='pinned-image'
    book=json.loads((tmp_path/'research.ipynb').read_text());main=''.join(book['cells'][-1]['source'])
    assert 'from v8_research import run' in main and 'trees=40' in main
    assert 'final' not in main and 'calibration' not in main
