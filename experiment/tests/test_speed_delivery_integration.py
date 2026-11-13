import json
from pathlib import Path
from speed_job import prepare_speed_job
from speed_report import write_speed_report
from watch_kaggle import tick
from test_speed_delivery_unit import fixture_output

def test_speed_notebook_has_a_separate_private_job_and_fixed_plan(tmp_path):
    manifest=prepare_speed_job('exampleuser',tmp_path,image='example-image')
    settings=json.loads((tmp_path/'kernel-metadata.json').read_text())
    notebook=json.loads((tmp_path/settings['code_file']).read_text())
    assert 'scorer-comparison' in settings['id']
    assert settings['is_private']=='true' and settings['enable_gpu']=='false'
    assert settings['docker_image']=='example-image'
    assert manifest['seed']==17 and manifest['trees']==70
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook>','exec')

def test_speed_completion_downloads_evidence_and_writes_report(tmp_path):
    def cli(args):
        if args[1]=='status':return 'KernelWorkerStatus.COMPLETE'
        fixture_output(Path(args[args.index('-p')+1])/'speed_results')
        return 'Downloaded'
    state=tick(tmp_path,'owner/slug/1',runner=cli,kind='speed')
    assert state['state']=='COMPLETE'
    report=Path(state['report']).read_text()
    assert '5.00 microseconds' in report
    assert 'no new independent detection evidence' in report
