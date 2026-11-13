import json
import pytest
from multi_job import prepare_multi_job
from multi_data import SPECS

@pytest.mark.parametrize('key',list(SPECS))
def test_job_is_private_single_source_and_executable(tmp_path,key):
    settings=prepare_multi_job('exampleuser',key,tmp_path,'pinned-image')
    assert settings['is_private']=='true' and settings['enable_internet']=='false'
    assert settings['dataset_sources']==[SPECS[key]['ref']]
    assert settings['docker_image']=='pinned-image'
    book=json.loads((tmp_path/'benchmark.ipynb').read_text())
    for cell in book['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<test>','exec')

def test_summary_requires_completed_runs(tmp_path):
    from multi_summary import completed_rows
    study=tmp_path/'paysim';study.mkdir()
    (study/'kaggle-status.json').write_text(json.dumps(dict(state='RUNNING')))
    with pytest.raises(ValueError,match='has not completed'):completed_rows(tmp_path,['paysim'])
