import json
import pandas as pd
import pytest
from validation_report import primary_rows,write_validation_summary
from validation_job import prepare_validation_job
from validation_data import SPECS

def test_primary_table_cannot_pool_training_repeats():
    rows=pd.DataFrame([dict(method='V6',seed=17,alpha=.001),dict(method='V6',seed=42,alpha=.001),dict(method='V6',seed=17,alpha=.01)])
    assert len(primary_rows(rows))==1
    with pytest.raises(ValueError,match='Duplicate'):primary_rows(pd.concat([rows,rows.iloc[:1]]))

@pytest.mark.parametrize('key',list(SPECS))
def test_validation_job_is_private_pinned_and_single_source(tmp_path,key):
    settings=prepare_validation_job('exampleuser',key,tmp_path,'pinned-image',dict(scope='synthetic test'))
    assert settings['is_private']=='true' and settings['enable_internet']=='false'
    assert settings['dataset_sources']==[SPECS[key]['ref']] and settings['docker_image']=='pinned-image'
    book=json.loads((tmp_path/'validation.ipynb').read_text())
    for cell in book['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<test>','exec')

def test_combined_report_requires_all_runs_to_complete(tmp_path):
    study=tmp_path/'paysim';study.mkdir();(study/'kaggle-status.json').write_text(json.dumps(dict(state='RUNNING')))
    with pytest.raises(ValueError,match='not complete'):write_validation_summary(tmp_path,keys=['paysim'])
