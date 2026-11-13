import json
from tail_job import prepare_tail_job

def test_candidate_job_is_private_and_has_only_declared_source(tmp_path):
    settings=prepare_tail_job('exampleuser','paysim',tmp_path,'pinned-image')
    assert settings['is_private']=='true' and settings['enable_internet']=='false'
    assert settings['dataset_sources']==['ealaxi/paysim1']
    book=json.loads((tmp_path/'candidate.ipynb').read_text())
    for cell in book['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<test>','exec')

def test_revised_pairwise_job_includes_its_tests_and_entry_point(tmp_path):
    settings=prepare_tail_job('exampleuser','paysim',tmp_path,'pinned-image',generation=6)
    assert '/altcboost-v6-' in settings['id']
    book=json.loads((tmp_path/'candidate.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
    assert 'from pairwise_experiment import run' in code and 'tests/test_pairwise_unit.py' in code
