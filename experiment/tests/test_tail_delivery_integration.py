import json
from tail_job import prepare_tail_job

def test_installed_candidate_is_self_contained_and_tests_precede_training(tmp_path,monkeypatch):
    prepare_tail_job('exampleuser','paysim',tmp_path,'pinned-image')
    book=json.loads((tmp_path/'candidate.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
    assert code.index('pytest')<code.index('from tail_experiment import run')
    install=''.join(book['cells'][1]['source']).replace('/kaggle/working/tail_experiment',str(tmp_path/'installed'))
    monkeypatch.chdir(tmp_path);exec(compile(install,'<embedded>','exec'),{})
    assert (tmp_path/'installed/altcboost_v5.py').exists()
    assert (tmp_path/'installed/tail_discovery.py').exists()
    assert (tmp_path/'installed/tail_summary.py').exists()
