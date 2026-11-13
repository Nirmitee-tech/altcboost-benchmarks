import json
from pathlib import Path
from kaggle_job import prepare

def test_prepared_job_contains_the_notebook_referenced_by_metadata(tmp_path):
    source=Path(__file__).resolve().parents[2]/'ALTCBoost_Kaggle.ipynb'
    manifest=prepare('exampleuser',source,tmp_path/'job')
    settings=json.loads((tmp_path/'job/kernel-metadata.json').read_text())
    notebook=json.loads((tmp_path/'job'/settings['code_file']).read_text())
    assert notebook['cells']
    combined=''.join(''.join(c['source']) for c in notebook['cells'])
    assert 'write_review.py' not in combined
    assert 'seeds=(17,42), trees=70' in combined
    assert all(c['outputs']==[] and c['execution_count'] is None
               for c in notebook['cells'] if c['cell_type']=='code')
    assert manifest['dataset']==['mlg-ulb/creditcardfraud']
    assert len(manifest['notebook_sha256'])==64
    assert sorted(p.name for p in (tmp_path/'job').iterdir())==[
        'ALTCBoost_Kaggle.ipynb','kernel-metadata.json','run-manifest.json']
