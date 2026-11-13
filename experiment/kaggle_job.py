"""Prepare a private Kaggle notebook job after the account owner is known."""
import argparse
import hashlib
import json
import re
from pathlib import Path

TITLE = 'ALTCBoost Fraud Benchmark 20261112'
SLUG = 'altcboost-fraud-benchmark-20261112'

def metadata(owner):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{1,49}', owner):
        raise ValueError('Use your Kaggle username without slashes or spaces')
    return dict(id=f'{owner}/{SLUG}', title=TITLE,
                code_file='ALTCBoost_Kaggle.ipynb', language='python',
                kernel_type='notebook', is_private='true', enable_gpu='false',
                enable_internet='false',
                dataset_sources=['mlg-ulb/creditcardfraud'],
                competition_sources=[], kernel_sources=[], model_sources=[])

def prepare(owner, notebook, destination):
    settings = metadata(owner)
    notebook = Path(notebook)
    contents = json.loads(notebook.read_text())
    if contents.get('nbformat') != 4:
        raise ValueError('Expected a version 4 notebook')
    for cell in contents['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<notebook>', 'exec')
            cell['outputs'] = []
            cell['execution_count'] = None
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    target = destination/settings['code_file']
    target.write_text(json.dumps(contents, indent=1))
    (destination/'kernel-metadata.json').write_text(json.dumps(settings, indent=2))
    manifest = dict(kernel=settings['id'], dataset=settings['dataset_sources'],
                    notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    scope='Source code and aggregate experiment outputs; no transaction upload',
                    seeds=[17,42], tree_limit=70)
    (destination/'run-manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('owner')
    parser.add_argument('--notebook', default='../ALTCBoost_Kaggle.ipynb')
    parser.add_argument('--destination', default='../kaggle-run')
    args=parser.parse_args()
    print(json.dumps(prepare(args.owner,args.notebook,args.destination), indent=2))
