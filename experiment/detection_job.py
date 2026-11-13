"""Package the detection experiment as a separate private Kaggle job."""
import argparse
import hashlib
import json
from pathlib import Path
from kaggle_job import metadata

def prepare_detection_job(owner,destination,image=None,incumbent=None):
    root=Path(__file__).resolve().parent
    names=['altcboost.py','altcboost_v2.py','altcboost_v3.py','altcboost_v4.py',
           'discovery.py','evaluation.py','quality_discovery.py','detection_objective.py',
           'detection_experiment.py','tests/test_unit.py','tests/test_detection_unit.py',
           'tests/test_detection_integration.py']
    files={name:(root/name).read_text() for name in names}
    cells=[]
    def add(kind,text):
        value=dict(cell_type=kind,id=hashlib.sha256(text.encode()).hexdigest()[:12],metadata={},source=text.splitlines(keepends=True))
        if kind=='code':compile(text,'<notebook>','exec');value.update(outputs=[],execution_count=None)
        cells.append(value)
    add('markdown','''# Detection quality development experiment

Goal: catch more fraud with fewer false alarms, measured at calibration targets of 0.05% and 0.1%.
This ULB dataset has already informed earlier work. Its final interval is development evidence now, not an independent final test.
No result from this notebook establishes production readiness or a general superiority claim.

The configurations and selection rule are frozen before the search. The standalone V4 branch remains independent of the baseline libraries.
A separate hybrid ensemble is labelled explicitly. Feature clipping, starting-model choice, leaf regularization and the old negative-weight heuristic are tested through prespecified candidates.
''')
    add('code','from pathlib import Path\nimport os,sys,json\nos.environ["OMP_NUM_THREADS"]="2"\nos.environ["OPENBLAS_NUM_THREADS"]="2"\nroot=Path("/kaggle/working/quality_experiment")\nroot.mkdir(exist_ok=True)\nfiles='+repr(files)+'\nfor name,text in files.items():\n    path=root/name\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(text)\nos.chdir(root)\nsys.path.insert(0,str(root))\n')
    add('code','''import subprocess
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_unit.py','tests/test_detection_unit.py'],check=True)
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_detection_integration.py'],check=True)
''')
    add('code','''from quality_discovery import discover,audit_previous_xgb
matches=list(Path('/kaggle/input').rglob('creditcard.csv'))
if len(matches)!=1:raise RuntimeError('Expected exactly one attached creditcard.csv')
output=Path('/kaggle/working/quality_results');output.mkdir(exist_ok=True)
discover(matches[0],output/'train-diagnostics.json')
audit_previous_xgb(matches[0],output/'historical-xgboost-audit.json')
''')
    add('code','from detection_experiment import run\nincumbent='+repr(incumbent)+'\nif incumbent is not None:\n    (output/"incumbent.json").write_text(json.dumps(incumbent,indent=2))\nrows,selected=run(matches[0],output,trees=100)\n')
    add('code','''import pandas as pd
display(pd.read_csv(output/'results.csv'))
print('Exploratory development evidence only. Independent IEEE-CIS validation is pending access and schema discovery.')
''')
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    target=destination/'ALTCBoost_Detection_Quality.ipynb';target.write_text(json.dumps(notebook,indent=1))
    settings=metadata(owner);settings.update(id=f'{owner}/altcboost-detection-quality-20261112',
        title='ALTCBoost Detection Quality 20261112',code_file=target.name)
    if image:settings.update(docker_image=image,docker_image_pinning_type='original')
    (destination/'kernel-metadata.json').write_text(json.dumps(settings,indent=2))
    manifest=dict(kernel=settings['id'],scope='ULB development-only; independent evaluation pending',
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),max_trees=100,
        individual_family_trials=4,additional_blend_trials=3,budgets=[.0005,.001])
    (destination/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('owner');p.add_argument('destination');p.add_argument('--image')
    args=p.parse_args();print(json.dumps(prepare_detection_job(args.owner,args.destination,args.image),indent=2))
