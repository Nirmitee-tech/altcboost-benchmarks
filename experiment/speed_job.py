"""Build a separate, private Kaggle inference experiment."""
import argparse
import hashlib
import json
from pathlib import Path
from kaggle_job import metadata

def prepare_speed_job(owner,destination,image=None):
    root=Path(__file__).resolve().parent
    names=['altcboost.py','altcboost_v2.py','altcboost_v3.py','discovery.py','evaluation.py',
           'run_experiment.py','speed_benchmark.py','tests/test_unit.py','tests/test_integration.py',
           'tests/test_speed_unit.py','tests/test_speed_integration.py']
    files={name:(root/name).read_text() for name in names}
    cells=[]
    def add(kind,text):
        cell=dict(cell_type=kind,id=hashlib.sha256(text.encode()).hexdigest()[:12],metadata={},source=text.splitlines(keepends=True))
        if kind=='code':
            compile(text,'<speed_notebook>','exec');cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown','''# ALTCBoost scorer comparison

This experiment tests whether the earlier single-row speed advantage persists when established models use lower-overhead interfaces.
It does not search for better detection settings. It fixes the seed and configurations chosen by the preceding validation exercise.
The earlier test interval is reused to check probability and threshold-decision equivalence, so this is not a fresh assessment of detection quality.

Every timed call starts from the same numeric feature matrix and includes the same shape, finite-value and contiguous-float64 boundary.
ALTCBoost feature scaling remains inside its scorer. Feature retrieval, HTTP, service queues, concurrency and process startup are outside this experiment.
Scoring is single-threaded. Training and C++ build times are recorded separately.
''')
    add('code','from pathlib import Path\nimport os,sys,json\nos.environ["OMP_NUM_THREADS"]="1"\nos.environ["OPENBLAS_NUM_THREADS"]="1"\nroot=Path("/kaggle/working/speed_experiment")\nroot.mkdir(exist_ok=True)\nfiles='+repr(files)+'\nfor name,text in files.items():\n    path=root/name\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(text)\nos.chdir(root)\nsys.path.insert(0,str(root))\n')
    add('code','''import subprocess
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_unit.py','tests/test_speed_unit.py'],check=True)
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_integration.py','tests/test_speed_integration.py'],check=True)
''')
    add('code','''from speed_benchmark import run
matches=list(Path('/kaggle/input').rglob('creditcard.csv'))
if len(matches)!=1:
    raise RuntimeError('Expected exactly one attached creditcard.csv')
timing,parity=run(matches[0],'/kaggle/working/speed_results',trees=70,single_calls=1000,rounds=5)
''')
    add('code','''import pandas as pd
display(pd.read_csv('/kaggle/working/speed_results/timing.csv'))
print(Path('/kaggle/working/speed_results/parity.json').read_text())
print('Rejected scorer paths are not timed. Timings alone do not establish deployment or fraud-detection superiority.')
''')
    add('markdown','''## Interpretation

Compare the lower-overhead paths with each other, and the public APIs with each other. Probability tolerance is 1e-6; flag decisions must agree exactly on all calibration and test rows at the three reference-calibrated thresholds.
Randomized adapter ordering and repeated rounds reduce ordering effects. The p99 values describe these calls and do not establish a service-level latency guarantee.

Primary references: [XGBoost prediction](https://xgboost.readthedocs.io/en/stable/prediction.html), [scikit-learn performance](https://scikit-learn.org/stable/computing/computational_performance.html).
''')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    target=destination/'ALTCBoost_Scorer_Comparison.ipynb'
    target.write_text(json.dumps(notebook,indent=1))
    settings=metadata(owner)
    settings.update(id=f'{owner}/altcboost-scorer-comparison-20261112',
        title='ALTCBoost Scorer Comparison 20261112',code_file=target.name)
    if image:settings.update(docker_image=image,docker_image_pinning_type='original')
    (destination/'kernel-metadata.json').write_text(json.dumps(settings,indent=2))
    manifest=dict(kernel=settings['id'],input='mlg-ulb/creditcardfraud',seed=17,trees=70,
        single_calls_per_round=1000,rounds=5,notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        selection='Prespecified depth5 baselines and ALTC crossFalse from prior validation; no new hyperparameter search')
    (destination/'run-manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('owner');p.add_argument('destination');p.add_argument('--image')
    args=p.parse_args();print(json.dumps(prepare_speed_job(args.owner,args.destination,args.image),indent=2))
