"""Package the candidate and its prespecified removal experiments privately."""
import hashlib
import json
from pathlib import Path
from kaggle_job import metadata
from multi_data import SPECS

def prepare_tail_job(owner,key,destination,image,generation=5):
    if generation not in {5,6}:raise ValueError('Unknown candidate generation')
    root=Path(__file__).resolve().parent
    names=['altcboost.py','altcboost_v2.py','altcboost_v3.py','altcboost_v5.py',
        'altcboost_v6.py','pairwise_experiment.py','pairwise_pilot.py',
        'evaluation.py','detection_objective.py','multi_data.py','multi_benchmark.py',
        'tail_objective.py','tail_discovery.py','tail_experiment.py','tail_summary.py','multi_summary.py',
        'tests/test_multi_unit.py','tests/test_tail_unit.py','tests/test_tail_integration.py',
        'tests/test_tail_experiment_unit.py','tests/test_tail_experiment_integration.py',
        'tests/test_pairwise_unit.py','tests/test_pairwise_integration.py']
    files={name:(root/name).read_text() for name in names};cells=[]
    def add(kind,source):
        value=dict(cell_type=kind,id=hashlib.sha256(source.encode()).hexdigest()[:12],metadata={},source=source.splitlines(True))
        if kind=='code':compile(source,'<notebook>','exec');value.update(outputs=[],execution_count=None)
        cells.append(value)
    add('markdown',f'# Temporal-tail candidate V{generation}: {key}\n\nFixed development experiment. Novelty and gains are unproven. No independent final-test or population FPR guarantee.')
    add('code','from pathlib import Path\nimport os,sys\nos.environ["OMP_NUM_THREADS"]="2"\nos.environ["OPENBLAS_NUM_THREADS"]="2"\nroot=Path("/kaggle/working/tail_experiment");root.mkdir(exist_ok=True)\nfiles='+repr(files)+'\nfor name,text in files.items():\n p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)\nos.chdir(root)\nsys.path[:0]=[str(root),str(root/"tests")]\n')
    unit=['tests/test_tail_unit.py','tests/test_tail_experiment_unit.py']
    integration=['tests/test_tail_integration.py','tests/test_tail_experiment_integration.py']
    if generation==6:unit+=['tests/test_pairwise_unit.py'];integration+=['tests/test_pairwise_integration.py']
    add('code','import subprocess\nsubprocess.run([sys.executable,"-m","pytest","-q",*'+repr(unit)+'],check=True)\nsubprocess.run([sys.executable,"-m","pytest","-q",*'+repr(integration)+'],check=True)\n')
    module='pairwise_experiment' if generation==6 else 'tail_experiment'
    add('code',f'from {module} import run\nfrom multi_data import find_sources\nrun({key!r},find_sources({key!r},"/kaggle/input"),"/kaggle/working/multi_results")\n')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    target=destination/'candidate.ipynb'
    target.write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5),indent=1))
    settings=metadata(owner);settings.update(id=f'{owner}/altcboost-v{generation}-{key.replace("_","-")}-20261112',
        title=f'ALTCBoost V{generation} {key} 20261112',code_file=target.name,dataset_sources=[SPECS[key]['ref']],
        docker_image=image,docker_image_pinning_type='original')
    (destination/'kernel-metadata.json').write_text(json.dumps(settings,indent=2))
    return settings
