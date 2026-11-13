"""Build separate private Kaggle notebooks for the frozen multi-source study."""
import hashlib
import json
from pathlib import Path
from kaggle_job import metadata
from multi_data import SPECS

def prepare_multi_job(owner,key,destination,image):
    root=Path(__file__).resolve().parent
    names=['altcboost.py','altcboost_v2.py','altcboost_v3.py','evaluation.py',
           'detection_objective.py','multi_data.py','multi_benchmark.py','multi_report.py',
           'tests/test_multi_unit.py','tests/test_multi_integration.py']
    files={name:(root/name).read_text() for name in names}
    cells=[]
    def add(kind,source):
        cell=dict(cell_type=kind,id=hashlib.sha256(source.encode()).hexdigest()[:12],metadata={},source=source.splitlines(True))
        if kind=='code':compile(source,'<notebook>','exec');cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown',f'# Frozen fraud benchmark: {key}\n\nSynthetic source. Separate temporal cohorts, label-blind caps, no per-dataset parameter search. Aggregate exports only.')
    add('code','from pathlib import Path\nimport os,sys\nos.environ["OMP_NUM_THREADS"]="2"\nos.environ["OPENBLAS_NUM_THREADS"]="2"\nroot=Path("/kaggle/working/multi_experiment");root.mkdir(exist_ok=True)\nfiles='+repr(files)+'\nfor name,text in files.items():\n p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)\nos.chdir(root)\nsys.path[:0]=[str(root),str(root/"tests")]\n')
    add('code','import subprocess\nsubprocess.run([sys.executable,"-m","pytest","-q","tests/test_multi_unit.py"],check=True)\nsubprocess.run([sys.executable,"-m","pytest","-q","tests/test_multi_integration.py"],check=True)\n')
    add('code',f'from multi_benchmark import run\nfrom multi_data import find_sources\nrun({key!r},find_sources({key!r},"/kaggle/input"),"/kaggle/working/multi_results")\n')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    target=destination/'benchmark.ipynb'
    target.write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5),indent=1))
    settings=metadata(owner);settings.update(id=f'{owner}/altcboost-multi-{key.replace("_","-")}-20261112',
        title=f'ALTCBoost Multi {key} 20261112',code_file=target.name,dataset_sources=[SPECS[key]['ref']],
        docker_image=image,docker_image_pinning_type='original')
    (destination/'kernel-metadata.json').write_text(json.dumps(settings,indent=2))
    return settings
