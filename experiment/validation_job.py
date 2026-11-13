"""Self-contained private jobs for the frozen validation study."""
import hashlib
import json
from pathlib import Path
from kaggle_job import metadata
from validation_data import SPECS

def prepare_validation_job(owner,key,destination,image,plan):
    root=Path(__file__).resolve().parent
    names=['altcboost.py','altcboost_v2.py','altcboost_v3.py','altcboost_v5.py','altcboost_v6.py',
        'evaluation.py','detection_objective.py','tail_objective.py','multi_data.py','multi_benchmark.py',
        'validation_data.py','cluster_uncertainty.py','tuned_models.py','validation_experiment.py',
        'tests/test_multi_unit.py','tests/test_validation_unit.py','tests/test_validation_integration.py']
    files={name:(root/name).read_text() for name in names};cells=[]
    def add(kind,source):
        cell=dict(cell_type=kind,id=hashlib.sha256(source.encode()).hexdigest()[:12],metadata={},source=source.splitlines(True))
        if kind=='code':compile(source,'<notebook>','exec');cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    add('markdown',f'# Frozen V6 validation: {key}\n\nFour baseline configurations per family, three model seeds, validation-only selection and paired cluster checks. No raw row export. Elliptic is a separate real AML transfer task; existing sources are consumed synthetic development evidence.')
    add('code','from pathlib import Path\nimport os,sys,json\nos.environ["OMP_NUM_THREADS"]="2"\nos.environ["OPENBLAS_NUM_THREADS"]="2"\nroot=Path("/kaggle/working/validation_experiment");root.mkdir(exist_ok=True)\nfiles='+repr(files)+'\nfor name,text in files.items():\n p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)\nos.chdir(root)\nsys.path[:0]=[str(root),str(root/"tests")]\n')
    add('code','import subprocess\nsubprocess.run([sys.executable,"-m","pytest","-q","tests/test_validation_unit.py"],check=True)\nsubprocess.run([sys.executable,"-m","pytest","-q","tests/test_validation_integration.py"],check=True)\n')
    add('code','output=Path("/kaggle/working/validation_results");output.mkdir(exist_ok=True)\n(output/"prespecified-plan.json").write_text(json.dumps('+repr(plan)+',indent=2))\nfrom validation_experiment import run\nfrom validation_data import sources\nrun('+repr(key)+',sources('+repr(key)+',"/kaggle/input"),output)\n')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    target=destination/'validation.ipynb'
    target.write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5),indent=1))
    settings=metadata(owner);settings.update(id=f'{owner}/altcboost-validation-{key.replace("_","-")}-20261112',
        title=f'ALTCBoost Validation {key} 20261112',code_file=target.name,dataset_sources=[SPECS[key]['ref']],
        docker_image=image,docker_image_pinning_type='original')
    (destination/'kernel-metadata.json').write_text(json.dumps(settings,indent=2))
    (destination/'source-manifest.json').write_text(json.dumps(dict(kernel=settings['id'],notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source_files=list(files)),indent=2))
    return settings
