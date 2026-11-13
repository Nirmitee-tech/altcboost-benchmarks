"""Package the prior-revision mechanism experiment without final scoring code."""
import hashlib
import json
from pathlib import Path
from kaggle_job import metadata
from validation_data import SPECS

def prepare_v7_job(owner,key,destination,image,plan,references):
    root=Path(__file__).resolve().parent
    names=['altcboost.py','altcboost_v2.py','altcboost_v3.py','altcboost_v5.py','altcboost_v6.py','altcboost_v7.py',
        'evaluation.py','detection_objective.py','tail_objective.py','prior_block.py',
        'multi_data.py','multi_benchmark.py','validation_data.py','cluster_uncertainty.py','tuned_models.py',
        'v7_research.py','v7_report.py','watch_kaggle.py','hosted_summary.py',
        'tests/test_multi_unit.py','tests/test_tail_integration.py','tests/test_prior_block_unit.py',
        'tests/test_v7_integration.py','tests/test_v7_research_unit.py','tests/test_v7_research_integration.py']
    files={name:(root/name).read_text() for name in names};cells=[]
    def add(kind,source):
        value=dict(cell_type=kind,id=hashlib.sha256(source.encode()).hexdigest()[:12],metadata={},source=source.splitlines(True))
        if kind=='code':compile(source,'<notebook>','exec');value.update(outputs=[],execution_count=None)
        cells.append(value)
    add('markdown',f'# Prior-revision research: {key}\n\nTraining/validation only. No calibration or final predictions. Adaptive development, not independent confirmation. Exact curvature, simple controls and known related work are evaluated explicitly.')
    add('code','from pathlib import Path\nimport os,sys,json\nos.environ["OMP_NUM_THREADS"]="2"\nos.environ["OPENBLAS_NUM_THREADS"]="2"\nroot=Path("/kaggle/working/v7_experiment");root.mkdir(exist_ok=True)\nfiles='+repr(files)+'\nfor name,text in files.items():\n p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)\nos.chdir(root)\nsys.path[:0]=[str(root),str(root/"tests")]\n')
    add('code','import subprocess\nsubprocess.run([sys.executable,"-m","pytest","-q","tests/test_prior_block_unit.py","tests/test_v7_research_unit.py"],check=True)\nsubprocess.run([sys.executable,"-m","pytest","-q","tests/test_v7_integration.py","tests/test_v7_research_integration.py"],check=True)\n')
    expected=plan['reference_cohorts'][key]
    cohorts=dict(train=expected['training_cohort'],validation=expected['validation_cohort'])
    add('code','output=Path("/kaggle/working/research_results");output.mkdir(exist_ok=True)\n(output/"prespecified-plan.json").write_text(json.dumps('+repr(plan)+',indent=2))\nfrom v7_research import run\nfrom validation_data import sources\nrun('+repr(key)+',sources('+repr(key)+',"/kaggle/input"),output,references='+repr(references)+',expected_cohorts='+repr(cohorts)+',expected_files='+repr(expected.get('source_files'))+')\n')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);target=destination/'research.ipynb'
    target.write_text(json.dumps(dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5),indent=1))
    settings=metadata(owner);settings.update(id=f'{owner}/altcboost-v7-research-{key.replace("_","-")}-20261113',
        title=f'ALTCBoost V7 Research {key} 20261113',code_file=target.name,dataset_sources=[SPECS[key]['ref']],
        docker_image=image,docker_image_pinning_type='original')
    (destination/'kernel-metadata.json').write_text(json.dumps(settings,indent=2))
    return settings
