"""Bundle source and tests into a notebook with no Drive dependency."""
import json
import hashlib
from pathlib import Path

root=Path(__file__).parent
names=['altcboost.py','altcboost_v2.py','altcboost_v3.py','discovery.py',
       'evaluation.py','run_experiment.py','tests/test_unit.py','tests/test_integration.py']
files={name:(root/name).read_text() for name in names}
cells=[]
def cell(kind,text):
    result=dict(cell_type=kind,id=hashlib.sha256(text.encode()).hexdigest()[:12],metadata={},source=text.splitlines(keepends=True))
    if kind=='code':result.update(execution_count=None,outputs=[])
    cells.append(result)

cell('markdown', '''# ALTCBoost: temporal fraud benchmark

Attach the Kaggle dataset `mlg-ulb/creditcardfraud` using Add Input, then Run All on CPU.
This notebook contains the model source, tests, discovery and benchmark runner.
It creates aggregate outputs only. It does not export transaction rows or predictions.

The proposed aligned variant removes hard-negative weighting and selects at 0.1% FPR.
It is an ablation whose performance must be measured. It is not an exact partial-AUC optimizer.

The dataset covers a short historical window and provides no card identities or label arrival times.
Time-ordered evaluation cannot establish entity independence, label-delay realism or production performance.
''')
cell('code', 'from pathlib import Path\nimport os, sys, json\nroot=Path("/kaggle/working/altcboost_experiment")\nroot.mkdir(parents=True,exist_ok=True)\nfiles='+repr(files)+'\nfor name,content in files.items():\n    path=root/name\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(content)\nsys.path.insert(0,str(root))\nos.chdir(root)\n')
cell('code', '''import subprocess
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_unit.py'],check=True)
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_integration.py'],check=True)
''')
cell('code', '''from discovery import load_data
matches=list(Path('/kaggle/input').rglob('creditcard.csv'))
if len(matches)!=1:
    raise RuntimeError('Attach exactly one mlg-ulb/creditcardfraud dataset containing creditcard.csv')
csv_path=matches[0]
_,_,discovery=load_data(csv_path)
print(json.dumps(discovery,indent=2))
''')
cell('markdown', '''## Prespecified experiment

The complete dataset is split into four time intervals. Equal timestamps stay together.
Identical complete rows are removed as a benchmark preprocessing choice, not evidence that those transactions were invalid.
The models see only the training interval. Validation selects configurations; a later calibration interval sets thresholds.
The final interval is used for reported results. Every model uses the same calibration rule.

The default below uses two seeds and a 70-tree limit. These are two training runs on one fixed split.
They do not constitute independent datasets. Equal configuration counts and tree limits do not establish equal tuning effort.
ALTCBoost uses internal early stopping; library baselines use their complete tree budget.
''')
cell('code', '''from run_experiment import run
rows=run(csv_path, '/kaggle/working/altcboost_results', seeds=(17,42), trees=70)
''')
cell('code', '''import pandas as pd
result=pd.read_csv('/kaggle/working/altcboost_results/results.csv')
display(result[['method','seed','alpha','status','recall','fpr','fpr_upper_95','precision','fit_s','single_p50_us','single_p99_us']])
print('Calibration bounds assume representative independent negatives. Temporal drift can violate that assumption.')
print('Latency measures model scoring on this machine; feature retrieval and network latency are excluded.')
''')
cell('markdown', '''## Research behind the experiment

* [Partial AUC minimax formulations](https://arxiv.org/abs/2512.01213): principled training objectives differ from the heuristic implemented here.
* [Large-scale partial AUC optimization](https://arxiv.org/abs/2203.01505): focuses ranking on a specified false-positive interval.
* [Neyman-Pearson classification](https://arxiv.org/abs/1802.02558): error control requires explicit distributional assumptions.
* [TabArena](https://arxiv.org/abs/2506.16791): validation, tuning and ensembles affect model comparisons.

Choose a candidate on validation data before inspecting final results. Repeatedly redesigning against the final holdout turns it into another validation set.
''')
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),nbformat=4,nbformat_minor=5)
(root.parent/'ALTCBoost_Kaggle.ipynb').write_text(json.dumps(notebook,indent=1))
for index,item in enumerate(cells):
    if item['cell_type']=='code':
        compile(''.join(item['source']),f'notebook_cell_{index}','exec')
assert all((root/name).read_text()==content for name,content in files.items())
print('Created',root.parent/'ALTCBoost_Kaggle.ipynb')
