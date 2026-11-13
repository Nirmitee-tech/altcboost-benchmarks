"""Rerunnable checks on aggregate research outputs and submitted code provenance."""
import argparse
import ast
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from v8_research import MODES

def check_results(rows,models,discovery,budget,reference_names=()):
    expected={'V8_'+name for name in MODES}|set(reference_names)
    if set(rows.method)!=expected or rows.duplicated(['method','alpha']).any():raise ValueError('Methods differ or records repeat')
    if set(rows.alpha)!={.0005,.001,.01} or len(rows)!=3*len(expected):raise ValueError('Operating points differ')
    val=discovery['partitions']['validation']
    if not (rows.tp+rows.fn).eq(val['sampled_positive']).all() or not (rows.fp+rows.tn).eq(val['sampled_negative']).all():raise ValueError('Validation population differs')
    if not np.allclose(rows.recall,rows.tp/(rows.tp+rows.fn),rtol=0,atol=1e-12):raise ValueError('Recall differs from counts')
    if not np.allclose(rows.fpr,rows.fp/(rows.fp+rows.tn),rtol=0,atol=1e-12):raise ValueError('FPR differs from counts')
    by_name={model['method']:model for model in models}
    if set(by_name)!={'V8_'+name for name in MODES}:raise ValueError('Candidate metadata differs')
    for model in models:
        attempts=model['attempt_diagnostics'];steps=model['step_diagnostics']
        if model['weak_learner_fits']!=budget or len(attempts)!=budget:raise ValueError('Proposal budget differs')
        if len(steps)!=sum(s['accepted'] for s in attempts):raise ValueError('Accepted step count differs')
        if not all(s['working_loss_after']<s['working_loss_before'] for s in steps):raise ValueError('Working loss did not descend')
        if not 0<=model['selected_prefix']<=len(steps):raise ValueError('Selected prefix outside fitted trees')
    blind=by_name['V8_adaptive_blind'];checked=by_name['V8_adaptive_checked']
    return dict(population_and_metrics_checked=True,working_loss_descent_checked=True,proposal_budget_checked=True,
        blind_checked_loss_history_matches=blind['training_loss']==checked['training_loss'],
        blind_checked_selected_prefix_matches=blind['selected_prefix']==checked['selected_prefix'],
        blind_objective_calls=blind['objective_evaluations'],checked_objective_calls=checked['objective_evaluations'],
        full_prediction_parity='Not checked on source rows; toy integration test covers it. Matching aggregate histories do not prove full prediction equality.')

def verify(study,key,plan):
    study=Path(study);folder=study/key
    if json.loads((folder/'kaggle-status.json').read_text())['state']!='COMPLETE':raise ValueError('Run is incomplete')
    paths=list((folder/'kaggle-output').rglob('research_results/validation.csv'))
    if len(paths)!=1:raise ValueError('Expected one research result')
    source=paths[0].parent;discovery=json.loads((source/'discovery.json').read_text());expected=plan['reference_cohorts'][key]
    for name,field in [('train','training_cohort'),('validation','validation_cohort')]:
        if discovery['partitions'][name]['cohort_sha256']!=expected[field]:raise ValueError('Cohort hash differs')
    if {r['name']:r['sha256'] for r in discovery['files']}!=expected['source_files']:raise ValueError('Source hashes differ')
    if json.loads((source/'prespecified-plan.json').read_text())!=plan:raise ValueError('Hosted plan differs')
    book=json.loads((folder/'job/research.ipynb').read_text());tree=ast.parse(''.join(book['cells'][1]['source']))
    files=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='files' for t in node.targets))
    installed=source.parent/'v8_experiment';mismatches=[name for name,value in files.items() if not (installed/name).exists() or (installed/name).read_text()!=value]
    if mismatches:raise ValueError('Executed embedded code differs: '+','.join(mismatches))
    references=json.loads((folder/'references.json').read_text())
    rows=pd.read_csv(paths[0]);models=json.loads((source/'models.json').read_text())
    if not rows.dataset.eq(key).all():raise ValueError('Source name differs')
    result=check_results(rows,models,discovery,plan['fixed_settings']['total_tree_proposals'],references)
    result.update(dataset=key,observed_utc=datetime.now(timezone.utc).isoformat(),source_and_cohort_hashes_match=True,
        executed_embedded_files_match=True,embedded_file_count=len(files),
        results_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),scope='Training and original validation only; no independent confirmation')
    (folder/'verification.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args();root=Path(args.root)
    plan=json.loads((root/'prespecified-plan.json').read_text())
    results={key:verify(root,key,plan) for key in plan['reference_cohorts']}
    (root/'verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
