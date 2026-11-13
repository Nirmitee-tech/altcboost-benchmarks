import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from multi_benchmark import run
from multi_report import write_multi_report
from test_multi_unit import fixture_csv

def test_runner_exports_aggregate_metrics_and_review(tmp_path):
    path,_=fixture_csv(tmp_path,n=40000)
    rows,discovery=run('paysim',[path],tmp_path/'output',train_cap=5000,eval_cap=8000,
        models=[('small_tree',dict(depth=2),DecisionTreeClassifier(max_depth=2,random_state=1))],chunksize=4001)
    assert len(rows)==3 and any(row['status']=='evaluated' for row in rows)
    for row in rows:
        if row['status']=='evaluated':
            assert row['tp']+row['fn']==row['test_positive']
            assert row['fp']+row['tn']==row['test_negative']
    exported=set(p.name for p in (tmp_path/'output').iterdir())
    assert exported=={'discovery.json','preprocessing.json','models.json','results.csv','manifest.json'}
    report=write_multi_report(tmp_path/'output',tmp_path/'review')
    assert 'synthetic' in report.read_text() and 'False positives' in report.read_text()
    assert 'synthetic0' not in ''.join(p.read_text() for p in (tmp_path/'output').iterdir())
    assert json.loads((tmp_path/'output/manifest.json').read_text())['common_cohort']

def test_small_calibration_reports_unsupported_budget(tmp_path):
    path,_=fixture_csv(tmp_path)
    rows,_=run('paysim',[path],tmp_path/'output',300,250,
        models=[('small_tree',{},DecisionTreeClassifier(max_depth=1))])
    assert all(row['status']=='unsupported' for row in rows)
