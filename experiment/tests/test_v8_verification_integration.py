import json
from pathlib import Path
from test_multi_unit import fixture_csv
from v8_research import run
from verify_v8 import check_results
import pandas as pd

def test_saved_research_outputs_pass_aggregate_verification(tmp_path):
    path,_=fixture_csv(tmp_path,8000);source=tmp_path/'output'
    run('paysim',[path],source,trees=3,train_cap=1000,eval_cap=2000)
    result=check_results(pd.read_csv(source/'validation.csv'),json.loads((source/'models.json').read_text()),json.loads((source/'discovery.json').read_text()),3)
    assert result['proposal_budget_checked'] and result['population_and_metrics_checked']
