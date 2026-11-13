import json
from pathlib import Path
from test_multi_unit import fixture_csv
from v8_research import run as research
from accuracy_audit import run

def test_audit_recomputes_saved_research_counts_and_writes_review(tmp_path):
    path,_=fixture_csv(tmp_path,8000);study=tmp_path/'study';folder=study/'paysim'
    research('paysim',[path],folder/'kaggle-output/research_results',trees=2,train_cap=1000,eval_cap=2000)
    (folder/'kaggle-status.json').write_text(json.dumps(dict(state='COMPLETE')))
    (study/'prespecified-plan.json').write_text(json.dumps(dict(reference_cohorts={'paysim':{}})))
    table,manifest=run(study,tmp_path/'audit')
    assert len(table)==1 and table.iloc[0]['method']=='V8_adaptive_checked'
    assert manifest['sources'][0]['unit']=='transaction row'
    text=(tmp_path/'audit/accuracy-review.md').read_text()
    assert 'Overall accuracy' in text and 'All-legitimate accuracy' in text
    assert 'Adaptive candidate' in text and '| V8_adaptive_checked |' not in text
    assert (tmp_path/'audit/all-operating-points.csv').exists()
