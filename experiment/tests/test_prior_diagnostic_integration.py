import json
from prior_diagnostic import run
from test_multi_unit import fixture_csv

def test_prior_diagnostic_exports_only_training_validation_aggregates(tmp_path):
    from validation_data import SPECS
    path,_=fixture_csv(tmp_path)
    source=tmp_path/SPECS['paysim']['files'][0];path.rename(source)
    output=tmp_path/'diagnostic.json';result=run('paysim',tmp_path,output)
    assert len(result['ceilings'])==3 and 'no calibration or final score' in result['scope']
    text=output.read_text();assert 'synthetic0' not in text and 'test_positive' not in text
