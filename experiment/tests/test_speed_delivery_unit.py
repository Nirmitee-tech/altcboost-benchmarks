import json
import pandas as pd
import pytest
from speed_report import write_speed_report

def fixture_output(folder):
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([dict(path='synthetic',method='synthetic',batch_size=1,p50_call_us=5.,p99_call_us=9.)]).to_csv(folder/'timing.csv',index=False)
    (folder/'parity.json').write_text(json.dumps([dict(path='synthetic',accepted=True)]))
    (folder/'manifest.json').write_text(json.dumps(dict(scope='synthetic test',unit='scoring call')))

def test_rejected_path_cannot_be_reported_as_timed(tmp_path):
    fixture_output(tmp_path)
    (tmp_path/'parity.json').write_text(json.dumps([dict(path='synthetic',accepted=False)]))
    with pytest.raises(ValueError,match='Timed paths differ'):
        write_speed_report(tmp_path,tmp_path/'report')

def test_report_ratio_uses_equivalent_single_row_calls(tmp_path):
    fixture_output(tmp_path)
    pd.DataFrame([
        dict(path='ALTCBoost_numba',method='altc',batch_size=1,p50_call_us=10.,p99_call_us=20.),
        dict(path='CatBoost_cpp_guarded',method='cat',batch_size=1,p50_call_us=25.,p99_call_us=40.)
    ]).to_csv(tmp_path/'timing.csv',index=False)
    (tmp_path/'parity.json').write_text(json.dumps([
        dict(path='ALTCBoost_numba',accepted=True),dict(path='CatBoost_cpp_guarded',accepted=True)]))
    summary=write_speed_report(tmp_path,tmp_path/'report')
    assert summary['single_call_speedup_vs_guarded_cpp']==2.5
