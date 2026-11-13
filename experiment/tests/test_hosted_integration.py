import json
from hosted_summary import write_report
from test_hosted_unit import rows

def test_report_and_aggregate_evidence_are_written(tmp_path):
    result=tmp_path/'results.csv'; rows().to_csv(result,index=False)
    discovery=tmp_path/'discovery.json'
    discovery.write_text(json.dumps({'splits':{'test':{'positive':4,'negative':1000}}}))
    write_report(result,discovery,tmp_path/'report')
    report=(tmp_path/'report/review.md').read_text()
    assert 'Counts are not added across seeds' in report
    assert '3-3' in report
    evidence=json.loads((tmp_path/'report/summary.json').read_text())
    assert evidence['source']==str(result.resolve())
    assert (tmp_path/'report/operating-point-0.001.csv').is_file()
