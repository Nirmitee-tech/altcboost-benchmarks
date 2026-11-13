import json
from pathlib import Path
from test_hosted_unit import rows
from watch_kaggle import tick

def test_completed_job_download_produces_a_readable_report(tmp_path):
    def cli(args):
        if args[1]=='status':
            return 'KernelWorkerStatus.COMPLETE'
        folder=Path(args[args.index('-p')+1])/'altcboost_results'
        folder.mkdir(parents=True)
        rows().to_csv(folder/'results.csv',index=False)
        (folder/'discovery.json').write_text(json.dumps({'splits':{'test':{'positive':4,'negative':1000}}}))
        return 'Downloaded'
    state=tick(tmp_path,'owner/slug/1',runner=cli)
    assert state['state']=='COMPLETE'
    assert Path(state['report']).is_file()
    assert (tmp_path/'hosted-report/summary.json').is_file()

def test_running_job_does_not_claim_a_report(tmp_path):
    state=tick(tmp_path,'owner/slug/1',runner=lambda _: 'KernelWorkerStatus.RUNNING')
    assert state['state']=='RUNNING'
    assert 'report' not in state
    assert not (tmp_path/'hosted-report').exists()
