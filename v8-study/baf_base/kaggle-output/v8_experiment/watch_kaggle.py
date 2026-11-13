"""Retrieve a completed Kaggle run and build its aggregate report."""
import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from hosted_summary import write_report

CLI='/Users/apple/.local/bin/kaggle'

def parse_status(text):
    match=re.search(r'KernelWorkerStatus\.([A-Z_]+)',text)
    if not match:
        raise ValueError('Kaggle returned no recognized worker status')
    state=match.group(1)
    if state not in {'QUEUED','RUNNING','COMPLETE','ERROR','CANCEL_REQUESTED','CANCEL_ACKNOWLEDGED','NEW_SCRIPT'}:
        raise ValueError('Unknown Kaggle worker status')
    return state

def run_cli(args):
    return subprocess.run([CLI,*args],text=True,capture_output=True,check=True,timeout=180).stdout

def tick(root,kernel,runner=run_cli,kind='fraud'):
    if kind not in {'fraud','speed','detection','multi','validation','v7','v8'}:raise ValueError('Unknown experiment kind')
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    state=parse_status(runner(['kernels','status',kernel]))
    record=dict(kernel=kernel,state=state,checked_utc=datetime.now(timezone.utc).isoformat())
    if state=='COMPLETE':
        out=root/'kaggle-output';out.mkdir(exist_ok=True)
        runner(['kernels','output',kernel,'-p',str(out)])
        pattern={'v8':'research_results/validation.csv','v7':'research_results/validation.csv','validation':'validation_results/results.csv','multi':'multi_results/results.csv','speed':'speed_results/timing.csv','detection':'quality_results/results.csv',
                 'fraud':'altcboost_results/results.csv'}[kind]
        results=list(out.rglob(pattern))
        if len(results)!=1:
            raise ValueError('Expected one completed benchmark results file')
        if kind=='v8':
            from v8_report import write_v8_report
            write_v8_report(results[0].parent,root/'hosted-report')
        elif kind=='v7':
            from v7_report import write_v7_report
            write_v7_report(results[0].parent,root/'hosted-report')
        elif kind=='validation':
            from validation_report import write_validation_report
            write_validation_report(results[0].parent,root/'hosted-report')
        elif kind=='multi':
            from multi_report import write_multi_report
            write_multi_report(results[0].parent,root/'hosted-report')
        elif kind=='speed':
            from speed_report import write_speed_report
            write_speed_report(results[0].parent,root/'hosted-report')
        elif kind=='detection':
            from detection_report import write_detection_report
            write_detection_report(results[0].parent,root/'hosted-report')
        else:
            discovery=results[0].with_name('discovery.json')
            write_report(results[0],discovery,root/'hosted-report')
        record['report']=str(root/'hosted-report/review.md')
    elif state in {'ERROR','CANCEL_ACKNOWLEDGED'}:
        (root/'kaggle-failure.log').write_text(runner(['kernels','logs',kernel]))
    (root/'kaggle-status.json').write_text(json.dumps(record,indent=2))
    return record

def watch(root,kernel,interval=45,kind='fraud'):
    root=Path(root)
    while True:
        try:
            record=tick(root,kernel,kind=kind)
            print(json.dumps(record),flush=True)
            if record['state'] in {'COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'}:
                return record
        except Exception as error:
            # Retry retrieval failures; a failed HTTP request is not a failed training run.
            record=dict(checked_utc=datetime.now(timezone.utc).isoformat(),
                        retrieval_error=str(error))
            (root/'kaggle-retrieval-error.json').write_text(json.dumps(record,indent=2))
            print(json.dumps(record),flush=True)
        time.sleep(interval)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kernel');parser.add_argument('--root',default='..')
    parser.add_argument('--kind',choices=['fraud','speed','detection','multi','validation','v7','v8'],default='fraud')
    args=parser.parse_args();watch(args.root,args.kernel,kind=args.kind)
