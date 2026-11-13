import json
from pathlib import Path
import pandas as pd
from detection_job import prepare_detection_job
from watch_kaggle import tick

def test_detection_notebook_is_private_and_exploratory(tmp_path):
    manifest=prepare_detection_job('exampleuser',tmp_path,image='example-image')
    settings=json.loads((tmp_path/'kernel-metadata.json').read_text())
    notebook=json.loads((tmp_path/settings['code_file']).read_text())
    combined=''.join(''.join(c['source']) for c in notebook['cells'])
    assert settings['is_private']=='true' and settings['enable_gpu']=='false'
    assert 'development-only' in manifest['scope']
    assert 'not an independent final test' in combined
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<notebook>','exec')

def test_completed_quality_job_writes_a_tradeoff_aware_report(tmp_path):
    def cli(args):
        if args[1]=='status':return 'KernelWorkerStatus.COMPLETE'
        output=Path(args[args.index('-p')+1])/'quality_results';output.mkdir(parents=True)
        pd.DataFrame([dict(method='synthetic',alpha=.001,status='exploratory',tp=45,fp=50)]).to_csv(output/'results.csv',index=False)
        (output/'manifest.json').write_text(json.dumps(dict(scope='synthetic development',unit='transaction row')))
        (output/'incumbent.json').write_text(json.dumps(dict(tp=40,fp=36)))
        (output/'historical-xgboost-audit.json').write_text(json.dumps(dict(rows=[
            dict(legacy_tp=40,legacy_fp=35,strict=dict(tp=40,fp=35))])))
        return 'Downloaded'
    status=tick(tmp_path,'owner/slug/1',runner=cli,kind='detection')
    report=Path(status['report']).read_text()
    assert 'tradeoff; not a joint improvement' in report
    assert 'does not provide independent confirmation' in report
    assert 'reproduced the earlier test TP/FP counts' in report
