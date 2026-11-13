import json
from pathlib import Path
from multi_job import prepare_multi_job
from watch_kaggle import tick

def test_embedded_job_runs_tests_before_benchmark(tmp_path,monkeypatch):
    prepare_multi_job('exampleuser','paysim',tmp_path,'pinned-image')
    book=json.loads((tmp_path/'benchmark.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
    assert code.index('pytest')<code.index('from multi_benchmark import run')
    # Execute the embedded source installation in an isolated local working directory.
    install=''.join(book['cells'][1]['source']).replace('/kaggle/working/multi_experiment',str(tmp_path/'installed'))
    monkeypatch.chdir(tmp_path)
    exec(compile(install,'<embedded>','exec'),{})
    assert (tmp_path/'installed/multi_report.py').exists()

def test_multi_watcher_waits_without_claiming_results(tmp_path):
    record=tick(tmp_path,'owner/kernel/1',runner=lambda args:'KernelWorkerStatus.RUNNING',kind='multi')
    assert record['state']=='RUNNING' and 'report' not in record

def test_multi_watcher_downloads_and_writes_completed_report(tmp_path):
    from test_multi_unit import fixture_csv
    from multi_benchmark import run
    from sklearn.tree import DecisionTreeClassifier
    path,_=fixture_csv(tmp_path)
    results=tmp_path/'kaggle-output/multi_results'
    run('paysim',[path],results,300,250,models=[('tree',{},DecisionTreeClassifier(max_depth=1))])
    calls=[]
    def runner(args):
        calls.append(args)
        return 'KernelWorkerStatus.COMPLETE' if args[1]=='status' else ''
    record=tick(tmp_path,'owner/kernel/1',runner=runner,kind='multi')
    assert Path(record['report']).exists()
    assert calls[1][1]=='output'
    from multi_summary import write_summary
    summary_root=tmp_path/'summary';summary_root.mkdir()
    import shutil
    shutil.copytree(tmp_path/'kaggle-output',summary_root/'paysim/kaggle-output')
    shutil.copy(tmp_path/'kaggle-status.json',summary_root/'paysim/kaggle-status.json')
    summary=write_summary(summary_root,keys=['paysim'])
    assert 'paysim' in summary.read_text() and 'Unavailable' in summary.read_text()
    assert (summary_root/'benchmark-results.csv').exists()
