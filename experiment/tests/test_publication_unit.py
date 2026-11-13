import pandas as pd
from publish_benchmark import table,render
from test_accuracy_audit_unit import fixture

def test_public_table_escapes_text_and_distinguishes_validation_from_final():
    assert '&lt;script&gt;' in table(['Heading'],[['<script>']])
    validation=fixture();final=fixture()
    page=render(validation,final,'synthetic observation time')
    assert 'consumed validation cohorts' in page and 'No V8 final-test result' in page
    assert 'V6: earlier final-test benchmarks' in page and '35 of 1,129' in page
    assert '0.30%' not in page and '3.00%' in page

def test_kaggle_runner_uses_the_command_from_path(monkeypatch):
    import watch_kaggle
    calls=[]
    class Result:stdout='KernelWorkerStatus.COMPLETE'
    monkeypatch.setattr(watch_kaggle.subprocess,'run',lambda args,**kwargs:(calls.append(args) or Result()))
    assert 'COMPLETE' in watch_kaggle.run_cli(['kernels','status','synthetic/run'])
    assert calls[0][0]=='kaggle'
