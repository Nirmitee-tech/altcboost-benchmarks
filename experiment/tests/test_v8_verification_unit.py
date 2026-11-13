import pandas as pd
import pytest
from verify_v8 import check_results
from v8_research import MODES

def fixture():
    rows=pd.DataFrame([dict(method='V8_'+name,alpha=a,tp=2,fn=2,fp=1,tn=99,recall=.5,fpr=.01) for name in MODES for a in [.0005,.001,.01]])
    models=[dict(method='V8_'+name,attempt_diagnostics=[dict(accepted=True)],step_diagnostics=[dict(working_loss_before=2,working_loss_after=1)],weak_learner_fits=1,selected_prefix=1,training_loss=[2,1],objective_evaluations=3) for name in MODES]
    discovery=dict(partitions=dict(validation=dict(sampled_positive=4,sampled_negative=100)))
    return rows,models,discovery

def test_result_verification_checks_counts_and_working_loss():
    rows,models,discovery=fixture();assert check_results(rows,models,discovery,1)['working_loss_descent_checked']
    rows.loc[0,'tp']=3
    with pytest.raises(ValueError,match='population differs'):check_results(rows,models,discovery,1)
    rows,models,discovery=fixture();models[0]['step_diagnostics'][0]['working_loss_after']=3
    with pytest.raises(ValueError,match='did not descend'):check_results(rows,models,discovery,1)
