import pandas as pd
import pytest
from hosted_summary import METHODS,summarize

def rows():
    return pd.DataFrame([dict(method=m,seed=s,alpha=.001,status='evaluated',tp=3,fn=1,fp=1,tn=999,
        recall=.75,fpr=.001,single_p50_us=10.,fit_s=1.) for m in sorted(METHODS) for s in (17,42)])

def test_repeated_seed_counts_are_not_pooled():
    summary,_=summarize(rows())
    assert all(r['tp_min']==3 and r['tp_max']==3 for r in summary)
    assert all(r['meets_empirical_budget_each_seed'] for r in summary)

def test_incomplete_run_is_rejected():
    data=rows()
    with pytest.raises(ValueError,match='seeds'):
        summarize(data[data.seed==17])

def test_duplicate_method_is_rejected():
    data=rows()
    with pytest.raises(ValueError,match='Duplicate'):
        summarize(pd.concat([data,data.iloc[:1]]))

def test_metrics_must_agree_with_counts():
    data=rows();data.loc[0,'recall']=1.
    with pytest.raises(ValueError,match='Recall disagrees'):
        summarize(data)
