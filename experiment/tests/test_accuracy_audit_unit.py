import pandas as pd
import pytest
from accuracy_audit import audit_counts

def fixture():
    return pd.DataFrame([dict(dataset='synthetic',method='candidate',alpha=.001,tp=3,fn=97,fp=9,tn=9891,recall=.03,fpr=9/9900,precision=.25)])

def test_audit_exposes_high_accuracy_with_poor_fraud_recall():
    row=audit_counts(fixture()).iloc[0]
    assert row.accuracy==pytest.approx(.9894) and row.all_legitimate_accuracy==pytest.approx(.99)
    assert row.recall_recomputed==.03 and row.precision_recomputed==.25
    assert row.accuracy_lift_percentage_points<0 and row.within_empirical_budget

@pytest.mark.parametrize('field,value',[('tp',-1),('fn',.5),('fp',float('nan')),('recall',.99),('precision',.99)])
def test_audit_rejects_invalid_counts_or_inconsistent_rates(field,value):
    rows=fixture();rows[field]=value
    with pytest.raises(ValueError):audit_counts(rows)

def test_all_legitimate_prediction_has_zero_precision_and_zero_recall():
    rows=fixture();rows[['tp','fp','fn','tn','recall','fpr','precision']]=[0,0,100,9900,0.,0.,0.]
    row=audit_counts(rows).iloc[0]
    assert row.accuracy==row.all_legitimate_accuracy and row.precision_recomputed==row.recall_recomputed==0
