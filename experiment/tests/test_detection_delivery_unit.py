from detection_report import compare_counts,audit_reading

def test_more_fraud_with_more_false_alarms_is_not_a_joint_win():
    assert compare_counts(45,50,40,36)=='tradeoff; not a joint improvement'

def test_fewer_alerts_with_more_missed_fraud_is_not_a_joint_win():
    assert compare_counts(35,10,40,36)=='tradeoff; not a joint improvement'

def test_joint_improvement_is_identified():
    assert compare_counts(42,20,40,36)=='more fraud and fewer false alarms'

def test_audit_preserves_unchanged_historical_test_counts():
    audit=dict(rows=[dict(legacy_tp=40,legacy_fp=35,strict=dict(tp=40,fp=35))])
    assert 'reproduced the earlier test TP/FP counts' in audit_reading(audit)

def test_audit_does_not_hide_changed_test_counts():
    audit=dict(rows=[dict(legacy_tp=40,legacy_fp=35,strict=dict(tp=40,fp=34))])
    assert 'changed some test counts' in audit_reading(audit)
