from v7_research import MODES

def test_mechanism_plan_includes_simpler_starting_model_controls():
    assert set(MODES)=={'zero','clipped','shrunk','adaptive_diagonal','adaptive_joint'}

def test_static_control_comparison_respects_the_primary_validation_key():
    import pandas as pd
    from v7_report import best_record
    rows=pd.DataFrame([dict(method='a',validation_mean_recall=.4,validation_mean_fpr=.0001,ap=.9),
                       dict(method='b',validation_mean_recall=.5,validation_mean_fpr=.0002,ap=.7)])
    assert best_record(rows,['a','b'])['method']=='b'
