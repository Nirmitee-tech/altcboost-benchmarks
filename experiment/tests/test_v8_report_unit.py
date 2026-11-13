from v8_report import summarize_attempts

def test_attempt_summary_separates_working_descent_from_target_increase():
    rows=summarize_attempts([dict(method='example',attempt_diagnostics=[
        dict(fraction=.01,accepted=False,reason='non_descent_direction'),
        dict(fraction=.02,accepted=True,target_loss_before=1.,target_loss_after=1.2),
        dict(fraction=.02,accepted=True,target_loss_before=1.2,target_loss_after=.9)])])
    assert rows.proposals.sum()==3 and rows.accepted.sum()==2
    assert rows.non_descent_rejections.sum()==1 and rows.target_loss_increases.sum()==1

def test_comparison_does_not_credit_adaptation_for_a_simple_control_gain():
    import pandas as pd
    from v8_report import compare_mechanisms
    rows=pd.DataFrame([dict(method='V8_'+mode,validation_mean_recall=score,validation_mean_fpr=.001,ap=.5)
        for mode,score in [('target',0.),('linear',.06),('adaptive_checked',.02)]])
    models=[dict(method='V8_adaptive_'+name,objective_evaluations=calls,fit_s=seconds)
        for name,calls,seconds in [('blind',300,100.),('checked',240,95.)]]
    result=compare_mechanisms(rows,models)
    assert not result['adaptive_beats_best_simple'] and result['best_simple_method']=='V8_linear'
    assert abs(result['objective_call_reduction_fraction']-.2)<1e-12
