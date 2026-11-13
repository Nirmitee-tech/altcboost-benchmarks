from tail_experiment import VARIANTS,candidate_models

def test_removal_experiments_isolate_time_tail_and_pointwise_control():
    assert VARIANTS['V5_temporal_tail']==dict(tail_fraction=.01,anchor_weight=.2,group_count=4)
    assert VARIANTS['V5_pooled_tail']['group_count']==1
    assert VARIANTS['V5_temporal_all_negatives']['tail_fraction']==1.
    assert VARIANTS['V5_balanced_logistic']['anchor_weight']==1.
    assert len(list(candidate_models(trees=2)))==7
