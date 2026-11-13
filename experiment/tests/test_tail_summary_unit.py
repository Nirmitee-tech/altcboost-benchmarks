import pytest
from tail_summary import observed_comparison

@pytest.mark.parametrize('candidate,reference,expected',[
    ((10,2),(9,3),'more or equal'),((8,4),(9,3),'fewer or equal'),
    ((10,4),(9,3),'trade off'),((9,3),(9,3),'same counts')])
def test_count_comparison_does_not_call_a_tradeoff_an_improvement(candidate,reference,expected):
    assert expected in observed_comparison(dict(zip(['tp','fp'],candidate)),dict(zip(['tp','fp'],reference)))

@pytest.mark.parametrize('iteration',[0,2])
def test_zero_tree_prefix_is_not_reported_as_a_tree_training_gain(iteration):
    from tail_summary import prefix_note
    result=prefix_note('synthetic',[dict(method='candidate',selected_iteration=iteration)],'candidate')
    assert (result is not None)==(iteration==0)
