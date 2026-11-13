import numpy as np
import pytest
from speed_benchmark import Adapter,BoundaryGuard,checked_input,check_parity,measure

@pytest.mark.parametrize('value',[np.ones((2,2)),np.array([[np.nan,1,2]]),np.ones(3)])
def test_common_boundary_rejects_invalid_rows(value):
    with pytest.raises(ValueError,match='finite numeric rows'):
        checked_input(value,3)

def test_boundary_creates_the_same_contiguous_float64_input():
    value=np.arange(12,dtype=np.float32).reshape(3,4)[:,::2]
    result=checked_input(value,2)
    assert result.dtype==np.float64 and result.flags.c_contiguous
    assert np.array_equal(result,value)

def test_close_probabilities_do_not_override_changed_flags():
    result=check_parity([.2,.5],[.2,.5-1e-8],{.001:.5})
    assert result['max_probability_error']<1e-6
    assert result['flag_disagreements']['0.001']==1
    assert not result['accepted']

def test_equal_probabilities_pass_the_gate():
    result=check_parity([.2,.8],[.2,.8],{.001:.5})
    assert result['accepted']

def test_timer_keeps_batch_latency_and_cost_per_row_separate():
    path=Adapter('sum','synthetic',3,lambda x:x.sum(axis=1))
    result=measure([path],np.ones((20,3)),single_calls=3,rounds=2)
    one=next(row for row in result if row['batch_size']==1)
    batch=next(row for row in result if row['batch_size']==20)
    assert one['calls']==6 and batch['calls']==40
    assert batch['p50_us_per_row']==pytest.approx(batch['p50_call_us']/20)
    assert one['p99_call_us']>=one['p50_call_us']

def test_boundary_guard_only_recomputes_ambiguous_rows():
    seen=[]
    def reference(x):
        seen.append(x.copy());return np.full(len(x),.5)
    guard=BoundaryGuard(lambda x:x[:,0].copy(),reference,{.001:.5})
    scores=guard(np.array([[.1],[.5-1e-16],[.9]]))
    assert np.array_equal(scores,[.1,.5,.9])
    assert len(seen)==1 and len(seen[0])==1 and guard.fallback_rows==1
