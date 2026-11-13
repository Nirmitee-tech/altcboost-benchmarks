import pytest
from watch_kaggle import parse_status

@pytest.mark.parametrize('state',['RUNNING','COMPLETE','ERROR','QUEUED'])
def test_actual_cli_worker_states_are_recognized(state):
    assert parse_status(f'owner/slug has status "KernelWorkerStatus.{state}"')==state

def test_missing_status_does_not_read_as_completion():
    with pytest.raises(ValueError,match='recognized'):
        parse_status('Request timed out')
