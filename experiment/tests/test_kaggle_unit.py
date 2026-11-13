import pytest
from kaggle_job import metadata

def test_job_is_private_cpu_with_only_the_public_dataset():
    value=metadata('exampleuser')
    assert value['id']=='exampleuser/altcboost-fraud-benchmark-20261112'
    assert value['is_private']=='true'
    assert value['enable_gpu']=='false'
    assert value['enable_internet']=='false'
    assert value['dataset_sources']==['mlg-ulb/creditcardfraud']
    assert value['competition_sources']==[]

@pytest.mark.parametrize('owner',['../someone','user/name','user name',''])
def test_owner_must_be_a_username(owner):
    with pytest.raises(ValueError,match='username'):
        metadata(owner)
