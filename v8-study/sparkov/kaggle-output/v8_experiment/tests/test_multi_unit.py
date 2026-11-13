import numpy as np
import pandas as pd
import pytest
from multi_data import SPECS,feature_names,load_cohort,priorities,read_chunks
from multi_benchmark import preprocessing

def fixture_csv(tmp_path,n=4000):
    row=np.arange(n)
    frame=pd.DataFrame(dict(step=row//40,type=np.where(row%2,'PAYMENT','TRANSFER'),
        amount=(row%37).astype(float),oldbalanceOrg=(row%53).astype(float),
        oldbalanceDest=(row%17).astype(float),nameOrig=['synthetic'+str(i%50) for i in row],
        isFraud=(row%13==0).astype(int)))
    path=tmp_path/'synthetic.csv';frame.to_csv(path,index=False);return path,frame

def test_sampling_is_chunk_invariant_and_label_blind(tmp_path):
    path,frame=fixture_csv(tmp_path)
    first,a=load_cohort('paysim',[path],300,250,chunksize=137)
    _,b=load_cohort('paysim',[path],300,250,chunksize=1001)
    assert a['partitions']==b['partitions']
    frame.isFraud=1-frame.isFraud;frame.to_csv(path,index=False)
    second,_=load_cohort('paysim',[path],300,250,chunksize=501)
    assert all(np.array_equal(x['__row'],y['__row']) for x,y in zip(first,second))
    assert all(x['__time'].max()<y['__time'].min() for x,y in zip(first,first[1:]))

def test_labels_checked_before_cast(tmp_path):
    path,frame=fixture_csv(tmp_path);frame.loc[0,'isFraud']=2;frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='binary'):list(read_chunks('paysim',[path]))

@pytest.mark.parametrize('key',list(SPECS))
def test_predictor_allowlist_excludes_targets_identifiers_and_future_fields(key):
    assert not set(feature_names(key))&{'isFraud','is_fraud','fraud_bool','nameOrig','cc_num',
        'newbalanceOrig','newbalanceDest','isFlaggedFraud','month','device_fraud_count'}

def test_training_preprocessing_does_not_learn_test_values(tmp_path):
    _,frame=fixture_csv(tmp_path);names=feature_names('paysim');processor=preprocessing('paysim')
    train=frame[names].copy();train.loc[0,'amount']=np.nan
    x=processor.fit_transform(train)
    test=train.iloc[:2].copy();test['type']='UNSEEN';test['amount']=np.nan
    transformed=processor.transform(test)
    assert transformed.shape[1]==x.shape[1] and np.isfinite(transformed).all()
    assert processor.named_steps['prepare'].named_transformers_['numeric'].statistics_[0]==np.nanmedian(train.amount)

def test_priorities_are_row_only():
    assert np.array_equal(priorities(np.arange(10),17),np.concatenate([priorities(np.arange(4),17),priorities(np.arange(4,10),17)]))

def test_sparkov_official_time_overlap_is_rejected(tmp_path):
    frame=pd.DataFrame(dict(trans_date_trans_time=pd.date_range('2020-01-01',periods=100,freq='h').astype(str),
        amt=np.arange(100),city_pop=100,category='synthetic',cc_num='synthetic',is_fraud=np.arange(100)%2))
    first=tmp_path/'train.csv';second=tmp_path/'test.csv';frame.to_csv(first,index=False);frame.to_csv(second,index=False)
    with pytest.raises(ValueError,match='overlap'):load_cohort('sparkov',[first,second])
