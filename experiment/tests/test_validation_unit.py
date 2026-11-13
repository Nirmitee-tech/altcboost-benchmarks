import json
import numpy as np
import pandas as pd
import pytest
from validation_data import SPECS,predictors,load_elliptic,Inputs
from tuned_models import GRIDS,positive_weight,raw_score,build_model
from cluster_uncertainty import paired_cluster_interval,cluster_schemes

def elliptic_fixture(tmp_path,n=1600):
    rng=np.random.default_rng(72);row=np.arange(n)
    frame=pd.DataFrame(rng.normal(size=(n,167)))
    frame[0]=[f'synthetic-id-{i}' for i in row];frame[1]=row%49+1
    features=tmp_path/'elliptic_txs_features.csv';frame.to_csv(features,index=False,header=False)
    labels=pd.DataFrame(dict(txId=frame[0],**{'class':np.where(row%11==0,'unknown',np.where(row%7==0,'1','2'))}))
    classes=tmp_path/'elliptic_txs_classes.csv';labels.to_csv(classes,index=False)
    return features,classes,frame,labels

def test_elliptic_reads_local_features_and_excludes_unknown_labels(tmp_path):
    features,classes,frame,labels=elliptic_fixture(tmp_path)
    frames,report=load_elliptic([features,classes],chunksize=37)
    assert report['source_rows']==len(labels)
    assert sum(len(part) for part in frames)==int((labels['class']!='unknown').sum())
    assert len(predictors('elliptic'))==93 and 'f_095' not in predictors('elliptic')
    assert not set(predictors('elliptic'))&{'txId','__time','class','is_illicit'}
    for a,b in zip(frames,frames[1:]):assert a['__time'].max()<b['__time'].min()
    _,second=load_elliptic([features,classes],chunksize=83)
    assert report['partitions']==second['partitions']

def test_future_neighbour_features_cannot_change_elliptic_predictors(tmp_path):
    features,classes,frame,_=elliptic_fixture(tmp_path)
    before,_=load_elliptic([features,classes])
    frame.loc[:,95:]=99999.;frame.to_csv(features,index=False,header=False)
    after,_=load_elliptic([features,classes])
    assert all(np.array_equal(a[predictors('elliptic')],b[predictors('elliptic')]) for a,b in zip(before,after))

def test_unknown_label_state_and_duplicate_ids_are_rejected(tmp_path):
    features,classes,_,labels=elliptic_fixture(tmp_path)
    labels.loc[0,'class']='3';labels.to_csv(classes,index=False)
    with pytest.raises(ValueError,match='label state'):load_elliptic([features,classes])
    labels.loc[0,'class']='unknown';labels.loc[0,'txId']=labels.loc[1,'txId'];labels.to_csv(classes,index=False)
    with pytest.raises(ValueError,match='unique present'):load_elliptic([features,classes])

def test_baseline_search_sizes_and_frozen_candidate_settings():
    assert all(len(grid)==4 for grid in GRIDS.values())
    assert positive_weight(dict(positive_weight='train_ratio'),[0,0,0,1])==3.
    model=build_model('V6',{},np.array([0,1]),[],17,100)
    assert model.learning_rate==.08 and model.max_leaf_nodes==17 and model.tail_fraction==.01
    assert model.anchor_weight==.2 and model.group_temperature==.25
    assert raw_score('XGBoost',None,np.ones((4,2)),0).tolist()==[0.]*4

def test_paired_cluster_intervals_use_common_draws_and_zero_for_identical_models():
    y=np.tile([0,1],30);clusters=np.repeat(np.arange(30),2)
    first=(np.arange(60)%3==0);second=(np.arange(60)%5==0)
    same=paired_cluster_interval(y,first,first,clusters,replicates=100)
    assert same['delta_recall_interval_95']==[0.,0.] and same['delta_fpr_interval_95']==[0.,0.]
    a=paired_cluster_interval(y,first,second,clusters,replicates=100)
    b=paired_cluster_interval(y,second,first,clusters,replicates=100)
    assert np.allclose(a['delta_recall_interval_95'],-np.array(b['delta_recall_interval_95'])[::-1])
    assert np.allclose(a['delta_fpr_interval_95'],-np.array(b['delta_fpr_interval_95'])[::-1])

def test_coarse_time_clusters_do_not_produce_a_confidence_claim():
    result=paired_cluster_interval([0,1]*10,[False,True]*10,[False]*20,np.repeat([0,1],10))
    assert result['status']=='unsupported' and 'delta_recall_interval_95' not in result

def test_cluster_unit_is_prespecified_not_chosen_from_outcomes():
    frame=pd.DataFrame({'__time':[4,5,7,8]})
    assert cluster_schemes('paysim',frame)['four_step_time_block'].tolist()==[1,1,1,2]
