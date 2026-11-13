import json
import numpy as np
import pandas as pd
import pytest
from validation_data import Inputs
from tuned_models import build_model,fit_model,staged_scores,raw_score,select_prefix,GRIDS
from validation_experiment import run
from test_multi_unit import fixture_csv

@pytest.mark.parametrize('family',list(GRIDS))
def test_native_and_cached_prefix_scores_match_direct_prediction(tmp_path,family):
    _,frame=fixture_csv(tmp_path,4000);train=frame.iloc[:2800];validation=frame.iloc[2800:]
    inputs=Inputs('paysim',train);x=inputs.transform(train,family);xv=inputs.transform(validation,family)
    y=train.isFraud.to_numpy();yv=validation.isFraud.to_numpy()
    model=build_model(family,GRIDS[family][0],y,inputs.cats,17,4)
    fit_model(family,model,x,y,xv,yv,np.zeros(len(y)))
    yielded=list(staged_scores(family,model,xv));assert yielded
    for count,scores in yielded:assert np.allclose(scores,raw_score(family,model,xv,count),rtol=0,atol=1e-12)
    prefix,key=select_prefix(family,model,xv,yv)
    assert 0<=prefix<=4 and len(key)==3

def test_native_categories_are_fitted_on_training_only(tmp_path):
    _,frame=fixture_csv(tmp_path);inputs=Inputs('paysim',frame.iloc[:1000])
    future=frame.iloc[1000:1003].copy();future['type']='unseen-category'
    native=inputs.transform(future,'LightGBM_native')
    assert native.type.cat.codes.eq(-1).all()
    cat=inputs.transform(future,'CatBoost_native');assert cat.type.eq('unseen-category').all()
    assert inputs.transform(future,'V6').shape[1]==inputs.transform(frame.iloc[:3],'V6').shape[1]

def test_search_repeat_and_uncertainty_pipeline_exports_aggregate_only(tmp_path):
    path,_=fixture_csv(tmp_path,40000)
    rows,selection=run('paysim',[path],tmp_path/'out',train_cap=1000,eval_cap=8000,seeds=(17,42),trees=3,replicates=30)
    assert len(rows)==24 and set(row['seed'] for row in rows)=={17,42}
    trials=pd.read_csv(tmp_path/'out/trials.csv')
    assert trials.groupby('family').size().eq(4).all()
    manifest=json.loads((tmp_path/'out/manifest.json').read_text());assert manifest['frozen_v6']
    assert set(selection['selected'])=={'V6',*GRIDS}
    text=''.join(p.read_text() for p in (tmp_path/'out').iterdir())
    assert 'synthetic0' not in text
    assert set(p.name for p in (tmp_path/'out').iterdir())=={'discovery.json','selection.json','trials.csv','results.csv','window-results.csv','models.json','uncertainty.json','manifest.json'}

def test_final_labels_do_not_select_configurations_or_prefixes(tmp_path):
    from multi_data import load_cohort
    path,frame=fixture_csv(tmp_path,24000)
    _,discovery=load_cohort('paysim',[path],1000,8000)
    _,first=run('paysim',[path],tmp_path/'first',1000,8000,seeds=(17,),trees=2,replicates=20)
    mask=frame.step>=discovery['time_edges'][-1];frame.loc[mask,'isFraud']=1-frame.loc[mask,'isFraud'];frame.to_csv(path,index=False)
    _,second=run('paysim',[path],tmp_path/'second',1000,8000,seeds=(17,),trees=2,replicates=20)
    clean=lambda selection:{family:(record['config'],record['prefix'],record['validation_key']) for family,record in selection['selected'].items()}
    assert clean(first)==clean(second)
