import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from speed_benchmark import CompiledCatBoost,checked_input,run
from discovery import FEATURES

def test_compiled_catboost_reproduces_probabilities_and_flags(tmp_path):
    rng=np.random.default_rng(8);x=rng.normal(size=(500,5));y=(x[:,0]*x[:,1]>0).astype(int)
    model=CatBoostClassifier(iterations=8,depth=3,verbose=False,allow_writing_files=False).fit(x[:350],y[:350])
    scorer=CompiledCatBoost(model,5,tmp_path)
    reference=model.predict_proba(x[350:])[:,1]
    actual=scorer(checked_input(x[350:],5))
    assert np.allclose(reference,actual,rtol=0,atol=1e-12)
    assert np.array_equal(reference>=.5,actual>=.5)

def test_runner_writes_parity_and_timing_without_exporting_rows(tmp_path):
    rng=np.random.default_rng(9);n=24000
    frame=pd.DataFrame(rng.normal(size=(n,len(FEATURES))),columns=FEATURES)
    frame['Time']=np.arange(n)
    frame['Class']=(rng.random(n)<.04).astype(int)
    csv=tmp_path/'synthetic.csv';frame.to_csv(csv,index=False)
    output=tmp_path/'out'
    timing,parity=run(csv,output,trees=2,single_calls=3,rounds=1,training_limit=1000)
    assert len(parity)==7
    assert next(p for p in parity if p['path']=='CatBoost_cpp_guarded')['accepted']
    assert {row['path'] for row in timing}=={p['path'] for p in parity if p['accepted']}
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['training_limit']==1000
    assert manifest['hypotheses']
    quality=pd.read_csv(output/'quality.csv')
    assert (quality[quality.alpha==.0001].status=='unsupported').all()
    assert 'Amount' not in pd.read_csv(output/'timing.csv').columns
