"""Fixed V6 direct-pairwise removal experiments after validation-only screening."""
import argparse
import json
from pathlib import Path
from altcboost_v6 import ALTCBoostV6
from multi_benchmark import frozen_models
from multi_data import SPECS,find_sources
from tail_experiment import VARIANTS,run as run_shared

def candidate_models(trees=100):
    for name,config in VARIANTS.items():
        parameters={k:v for k,v in config.items() if k!='group_count'}
        yield name.replace('V5_','V6_'),dict(config,trees=trees,leaves=17,temperature=1.,group_temperature=.25),ALTCBoostV6(n_estimators=trees,**parameters)
    for name,config,model in frozen_models():
        if name in {'ALTCBoost_tuned','CatBoost','XGBoost'}:yield name,config,model

def run(key,paths,output,train_cap=150000,eval_cap=100000,models=None,trees=100):
    result=run_shared(key,paths,output,train_cap,eval_cap,
        models=candidate_models(trees) if models is None else models,trees=trees)
    path=Path(output)/'manifest.json';manifest=json.loads(path.read_text())
    manifest['candidate']='V6 exact mean logistic pair loss over the selected negative tail, with temporal group loss aggregation'
    manifest['scope']='Revised candidate after V5 development failure and a PaySim validation-only screen; all final intervals remain consumed development evidence'
    manifest['limitations']=manifest['limitations'].replace('Only V5 checks','Only the candidate family checks')
    manifest['arithmetic']='Chunked pair evaluation bounds memory; arithmetic still scales with positive-negative pair counts'
    path.write_text(json.dumps(manifest,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SPECS);p.add_argument('root');p.add_argument('output');args=p.parse_args()
    run(args.dataset,find_sources(args.dataset,args.root),args.output)
