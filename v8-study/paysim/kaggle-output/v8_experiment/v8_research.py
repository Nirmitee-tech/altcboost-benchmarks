"""Controlled tail-continuation research; no calibration or final scoring."""
import argparse
import json
from pathlib import Path
from altcboost_v8 import ALTCBoostV8
from validation_data import SPECS,sources
from v7_research import run as shared_run

MODES=['target','all_negative','geometric','linear','adaptive_blind','adaptive_checked']

def candidates(budget=40):
    return [('V8_'+mode,ALTCBoostV8(continuation=mode,n_estimators=budget)) for mode in MODES]

def run(key,paths,output,references=None,trees=40,train_cap=150000,eval_cap=100000,expected_cohorts=None,expected_files=None):
    result=shared_run(key,paths,output,references,trees,train_cap,eval_cap,expected_cohorts,expected_files,candidate_factory=candidates,reference_policy='budget_prefix')
    path=Path(output)/'manifest.json';manifest=json.loads(path.read_text())
    manifest.update(candidate='Prior-free adaptive tail continuation with a tie-aware one-sided directional test',
        modes=['V8_'+mode for mode in MODES],weak_learner_proposal_budget=trees,
        reference_policy='Previously selected baseline hyperparameters, refitted with the same tree cap; validation reselects a prefix within that cap. Earlier hyperparameter tuning gives baselines additional search effort.',
        descent_scope='Each accepted step decreases its current working-tail objective. The fixed target-tail loss may increase when training uses a relaxed fraction.',
        limitations='Adaptive validation-only development on consumed sources. No independent confirmation or established novelty. Shared budget counts all fitted tree proposals, including rejected ones. Numerical directional signs are not interval-arithmetic certificates. Nonnegative exact directional derivative rules out descent only along the proposed tree, not all possible trees. Different working fractions define different objectives; no global target-loss monotonicity or future FPR guarantee. Loss evaluation counts are implementation-specific and vary in cost with tail fraction.')
    path.write_text(json.dumps(manifest,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SPECS);p.add_argument('root');p.add_argument('output');args=p.parse_args()
    run(args.dataset,sources(args.dataset,args.root),args.output)
