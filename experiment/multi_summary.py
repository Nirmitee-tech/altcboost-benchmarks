"""Combine completed aggregate reviews without pooling different source populations."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

def completed_rows(root,keys):
    rows=[];evidence=[]
    for key in keys:
        study=Path(root)/key
        state=json.loads((study/'kaggle-status.json').read_text())
        if state['state']!='COMPLETE':raise ValueError(f'{key} has not completed')
        output=study/'kaggle-output/multi_results'
        manifest=json.loads((output/'manifest.json').read_text())
        if manifest['dataset']!=key:raise ValueError('Dataset identity mismatch')
        discovery=json.loads((output/'discovery.json').read_text())
        result=pd.read_csv(output/'results.csv')
        if not result.dataset.eq(key).all():raise ValueError('Result dataset identity mismatch')
        rows.extend(result.to_dict('records'))
        evidence.append(dict(dataset=key,kernel=state['kernel'],manifest=manifest,discovery=discovery))
    return pd.DataFrame(rows),evidence

def write_summary(root,keys=('baf_base','paysim','sparkov')):
    root=Path(root);rows,evidence=completed_rows(root,keys)
    lines=['# Multi-dataset benchmark results','',
        'All listed runs completed on Kaggle. These are measured results on capped synthetic datasets using frozen configurations. They do not establish production performance or a universal winner.','',
        'The table shows the operating point selected on calibration data for a 0.1% false-positive target. Each cell gives fraud caught / false positives. Actual test false-positive rates can differ from that target.','',
        '| Dataset | Original ALTCBoost | Tuned ALTCBoost | CatBoost | XGBoost | Logistic |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    methods=['ALTCBoost_incumbent','ALTCBoost_tuned','CatBoost','XGBoost','Logistic']
    for key in keys:
        cells=[key]
        for method in methods:
            matches=rows[(rows.dataset==key)&(rows.method==method)&(abs(rows.alpha-.001)<1e-12)]
            if len(matches)!=1 or matches.iloc[0].status!='evaluated':cells.append('Unavailable');continue
            row=matches.iloc[0];cells.append(f'{int(row.tp)} / {int(row.fp)}')
        lines.append('| '+' | '.join(cells)+' |')
    lines.extend(['',
        'Read the individual reviews for the 0.05% and 1% targets. A gain in recall with more false positives is a tradeoff. Score ranking metrics and confidence intervals are saved with the counts. No significance test of model differences was performed.','',
        '## For whoever builds this','',
        'All models share each source cohort and its training-fitted preprocessing. This run does not assess native CatBoost categorical handling or per-dataset tuning. ALTCBoost uses validation stopping; the other methods use fixed iteration counts. The original ULB holdout has already informed development and is excluded from this new comparison. IEEE-CIS remains outside this benchmark pending competition access.',''])
    for item in evidence:
        test=item['discovery']['partitions']['test']
        lines.append(f"- {item['dataset']}: {test['sampled_positive']} fraud and {test['sampled_negative']} normal {item['manifest']['unit']}s; {item['manifest']['dataset_kind']}; {item['manifest']['source_url']}; Kaggle run {item['kernel']}.")
    lines.extend(['','Counts come from each dataset folder at kaggle-output/multi_results/results.csv. Source files, cohort hashes and observation times are in discovery.json. Individual reports are in hosted-report/review.md. The combined counts are also saved as benchmark-results.csv.','',
        'Decision still needed: the project owner must set the acceptable recall and false-positive tradeoff. Further improvements need validation data for selection and an untouched future interval for confirmation.'])
    target=root/'results-review.md';target.write_text('\n'.join(lines)+'\n')
    rows.to_csv(root/'benchmark-results.csv',index=False)
    (root/'results-evidence.json').write_text(json.dumps(dict(generated_utc=datetime.now(timezone.utc).isoformat(),sources=evidence),indent=2))
    return target

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
    print(write_summary(args.root))
