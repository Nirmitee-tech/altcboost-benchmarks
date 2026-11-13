"""Write a per-source aggregate benchmark review."""
import json
from pathlib import Path
import pandas as pd

def write_multi_report(source,destination):
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    discovery=json.loads((source/'discovery.json').read_text())
    rows=pd.read_csv(source/'results.csv')
    lines=[f"# {manifest['dataset']} benchmark",'',
        'These results measure frozen model configurations on a capped temporal cohort. No per-dataset tuning was performed.',
        f"Source type: {manifest['dataset_kind']}. Counting unit: {manifest['unit']}.",
        f"Source: {manifest['source_url']}. Licence: {manifest['licence']}.",'',
        '| Model | Calibration FPR target | Fraud caught | False positives | Recall | Observed FPR |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows.to_dict('records'):
        if row['status']=='evaluated':
            lines.append(f"| {row['method']} | {row['alpha']:.2%} | {int(row['tp'])} | {int(row['fp'])} | {row['recall']:.2%} | {row['fpr']:.3%} |")
        else:lines.append(f"| {row['method']} | {row['alpha']:.2%} | Unsupported calibration sample | | | |")
    test=discovery['partitions']['test']
    lines.extend(['',f"The test cohort contains {test['sampled_positive']} labelled fraud rows and {test['sampled_negative']} normal rows. Calibration targets are not guarantees for a later time window.",
        '', 'Average precision and partial ROC area are included in results.csv to compare score rankings independently of the calibrated cutoff. A higher recall accompanied by more false positives is a tradeoff, not evidence of an overall improvement.',
        '',manifest['limitations'],'','## For whoever builds this','',
        f"Evidence observed at {discovery['observed_utc']}. Full source rows: {discovery['source_rows']}. Sampling: {json.dumps(discovery['sampling'])}.",
        'Source hashes, cohort hashes, missing values, time boundaries and entity overlap are in discovery.json. Counts and confidence intervals are in results.csv. Model settings and fit durations are in models.json.',
        '', 'Decision still needed: the project owner must choose the acceptable fraud recall and false-positive tradeoff before selecting a deployment candidate.'])
    target=destination/'review.md';target.write_text('\n'.join(lines)+'\n');return target
