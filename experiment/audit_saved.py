"""Rerunnable audit of the supplied results, without rerunning training."""
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

def audit(path):
    rows=list(csv.DictReader(Path(path).open()))
    cases=sorted({(r['dataset'],r['seed']) for r in rows})
    comparisons=[]
    for dataset,seed in cases:
        group={r['method']:r for r in rows if r['dataset']==dataset and r['seed']==seed}
        comparisons.append(dict(dataset=dataset,seed=seed,
            ap={name:float(r['ap']) for name,r in group.items()},
            fpr={name:float(r['test_fpr']) for name,r in group.items()},
            hard2_beats_v2_ap=float(group['ALTCBoostV3_hard2']['ap'])>float(group['ALTCBoostV2']['ap']),
            hard4_beats_v2_ap=float(group['ALTCBoostV3_hard4']['ap'])>float(group['ALTCBoostV2']['ap'])))
    return dict(observed_utc=datetime.now(timezone.utc).isoformat(), source=str(Path(path).resolve()),
        scope='Supplied loose CSV, all rows; historical results not independently reproduced',
        unit='dataset and seed case; repeated methods within a case',
        result_rows=len(rows), cases=len(cases), datasets=len({x[0] for x in cases}),
        comparisons=comparisons)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('output');args=p.parse_args()
    report=audit(args.csv)
    archive=Path(args.csv).resolve().parent.parent/'original/altcboost/benchmark_v3_results.csv'
    if archive.exists():
        report['archive_comparison']=audit(archive)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
