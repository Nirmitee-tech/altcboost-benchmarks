"""Build the public benchmark page from saved aggregate measurements."""
import argparse
from datetime import datetime,timezone
from html import escape
from pathlib import Path
import pandas as pd
from accuracy_audit import audit_counts

REPOSITORY='https://github.com/Nirmitee-tech/altcboost-benchmarks'
DATASETS={'elliptic':'Elliptic (known-label AML)','paysim':'PaySim (synthetic)','sparkov':'Sparkov (synthetic)','baf_base':'BAF Base (synthetic)'}
METHODS={'V8_adaptive_checked':'V8 adaptive candidate','V6':'V6 temporal tail','CatBoost_native':'CatBoost','LightGBM_native':'LightGBM'}

def table(headers,rows):
    head=''.join('<th scope="col">'+escape(str(value))+'</th>' for value in headers)
    body=''.join('<tr>'+''.join('<td>'+escape(str(value))+'</td>' for value in row)+'</tr>' for row in rows)
    return '<div class="scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'

def render(validation,final,observed):
    validation=audit_counts(validation)
    latest=[]
    for r in validation.to_dict('records'):
        latest.append([DATASETS.get(r['dataset'],r['dataset']),METHODS.get(r['method'],r['method']),
            f"{int(r['tp'])} / {int(r['fraud_cases'])}",f"{r['recall_recomputed']:.2%}",f"{r['precision_recomputed']:.2%}",
            f"{int(r['fp'])} / {int(r['legitimate_cases'])}",f"{r['accuracy']:.3%}"])
    historical=[]
    for r in final.to_dict('records'):
        historical.append([DATASETS.get(r['dataset'],r['dataset']),METHODS.get(r['method'],r['method']),
            f"{int(r['tp'])} / {int(r['tp']+r['fn'])}",f"{r['recall']:.2%}",int(r['fp']),f"{r['fpr']:.3%}"])
    latest_table=table(['Source','Model','Fraud caught / fraud cases','Recall','Precision','False alerts / legitimate cases','Overall accuracy'],latest)
    final_table=table(['Source','Model','Positive cases caught / positive cases','Recall','False positives','Observed FPR'],historical)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Reproducible ALTCBoost fraud-detection research benchmarks, with measured counts, baselines and limitations.">
<title>ALTCBoost benchmarks | Nirmitee</title>
<style>body{{margin:0;background:#f7f8fa;color:#18252c;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:40px 24px}}a{{color:#075e9a}}h1{{font-size:clamp(2rem,5vw,3rem);line-height:1.15;margin:8px 0 16px}}h2{{margin-top:40px;line-height:1.25}}.brand{{letter-spacing:.1em;text-transform:uppercase;font-size:.8rem;font-weight:700}}.scope{{border-left:4px solid #a34c20;background:#fff3e8;padding:16px 20px}}.scroll{{overflow-x:auto;background:white;border:1px solid #d6dce1;border-radius:6px}}table{{border-collapse:collapse;width:100%;font-size:.86rem}}th,td{{padding:10px 12px;border-bottom:1px solid #e2e7eb;text-align:left;vertical-align:top}}th{{background:#edf1f4}}tbody tr:nth-child(even){{background:#fafbfd}}footer{{margin-top:40px;padding-top:20px;border-top:1px solid #ccd5dc;font-size:.9rem}}code{{background:#e8edf1;padding:2px 4px}}li{{margin:8px 0}}</style></head>
<body><main><div class="brand">Nirmitee · Research benchmarks</div><h1>ALTCBoost</h1>
<p>Measured fraud detection at low false-positive budgets, with the code and aggregate results available for review.</p>
<p><a href="{REPOSITORY}">Source code</a> · <a href="{REPOSITORY}/blob/main/docs/novelty-accuracy-audit.md">Novelty and accuracy audit</a> · <a href="data/v8-validation.csv">Validation CSV</a> · <a href="data/v6-final-results.csv">Earlier final-test CSV</a></p>
<div class="scope"><strong>Research prototype. Novelty and general superiority remain unproven.</strong>
V8 has not beaten the strongest simple training policy on any of the four development sources. Its tie check reduced objective evaluations on BAF from 296 to 236 with matching validation metrics. That is one observed implementation result, with no general speedup claim.</div>
<h2>V8: validation development results</h2>
<p>At an empirical validation false-positive budget of 0.1%. Recall is the share of fraud caught; precision is the share of alerts that are fraud. The native reference was selected by the original mean-recall rule across 0.05% and 0.1% budgets. The table below uses 0.1% alone.</p>
{latest_table}
<p><strong>These are consumed validation cohorts.</strong> Validation selects prefixes and thresholds and informed the redesign. No V8 final-test result or independent confirmation is reported. One seed and 40 tree proposals per policy were used. References reuse previously tuned hyperparameters with a 40-tree cap; prior search effort, tree capacity and wall time are not matched.</p>
<p>BAF demonstrates the limit of overall accuracy: the candidate has 98.808% accuracy while catching only 35 of 1,129 fraud cases. Labelling every case legitimate would have 98.871% accuracy and catch no fraud.</p>
<h2>V6: earlier final-test benchmarks</h2>
<p>Seed 17, with thresholds selected on an earlier calibration interval targeting 0.1% false positives. The observed later FPR can exceed the target. These final intervals have since been consumed and informed later research. This table concerns V6, a different model and protocol from V8.</p>
{final_table}
<p>The V6 Elliptic failure remains part of the record. Elliptic is a known-label Bitcoin AML transfer task, with unknown labels excluded. It is not independent payment-fraud validation. The other three sources are synthetic.</p>
<h2>Reproduce and assess</h2><ul>
<li>Run the unit and integration suites using the commands in the repository README.</li>
<li>Recompute the validation audit from saved confusion counts. Source and cohort hashes are included alongside the aggregate data.</li>
<li>Download datasets separately from their providers and follow their terms. Raw datasets, identifiers, per-row predictions and credentials are not distributed here.</li>
<li>Freeze any new candidate and its calibration thresholds before testing on fresh confirmation data. Existing validation intervals and repeated seeds do not establish independent superiority.</li></ul>
<p><a href="{REPOSITORY}/blob/main/DATASETS.md">Dataset sources and licenses</a> · <a href="{REPOSITORY}/blob/main/docs/tail-continuation-research.md">Method and derivation</a> · <a href="{REPOSITORY}/blob/main/validation-study/results-review.md">Earlier benchmark protocol</a></p>
<footer>Measurements: 12–13 November 2026. Page generated {escape(observed)} from saved aggregate CSVs. Source manifests record the counting units, cohort hashes and observation times. Statistical significance, future FPR control and production readiness are not established.</footer>
</main></body></html>'''

def build(root):
    root=Path(root);validation=pd.read_csv(root/'accuracy-novelty-audit/candidate-and-reference.csv')
    all_final=pd.read_csv(root/'validation-study/benchmark-results.csv')
    final=all_final[(all_final.seed==17)&((all_final.alpha-.001).abs()<1e-12)]
    if len(final)!=16 or final.duplicated(['dataset','method']).any():raise ValueError('Expected four primary models on four final-test sources')
    audited_final=audit_counts(final)
    destination=root/'docs';(destination/'data').mkdir(exist_ok=True)
    page=render(validation,audited_final,datetime.now(timezone.utc).isoformat())
    (destination/'index.html').write_text(page);(destination/'.nojekyll').write_text('')
    validation.to_csv(destination/'data/v8-validation.csv',index=False)
    all_final.to_csv(destination/'data/v6-final-results.csv',index=False)
    return destination/'index.html'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',nargs='?',default='.');args=parser.parse_args();print(build(args.root))
