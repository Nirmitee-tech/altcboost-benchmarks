"""Report exploratory detection results without claiming an unseen holdout."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

def compare_counts(tp,fp,reference_tp,reference_fp):
    if tp>reference_tp and fp<reference_fp:return 'more fraud and fewer false alarms'
    if tp>=reference_tp and fp<=reference_fp and (tp>reference_tp or fp<reference_fp):
        return 'improves one measure without worsening the other'
    if tp==reference_tp and fp==reference_fp:return 'unchanged'
    if tp<=reference_tp and fp>=reference_fp:return 'worse or equal on both measures'
    return 'tradeoff; not a joint improvement'

def audit_reading(audit):
    if audit is None:return 'The historical XGBoost audit is not present in this output.'
    changed=any(row['legacy_tp']!=row['strict']['tp'] or row['legacy_fp']!=row['strict']['fp'] for row in audit['rows'])
    return ('The historical XGBoost audit changed some test counts; use the strict results in its evidence file.' if changed else
            'The historical XGBoost audit reproduced the earlier test TP/FP counts at all checked targets. The dtype fix corrects tied calibration decisions and prevents the degenerate all-alert case.')

def write_detection_report(source,output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    results=pd.read_csv(source/'results.csv')
    manifest=json.loads((source/'manifest.json').read_text())
    baseline=json.loads((source/'incumbent.json').read_text()) if (source/'incumbent.json').exists() else None
    audit=json.loads((source/'historical-xgboost-audit.json').read_text()) if (source/'historical-xgboost-audit.json').exists() else None
    audit_note=audit_reading(audit)
    champion=manifest.get('chosen_before_final_scoring')
    selection_note=(f'The validation-selected champion was {champion}, frozen before calibration and final scoring.' if champion else
                    'Individual-family selections were frozen before calibration. A global champion was not recorded in this local run.')
    table=['| Model | Calibration target | Fraud caught | False alarms |',
           '|---|---:|---:|---:|']
    comparisons=[]
    for row in results[results.status=='exploratory'].itertuples():
        table.append(f'| {row.method} | {row.alpha:.2%} | {int(row.tp)} | {int(row.fp)} |')
        if baseline and abs(row.alpha-.001)<1e-12:
            comparisons.append(dict(method=row.method,tp=int(row.tp),fp=int(row.fp),
                reading=compare_counts(row.tp,row.fp,baseline['tp'],baseline['fp'])))
    summary=dict(source=str(source.resolve()),observed_utc=datetime.now(timezone.utc).isoformat(),
        scope=manifest['scope'],unit=manifest['unit'],independent_validation_complete=False,
        baseline=baseline,comparisons=comparisons,results=results.to_dict(orient='records'),historical_xgboost_audit=audit,
        champion_recorded_before_final_scoring=champion)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    observations='\n'.join(f"- {row['method']}: {row['reading']} compared with the historical incumbent at the same calibration target."
                           for row in comparisons)
    report=f'''# Detection improvement experiment

Treat the selected models as development candidates. The objective is to catch more fraud while reducing false alarms. This experiment does not provide independent confirmation because the existing final interval has already informed earlier design choices.

The search compares a regularized Newton-leaf branch, the previous algorithm, and two established tree libraries. Each individual model family receives four prespecified configurations and the same maximum tree count. Separately labelled blends test whether combining models helps. A shared low-false-positive validation rule selects configurations before separate threshold calibration.

{chr(10).join(table)}

{observations}

A reduction in alerts accompanied by fewer fraud cases caught is a tradeoff. It is not counted as a joint improvement. Likewise, extra fraud caught with extra false alarms is not counted as a joint improvement. The data provide limited positive cases, and the threshold target is not a guarantee of the future false-positive rate.

## For whoever builds this

Skip this section unless you are changing or running the code.

Source directory: `{source.resolve()}`. Evidence files are `trials.csv`, `selected.json`, `results.csv`, `discovery.json` and `manifest.json`, with `champion.json` when present. The model-selection rule averages validation recall at empirical FPR 0.0005 and 0.001, then breaks ties using fewer validation false positives and average precision. {selection_note} Calibration uses a separate time interval and one-sided order-statistic bounds under representative-independent-negative assumptions. All comparisons explicitly convert scores to float64 so a nextafter threshold cannot round back onto tied float32 scores.

The V4 branch recomputes regularized Newton leaf values from summed gradients and curvature after tree-structure selection. The leaf step is bounded after aggregation. Four variants test clipping at six standard deviations versus no clipping, and a linear versus constant starting model. V4 does not contain XGBoost, CatBoost, pairwise cross features or the V3 adaptive negative-weight heuristic. Its previous compiled-scoring speed has not been established for this branch. The ensemble is explicitly a hybrid of V4 and CatBoost, not a standalone ALTC model.

The short smoke run's apparent perfect-recall XGBoost result was withdrawn: a float32 threshold-comparison bug could flag all tied-score rows. A regression test exercises that case. {audit_note} The historical CatBoost and ALTC comparisons used float64 scores and are not affected by that specific dtype mechanism.

Equal configuration counts and tree caps do not establish equal wall-clock tuning budgets. The custom learners use early stopping while the library learners use their full tree budget. The four-configuration design is a bounded development experiment, not an exhaustive model comparison. Repeatedly reusing the ULB interval means improvements must remain labelled exploratory.

Decisions still needed: Jitendra must select the operational false-positive budget and alert capacity. The Kaggle account owner must enable IEEE-CIS download access in account `jitendra93266`; download still returned 403 after the access-step confirmation. An independent data owner must supply a fresh final evaluation window before a performance or deployment claim.
'''
    (output/'review.md').write_text(report)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');args=p.parse_args()
    print(json.dumps(write_detection_report(args.source,args.output),indent=2))
