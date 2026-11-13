"""Report a mechanism experiment without relabelling validation as confirmation."""
import json
from pathlib import Path
import pandas as pd

STATIC=['V6_fixed','V7_zero','V7_clipped','V7_shrunk']

def strict_rows(rows):
    selected=rows[(rows.alpha-.001).abs()<1e-12]
    if selected.method.duplicated().any():raise ValueError('Duplicate validation method records')
    return selected

def best_record(rows,names):
    selected=rows[rows.method.isin(names)]
    if selected.empty:return None
    records=selected.to_dict('records')
    return max(records,key=lambda row:(row['validation_mean_recall'],-row['validation_mean_fpr'],row['ap']))

def write_v7_report(source,destination):
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    rows=pd.read_csv(source/'validation.csv');manifest=json.loads((source/'manifest.json').read_text())
    lines=[f"# Prior-revision experiment: {manifest['dataset']}",'',
        'This is a training/validation mechanism experiment. The same validation labels select model prefixes and compare the hypotheses. These scores are development evidence and do not establish independent confirmation or novelty.','',
        '| Method | Mean recall at two strict validation budgets | Caught at 0.1% | False positives at 0.1% | Selected prior weight | Selected prefix |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows[(rows.alpha-.001).abs()<1e-12].to_dict('records'):
        weight=f"{row['selected_prior_weight']:.4f}" if pd.notna(row['selected_prior_weight']) else 'N/A'
        lines.append(f"| {row['method']} | {row['validation_mean_recall']:.2%} | {int(row['tp'])} | {int(row['fp'])} | {weight} | {int(row['selected_prefix'])} |")
    lines.extend(['','The primary mechanism question is whether the coupled update improves over zero, clipping, fixed shrinkage and diagonal-block controls. A simpler control matching or beating it weakens the claim for the extra mechanism. The 0.05% and 1% validation results remain in validation.csv.','',
        '## For whoever builds this','',manifest['scope'],manifest['limitations'],'',
        'The prior weight is constrained to [0,1]. Every accepted update is checked against the actual empirical training loss. Exact directional curvature concerns a two-coordinate restriction away from tail membership ties; the tree structure still uses a diagonal Newton approximation. Neither this check nor a validation gain proves future low-FPR performance.',
        '', 'Decision still needed: the research lead must assess whether the coupled update earns its complexity over the controls and the closest published methods. A revised method needs a new confirmation protocol before a publication claim.'])
    target=destination/'review.md';target.write_text('\n'.join(lines)+'\n');return target

def write_v7_summary(root,keys=('elliptic','paysim','sparkov','baf_base')):
    root=Path(root);all_rows=[];lines=['# Prior-revision research results','',
        'These are adaptive training/validation experiments. Validation both selects model prefixes and compares the hypotheses. No calibration or final predictions were computed. Existing sources informed this redesign, so these results are not independent confirmation.','',
        'The table reports mean recall across empirical validation false-positive budgets of 0.05% and 0.1%. The full three-component validation key breaks ties by false-positive rate and average precision.','',
        '| Source | Fixed V6 | Best static control | Diagonal adaptive update | Coupled adaptive update | Highest tested reference |',
        '| --- | ---: | --- | ---: | ---: | --- |']
    findings=[]
    for key in keys:
        study=root/key;state=json.loads((study/'kaggle-status.json').read_text())
        if state['state']!='COMPLETE':raise ValueError(f'{key} has not completed')
        paths=list((study/'kaggle-output').rglob('research_results/validation.csv'))
        if len(paths)!=1:raise ValueError('Expected one research result file')
        rows=pd.read_csv(paths[0]);all_rows.append(rows);primary=strict_rows(rows)
        if not primary.dataset.eq(key).all():raise ValueError('Source identity mismatch')
        indexed=primary.set_index('method');static=best_record(primary,STATIC)
        ref=best_record(primary,['CatBoost_native','XGBoost','LightGBM_native'])
        def score(method):return float(indexed.loc[method,'validation_mean_recall'])
        reference=f"{ref['validation_mean_recall']:.2%} ({ref['method']})" if ref else 'Not included'
        lines.append(f"| {key} | {score('V6_fixed'):.2%} | {static['validation_mean_recall']:.2%} ({static['method']}) | {score('V7_adaptive_diagonal'):.2%} | {score('V7_adaptive_joint'):.2%} | {reference} |")
        joint=score('V7_adaptive_joint');diagonal=score('V7_adaptive_diagonal');simple=static['validation_mean_recall']
        findings.append(f"- {key}: coupled-update mean recall differs by {(joint-simple)*100:+.2f} percentage points from the strongest static control and {(joint-diagonal)*100:+.2f} percentage points from the diagonal adaptive control. These are validation differences, not significance claims.")
    lines.extend(['','The static controls include zero start, a smooth bounded prior and fixed shrinkage. A positive result against fixed V6 alone cannot establish that the extra optimizer is needed.','',*findings,'',
        '## For whoever builds this','',
        'The plan is v7-study/prespecified-plan.json. Individual hosted reports include the validation counts, selected prior coefficients and prefixes. Raw row data and future prediction scores are not exported. The preceding Elliptic final-test failure remains unchanged.','',
        'Related-work overlap includes fully corrective partial-AUC ensembles and variable-projection boosting. Exact small-block curvature, coefficient correction and empirical descent are not individually claimed as new inventions. A publishable novelty argument requires direct comparison with the closest algorithms and fresh confirmation after model selection.','',
        'Decision still needed: the research lead must decide whether the coupled update earns its complexity over the controls. Jitendra must choose whether to continue this mechanism or withdraw it based on the complete evidence.'])
    target=root/'results-review.md';target.write_text('\n'.join(lines)+'\n')
    pd.concat(all_rows,ignore_index=True).to_csv(root/'validation-results.csv',index=False)
    return target
