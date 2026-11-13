"""Validation-only report for descent-driven tail continuation."""
import json
from pathlib import Path
import pandas as pd
from v7_report import strict_rows,best_record

def summarize_attempts(models):
    rows=[]
    for model in models:
        for fraction in sorted({r['fraction'] for r in model['attempt_diagnostics']}):
            attempts=[r for r in model['attempt_diagnostics'] if r['fraction']==fraction]
            rows.append(dict(method=model['method'],working_fraction=fraction,proposals=len(attempts),
                accepted=sum(r['accepted'] for r in attempts),
                non_descent_rejections=sum(r.get('reason') in {'non_descent_direction','near_stationary_direction'} for r in attempts),
                target_loss_increases=sum(r['accepted'] and r['target_loss_after']>r['target_loss_before'] for r in attempts)))
    return pd.DataFrame(rows)

def compare_mechanisms(primary,models):
    simple=best_record(primary,['V8_target','V8_all_negative','V8_geometric','V8_linear'])
    adaptive=primary.set_index('method').loc['V8_adaptive_checked']
    by_name={r['method']:r for r in models};blind=by_name['V8_adaptive_blind'];checked=by_name['V8_adaptive_checked']
    key=lambda row:(float(row['validation_mean_recall']),-float(row['validation_mean_fpr']),float(row['ap']))
    return dict(best_simple_method=simple['method'],best_simple_mean_recall=float(simple['validation_mean_recall']),
        adaptive_mean_recall=float(adaptive['validation_mean_recall']),adaptive_beats_best_simple=bool(key(adaptive)>key(simple)),
        blind_objective_calls=int(blind['objective_evaluations']),checked_objective_calls=int(checked['objective_evaluations']),
        objective_call_reduction_fraction=float(1-checked['objective_evaluations']/blind['objective_evaluations']),
        blind_fit_s=float(blind['fit_s']),checked_fit_s=float(checked['fit_s']),
        runtime_scope='One fit per policy and source; no repeated timing study or general speedup claim')

def write_v8_report(source,destination):
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    rows=strict_rows(pd.read_csv(source/'validation.csv'))
    summarize_attempts(json.loads((source/'models.json').read_text())).to_csv(destination/'mechanism-counts.csv',index=False)
    manifest=json.loads((source/'manifest.json').read_text())
    lines=[f"# Tail-continuation experiment: {manifest['dataset']}",'',
        'These are development results. Validation selects prefixes and compares methods on sources already used in development. They do not establish independent confirmation or novelty.','',
        '| Method | Mean strict recall | Caught at 0.1% | False positives at 0.1% | Selected prefix | Tree proposals | Objective calls | Fit seconds |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows.to_dict('records'):
        count=lambda field:str(int(r[field])) if pd.notna(r.get(field)) else 'N/A'
        lines.append(f"| {r['method']} | {r['validation_mean_recall']:.2%} | {int(r['tp'])} | {int(r['fp'])} | {int(r['selected_prefix'])} | {count('weak_learner_fits')} | {count('objective_evaluations')} | {r['fit_s']:.2f} |")
    lines.extend(['','Mean strict recall averages recall at empirical validation false-positive budgets of 0.05% and 0.1%. Thresholds use validation negative labels, so these counts do not measure future false-positive control.','',
        '## For whoever builds this','',manifest['descent_scope'],manifest['reference_policy'],manifest['limitations'],'',
        'Decision still needed: the research lead must determine whether continuation improves beyond fixed and clock schedules. A separate confirmation source and comparison with the closest published optimizers are required before a publication claim.'])
    target=destination/'review.md';target.write_text('\n'.join(lines)+'\n');return target

def write_v8_summary(root,keys=('elliptic','paysim','sparkov','baf_base')):
    root=Path(root);collected=[];lines=['# Tail-continuation research results','',
        'Validation both selects prefixes and compares hypotheses on consumed sources. All predictions in this study are training or validation predictions. The table is development evidence.','',
        '| Source | Strict target | All negatives | Geometric schedule | Linear schedule | Adaptive checked | Highest tested reference |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    checks=[];comparisons={}
    for key in keys:
        folder=root/key
        if json.loads((folder/'kaggle-status.json').read_text())['state']!='COMPLETE':raise ValueError('Research job incomplete')
        paths=list((folder/'kaggle-output').rglob('research_results/validation.csv'))
        if len(paths)!=1:raise ValueError('Expected one validation result')
        rows=pd.read_csv(paths[0]);collected.append(rows);primary=strict_rows(rows)
        if not primary.dataset.eq(key).all():raise ValueError('Source identity mismatch')
        indexed=primary.set_index('method');ref=best_record(primary,['CatBoost_native','XGBoost','LightGBM_native'])
        scores=[f"{indexed.loc['V8_'+mode,'validation_mean_recall']:.2%}" for mode in ['target','all_negative','geometric','linear','adaptive_checked']]
        lines.append('| '+key+' | '+' | '.join(scores)+' | '+(f"{ref['validation_mean_recall']:.2%} ({ref['method']})" if ref else 'Not included')+' |')
        models=json.loads(paths[0].with_name('models.json').read_text());by_name={r['method']:r for r in models}
        comparisons[key]=compare_mechanisms(primary,models)
        blind=by_name['V8_adaptive_blind'];checked=by_name['V8_adaptive_checked']
        same=blind['training_loss']==checked['training_loss'] and blind['selected_prefix']==checked['selected_prefix']
        checks.append(f"- {key}: blind and checked accepted target-loss histories and selected prefixes match: {same}. Objective calls: {blind['objective_evaluations']} versus {checked['objective_evaluations']}. Fit seconds: {blind['fit_s']:.2f} versus {checked['fit_s']:.2f}. Counts describe this implementation and do not establish a general speedup.")
    lines.extend(['',*checks,'','## For whoever builds this','',
        'The plan is v8-study/prespecified-plan.json. Each policy has 40 total fitted-tree proposals, including rejected proposals. Baselines use previous hyperparameters and a 40-tree cap, with prefixes reselected on validation. This does not match wall time or earlier tuning effort.',
        'Each accepted step decreases the current working-tail loss. Changing the working fraction can increase the fixed target-tail loss. The directional check rules out improvement only along a particular proposed tree, using floating-point arithmetic.','',
        'Decision still needed: the research lead must accept or withdraw the claimed benefit against the strongest simple control, and specify a fresh confirmation source before freezing a publication candidate.'])
    (root/'conclusion.json').write_text(json.dumps(dict(scope='Consumed original validation; no independent confirmation',
        comparisons=comparisons,adaptive_exceeds_best_simple_count=sum(r['adaptive_beats_best_simple'] for r in comparisons.values()),
        sources_compared=len(comparisons),novelty='Unproven; primary-source overlap review is in docs/tail-continuation-research.md'),indent=2))
    target=root/'results-review.md';target.write_text('\n'.join(lines)+'\n');pd.concat(collected).to_csv(root/'validation-results.csv',index=False);return target
