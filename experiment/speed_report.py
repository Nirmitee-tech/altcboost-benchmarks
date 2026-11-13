"""Create a source-labelled report from scorer-comparison artifacts."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

def write_speed_report(source,output):
    source,output=Path(source),Path(output)
    timing=pd.read_csv(source/'timing.csv')
    parity=json.loads((source/'parity.json').read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    accepted={p['path'] for p in parity if p['accepted']}
    if set(timing.path)!=accepted:
        raise ValueError('Timed paths differ from paths accepted by parity checks')
    one=timing[timing.batch_size==1]
    values={row.path:float(row.p50_call_us) for row in one.itertuples()}
    ratio=None
    if 'ALTCBoost_numba' in values and 'CatBoost_cpp_guarded' in values:
        ratio=values['CatBoost_cpp_guarded']/values['ALTCBoost_numba']
    table=['| Scorer | Median single-call time | p99 single-call time |',
           '|---|---:|---:|']
    for row in one.itertuples():
        table.append(f'| {row.path} | {row.p50_call_us:.2f} microseconds | {row.p99_call_us:.2f} microseconds |')
    rejected=[p['path'] for p in parity if not p['accepted']]
    summary=dict(source=str(source.resolve()),observed_utc=datetime.now(timezone.utc).isoformat(),
        scope=manifest['scope'],unit=manifest['unit'],timing=timing.to_dict(orient='records'),
        rejected_paths=rejected,parity=parity,single_call_speedup_vs_guarded_cpp=ratio)
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    text=f'''# Scorer comparison

Use this experiment to assess the earlier speed claim before choosing an inference approach. Detection settings were fixed. This does not establish a better fraud classifier.

{f'ALTCBoost was {ratio:.2f} times faster per single-row call than the guarded compiled CatBoost path in this experiment. This supports further work on the fast CPU scorer. The next quality experiment needs matched tuning effort and a fresh final holdout.' if ratio is not None else 'The available paths do not support a direct guarded-CatBoost speed ratio.'}

The measurements include a common input boundary and model scoring. They exclude feature retrieval, network calls, queues, concurrency and process startup. Alternative scorers were checked against each model's reference probabilities and threshold decisions before timing.

{chr(10).join(table)}

Rejected paths: {', '.join(rejected) if rejected else 'none'}. Rejection means the output-equivalence gate failed; no speed claim is made for those paths. Batch measurements are recorded separately in the evidence files.

The benchmark repeats calls on one machine. It does not represent all deployment environments. The prior holdout was reused for equivalence checks, so these results provide no new independent detection evidence. Exported C++ and native prediction are two concrete alternative paths; they do not exhaust every possible optimization of the baseline libraries.

## For whoever builds this

Skip this section unless you are running or changing the code.

Source directory: `{source.resolve()}`. Files: `timing.csv`, `quality.csv`, `parity.json`, `manifest.json`, and `discovery.json`. The fixed selection is seed 17, depth 5 for XGBoost and CatBoost, and `use_cross=False` for ALTCBoost V3, with a 70-tree limit. The reference model keeps its original early-stopping rule. Prediction comparisons use a 1e-6 probability tolerance and require identical flags at all supported reference-calibrated thresholds on calibration and test rows.

The common boundary converts to contiguous float64 and checks finite values and feature count. This checks input form, not correctness of upstream feature values. ALTCBoost includes its own scaling and clipping. The C++ CatBoost bridge converts features to float32 internally, matching its exported model interface. A guarded variant recomputes rows within 1e-12 of the three calibration cutoffs through the reference CatBoost API. That guard is included in timing, and its observed fallback count is recorded in the manifest. Numerical equivalence is checked on observed rows; it does not establish equivalence on every possible future input. Setup and compilation are excluded from warm timings and recorded separately. No database, HTTP or concurrent-worker benchmark was performed.

Regenerate using `python speed_report.py SOURCE_DIRECTORY REPORT_DIRECTORY`. Timings are repeated scoring-call samples, not new fraud cases. The p99 values are descriptive estimates from those samples.

Decisions still needed: Jitendra must choose whether single-row latency or batch throughput matters for the target workflow. An implementation owner must measure the complete deployed feature and inference path before making a service-level performance claim. Detection redesign requires a fresh holdout.
'''
    (output/'review.md').write_text(text)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');args=p.parse_args()
    print(json.dumps(write_speed_report(args.source,args.output),indent=2))
