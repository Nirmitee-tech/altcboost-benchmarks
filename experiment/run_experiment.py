"""CPU benchmark with disjoint temporal model selection, calibration and test."""
import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from discovery import FEATURES, load_data
from evaluation import calibrate, metrics

def candidates(seed, trees, full=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    yield 'Logistic', 'linear', make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    if not full:
        return
    from altcboost_v2 import ALTCBoostV2
    from altcboost_v3 import ALTCBoostV3
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    # Each tree family gets two prespecified candidates and the same tree limit.
    for depth in (3, 5):
        yield 'XGBoost', f'depth{depth}', XGBClassifier(n_estimators=trees, max_depth=depth,
            learning_rate=.06, n_jobs=2, tree_method='hist', random_state=seed)
    for leaves in (15, 31):
        yield 'LightGBM', f'leaves{leaves}', LGBMClassifier(n_estimators=trees,
            num_leaves=leaves, learning_rate=.06, n_jobs=2, verbosity=-1, random_state=seed)
    for depth in (3, 5):
        yield 'CatBoost', f'depth{depth}', CatBoostClassifier(iterations=trees,
            depth=depth, learning_rate=.06, thread_count=2, verbose=False,
            allow_writing_files=False, random_seed=seed)
    common = dict(n_estimators=trees, random_state=seed, min_samples_leaf=40)
    for cross in (False, True):
        yield 'ALTCBoostV2', f'cross{cross}', ALTCBoostV2(**common, use_cross=cross)
        yield 'ALTCBoostV3_original', f'cross{cross}', ALTCBoostV3(**common, use_cross=cross)
        # A proposed ablation, not an implementation of an exact pAUC objective.
        yield 'ALTCBoostV3_aligned', f'cross{cross}', ALTCBoostV3(**common,
            use_cross=cross, hard_negative_strength=0, low_fpr_validation=.001,
            validation_ap_weight=.5)

def run(csv, output, seeds=(17, 42), trees=70, full=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frame, parts, discovery = load_data(csv)
    (output/'discovery.json').write_text(json.dumps(discovery, indent=2))
    arrays = [(frame.iloc[i][FEATURES].to_numpy(dtype=float), frame.iloc[i].Class.to_numpy()) for i in parts]
    (xt, yt), (xv, yv), (xc, yc), (xe, ye) = arrays
    rows = []; selections = []
    for seed in seeds:
        best = {}
        for family, variant, model in candidates(seed, trees, full):
            started = time.perf_counter()
            with threadpool_limits(limits=2):
                if family.startswith('ALTC'):
                    model.fit(xt, yt, xv, yv)
                else:
                    model.fit(xt, yt)
            elapsed = time.perf_counter()-started
            pv = model.predict_proba(xv)[:, 1]
            objective = .5*average_precision_score(yv, pv) + .5*roc_auc_score(yv, pv, max_fpr=.001)
            selections.append(dict(seed=seed, family=family, variant=variant,
                                   validation_objective=float(objective), fit_s=elapsed))
            print(seed, family, variant, 'validation', round(objective, 4), flush=True)
            if family not in best or objective > best[family][0]:
                best[family] = (objective, variant, model, elapsed)
        # Calibrate and touch the test set only after validation selection.
        for family, (_, variant, model, elapsed) in best.items():
            pc = model.predict_proba(xc)[:, 1]
            pe = model.predict_proba(xe)[:, 1]
            model.predict_proba(xe[:1])
            single = []
            for row in xe[:200]:
                start = time.perf_counter_ns(); model.predict_proba(row.reshape(1, -1))
                single.append((time.perf_counter_ns()-start)/1000)
            batch = []
            for _ in range(5):
                start = time.perf_counter(); model.predict_proba(xe)
                batch.append((time.perf_counter()-start)*1e6/len(xe))
            for alpha in (.0001, .001, .01):
                try:
                    cut, details = calibrate(yc, pc, alpha)
                except ValueError as e:
                    rows.append(dict(seed=seed, method=family, variant=variant, alpha=alpha,
                                     status='unsupported', reason=str(e)))
                    continue
                row = dict(seed=seed, method=family, variant=variant,
                    status='evaluated', fit_s=elapsed, threshold=cut,
                    batch_us_per_row=float(np.median(batch)), single_p50_us=float(np.median(single)),
                    single_p99_us=float(np.quantile(single, .99)), **details,
                    **metrics(ye, pe, cut, alpha))
                rows.append(row)
                print(family, alpha, 'recall', round(row['recall'], 4), 'FPR', round(row['fpr'], 6), flush=True)
        pd.DataFrame(rows).to_csv(output/'results.csv', index=False)
        pd.DataFrame(selections).to_csv(output/'validation.csv', index=False)
    packages=['numpy','scipy','scikit-learn','pandas']
    if full:
        packages+=['numba','xgboost','lightgbm','catboost']
    versions = {p: importlib.metadata.version(p) for p in packages}
    (output/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),
        platform=platform.platform(), versions=versions, seeds=list(seeds), trees=trees,
        note='Temporal split is fixed across seeds; seeds are model randomness, not independent datasets'), indent=2))
    return rows

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('csv'); p.add_argument('--output', default='results')
    p.add_argument('--trees', type=int, default=70); p.add_argument('--seeds', nargs='+', type=int, default=[17,42])
    args = p.parse_args(); run(args.csv, args.output, tuple(args.seeds), args.trees)
