"""Read-only, aggregate discovery for the public ULB Kaggle dataset."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

FEATURES = [f'V{i}' for i in range(1, 29)] + ['Amount']

def load_data(path):
    frame = pd.read_csv(path, usecols=['Time', 'Class'] + FEATURES)
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Missing or nonfinite values require an explicit preprocessing plan')
    if set(frame.Class.unique()) != {0, 1}:
        raise ValueError('Expected both binary labels')
    # Identical records are excluded once before partitioning. No rows are exported.
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().sort_values('Time', kind='stable').reset_index(drop=True)
    unique_times = np.sort(frame.Time.unique())
    edges = [unique_times[int(len(unique_times) * q)] for q in (.55, .70, .85)]
    groups = np.searchsorted(edges, frame.Time.to_numpy(), side='right')
    parts = [np.flatnonzero(groups == i) for i in range(4)]
    if any(set(frame.iloc[idx].Class.unique()) != {0, 1} for idx in parts):
        raise ValueError('Every temporal partition needs both classes')
    report = dict(source='https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
                  observed_utc=datetime.now(timezone.utc).isoformat(),
                  sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  scope='Complete CSV; identical complete rows removed; time boundaries keep equal timestamps together',
                  unit='transaction row; identities unavailable', duplicates_removed=duplicate_count,
                  rows_after_dedup=len(frame), features=FEATURES,
                  splits={name: dict(rows=len(idx), positive=int(frame.iloc[idx].Class.sum()),
                                    negative=int((frame.iloc[idx].Class == 0).sum()),
                                    time_min=float(frame.iloc[idx].Time.min()),
                                    time_max=float(frame.iloc[idx].Time.max()))
                          for name, idx in zip(['train', 'validation', 'calibration', 'test'], parts)})
    return frame, parts, report

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('csv'); p.add_argument('output')
    args = p.parse_args()
    _, _, report = load_data(args.csv)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
