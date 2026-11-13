"""Read-only discovery and deterministic cohorts for three synthetic fraud sources."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd

BAF_NUM=['income','name_email_similarity','prev_address_months_count','current_address_months_count',
 'customer_age','days_since_request','intended_balcon_amount','zip_count_4w','velocity_6h',
 'velocity_24h','velocity_4w','bank_branch_count_8w','date_of_birth_distinct_emails_4w',
 'credit_risk_score','email_is_free','phone_home_valid','phone_mobile_valid','bank_months_count',
 'has_other_cards','proposed_credit_limit','foreign_request','session_length_in_minutes',
 'keep_alive_session','device_distinct_emails_8w']
SPECS={
 'baf_base':dict(ref='sgpjesus/bank-account-fraud-dataset-neurips-2022',files=['Base.csv'],
  label='fraud_bool',time='month',numeric=BAF_NUM,
  categorical=['payment_type','employment_status','housing_status','source','device_os'],group=None,
  kind='synthetic, generated from anonymized real application data',unit='bank-account application row',
  licence='CC-BY-NC-SA-4.0',source='https://github.com/Feedzai/bank-account-fraud',
  excluded_reason='month is used only for partitioning; device_fraud_count is excluded as a fraud-history field'),
 'paysim':dict(ref='ealaxi/paysim1',files=['PS_20174392719_1491204439457_log.csv'],
  label='isFraud',time='step',numeric=['amount','oldbalanceOrg','oldbalanceDest'],
  categorical=['type'],group='nameOrig',kind='synthetic mobile-money transactions',unit='transaction row',
  licence='CC-BY-SA-4.0',source='https://github.com/EdgarLopezPhD/PaySim',
  excluded_reason='newbalanceOrig/newbalanceDest are post-event balances; isFlaggedFraud is another rule decision; names are not predictors'),
 'sparkov':dict(ref='kartik2112/fraud-detection',files=['fraudTrain.csv','fraudTest.csv'],
  label='is_fraud',time='trans_date_trans_time',numeric=['amt','city_pop'],
  categorical=['category'],group='cc_num',kind='synthetic card transactions generated with Sparkov',unit='card-transaction row',
  licence='CC0-1.0',source='https://github.com/namebrandon/Sparkov_Data_Generation',
  excluded_reason='IDs, names, addresses, coordinates, birth dates and merchant identities are excluded; card ID is used only for overlap auditing')}
PARTS=['train','validation','calibration','test']

def feature_names(key):
    spec=SPECS[key]
    return spec['numeric']+(['hour_of_day','day_of_week'] if key=='sparkov' else [])+spec['categorical']

def read_columns(key):
    spec=SPECS[key]
    return list(dict.fromkeys([spec['label'],spec['time'],*spec['numeric'],*spec['categorical'],
                              *([spec['group']] if spec['group'] else [])]))

def read_chunks(key,paths,chunksize=100000):
    spec=SPECS[key];offset=0
    for file_number,path in enumerate(paths):
        dtype={column:'str' for column in [*spec['categorical'],*([spec['group']] if spec['group'] else [])]}
        for frame in pd.read_csv(path,usecols=read_columns(key),dtype=dtype,chunksize=chunksize):
            if not set(frame[spec['label']].dropna().unique()).issubset({0,1}) or frame[spec['label']].isna().any():
                raise ValueError('Source labels must be present and binary')
            if key=='sparkov':
                stamp=pd.to_datetime(frame[spec['time']],utc=True,errors='raise').dt.as_unit('ns')
                if stamp.isna().any():raise ValueError('Source time must be present')
                frame['__time']=stamp.astype('int64')
                frame['hour_of_day']=stamp.dt.hour.astype(float)
                frame['day_of_week']=stamp.dt.day_of_week.astype(float)
            else:
                frame['__time']=pd.to_numeric(frame[spec['time']],errors='raise')
            if not np.isfinite(frame['__time'].to_numpy(dtype=float)).all():
                raise ValueError('Source time must be finite')
            for column in spec['numeric']:
                frame[column]=pd.to_numeric(frame[column],errors='raise').astype(float)
                if np.isinf(frame[column].to_numpy()).any():raise ValueError('Infinite predictor values are unsupported')
            for column in spec['categorical']:
                frame[column]=frame[column].astype(object).where(frame[column].notna(),np.nan)
            frame['__row']=np.arange(offset,offset+len(frame),dtype=np.int64)
            offset+=len(frame)
            frame['__file']=file_number
            yield frame

def file_sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()

def priorities(row_ids,seed):
    # Stable row-only priorities allow label-blind sampling independent of CSV chunk size.
    with np.errstate(over='ignore'):
        value=np.asarray(row_ids,dtype=np.uint64)+np.uint64(seed)
        value=(value^(value>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        value=(value^(value>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        return value^(value>>np.uint64(31))

def partition_ids(key,frame,edges):
    if key=='sparkov':
        ids=np.searchsorted(edges,frame['__time'].to_numpy(),side='right')
        ids[frame['__file'].to_numpy()==1]=3
        return ids
    return np.searchsorted(edges,frame['__time'].to_numpy(),side='right')

def load_cohort(key,paths,train_cap=150000,eval_cap=100000,seed=20261112,chunksize=100000):
    if key not in SPECS:raise ValueError('Unknown dataset')
    if train_cap<1 or eval_cap<1:raise ValueError('Cohort caps must be positive')
    spec=SPECS[key];paths=list(map(Path,paths))
    if len(paths)!=len(spec['files']):raise ValueError('Expected all source files in the declared order')
    times=[];files=[dict(name=path.name,sha256=file_sha256(path),rows=0,positive=0,
                       time_min=None,time_max=None) for path in paths]
    for chunk in read_chunks(key,paths,chunksize):
        index=int(chunk['__file'].iloc[0]);info=files[index]
        info['rows']+=len(chunk);info['positive']+=int(chunk[spec['label']].sum())
        low,high=int(chunk.__time.min()),int(chunk.__time.max())
        info['time_min']=low if info['time_min'] is None else min(info['time_min'],low)
        info['time_max']=high if info['time_max'] is None else max(info['time_max'],high)
        if key!='sparkov' or index==0:times.append(np.unique(chunk['__time']))
    unique=np.unique(np.concatenate(times))
    fractions=(.70,.85) if key=='sparkov' else (.55,.70,.85)
    if len(unique)<8:raise ValueError('Insufficient distinct time groups for this protocol')
    edges=np.array([unique[int(len(unique)*q)] for q in fractions])
    if key=='sparkov' and files[0]['time_max']>=files[1]['time_min']:
        raise ValueError('Official card training and test time windows overlap')
    selected=[None]*4
    totals=[dict(rows=0,positive=0) for _ in PARTS]
    missing={column:0 for column in feature_names(key)}
    for chunk in read_chunks(key,paths,chunksize):
        chunk['__priority']=priorities(chunk['__row'],seed)
        ids=partition_ids(key,chunk,edges)
        for column in missing:missing[column]+=int(chunk[column].isna().sum())
        for index,name in enumerate(PARTS):
            part=chunk.loc[ids==index]
            totals[index]['rows']+=len(part);totals[index]['positive']+=int(part[spec['label']].sum())
            if not len(part):continue
            capacity=train_cap if index==0 else eval_cap
            combined=part if selected[index] is None else pd.concat([selected[index],part],ignore_index=True)
            selected[index]=combined.nsmallest(capacity,['__priority','__row']).copy()
    reports={};frames=[]
    for index,name in enumerate(PARTS):
        frame=selected[index]
        if frame is None or set(frame[spec['label']].unique())!={0,1}:
            raise ValueError(f'Both classes are required in sampled partition {name}')
        frame=frame.sort_values('__row').reset_index(drop=True);frames.append(frame)
        reports[name]=dict(source_rows=totals[index]['rows'],source_positive=totals[index]['positive'],
            sampled_rows=len(frame),sampled_positive=int(frame[spec['label']].sum()),
            sampled_negative=int((frame[spec['label']]==0).sum()),
            time_min=int(frame.__time.min()),time_max=int(frame.__time.max()),
            cohort_sha256=hashlib.sha256(frame['__row'].to_numpy(dtype=np.int64).tobytes()).hexdigest(),
            equivalent_predictor_label_rows=int(frame.duplicated(subset=[spec['label'],*feature_names(key)]).sum()))
    for first,second in zip(frames,frames[1:]):
        if first.__time.max()>=second.__time.min():raise ValueError('Sampled temporal partitions overlap')
    entity=dict(status='unavailable; source has no applicant identifier')
    if spec['group']:
        train_groups=set(frames[0][spec['group']].dropna().astype(str))
        test_groups=set(frames[3][spec['group']].dropna().astype(str))
        entity=dict(scope='sampled rows only',train_groups=len(train_groups),test_groups=len(test_groups),
                    overlapping_groups=len(train_groups&test_groups),ids_used_as_predictors=False)
    report=dict(dataset=key,dataset_ref=spec['ref'],dataset_kind=spec['kind'],unit=spec['unit'],
        source_url=spec['source'],licence=spec['licence'],observed_utc=datetime.now(timezone.utc).isoformat(),
        files=files,source_rows=sum(info['rows'] for info in files),partitions=reports,
        time_edges=edges.tolist(),time_unit='nanoseconds since Unix epoch' if key=='sparkov' else spec['time'],
        raw_read_column_allowlist=read_columns(key),predictor_allowlist=feature_names(key),
        excluded_reason=spec['excluded_reason'],source_missing_predictor_values=missing,entity_overlap=entity,
        scope='Full-source metadata with capped, deterministic, label-blind temporal cohorts; no row export',
        sampling=dict(seed=seed,train_cap=train_cap,eval_cap=eval_cap,method='lowest stable hash priorities within each time partition'),
        duplicate_policy='Equal model-input/label rows are counted but retained; equality does not prove a duplicate transaction',
        hypotheses=['All four sampled time partitions contain both classes',
                    'No time group crosses partition boundaries',
                    'Identifiers, labels and specified post-event fields are absent from predictors',
                    'Identical source files and sampling seed reproduce the cohort hashes'])
    return frames,report

def find_sources(key,root):
    root=Path(root);result=[]
    for name in SPECS[key]['files']:
        matches=list(root.rglob(name))
        if len(matches)!=1:raise ValueError(f'Expected exactly one source file named {name}')
        result.append(matches[0])
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=SPECS);parser.add_argument('root')
    parser.add_argument('output');parser.add_argument('--train-cap',type=int,default=150000)
    parser.add_argument('--eval-cap',type=int,default=100000)
    args=parser.parse_args();_,report=load_cohort(args.dataset,find_sources(args.dataset,args.root),args.train_cap,args.eval_cap)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
