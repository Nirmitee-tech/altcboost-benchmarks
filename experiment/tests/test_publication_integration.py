import functools
import http.server
import threading
import urllib.request
from pathlib import Path
import pandas as pd
from publish_benchmark import build
from test_accuracy_audit_unit import fixture

def test_published_page_serves_visible_counts_and_downloadable_data(tmp_path):
    (tmp_path/'docs').mkdir();(tmp_path/'accuracy-novelty-audit').mkdir();(tmp_path/'validation-study').mkdir()
    rows=fixture();rows.to_csv(tmp_path/'accuracy-novelty-audit/candidate-and-reference.csv',index=False)
    historical=[]
    for dataset in ['elliptic','paysim','sparkov','baf_base']:
        for method in ['V6','XGBoost','CatBoost_native','LightGBM_native']:
            r=rows.iloc[0].to_dict();r.update(dataset=dataset,method=method,seed=17);historical.append(r)
    pd.DataFrame(historical).to_csv(tmp_path/'validation-study/benchmark-results.csv',index=False)
    page=build(tmp_path);assert page.exists()
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(tmp_path/'docs'))
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try:
        base='http://127.0.0.1:'+str(server.server_port)
        text=urllib.request.urlopen(base+'/').read().decode()
        assert 'ALTCBoost benchmarks | Nirmitee' in text and '3 / 100' in text
        csv=urllib.request.urlopen(base+'/data/v8-validation.csv').read().decode()
        assert 'tp' in csv and 'synthetic' in csv
    finally:server.shutdown();server.server_close();worker.join()
