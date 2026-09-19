"""Rerun only timed-out candidates with 5s per phase; preserve main evidence."""
import argparse,json,multiprocessing,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import run

def retry(job):
    task,source,site,feature,train,queries=job
    original=run.bounded
    run.bounded=lambda fn,budget:original(fn,5)
    result=run.assess(source,site,feature,train,queries,[run.features(e['input']) for e in train],list(map(run.features,queries)))
    return task,result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corpus',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    corpus=json.loads(Path(args.corpus).read_text());out=Path(args.out);pred=json.loads((out/'predictions.json').read_text());jobs=[]
    for task,r in pred.items():
        data=corpus[task]['data']
        for c in r['candidates']:
            if c['fit_error']=='cpu_timeout' or c.get('query_error')=='cpu_timeout':jobs.append((task,corpus[task]['source'],c['site'],c['feature'],data['train'],[e['input'] for e in data['test']]))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(retry,jobs))
    # Scoring starts only after every new candidate prediction is fixed.
    scored=[]
    for task,c in rows:
        c['test_correct']=c['fit'] and c.get('query_error') is None and c['predictions']==[e['output'] for e in corpus[task]['data']['test']]
        scored.append(dict(task=task,candidate=c))
    result=dict(budget_seconds=5,workers=12,seconds=time.perf_counter()-start,source_sha256=run.sha(Path(__file__).read_text()),prediction_sha256=run.sha((out/'predictions.json').read_text()),candidates=scored)
    (out/'sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(retried=len(scored),fits=sum(x['candidate']['fit'] for x in scored),correct=sum(x['candidate']['test_correct'] for x in scored),still_timeout=sum(x['candidate']['fit_error']=='cpu_timeout' for x in scored),seconds=result['seconds'])))
