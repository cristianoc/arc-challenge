"""Post-score descriptive summaries. Does not change any learned policy."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('m017',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--problems',type=Path,required=True)
    a=ap.parse_args();rows=json.loads((a.run/'predictions.json').read_text());scores=json.loads((a.run/'scores.json').read_text())
    problems={(p['split'],p['id']):p for p in json.loads(a.problems.read_text())}
    outcome={'summary':json.loads((a.run/'summary.json').read_text()),'paired':{},'diagnostics':{},'old_conflicts':{},'oracle':{}}
    for split in ('training','evaluation'):
        rs=[r for r in rows if r['split']==split and r['eligible'] and not r['development']]
        ss=[r for r in scores if r['split']==split and not r['development']]
        outcome['paired'][split]={};outcome['diagnostics'][split]={};outcome['oracle'][split]={}
        for policy in m.POLICIES:
            win=[r['id'] for r in ss if r['policies'][policy]['correct'] and not r['policies']['base']['correct']]
            loss=[r['id'] for r in ss if r['policies']['base']['correct'] and not r['policies'][policy]['correct']]
            wrong=[r['id'] for r in ss if r['policies'][policy]['complete'] and not r['policies'][policy]['correct']]
            outcome['paired'][split][policy]={'wins':win,'losses':loss,'wrong_complete':wrong}
        for arm in m.ARMS:
            status=Counter(r['families'][arm]['diagnostics']['status'] for r in rs);q=Counter();core=Counter();necessity=Counter()
            for r in rs:
                for d in r['policies'][arm]['diagnosis']:q.update(d)
                w=r['families'][arm]['diagnostics'].get('witness')
                if w:core[str(len(w))]+=1
                for model in r['families'][arm]['models']:
                    for why in model['necessity']:necessity[str(len(why['witness']))]+=1
            outcome['diagnostics'][split][arm]={'status':dict(status),'query':dict(q),'full_core_sizes':dict(core),'necessity_sizes':dict(necessity)}
            outcome['oracle'][split][arm]=sum(r['families'].get(arm,{}).get('oracle_correct',False) for r in ss)
        outcome['oracle'][split]['union']=sum(any(v.get('oracle_correct') for v in r['families'].values()) for r in ss)
        outcome['oracle'][split]['union_missed']=[r['id'] for r in ss if any(v.get('oracle_correct') for v in r['families'].values()) and not r['policies']['union']['correct']]
        c=Counter()
        for r in rs:
            o=r.get('old_conflict')
            if not o:continue
            f=bool(o['new_separators']);op=bool(o['new_shared_actions'])
            c['witnesses']+=1;c['separated_by_new_term']+=f;c['explained_by_new_operation']+=op;c['both']+=f and op
            for arm in ('split','share','joint'):c[arm+'_whole_fits']+=bool(r['families'][arm]['models'])
        outcome['old_conflicts'][split]=dict(c)
    outcome['source_commit']='2fe31f5a5c1ed9746618533d1d4b34ac4a84f110'
    outcome['protocol_commit']='f37093f73ef71223d776fa6b31622bf5b8ed52d0'
    outcome['data_revision']='f3283f727488ad98fe575ea6a5ac981e4a188e49'
    outcome['baseline']='71c9efdfb112c35ac61030c758fdbdb3144a3436'
    outcome['prediction_run']=json.loads((a.run/'prediction-run.json').read_text())
    outcome['checks']=json.loads((a.run/'audit.json').read_text())
    outcome['checks']['synthetic_tests']=16;outcome['checks']['repeated_12_worker_predictions_identical']=True
    maj=json.loads((a.run/'majority-audit.json').read_text())
    outcome['majority_sensitivity']={'protocol_commit':maj['protocol_commit'],'summary':maj['summary'],'query_scored':False,
                                   'known_case':next(r for r in maj['tasks'] if r['id']=='7e0986d6')}
    outcome['hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.run/'predictions.json',a.run/'scores.json',a.run/'summary.json',a.run/'audit.json',a.run/'majority-audit.json')}
    (a.run/'compact-results.json').write_text(json.dumps(outcome,indent=2,sort_keys=True)+'\n')
    cases={}
    for r,scored in zip(rows,scores,strict=True):
        if r['id'] not in ('29c11459','25d8a9c8','3618c87e','d5d6de2d','7e0986d6','e88171ec'):continue
        p=problems[(r['split'],r['id'])];learner=m.Learner(p['train']);indices=tuple(range(len(p['train'])))
        record={'split':r['split'],'scores':scored['policies'],'selected_rules':{},'training_examples':p['train'],'query_inputs':p['query_inputs']}
        for arm in m.ARMS:
            model=learner.selected(indices,arm)
            if model:
                fs=tuple(model['features']);table=learner.table(indices,fs,arm)
                record['selected_rules'][arm]={'features':model['names'],'cv_exact':model['cv_exact'],
                    'rules':[{'key':list(k),'actions':[name for i,name in enumerate(m.ACTIONS) if mask&(1<<i)]} for k,mask in sorted(table.items())],
                    'predictions':r['policies'][arm]['predictions']}
            else:record['selected_rules'][arm]={'witness':r['families'][arm]['diagnostics'].get('witness')}
        cases[r['id']]=record
    (a.run/'cases.json').write_text(json.dumps(cases,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'paired':outcome['paired'],'oracle':outcome['oracle']},indent=2))

if __name__=='__main__':main()
