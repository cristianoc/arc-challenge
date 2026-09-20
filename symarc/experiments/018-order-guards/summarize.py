"""Describe saved primary predictions. No selector, feature or query is changed."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import argparse
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r018',HERE/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--problems',type=Path,required=True)
    a=p.parse_args();read=lambda p:json.loads(p.read_text())
    rows=read(a.run/'predictions.json');scores=read(a.run/'scores.json');problems=read(a.problems)
    out={'summary':read(a.run/'summary.json'),'paired':{},'diagnostics':{},'oracle':{},'cases':{},'controlled':[]}
    for sp in ('training','evaluation'):
        cohort=[(r,s) for r,s in zip(rows,scores,strict=True) if r['split']==sp and not r['development']]
        out['paired'][sp]={}
        for arm in m.POLICIES:
            out['paired'][sp][arm]={
                'gains':[s['id'] for r,s in cohort if s['policies'][arm]['correct'] and not s['policies']['exact']['correct']],
                'losses':[s['id'] for r,s in cohort if not s['policies'][arm]['correct'] and s['policies']['exact']['correct']],
                'complete_errors':[s['id'] for r,s in cohort if s['policies'][arm]['complete'] and not s['policies'][arm]['correct']]}
        out['oracle'][sp]={arm:sum(any(d['oracle_correct'] for n,d in s['families'].items() if arm=='union' or n==arm) for r,s in cohort) for arm in m.POLICIES}
        out['diagnostics'][sp]={}
        for arm in m.ARMS:
            statuses=Counter(r['families'][arm]['diagnostics']['status'] for r,s in cohort if r['eligible'])
            abstention=Counter()
            for r,s in cohort:
                for d in r['policies'][arm].get('diagnosis',[]):abstention.update(d)
            out['diagnostics'][sp][arm]={'training_status':dict(statuses),'query_cells':dict(abstention)}
    for tid in ('ce9e57f2','2281f1f4','29c11459'):
        r=next(r for r in rows if r['id']==tid);prob=next(p for p in problems if p['id']==tid)
        learner=m.Learner(prob['train']);idx=tuple(range(len(prob['train'])));models={}
        for arm in ('exact','union'):
            sel=r['policies'][arm]['selected'];model=next(x for x in r['families'][sel[0]]['models'] if x['features']==sel[1])
            table=learner.table(idx,tuple(sel[1]),sel[0])
            models[arm]={'selected':sel,'names':model['names'],'cv_exact':model['cv_exact'],'cv_fraction':model['cv_fraction'],
                'outer_pass':all(s['exact'] for f in r['policies'][arm]['outer'] for s in f['scores']),
                'action_table':[{'key':list(k),'actions':[m.s17.ACTIONS[i] for i in range(19) if mask&(1<<i)]} for k,mask in sorted(table.items())]}
        out['cases'][tid]={'models':models,'train_shapes':[(len(p['input']),len(p['input'][0])) for p in prob['train']],
                            'query_shapes':[(len(g),len(g[0])) for g in prob['query_inputs']]}
    for rec in read(a.run/'controlled.json'):
        r=read(a.run/(rec['id']+'-predictions.json'));modes=[]
        for fs in ([4,24],[24,27,28]):
            model=next(x for x in r['families']['guarded_order']['models'] if x['features']==fs)
            scored=[m.b.metrics(g,t['output'],t['input']) for g,t in zip(model['predictions'],rec['tests'],strict=True)]
            failures=[]
            for j,(g,t) in enumerate(zip(model['predictions'],rec['tests'],strict=True)):
                if g is not None:
                    for ri,(pr,yr) in enumerate(zip(g,t['output'])):
                        for ci,(v,y) in enumerate(zip(pr,yr)):
                            if v>=0 and v!=y:failures.append({'query':j,'cell':[ri,ci],'predicted':v,'expected':y})
            modes.append({'features':fs,'names':model['names'],'cv_exact':model['cv_exact'],
                'cv_fraction':model['cv_fraction'],'correct_grids':sum(s['exact'] for s in scored),
                'complete_grids':sum(s['complete'] for s in scored),'first_wrong_cell':failures[0] if failures else None})
        l=m.Learner(rec['train']);tables={}
        for fs in ((4,24),(24,27,28)):
            tab=l.table((0,),fs,'guarded_order')
            tables[str(fs)]=[{'key':list(k),'actions':[m.s17.ACTIONS[i] for i in range(19) if mask&(1<<i)]} for k,mask in sorted(tab.items())]
        out['controlled'].append({'id':rec['id'],'selected_policies':rec['policies'],'contrasting_models':modes,'one_demonstration_action_tables':tables})
    out['hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.run.iterdir() if p.name in
        ('predictions.json','summary.json','scores.json','audit.json','controlled.json','prediction-run.json','reproduction.json')}
    out['protocol_commit']='8edf63429f580a91e8d25a8606e2232f974dde88'
    out['active_registration_commit']='f50726ed341a28dd2326fb70185a28fdec239053'
    out['data_revision']='f3283f727488ad98fe575ea6a5ac981e4a188e49'
    m.b.write_json(a.run/'compact-results.json',out)
    print(json.dumps({'oracle':out['oracle'],'diagnostics':out['diagnostics'],'cases':out['cases']},indent=2))

if __name__=='__main__':main()
