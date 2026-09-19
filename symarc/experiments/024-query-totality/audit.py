"""Independent explicit-set audit of query-total programs, projections and choice.

Reuses only 018 input feature/action extraction. Does not call 024 condition,
023 decide, or learner fit/apply/compress. Search/internal evidence is frozen
and compared against archived 023 pools, not independently resynthesized here.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

spec = importlib.util.spec_from_file_location('extract018', Path(__file__).resolve().parents[1]/'018-order-guards/run.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
OPS = set(range(19))
POLICIES = ('baseline', 'tie_complete', 'total_ranked', 'total_tie_complete')


def read(p):
    with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f: return json.load(f)
def bits(s): return sum(2**a for a in s)
def full(ps): return bool(ps) and all(g is not None and g and g[0] and all(len(r)==len(g[0]) for r in g) and all(type(v)==int and 0<=v<=9 for r in g for v in r) for g in ps)
def rank(m): return (-m['cv_exact'], -Fraction(*m['cv_fraction']), len(m['features']), m['cost'], ('exact','order','guarded_exact','guarded_order').index(m['arm']), tuple(m['features']))
def ident(m): return [m['arm'], m['features']]


def set_condition(raw, domains, mode):
    unknown = set(domains)-set(raw)
    ds = []
    if mode == 'default' and raw:
        costs = {a: sum(a not in options for options in raw.values()) for a in OPS}
        ds = [a for a in sorted(OPS) if costs[a]==min(costs.values())]
        branches = [(d, {k: {d} if d in options else set(options) for k, options in raw.items()}) for d in ds]
    else:
        branches = [(None, raw)]
    prior = {k: set().union(*(b[k] for d,b in branches)) for k in raw}
    filtered = []
    if not unknown:
        for d, branch in branches:
            mapping = {k: values & domains.get(k, OPS) for k,values in branch.items()}
            if all(mapping[k] for k in domains): filtered.append((d,mapping))
    kept = {k: set().union(*(b[k] for d,b in filtered)) for k in raw}
    naive = {k: v & domains.get(k, OPS) for k,v in prior.items()}
    return dict(feasible=bool(filtered), unknown=[list(z) for z in sorted(unknown)], defaults=ds,
        surviving_defaults=[d for d,b in filtered if d is not None],
        naive_feasible=not unknown and all(naive[z] for z in domains),
        entries=[[list(z),bits(raw[z]),bits(prior[z]),bits(domains.get(z,OPS)),bits(kept[z]),bits(naive[z])] for z in sorted(raw)]), prior, kept


def render(grids, scenes, features, table):
    result=[]
    for g,sc in zip(grids,scenes,strict=True):
        flat=[]
        for f,values in sc:
            possible={values[a] for a in table.get(tuple(f[i] for i in features),set())}
            flat.append(next(iter(possible)) if len(possible)==1 and min(possible)>=0 else -1)
        w=len(g[0]);result.append([flat[i:i+w] for i in range(0,len(flat),w)])
    return result


def choose(models, count):
    def select(pool, tied, total=False):
        if not pool: return [],[None]*count
        ordered=sorted(pool,key=rank);winner=ordered[0]
        if tied:
            candidates=[m for m in ordered if rank(m)[:2]==rank(winner)[:2] and full(m['total_predictions'] if total else m['predictions'])]
            if candidates: winner=candidates[0]
        return ident(winner), winner['total_predictions'] if total else winner['predictions']
    feasible=[m for m in models if m['feasible']]
    return {'baseline':select(models,False),'tie_complete':select(models,True),
            'total_ranked':select(feasible,False,True),'total_tie_complete':select(feasible,True,True)}


def metrics(pred,y,x):
    complete=full([pred]);h,w=len(y),len(y[0]);aligned=pred is not None and len(pred)==h and all(len(r)==w for r in pred)
    same=len(x)==h and all(len(r)==w for r in x)
    known=correct=ck=cc=0
    if aligned:
        for r in range(h):
            for c in range(w):
                if pred[r][c]>=0:
                    known+=1;correct+=pred[r][c]==y[r][c]
                    if same and x[r][c]!=y[r][c]: ck+=1;cc+=pred[r][c]==y[r][c]
    changed=sum(x[r][c]!=y[r][c] for r in range(h) for c in range(w)) if same else None
    return dict(complete=complete,exact=pred==y,cells=h*w,known=known,correct=correct,wrong=known-correct,
                changed=changed,changed_known=ck,changed_correct=cc,shape_changed=not same)


class Checker:
    def __init__(self,problem,mode):
        self.train=problem['train']; self.mode=mode
        self.scenes=[e.scene(p['input']) for p in self.train]
        self.counts=Counter()
        self.cache={}

    def pool(self,pool,old,indices,grids,scenes):
        assert len(pool['models'])==len(old)
        for actual,previous in zip(pool['models'],old,strict=True):
            assert {k:actual[k] for k in previous}==previous
            self.counts['unchanged_candidate_records']+=1
        for table in pool['tables']:
            fs=tuple(table['features']);raw={}
            for i in indices:
                labels=[v for r in self.train[i]['output'] for v in r]
                for (f,values),label in zip(self.scenes[i],labels,strict=True):
                    key=tuple(f[a] for a in fs); options={a for a,v in enumerate(values) if v==label}
                    raw[key]=raw.get(key,OPS)&options
            assert all(raw.values())
            domains={}
            for sc in scenes:
                for f,values in sc:
                    key=tuple(f[a] for a in fs);valid={a for a,v in enumerate(values) if v>=0}
                    domains[key]=domains.get(key,OPS)&valid
            calculated,prior,kept=set_condition(raw,domains,self.mode)
            assert calculated=={k:table[k] for k in calculated}
            ordinary=render(grids,scenes,fs,prior)
            total=render(grids,scenes,fs,kept) if calculated['feasible'] else [None]*len(grids)
            assert table['predictions']==ordinary and table['total_predictions']==total
            self.counts['tables']+=1;self.counts['context_projections']+=len(raw)
            self.counts['default_branches_checked']+=len(calculated['defaults'])
            self.counts['naive_false_feasibility']+=calculated['naive_feasible'] and not calculated['feasible']
            self.counts['coupled_projection_differences']+=calculated['feasible'] and any(x[4]!=x[5] for x in calculated['entries'])
        for model in pool['models']:
            t=pool['tables'][model['table_id']]
            assert t['features']==model['features']
            assert model['predictions']==t['predictions'] and model['total_predictions']==t['total_predictions']
            assert model['feasible']==t['feasible']
            self.counts['candidate_grid_predictions']+=2*len(grids)
        expected=choose(pool['models'],len(grids))
        for policy,(which,preds) in expected.items():
            got=pool['policies'][policy]
            assert got['selected']==which and got['predictions']==preds
            self.counts['policy_choices']+=1
        return expected


def main():
    p=argparse.ArgumentParser(__doc__)
    for arg in ('problems','predictions','baseline','answers','out'):p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--stable',type=Path);p.add_argument('--start',type=int,default=0);p.add_argument('--limit',type=int)
    a=p.parse_args();start=time.perf_counter()
    jobs={(p['split'],p['id']):p for p in read(a.problems)}
    previous={(r['split'],r['id']):r for r in read(a.baseline)}
    rows=read(a.predictions);rows=rows[a.start:] if a.limit is None else rows[a.start:a.start+a.limit]
    scores={(s['split'],s['id']):s for s in read(a.predictions.with_name('scores.json'))}
    answers=read(a.answers)
    stable={(s['split'],s['id']):s for s in map(json.loads,a.stable.read_text().splitlines())} if a.stable else None
    counts=Counter()
    for row in rows:
        key=(row['split'],row['id']);problem=jobs[key];old=previous[key];c=Checker(problem,row['mode'])
        indices=tuple(range(len(problem['train'])));queries=problem['query_inputs'];qsc=[e.scene(g) for g in queries]
        actual=c.pool(row,old['models'],indices,queries,qsc)
        for ref in ('baseline','tie_complete'):
            assert row['policies'][ref]['predictions']==old['policies'][ref]['predictions']
            assert row['policies'][ref]['selected']==old['policies'][ref]['selected']
        assert len(row['outer'])==len(old['outer'])
        for f,of in zip(row['outer'],old['outer'],strict=True):
            assert f['excluded']==of['excluded'];ex=f['excluded'];rest=tuple(i for i in indices if i not in ex)
            xs=[problem['train'][i]['input'] for i in ex]
            choices=c.pool(f,of['models'],rest,xs,[c.scenes[i] for i in ex])
            for pol,(which,ps) in choices.items():
                for pred,i,stored in zip(ps,ex,f['scores'][pol],strict=True):
                    expected=metrics(pred,problem['train'][i]['output'],problem['train'][i]['input'])
                    assert stored==expected;c.counts['outer_grid_scores']+=1
            for pol in ('baseline','tie_complete'):
                assert f['policies'][pol]['predictions']==of['policies'][pol]['predictions']
                assert f['policies'][pol]['selected']==of['policies'][pol]['selected']
        targets=answers.get('/'.join(key),answers.get(row['id']))
        for pol,(which,ps) in actual.items():
            s=scores[key]['policies'][pol]
            assert s['selected']==which and s['correct']==(ps==targets) and s['complete']==full(ps)
            assert s['wrong_complete']==(full(ps) and ps!=targets)
            assert s['grids']==[metrics(pred,y,x) for pred,y,x in zip(ps,targets,queries,strict=True)]
            outer_ok=len(row['outer'])>=2 and all(v['exact'] for f in row['outer'] for v in f['scores'][pol])
            assert s['outer_pass']==outer_ok;c.counts['policy_task_scores']+=1
            if stable is not None:
                st=stable[key];take=not st['fitted'] and full(ps);hp=ps if take else st['predictions']
                assert s['hybrid']==dict(correct=hp==targets,use_relational=take,correct_grids=sum(x==y for x,y in zip(hp,targets,strict=True)))
                c.counts['hybrid_task_scores']+=1
        counts.update(c.counts)
    result={'counts':dict(counts),'tasks':len(rows),'start':a.start,'seconds':time.perf_counter()-start,
            'predictions_sha256':hashlib.sha256(a.predictions.read_bytes()).hexdigest(),
            'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
