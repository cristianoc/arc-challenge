"""Independent post-run selection and explicit-set operation audit for 023.

Reuses the frozen input-feature/action evaluator. Does not call learner
fit/apply/compress or 023 selection/scoring. It is not an independent search
completeness proof: candidate-pool replication is checked separately.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference018',HERE.parent/'018-order-guards/run.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
POLICIES=('baseline','tie_complete','complete_first','tie_consensus')


def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def whole(ps):
    return len(ps)>0 and all(g and g[0] and all(len(r)==len(g[0]) for r in g)
        and all(type(v)==int and v in range(10) for r in g for v in r) for g in ps)
def ident(m):return [m['arm'],m['features']]
def ordered_score(m):return (-m['cv_exact'],-Fraction(m['cv_fraction'][0],m['cv_fraction'][1]),
    len(m['features']),m['cost'],('exact','order','guarded_exact','guarded_order').index(m['arm']),tuple(m['features']))

def choices(pool,count):
    if not pool:return {p:([],[None]*count) for p in POLICIES}
    sorted_pool=sorted(pool,key=ordered_score);first=sorted_pool[0]
    evidence=ordered_score(first)[:2]
    tied=[m for m in sorted_pool if ordered_score(m)[:2]==evidence and whole(m['predictions'])]
    complete=[m for m in sorted_pool if whole(m['predictions'])]
    t=tied[0] if tied else first; a=complete[0] if complete else first
    consensus=(ident(t),t['predictions'])
    if tied and any(m['predictions']!=t['predictions'] for m in tied):consensus=([],[None]*count)
    return {'baseline':(ident(first),first['predictions']),'tie_complete':(ident(t),t['predictions']),
            'complete_first':(ident(a),a['predictions']),'tie_consensus':consensus}


class Checker:
    def __init__(self,problem,mode):
        self.problem=problem; self.train=problem['train']; self.mode=mode;self.counts=Counter()
        self.scenes={};self.tables={};self.cache={};self.cvs={};groups=defaultdict(list)
        for i,p in enumerate(self.train):groups[self.signature(p['input'])].append(i)
        self.groups=list(groups.values())
    @staticmethod
    def signature(g):return tuple(tuple(r) for r in g)
    def scene(self,g):
        key=self.signature(g)
        if key not in self.scenes:self.scenes[key]=e.scene(g)
        return self.scenes[key]
    def table(self,indices,fs):
        key=(tuple(indices),tuple(fs))
        if key in self.tables:return self.tables[key]
        table={}
        for i in indices:
            grid=self.train[i]['input'];target=self.train[i]['output']
            for (fv,av),y in zip(self.scene(grid),(y for r in target for y in r),strict=True):
                ctx=tuple(fv[f] for f in fs);allowed={a for a,v in enumerate(av) if v==y}
                table[ctx]=table.get(ctx,set(range(19))) & allowed
        self.counts['tables']+=1;self.counts['context_action_sets']+=len(table)
        if any(not v for v in table.values()):
            self.tables[key]=None;return None
        if self.mode=='default' and table:
            costs={a:len([v for v in table.values() if a not in v]) for a in range(19)}
            defaults=[a for a in costs if costs[a]==min(costs.values())]
            retained={}
            for ctx,allowed in table.items():
                kept=set()
                for d in defaults:kept.update({d} if d in allowed else allowed)
                assert kept and kept<=allowed
                retained[ctx]=kept
            table=retained
        self.tables[key]=table;return table
    def predict(self,indices,fs,grid):
        key=(tuple(indices),tuple(fs),self.signature(grid))
        if key in self.cache:return self.cache[key]
        table=self.table(indices,fs)
        if table is None:self.cache[key]=None;return None
        values=[]
        for fv,av in self.scene(grid):
            ctx=tuple(fv[f] for f in fs);possible={av[a] for a in table.get(ctx,set())}
            values.append(next(iter(possible)) if len(possible)==1 and next(iter(possible))>=0 else -1)
        w=len(grid[0]);p=[values[i:i+w] for i in range(0,len(values),w)]
        self.cache[key]=p;self.counts['independent_grid_evaluations']+=1;return p
    def cv(self,indices,fs):
        key=(tuple(indices),tuple(fs))
        if key in self.cvs:return self.cvs[key]
        exact=0;frac=Fraction(0)
        for group in self.groups:
            if group[0] not in indices:continue
            remaining=[i for i in indices if i not in group]
            gs=[self.predict(remaining,fs,self.train[i]['input']) for i in group]
            exact+=all(p==self.train[i]['output'] for p,i in zip(gs,group,strict=True))
            p,y=gs[0],self.train[group[0]]['output']
            if p is not None:
                frac+=Fraction(sum(v==y[r][c] for r,row in enumerate(p) for c,v in enumerate(row)),sum(len(r) for r in y))
        self.cvs[key]=(exact,frac);return exact,frac
    def pool(self,pool,indices,inputs):
        for m in pool:
            fs=m['features'];score=self.cv(indices,fs)
            assert score==(m['cv_exact'],Fraction(*m['cv_fraction'])),(self.problem['id'],fs,'inner score')
            assert m['cost']==sum(e.TERMS[f][1] for f in fs)
            assert m['predictions']==[self.predict(indices,fs,g) for g in inputs],(self.problem['id'],fs,'prediction')
            self.counts['candidate_models']+=1
    def decisions(self,pool,actual,inputs):
        expected=choices(pool,len(inputs))
        for name,(selected,pred) in expected.items():
            assert actual[name]['selected']==selected and actual[name]['predictions']==pred,(self.problem['id'],name,'selection')
            self.counts['policy_choices']+=1
        baseline=actual['baseline']['predictions']
        if whole(baseline):
            for p in ('tie_complete','complete_first'):
                assert actual[p]['predictions']==baseline
                self.counts['complete_preservation_checks']+=1


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--problems',type=Path,required=True);ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--scores',type=Path,required=True);ap.add_argument('--answers',type=Path,required=True)
    ap.add_argument('--stable',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--limit',type=int)
    a=ap.parse_args();start=time.perf_counter()
    problems={(p['split'],p['id']):p for p in read(a.problems)};rows=read(a.predictions)
    rows=rows[a.start:] if a.limit is None else rows[a.start:a.start+a.limit]
    scores={(s['split'],s['id']):s for s in read(a.scores)};answers=read(a.answers)
    stable={s['split']+'/'+s['id']:s for s in map(json.loads,a.stable.read_text().splitlines())} if a.stable else {}
    counts=Counter();ids=[]
    for row in rows:
        problem=problems[(row['split'],row['id'])];c=Checker(problem,row['mode']);ids.append([row['split'],row['id']])
        if row['eligible']:
            indices=list(range(len(problem['train'])))
            c.pool(row['models'],indices,problem['query_inputs']);c.decisions(row['models'],row['policies'],problem['query_inputs'])
            assert [f['excluded'] for f in row['outer']]==c.groups
            for fold in row['outer']:
                group=fold['excluded'];remain=[i for i in indices if i not in group];inputs=[problem['train'][i]['input'] for i in group]
                c.pool(fold['models'],remain,inputs);c.decisions(fold['models'],fold['policies'],inputs)
                for policy in POLICIES:
                    for p,i,s in zip(fold['policies'][policy]['predictions'],group,fold['scores'][policy],strict=True):
                        y=problem['train'][i]['output']
                        assert (p==y)==s['exact'] and bool(whole([p]))==s['complete']
                        c.counts['outer_grid_scores']+=1
            for i,record in enumerate(row['per_query']):
                pool=[{**m,'predictions':[m['predictions'][i]]} for m in row['models']]
                c.decisions(pool,record,[problem['query_inputs'][i]])
        else:c.decisions([],row['policies'],problem['query_inputs'])
        key=row['split']+'/'+row['id'];ys=answers[key] if key in answers else answers[row['id']]
        saved=scores[(row['split'],row['id'])]
        for policy in POLICIES:
            ps=row['policies'][policy]['predictions'];s=saved['policies'][policy]
            assert bool(whole(ps))==s['complete'] and (ps==ys)==s['correct']
            assert (bool(whole(ps)) and ps!=ys)==s['wrong_complete']
            assert sum(p==y for p,y in zip(ps,ys,strict=True))==s['correct_grids']
            assert s['outer_pass']==(len(row['outer'])>=2 and all(z['exact'] for f in row['outer'] for z in f['scores'][policy]))
            if stable:
                old=stable[key];take=(not old['fitted']) and bool(whole(ps));hybrid=ps if take else old['predictions']
                assert take==s['hybrid']['use_relational'] and (hybrid==ys)==s['hybrid']['correct']
                assert sum(p==y for p,y in zip(hybrid,ys,strict=True))==s['hybrid']['correct_grids']
                c.counts['hybrid_task_scores']+=1
            c.counts['task_scores']+=1
        counts.update(c.counts)
    write(a.out,{'counts':dict(counts),'ids':ids,'start':a.start,'tasks':len(rows),'seconds':time.perf_counter()-start,
        'predictions_sha256':sha(a.predictions),'source_sha256':sha(Path(__file__)),
        'boundary':'Reuses feature/action evaluator; independently checks action sets, default projections, internal evidence, query-aware choices and scores. Does not independently search feature subsets.'})
    print(json.dumps({'tasks':len(rows),'seconds':time.perf_counter()-start,'counts':dict(counts)}),flush=True)

if __name__=='__main__':main()
