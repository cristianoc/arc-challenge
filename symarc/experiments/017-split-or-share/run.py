"""017: exact conflict-set refinement of features versus shared actions.

The directional constructor is supplied. Feature subsets and compatible action
sets are learned from demonstrations; separate prediction and scoring commands
prevent query labels from entering construction or selection. Standard library.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('refine016', HERE.parent/'016-conflict-refinement/run.py')
s16 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s16)
b = s16.b
DIRECTIONS = (('N',-1,0),('S',1,0),('W',0,-1),('E',0,1))
TERMS = s16.TERMS + [('ray_'+d+'.count',4) for d,_,_ in DIRECTIONS]
ACTIONS = tuple(['constant:'+str(c) for c in range(10)] + ['copy:C'] +
                ['copy:'+d for d,_,_ in DIRECTIONS] + ['after_ray:'+d for d,_,_ in DIRECTIONS])
BASE_MASK = (1<<11)-1
FULL_MASK = (1<<19)-1
ARMS = {'base':(tuple(range(15)),BASE_MASK),
        'split':(tuple(range(19)),BASE_MASK),
        'share':(tuple(range(15)),FULL_MASK),
        'joint':(tuple(range(19)),FULL_MASK)}
POLICIES = (*ARMS,'union')
MAX_FEATURES = 3
DEVELOPMENT = s16.DEVELOPMENT


def scene(grid):
    """All feature/action values depend only on the input, never on labels."""
    h,w = b.shape(grid); fs=s16.feature_rows(grid); rows=[]
    for j,f in enumerate(fs):
        r,c=divmod(j,w); colour=grid[r][c]; counts=[]; adjacent=[]; beyond=[]
        for _,dr,dc in DIRECTIONS:
            rr,cc=r+dr,c+dc
            adjacent.append(grid[rr][cc] if 0<=rr<h and 0<=cc<w else -1)
            n=1
            while 0<=rr<h and 0<=cc<w and grid[rr][cc]==colour:
                n+=1;rr+=dr;cc+=dc
            counts.append(n)
            beyond.append(grid[rr][cc] if 0<=rr<h and 0<=cc<w else -1)
        rows.append((tuple(f)+tuple(counts),tuple(range(10))+(colour,)+tuple(adjacent)+tuple(beyond)))
    return rows


def intersection(records, mask):
    for record in records: mask &= record[4]
    return mask


def minimal_core(records, mask):
    """Deletion-minimal, not minimum-cardinality, empty action intersection."""
    core=list(records)
    assert intersection(core,mask)==0
    i=0
    while i<len(core):
        if intersection(core[:i]+core[i+1:],mask)==0: core.pop(i)
        else: i+=1
    return core


def conflict(records, selected, mask):
    """Return a minimal inconsistent set in one feature class, not just a pair."""
    seen={}
    for r in records:
        key=tuple(r[0][i] for i in selected); allowed=r[4]&mask
        if key not in seen:
            seen[key]=(allowed,[r])
            if not allowed: return [r]
            continue
        previous,contributors=seen[key]; common=previous&allowed
        if common==previous: continue
        contributors=contributors+[r]
        if not common: return minimal_core(contributors,mask)
        seen[key]=(common,contributors)
    return None


def certificate(core):
    return [{'origin':list(r[3]),'input':r[1][10],'output':r[2],
             'features':list(r[0]),'action_values':list(r[1]),'allowed':r[4]} for r in core]


def complexity(selected):
    return (len(selected),sum(TERMS[i][1] for i in selected),tuple(selected))


def search(records,vocabulary,mask,maximum=MAX_FEATURES):
    # Only input features and action compatibility affect consistency. Preserve
    # the earliest labelled origin for a compact exact representative table.
    unique={}
    for r in records: unique.setdefault((r[0],r[4]&mask),r)
    records=list(unique.values())
    impossible=conflict(records,vocabulary,mask)
    if impossible is not None:
        return [],dict(status='vocabulary_conflict',states=1,witness=certificate(impossible))
    visited=set();solutions=set();sizes=Counter()
    def visit(selected):
        selected=tuple(sorted(selected))
        if selected in visited or any(set(s).issubset(selected) for s in solutions):return
        visited.add(selected); core=conflict(records,selected,mask)
        if core is None:
            solutions.add(selected);return
        sizes[len(core)]+=1
        if len(selected)==maximum:return
        for i in vocabulary:
            if len({r[0][i] for r in core})>1:visit((*selected,i))
    visit(())
    minimal=sorted((s for s in solutions if not any(set(t)<set(s) for t in solutions)),key=complexity)
    return minimal,dict(status='fit' if minimal else 'feature_bound',states=len(visited),
                        core_sizes={str(k):v for k,v in sorted(sizes.items())})


def fit(records,selected,mask):
    table={}
    for r in records:
        key=tuple(r[0][i] for i in selected)
        table[key]=table.get(key,mask)&r[4]
        if not table[key]:return None
    return table


def apply(grid,rows,selected,table):
    if table is None:return None
    predictions=[]
    for fs,values in rows:
        key=tuple(fs[i] for i in selected); survivors=table.get(key,0)
        possible={values[i] for i in range(len(ACTIONS)) if survivors&(1<<i)}
        # Undefined surviving actions are NOT dropped as inconvenient evidence.
        predictions.append(next(iter(possible)) if len(possible)==1 and -1 not in possible else -1)
    w=len(grid[0]);return [predictions[i:i+w] for i in range(0,len(predictions),w)]


class Learner:
    def __init__(self,pairs):
        self.pairs=pairs;self.scenes=[scene(p['input']) for p in pairs];self.records=[]
        for i,(p,sc) in enumerate(zip(pairs,self.scenes,strict=True)):
            w=len(p['input'][0]); records=[]
            for j,((fs,values),y) in enumerate(zip(sc,itertools.chain.from_iterable(p['output']),strict=True)):
                allowed=sum(1<<k for k,v in enumerate(values) if v==y)
                records.append((fs,values,y,(i,j//w,j%w),allowed))
            self.records.append(records)
        groups=defaultdict(list)
        for i,p in enumerate(pairs):groups[b.signature(p['input'])].append(i)
        self.groups=tuple(tuple(g) for g in groups.values());self.table_cache={};self.models_cache={}

    def table(self,indices,selected,arm):
        k=(indices,selected,ARMS[arm][1])
        if k not in self.table_cache:
            self.table_cache[k]=fit([r for i in indices for r in self.records[i]],selected,ARMS[arm][1])
        return self.table_cache[k]

    def groups_inside(self,indices):return [g for g in self.groups if g[0] in indices]

    def fixed_cv(self,indices,selected,arm):
        folds=[]
        for excluded in self.groups_inside(indices):
            reduced=tuple(i for i in indices if i not in excluded);table=self.table(reduced,selected,arm)
            ss=[]
            for i in excluded:
                p=self.pairs[i];pred=apply(p['input'],self.scenes[i],selected,table)
                ss.append(b.metrics(pred,p['output'],p['input']))
            folds.append({'excluded':list(excluded),'scores':ss})
        exact=sum(all(s['exact'] for s in f['scores']) for f in folds)
        fractions=sum((Fraction(f['scores'][0]['correct'],f['scores'][0]['cells']) for f in folds),Fraction(0))
        return (exact,fractions),folds

    def discover(self,indices,arm):
        k=(indices,arm)
        if k in self.models_cache:return self.models_cache[k]
        records=[r for i in indices for r in self.records[i]];vocabulary,mask=ARMS[arm]
        subsets,diagnostics=search(records,vocabulary,mask);models=[]
        for fs in subsets:
            assert self.table(indices,fs,arm) is not None
            score,folds=self.fixed_cv(indices,fs,arm);reasons=[]
            for f in fs:
                core=conflict(records,tuple(i for i in fs if i!=f),mask)
                assert core and len({r[0][f] for r in core})>1
                reasons.append({'necessary_feature':f,'witness':certificate(core)})
            models.append({'arm':arm,'features':list(fs),'names':[TERMS[i][0] for i in fs],
                'cost':sum(TERMS[i][1] for i in fs),'cv_exact':score[0],
                'cv_fraction':[score[1].numerator,score[1].denominator],
                'validation':folds,'necessity':reasons})
        result=(models,diagnostics);self.models_cache[k]=result;return result

    def selected(self,indices,policy):
        arms=ARMS if policy=='union' else (policy,)
        models=[m for a in arms for m in self.discover(indices,a)[0]]
        if not models:return None
        def rank(m):
            return (-m['cv_exact'],-Fraction(*m['cv_fraction']),len(m['features']),m['cost'],
                    ARMS[m['arm']][1].bit_count(),list(ARMS).index(m['arm']),tuple(m['features']))
        return min(models,key=rank)

    def predict(self,indices,model,grid,rows):
        if model is None:return None
        fs=tuple(model['features']);return apply(grid,rows,fs,self.table(indices,fs,model['arm']))

    def diagnosis(self,indices,model,rows):
        result=Counter()
        if model is None:
            result['no_model']=len(rows);return dict(result)
        fs=tuple(model['features']);table=self.table(indices,fs,model['arm'])
        for features,values in rows:
            key=tuple(features[i] for i in fs);mask=table.get(key,0)
            if not mask:result['unknown_key']+=1;continue
            vals={values[i] for i in range(len(ACTIONS)) if mask&(1<<i)}
            if -1 in vals:result['undefined_action']+=1
            elif len(vals)>1:result['action_disagreement']+=1
            else:result['determined']+=1
        return dict(result)


def investigate(problem):
    pairs=problem['train'];xs=problem['query_inputs'];groups=defaultdict(list)
    for i,p in enumerate(pairs):groups[b.signature(p['input'])].append(i)
    eligible=len(groups)>=2 and all(b.shape(p['input'])==b.shape(p['output']) for p in pairs)
    row=dict(id=problem['id'],split=problem['split'],development=problem['id'] in DEVELOPMENT,
             eligible=eligible,query_inputs=xs,families={},policies={})
    if not eligible:
        row['policies']={p:dict(predictions=[None]*len(xs),selected=[],outer=[]) for p in POLICIES};return row
    learner=Learner(pairs);indices=tuple(range(len(pairs)));qsc=[scene(g) for g in xs]
    for arm in ARMS:
        models,diagnostics=learner.discover(indices,arm)
        row['families'][arm]={'models':models,'diagnostics':diagnostics}
        for m in models:m['predictions']=[learner.predict(indices,m,g,sc) for g,sc in zip(xs,qsc,strict=True)]
    for policy in POLICIES:
        selected=learner.selected(indices,policy);outer=[]
        for excluded in learner.groups:
            reduced=tuple(i for i in indices if i not in excluded);chosen=learner.selected(reduced,policy);ss=[]
            for i in excluded:
                p=pairs[i];pred=learner.predict(reduced,chosen,p['input'],learner.scenes[i])
                ss.append(b.metrics(pred,p['output'],p['input']))
            outer.append({'excluded':list(excluded),'selected':[] if chosen is None else [chosen['arm'],chosen['features']],
                          'scores':ss})
        row['policies'][policy]={'selected':[] if selected is None else [selected['arm'],selected['features']],
            'outer':outer,'predictions':[learner.predict(indices,selected,g,sc) for g,sc in zip(xs,qsc,strict=True)],
            'diagnosis':[learner.diagnosis(indices,selected,sc) for sc in qsc]}
    # Diagnose the old full-vocabulary witness independently of query labels.
    old=row['families']['base']['diagnostics'].get('witness')
    if old:
        shared=FULL_MASK
        for x in old:shared&=x['allowed']
        row['old_conflict']={'witness':old,'new_separators':[i for i in range(15,19) if len({x['features'][i] for x in old})>1],
            'new_shared_actions':[ACTIONS[i] for i in range(11,19) if shared&(1<<i)]}
    return row


def predict(problems,out,workers):
    jobs=json.loads(problems.read_text());start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(investigate,jobs))
    b.write_json(out/'predictions.json',rows)
    dependencies=[HERE.parent/f'{e}/run.py' for e in ('016-conflict-refinement','015-structural-roles','014-cross-demonstration-transport')]
    dependencies.append(HERE.parent/'014-cross-demonstration-transport/case_e88171ec.py')
    b.write_json(out/'prediction-run.json',dict(workers=workers,seconds=time.perf_counter()-start,seed=None,
        python=sys.version,platform=platform.platform(),terms=TERMS,actions=ACTIONS,development=sorted(DEVELOPMENT),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependencies={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
        problems_sha256=hashlib.sha256(problems.read_bytes()).hexdigest(),
        predictions_sha256=hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest()))
    print(json.dumps({'tasks':len(rows),'seconds':time.perf_counter()-start}))


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('predict');q.add_argument('--problems',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--workers',type=int,default=12)
    q=sub.add_parser('score');q.add_argument('--predictions',type=Path,required=True);q.add_argument('--answers',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='predict':predict(a.problems,a.out,a.workers)
    else:s16.score(a.predictions,a.answers,a.out)

if __name__=='__main__':main()
