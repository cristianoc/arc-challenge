"""018: learn order relations and explicit applicability guards.

Reuses 017's feature/action evaluator, exact conflict extraction, fitting,
prediction semantics and scoring. No changes to previous experiments or core.
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
spec = importlib.util.spec_from_file_location('split017', HERE.parent/'017-split-or-share/run.py')
s17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s17)
b = s17.b
PAIRS = tuple(itertools.combinations(range(4), 2))
TERMS = s17.TERMS + [
    (f'compare({s17.DIRECTIONS[i][0]},{s17.DIRECTIONS[j][0]})', 9) for i,j in PAIRS
] + [(f'defined(after_ray:{d})',5) for d,_,_ in s17.DIRECTIONS]
ARMS = {'exact':tuple(range(19)), 'order':tuple(range(25)),
        'guarded_exact':tuple(range(19))+tuple(range(25,29)),
        'guarded_order':tuple(range(29))}
POLICIES = (*ARMS,'union')
DEVELOPMENT = set(s17.DEVELOPMENT) | {'29c11459'}
MASK = s17.FULL_MASK


def extend(rows):
    return [(tuple(f)+tuple((f[15+i]>f[15+j])-(f[15+i]<f[15+j]) for i,j in PAIRS)
             +tuple(int(v[k]>=0) for k in range(15,19)),v) for f,v in rows]


def scene(grid):
    return extend(s17.scene(grid))


def complexity(fs):
    return (len(fs),sum(TERMS[i][1] for i in fs),tuple(fs))


def search(records,vocabulary,maximum=3):
    """017's exact conflict-clause search with the extended term cost table."""
    unique={}
    for r in records:unique.setdefault((r[0],r[4]),r)
    records=list(unique.values())
    bad=s17.conflict(records,vocabulary,MASK)
    if bad is not None:return [],{'status':'vocabulary_conflict','states':1,'witness':s17.certificate(bad)}
    visited=set();solutions=set()
    def visit(fs):
        fs=tuple(sorted(fs))
        if fs in visited or any(set(s).issubset(fs) for s in solutions):return
        visited.add(fs);bad=s17.conflict(records,fs,MASK)
        if bad is None:solutions.add(fs);return
        if len(fs)==maximum:return
        for f in vocabulary:
            if len({r[0][f] for r in bad})>1:visit((*fs,f))
    visit(())
    minimal=sorted((s for s in solutions if not any(set(t)<set(s) for t in solutions)),key=complexity)
    return minimal,{'status':'fit' if minimal else 'feature_bound','states':len(visited)}


class Learner(s17.Learner):
    def __init__(self,pairs):
        super().__init__(pairs)
        self.scenes=[extend(rows) for rows in self.scenes]
        self.records=[[(fs,v,r[2],r[3],r[4]) for (fs,v),r in zip(sc,rs,strict=True)]
                      for sc,rs in zip(self.scenes,self.records,strict=True)]

    def table(self,indices,selected,arm):
        k=(indices,selected)
        if k not in self.table_cache:
            self.table_cache[k]=s17.fit([r for i in indices for r in self.records[i]],selected,MASK)
        return self.table_cache[k]

    def discover(self,indices,arm):
        k=(indices,arm)
        if k in self.models_cache:return self.models_cache[k]
        records=[r for i in indices for r in self.records[i]]
        subsets,diag=search(records,ARMS[arm]);models=[]
        for fs in subsets:
            score,folds=self.fixed_cv(indices,fs,arm);reasons=[]
            for f in fs:
                core=s17.conflict(records,tuple(i for i in fs if i!=f),MASK)
                assert core is not None and len({r[0][f] for r in core})>1
                reasons.append({'necessary_feature':f,'witness':s17.certificate(core)})
            models.append({'arm':arm,'features':list(fs),'names':[TERMS[f][0] for f in fs],
                           'cost':complexity(fs)[1], 'cv_exact':score[0],
                           'cv_fraction':[score[1].numerator,score[1].denominator],
                           'validation':folds,'necessity':reasons})
        self.models_cache[k]=(models,diag);return models,diag

    def selected(self,indices,policy):
        models=[m for a in (ARMS if policy=='union' else (policy,)) for m in self.discover(indices,a)[0]]
        if not models:return None
        return min(models,key=lambda m:(-m['cv_exact'],-Fraction(*m['cv_fraction']),len(m['features']),
                                         m['cost'],list(ARMS).index(m['arm']),tuple(m['features'])))


def investigate(problem):
    pairs=problem['train'];xs=problem['query_inputs'];groups=defaultdict(list)
    for i,p in enumerate(pairs):groups[b.signature(p['input'])].append(i)
    eligible=len(groups)>=2 and all(b.shape(p['input'])==b.shape(p['output']) for p in pairs)
    row={'id':problem['id'],'split':problem['split'],'development':problem['id'] in DEVELOPMENT,
         'eligible':eligible,'query_inputs':xs,'families':{},'policies':{}}
    if not eligible:
        row['policies']={p:{'predictions':[None]*len(xs),'selected':[],'outer':[]} for p in POLICIES}
        return row
    learner=Learner(pairs);idx=tuple(range(len(pairs)));qsc=[scene(g) for g in xs]
    for arm in ARMS:
        models,diag=learner.discover(idx,arm)
        row['families'][arm]={'models':models,'diagnostics':diag}
        for m in models:m['predictions']=[learner.predict(idx,m,g,sc) for g,sc in zip(xs,qsc,strict=True)]
    for policy in POLICIES:
        chosen=learner.selected(idx,policy);outer=[]
        for excluded in learner.groups:
            reduced=tuple(i for i in idx if i not in excluded);m=learner.selected(reduced,policy)
            ss=[]
            for i in excluded:
                p=pairs[i];pred=learner.predict(reduced,m,p['input'],learner.scenes[i])
                ss.append(b.metrics(pred,p['output'],p['input']))
            outer.append({'excluded':list(excluded),'selected':[] if m is None else [m['arm'],m['features']],
                          'scores':ss})
        row['policies'][policy]={'selected':[] if chosen is None else [chosen['arm'],chosen['features']],
            'outer':outer,'predictions':[learner.predict(idx,chosen,g,sc) for g,sc in zip(xs,qsc,strict=True)],
            'diagnosis':[learner.diagnosis(idx,chosen,sc) for sc in qsc]}
    return row


def teacher_grid(width,colours,height=5,farther=False):
    """Explicit constructed teacher, not a new ARC ground-truth annotation."""
    x=[[0]*width for _ in range(height)];r=height//2
    x[r][0],x[r][-1]=colours;y=[row[:] for row in x]
    for c in range(1,width-1):
        y[r][c]=5 if c==width-1-c else colours[int((c>width-1-c) != farther)]
    return {'input':x,'output':y}


def controlled(out):
    # Both teachers and every width/palette are declared before any corpus run.
    results=[]
    for farther in (False,True):
        train=[teacher_grid(7,(1,2),farther=farther),teacher_grid(11,(3,4),farther=farther)]
        tests=[teacher_grid(w,cs,h,farther) for w in (5,9,13,17,21,25,29)
               for cs,h in (((6,8),5),((7,9),7))]
        row=investigate({'id':'synthetic_far' if farther else 'synthetic_near','split':'controlled',
                         'train':train,'query_inputs':[p['input'] for p in tests]})
        # Persist predictions before scoring this explicitly supplied teacher.
        b.write_json(out/(row['id']+'-predictions.json'),row)
        results.append({'id':row['id'],'train':train,'tests':tests,
            'policies':{name:{'selected':p['selected'],
                    'complete_grids':sum(b.metrics(g,t['output'],t['input'])['complete'] for g,t in zip(p['predictions'],tests)),
                    'correct_grids':sum(g==t['output'] for g,t in zip(p['predictions'],tests)),
                    'outer_pass':all(s['exact'] for f in p['outer'] for s in f['scores'])}
                        for name,p in row['policies'].items()}})
    b.write_json(out/'controlled.json',results)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('train','tests')} for r in results],indent=2))


def predict(problems,out,workers):
    jobs=json.loads(problems.read_text());start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(investigate,jobs))
    b.write_json(out/'predictions.json',rows)
    deps=[HERE.parent/e/'run.py' for e in ('017-split-or-share','016-conflict-refinement','015-structural-roles','014-cross-demonstration-transport')]
    deps.append(HERE.parent/'014-cross-demonstration-transport/case_e88171ec.py')
    b.write_json(out/'prediction-run.json',{'workers':workers,'seconds':time.perf_counter()-start,'seed':None,
        'python':sys.version,'platform':platform.platform(),'terms':TERMS,'actions':s17.ACTIONS,
        'development':sorted(DEVELOPMENT),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependencies':{str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in deps},
        'problems_sha256':hashlib.sha256(problems.read_bytes()).hexdigest(),
        'predictions_sha256':hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest()})
    print(json.dumps({'tasks':len(rows),'seconds':time.perf_counter()-start}))


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('predict');q.add_argument('--problems',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--workers',type=int,default=12)
    q=sub.add_parser('score');q.add_argument('--predictions',type=Path,required=True);q.add_argument('--answers',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('controlled');q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='predict':predict(a.problems,a.out,a.workers)
    elif a.cmd=='controlled':controlled(a.out)
    else:s17.s16.score(a.predictions,a.answers,a.out)

if __name__=='__main__':main()
