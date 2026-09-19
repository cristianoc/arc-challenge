"""016: witness-guided conjunctions of relational features; nested validation.

The scene vocabulary is supplied, not inferred from nothing. This file reuses
015's action learner and 014's grid/prediction/scoring utilities. Prediction
reads projected problems only. Query labels are read by a separate command.
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
spec = importlib.util.spec_from_file_location('roles015', HERE.parent/'015-structural-roles/run.py')
s15 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s15)
b = s15.b
DEVELOPMENT = {'00d62c1b', 'e88171ec'}
# Constructor costs are supplied syntax costs, not measured MDL or complexity.
TERMS = [('colour', 1), ('border', 1), ('degree_same', 2)] + [
    (scope + '.' + agg, scost + acost)
    for scope, scost in [('near', 2), ('reach', 3), ('colour_peers', 2)]
    for agg, acost in [('count', 1), ('any_border', 2), ('min_degree', 3), ('max_degree', 3)]]
ALL = tuple(range(len(TERMS)))
NO_REACH = tuple(i for i,(name,_) in enumerate(TERMS) if not name.startswith('reach.'))
FAMILIES = {'atoms': (ALL,1), 'refined': (ALL,3), 'no_reach': (NO_REACH,3)}
POLICIES = ('atoms_cv','refined_cv','refined_cost','refined_consensus','no_reach_cv','coordinates','identity')


def feature_rows(grid):
    """Evaluate the frozen scope/aggregate grammar, without output labels."""
    h,w = b.shape(grid); positions = [(r,c) for r in range(h) for c in range(w)]
    neighbours = {p:[q for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1))
                     if 0 <= q[0] < h and 0 <= q[1] < w and grid[q[0]][q[1]] == grid[p[0]][p[1]]]
                  for p in positions}
    degree = {p:len(neighbours[p]) for p in positions}
    edge = {p:int(p[0] in (0,h-1) or p[1] in (0,w-1)) for p in positions}
    peers = defaultdict(list)
    for p in positions: peers[grid[p[0]][p[1]]].append(p)
    def aggregate(cells):
        return (len(cells), int(any(edge[p] for p in cells)), min(degree[p] for p in cells), max(degree[p] for p in cells))
    peer_values = {k:aggregate(v) for k,v in peers.items()}
    reach_values = {}
    for p in positions:
        if p in reach_values: continue
        seen = {p}; stack = [p]
        while stack:
            for q in neighbours[stack.pop()]:
                if q not in seen: seen.add(q); stack.append(q)
        values = aggregate(seen)
        for q in seen: reach_values[q] = values
    return [(grid[r][c],edge[(r,c)],degree[(r,c)],
             *aggregate([(r,c)]+neighbours[(r,c)]),*reach_values[(r,c)],*peer_values[grid[r][c]])
            for r,c in positions]


def allowed(x,y):
    return (1 << y) | ((1 << s15.COPY) if x == y else 0)


def conflict(records, selected):
    """Return an incompatible pair, or None. No pairwise quadratic scan."""
    seen = {}
    for record in records:
        f,x,y,origin = record; key = tuple(f[i] for i in selected)
        if key not in seen:
            seen[key] = ({y:record}, record if x != y else None)
            continue
        by_output, changed = seen[key]
        if changed is not None and changed[2] != y:
            return changed, record
        if x != y:
            for previous_y, previous in by_output.items():
                if previous_y != y: return record, previous
            changed = record
        by_output.setdefault(y,record)
        seen[key] = (by_output,changed)
    return None


def search(records, vocabulary, max_features):
    """Enumerate ALL inclusion-minimal sufficient subsets within a size bound.

Every conflict gives a clause: choose a feature that separates this pair.
Branches cover every such feature; no heuristic pruning or node timeout.
"""
    records = list({(tuple(f),x,y):(tuple(f),x,y,o) for f,x,y,o in reversed(records)}.values())
    records.reverse()
    impossible = conflict(records,vocabulary)
    if impossible is not None:
        return [], {'states':1,'status':'vocabulary_conflict','witness':certificate(impossible)}
    visited = set(); solutions = set(); witnesses = 0
    def visit(selected):
        nonlocal witnesses
        selected = tuple(sorted(selected))
        if selected in visited or any(set(s).issubset(selected) for s in solutions): return
        visited.add(selected)
        pair = conflict(records,selected)
        if pair is None:
            solutions.add(selected); return
        witnesses += 1
        if len(selected) == max_features: return
        a,c = pair
        for i in vocabulary:
            if a[0][i] != c[0][i]: visit((*selected,i))
    visit(())
    minimal = [s for s in solutions if not any(set(t) < set(s) for t in solutions)]
    minimal.sort(key=complexity)
    return minimal, {'states':len(visited),'conflicts':witnesses,
                     'status':'fit' if minimal else 'feature_bound'}


def certificate(pair):
    return [{'origin':list(o),'input':x,'output':y,'features':list(f)} for f,x,y,o in pair]


def complexity(selected):
    return (len(selected),sum(TERMS[i][1] for i in selected),tuple(selected))


class Learner:
    def __init__(self, pairs):
        self.pairs = pairs
        self.features = [feature_rows(p['input']) for p in pairs]
        self.records = []
        for i,(p,fs) in enumerate(zip(pairs,self.features,strict=True)):
            w = len(p['input'][0])
            self.records.append([(f,x,y,(i,j//w,j%w)) for j,(f,x,y) in enumerate(zip(
                fs,(v for row in p['input'] for v in row),(v for row in p['output'] for v in row),strict=True))])
        groups = defaultdict(list)
        for i,p in enumerate(pairs): groups[b.signature(p['input'])].append(i)
        self.groups = tuple(tuple(v) for v in groups.values())
        self.key_cache = {}; self.table_cache = {}; self.models_cache = {}

    def keys(self, i, selected):
        k = (i,selected)
        if k not in self.key_cache:
            self.key_cache[k] = [tuple(f[j] for j in selected) for f in self.features[i]]
        return self.key_cache[k]

    def table(self, indices, selected):
        k = (indices,selected)
        if k not in self.table_cache:
            self.table_cache[k] = s15.fit([self.pairs[i] for i in indices],
                                        [self.keys(i,selected) for i in indices],'copy')
        return self.table_cache[k]

    def groups_inside(self, indices):
        return [g for g in self.groups if g[0] in indices]

    def fixed_cv(self, indices, selected):
        folds = []
        for excluded in self.groups_inside(indices):
            reduced = tuple(i for i in indices if i not in excluded)
            table = self.table(reduced,selected)
            scores = []
            for i in excluded:
                p = self.pairs[i]
                prediction = s15.apply(p['input'],self.keys(i,selected),table,'copy')
                scores.append(b.metrics(prediction,p['output'],p['input']))
            folds.append({'excluded':list(excluded),'scores':scores})
        # Equal weights for distinct input grids; all duplicate labels still checked.
        exact = sum(all(s['exact'] for s in f['scores']) for f in folds)
        correct = sum((Fraction(f['scores'][0]['correct'],f['scores'][0]['cells']) for f in folds), Fraction(0))
        return (exact,correct),folds

    def discover(self, indices, family):
        k = (indices,family)
        if k in self.models_cache: return self.models_cache[k]
        records = [r for i in indices for r in self.records[i]]
        vocab,maximum = FAMILIES[family]
        subsets,diagnostics = search(records,vocab,maximum)
        models = []
        for f in subsets:
            assert self.table(indices,f) is not None
            score,folds = self.fixed_cv(indices,f)
            reasons = []
            for j in f:
                pair = conflict(records,tuple(k for k in f if k != j))
                assert pair is not None and pair[0][0][j] != pair[1][0][j]
                reasons.append({'necessary_feature':j,'witness':certificate(pair)})
            models.append({'features':list(f),'names':[TERMS[j][0] for j in f],
                'cost':sum(TERMS[j][1] for j in f),'cv_exact':score[0],
                'cv_fraction':[score[1].numerator,score[1].denominator],
                'validation':folds,'necessity':reasons})
        result = models,diagnostics
        self.models_cache[k] = result
        return result

    def selected(self, indices, policy):
        family,strategy = policy.rsplit('_',1)
        models,diagnostics = self.discover(indices,family)
        if not models: return []
        if strategy == 'cost': return [models[0]]
        def score(m): return (m['cv_exact'],Fraction(*m['cv_fraction']))
        best = max(map(score,models)); top = [m for m in models if score(m) == best]
        return top if strategy == 'consensus' else top[:1]

    def apply_selected(self, indices, models, grid, fs):
        if not models: return None
        ps = [s15.apply(grid,[tuple(f[j] for j in m['features']) for f in fs],
                        self.table(indices,tuple(m['features'])),'copy') for m in models]
        return s15.consensus([[p] for p in ps],[grid])[0]


def investigate(problem):
    pairs = problem['train']; xs = problem['query_inputs']
    groups = defaultdict(list)
    for i,p in enumerate(pairs): groups[b.signature(p['input'])].append(i)
    eligible = len(groups) >= 2 and all(b.shape(p['input']) == b.shape(p['output']) for p in pairs)
    row = {'id':problem['id'],'split':problem['split'],'development':problem['id'] in DEVELOPMENT,
           'eligible':eligible,'query_inputs':xs,'families':{},'policies':{}}
    if not eligible:
        row['policies'] = {p:{'predictions':[None]*len(xs),'selected':[],'outer':[]} for p in POLICIES}
        return row
    learner = Learner(pairs); indices = tuple(range(len(pairs))); qfs = [feature_rows(g) for g in xs]
    for family in FAMILIES:
        models,diagnostics = learner.discover(indices,family)
        row['families'][family] = {'models':models,'diagnostics':diagnostics}
        for m in models:
            m['predictions'] = [learner.apply_selected(indices,[m],g,fs) for g,fs in zip(xs,qfs,strict=True)]
    for policy in POLICIES:
        if policy in ('coordinates','identity'): continue
        selected = learner.selected(indices,policy)
        outer = []
        for excluded in learner.groups:
            reduced = tuple(i for i in indices if i not in excluded)
            chosen = learner.selected(reduced,policy)
            ss = []
            for i in excluded:
                p = pairs[i]; pred = learner.apply_selected(reduced,chosen,p['input'],learner.features[i])
                ss.append(b.metrics(pred,p['output'],p['input']))
            outer.append({'excluded':list(excluded),'selected':[m['features'] for m in chosen],
                          'scores':ss})
        row['policies'][policy] = {'selected':[m['features'] for m in selected], 'outer':outer,
            'predictions':[learner.apply_selected(indices,selected,g,fs) for g,fs in zip(xs,qfs,strict=True)]}
    def coords(g):
        h,w=b.shape(g); return [(h,w,r,c) for r in range(h) for c in range(w)]
    table = s15.fit(pairs,[coords(p['input']) for p in pairs],'copy')
    row['policies']['coordinates'] = {'selected':['coordinates'] if table is not None else [],'outer':[],
        'predictions':[s15.apply(g,coords(g),table,'copy') for g in xs]}
    row['policies']['identity'] = {'selected':['identity'],'outer':[],'predictions':xs}
    return row


def predict(problems,out,workers):
    start=time.perf_counter(); jobs=json.loads(problems.read_text())
    with ProcessPoolExecutor(max_workers=workers) as pool: rows=list(pool.map(investigate,jobs))
    b.write_json(out/'predictions.json',rows)
    dependencies = [HERE.parent/'015-structural-roles/run.py',HERE.parent/'014-cross-demonstration-transport/run.py',
                    HERE.parent/'014-cross-demonstration-transport/case_e88171ec.py']
    b.write_json(out/'prediction-run.json',{'workers':workers,'seconds':time.perf_counter()-start,'seed':None,
        'python':sys.version,'platform':platform.platform(),'terms':TERMS,'development':sorted(DEVELOPMENT),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependencies':{str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
        'problems_sha256':hashlib.sha256(problems.read_bytes()).hexdigest(),
        'predictions_sha256':hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest()})
    print(json.dumps({'tasks':len(rows),'seconds':time.perf_counter()-start}))


def score(predictions,answers,out):
    rows=json.loads(predictions.read_text()); ys=json.loads(answers.read_text()); counts=defaultdict(Counter); scores=[]
    for row in rows:
        labels=ys[row['split']+'/'+row['id']]; rr={'id':row['id'],'split':row['split'], 'development':row['development'], 'policies':{},'families':{}}
        for name,p in row['policies'].items():
            ss=[b.metrics(g,y,x) for g,y,x in zip(p['predictions'],labels,row['query_inputs'],strict=True)]
            exact=all(s['exact'] for s in ss); complete=all(s['complete'] for s in ss)
            outer=bool(p['outer']) and all(s['exact'] for fold in p['outer'] for s in fold['scores'])
            rr['policies'][name]={'correct':exact,'complete':complete,'outer_pass':outer,'selected':p['selected'],'grids':ss}
            if row['development']: continue
            c=counts[(row['split'],name)]
            c['tasks']+=1; c['eligible']+=row['eligible']; c['fits']+=bool(p['selected']); c['complete']+=complete
            c['correct']+=exact; c['wrong_complete']+=complete and not exact; c['outer_pass']+=outer
            c['gated_complete']+=outer and complete; c['gated_correct']+=outer and exact; c['gated_wrong']+=outer and complete and not exact
            c['any_wrong_known']+=any(s['wrong'] for s in ss)
            for s in ss:
                for k in ['known','wrong','changed','changed_known','changed_correct']: c[k]+=s[k] or 0
        for family,details in row['families'].items():
            ms=details['models']; correct=[m['features'] for m in ms if all(g==y for g,y in zip(m['predictions'],labels,strict=True))]
            rr['families'][family]={'fits':bool(ms),'models':len(ms),'oracle_correct':bool(correct),'correct_models':correct,'status':details['diagnostics']['status']}
        scores.append(rr)
    summary={sp:{p:dict(c) for (s,p),c in counts.items() if s==sp} for sp in ('training','evaluation')}
    b.write_json(out/'scores.json',scores); b.write_json(out/'summary.json',summary)
    b.write_json(out/'score-run.json',{'predictions_sha256':hashlib.sha256(predictions.read_bytes()).hexdigest(),
        'answers_sha256':hashlib.sha256(answers.read_bytes()).hexdigest(),
        'summary_sha256':hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()})
    print('split policy fits complete correct wrong_complete outer_pass gated_correct')
    for sp,policies in summary.items():
        for p,c in policies.items(): print(sp,p,*[c[k] for k in ['fits','complete','correct','wrong_complete','outer_pass','gated_correct']])


def main():
    p=argparse.ArgumentParser(__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('predict');q.add_argument('--problems',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--workers',type=int,default=12)
    q=sub.add_parser('score');q.add_argument('--predictions',type=Path,required=True);q.add_argument('--answers',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='predict': predict(a.problems,a.out,a.workers)
    else: score(a.predictions,a.answers,a.out)

if __name__=='__main__': main()
