"""025: bounded OrElse repairs of frozen base context tuples.

Training-only branch support is refitted within every inner/outer fold. Query
inputs impose joint definedness; no prediction command reads query answers.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import lru_cache
import importlib.util
from pathlib import Path
import platform
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('query024', HERE.parent/'024-query-totality/run.py')
e24 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e24)
e23, e18 = e24.e23, e24.e18
read, write, sha = e23.read, e23.write, e23.sha
PAIRS = tuple((a,b) for a in range(11,19) for b in range(19) if a != b)
ACTIONS = list(e18.s17.ACTIONS) + [f'OrElse({e18.s17.ACTIONS[a]},{e18.s17.ACTIONS[b]})' for a,b in PAIRS]
BASE = (1 << 19)-1
FULL = (1 << len(ACTIONS))-1
BY_PRIMARY = {a: sum(1 << (19+i) for i,(x,b) in enumerate(PAIRS) if x == a) for a in range(11,19)}
LIBRARIES = ('base','all','observed')
POLICIES = (*LIBRARIES,'union_observed')


@lru_cache(maxsize=8192)
def outputs(values: tuple) -> tuple:
    if len(values) != 19 or any(type(v) is not int or not -1 <= v <= 9 for v in values):
        raise ValueError('Expected nineteen primitive colour/undefined values')
    return values + tuple(values[a] if values[a] >= 0 else values[b] for a,b in PAIRS)


@lru_cache(maxsize=8192)
def masks(values: tuple) -> tuple:
    extended = outputs(values); by_label = [0]*10
    for i,v in enumerate(extended):
        if v >= 0: by_label[v] |= 1 << i
    primary = sum(1 << (a-11) for a in range(11,19) if values[a] >= 0)
    return tuple(by_label), primary, sum(by_label)


def fit(records: list, features: tuple) -> dict:
    """Rows are (features, primitive values, target colour, labelled origin)."""
    table = {}
    for fs, values, y, origin in records:
        if type(y) is not int or not 0 <= y <= 9: raise ValueError('Invalid target colour')
        key = tuple(fs[f] for f in features)
        allowed, primary, _ = masks(values)
        if key not in table: table[key] = [FULL,0,0]
        state = table[key]
        state[0] &= allowed[y]; state[1] |= primary; state[2] |= 255 ^ primary
    result = {}
    for key,(all_mask,yes,no) in table.items():
        supported = BASE
        for a, group in BY_PRIMARY.items():
            if yes & no & (1 << (a-11)): supported |= group
        result[key] = (all_mask & BASE, all_mask, all_mask & supported, yes, no)
    return result


def apply(grid: list, scene: list, features: tuple, table: dict) -> list:
    predicted = []
    for fs, values in scene:
        remaining = table.get(tuple(fs[f] for f in features),0)
        vs = outputs(values); possible = set()
        while remaining:
            bit = remaining & -remaining; remaining -= bit
            possible.add(vs[bit.bit_length()-1])
            if len(possible) > 1: break
        predicted.append(next(iter(possible)) if len(possible) == 1 and -1 not in possible else -1)
    w = len(grid[0]); return [predicted[i:i+w] for i in range(0,len(predicted),w)]


def condition(table: dict, scenes: list, features: tuple) -> dict:
    valid = {}
    for scene in scenes:
        for fs,values in scene:
            key = tuple(fs[f] for f in features)
            valid[key] = valid.get(key,FULL) & masks(values)[2]
    unknown = sorted(set(valid)-set(table))
    kept = {k: table[k] & v for k,v in valid.items() if k in table}
    feasible = not unknown and all(table.values()) and all(kept.values())
    return {'feasible': feasible, 'unknown': [list(k) for k in unknown],
            'valid': [[list(k),v] for k,v in sorted(valid.items())],
            'kept': [[list(k),v] for k,v in sorted(kept.items())] if feasible else []}


def rank(model: dict) -> tuple:
    return (-model['cv_exact'], -Fraction(*model['cv_fraction']), len(model['features']),
            model['cost'], LIBRARIES.index(model['library']),
            e23.ARMS.index(model['arm']), tuple(model['features']))


def decide(models: list, count: int, policy: str) -> dict:
    libraries = ('base','observed') if policy == 'union_observed' else (policy,)
    candidates = [m for m in models if m['library'] in libraries and m['condition']['feasible']]
    best = min(candidates,key=rank) if candidates else None
    return {'selected': [] if best is None else [best['library'],best['arm'],best['features']],
            'predictions': [None]*count if best is None else best['predictions'],
            'feasible_models': len(candidates)}


class Engine:
    def __init__(self, train: list):
        self.train = train; self.scenes = [e18.scene(p['input']) for p in train]
        self.records = []
        groups = {}
        for i,(p,sc) in enumerate(zip(train,self.scenes,strict=True)):
            w = len(p['input'][0]); labels = [v for r in p['output'] for v in r]
            self.records.append([(fs,vs,y,(i,j//w,j%w)) for j,((fs,vs),y) in enumerate(zip(sc,labels,strict=True))])
            groups.setdefault(e18.b.signature(p['input']),[]).append(i)
        self.groups = list(groups.values()); self.cache = {}; self.tables = []
        self.validation_cache = {}

    def table(self, indices: tuple, features: tuple, library: str) -> tuple:
        cache_key = (indices,features)
        if cache_key not in self.cache:
            raw = fit([r for i in indices for r in self.records[i]],features)
            self.cache[cache_key] = (len(self.tables),raw)
            self.tables.append({'indices': list(indices), 'features': list(features),
                'entries': [[list(k),*v] for k,v in sorted(raw.items())]})
        tid, raw = self.cache[cache_key]; column = LIBRARIES.index(library)
        return tid,{k:v[column] for k,v in raw.items()}

    def cv(self, indices: tuple, features: tuple, library: str) -> list:
        cache_key = (indices,features,library)
        if cache_key not in self.validation_cache:
            folds = []
            for excluded in self.groups:
                if excluded[0] not in indices: continue
                reduced = tuple(i for i in indices if i not in excluded)
                tid, table = self.table(reduced,features,library)
                scores = [e18.b.metrics(apply(self.train[i]['input'],self.scenes[i],features,table),
                    self.train[i]['output'],self.train[i]['input']) for i in excluded]
                folds.append({'excluded':excluded,'table_id':tid,'scores':scores})
            self.validation_cache[cache_key] = folds
        return self.validation_cache[cache_key]

    def pool(self, indices: tuple, grids: list, query_scenes: list, old_models: list) -> list:
        # Old scores/predictions are not consumed: only the frozen tuple pool.
        unique = {}
        for old in sorted(old_models,key=lambda m:(e23.ARMS.index(m['arm']),tuple(m['features']))):
            unique.setdefault(tuple(old['features']),old)
        models = []
        for features,old in unique.items():
            for library in LIBRARIES:
                tid, table = self.table(indices,features,library)
                if not all(table.values()): raise AssertionError('Old fitting tuple was lost')
                folds = self.cv(indices,features,library)
                fraction = sum((Fraction(f['scores'][0]['correct'],f['scores'][0]['cells']) for f in folds),Fraction())
                domain = condition(table,query_scenes,features)
                kept = {tuple(k):v for k,v in domain['kept']}
                predictions = ([apply(g,sc,features,kept) for g,sc in zip(grids,query_scenes,strict=True)]
                               if domain['feasible'] else [None]*len(grids))
                models.append({'library':library, 'arm':old['arm'], 'features':list(features),
                    'cost':sum(e18.TERMS[f][1] for f in features), 'table_id':tid,
                    'cv_exact':sum(all(s['exact'] for s in f['scores']) for f in folds),
                    'cv_fraction':[fraction.numerator,fraction.denominator],
                    'validation':folds, 'condition':domain, 'predictions':predictions})
        return models

    def witnesses(self, indices: tuple, query_scenes: list, models: list, choices: dict) -> list:
        chosen = {tuple([c['selected'][0],c['selected'][1],tuple(c['selected'][2])])
                  for c in choices.values() if c['selected']}
        result = []
        for model in models:
            mid = (model['library'],model['arm'],tuple(model['features']))
            if mid not in chosen: continue
            fs = tuple(model['features']); origins = {}
            for i in indices:
                for f,v,y,o in self.records[i]:
                    key = tuple(f[j] for j in fs)
                    for a in range(11,19): origins.setdefault((key,a,v[a]>=0),list(o))
            events = {}
            for sc in query_scenes:
                for f,v in sc:
                    key = tuple(f[j] for j in fs)
                    for a in range(11,19): events[(key,a,v[a]>=0)] = events.get((key,a,v[a]>=0),0)+1
            details = []
            for k,mask in model['condition']['kept']:
                key = tuple(k)
                for a,group in BY_PRIMARY.items():
                    operations = [i for i in range(19,len(ACTIONS)) if mask & group & (1 << i)]
                    if operations:
                        details.append({'key':k,'primary':a,'operations':operations,
                            'primary_training':origins.get((key,a,True)), 'fallback_training':origins.get((key,a,False)),
                            'query_primary':events.get((key,a,True),0),'query_fallback':events.get((key,a,False),0)})
            result.append({'selected':list(mid[:2])+[list(fs)],'branches':details})
        return result


def strip(row: dict) -> dict:
    def pool(ms): return [{'features':m['features'],'arm':m['arm']} for m in ms]
    if row.get('mode') != 'base': raise ValueError('Only the frozen base pools are permitted')
    return {'id':row['id'],'split':row['split'],'models':pool(row['models']),
            'outer':[{'excluded':f['excluded'],'models':pool(f['models'])} for f in row['outer']]}


def investigate(job: tuple) -> dict:
    problem,saved = job
    if saved is None: saved = strip(e23.investigate(problem,'base'))
    if (problem['id'],problem['split']) != (saved['id'],saved['split']): raise ValueError('Wrong cached problem')
    train,queries = problem['train'],problem['query_inputs']
    eligible = len({e18.b.signature(p['input']) for p in train}) >= 2 and all(
        e18.b.shape(p['input']) == e18.b.shape(p['output']) for p in train)
    row = {'id':problem['id'],'split':problem['split'],'eligible':eligible,
        'development':problem['id'] in e18.DEVELOPMENT,'queries':queries,'models':[],'tables':[],'outer':[],'branch_witnesses':[]}
    if not eligible:
        if saved['models']: raise AssertionError('Eligibility changed')
        row['policies'] = {p:decide([],len(queries),p) for p in POLICIES}; return row
    engine = Engine(train); indices = tuple(range(len(train))); qsc = [e18.scene(g) for g in queries]
    row['models'] = engine.pool(indices,queries,qsc,saved['models'])
    row['policies'] = {p:decide(row['models'],len(queries),p) for p in POLICIES}
    row['branch_witnesses'] = engine.witnesses(indices,qsc,row['models'],row['policies'])
    if [f['excluded'] for f in saved['outer']] != engine.groups: raise AssertionError('Duplicate groups differ')
    for old in saved['outer']:
        excluded = old['excluded']; reduced = tuple(i for i in indices if i not in excluded)
        xs = [train[i]['input'] for i in excluded]
        models = engine.pool(reduced,xs,[engine.scenes[i] for i in excluded],old['models'])
        decisions = {p:decide(models,len(xs),p) for p in POLICIES}
        scores = {p:[e18.b.metrics(g,train[i]['output'],train[i]['input'])
                     for i,g in zip(excluded,d['predictions'],strict=True)] for p,d in decisions.items()}
        row['outer'].append({'excluded':excluded,'models':models,'policies':decisions,'scores':scores})
    row['tables'] = engine.tables
    return row


def predict(problems: Path, cached: Path | None, out: Path, start: int, limit: int | None, workers: int):
    all_problems = read(problems)
    saved = {r['split']+'/'+r['id']:strip(r) for r in read(cached)} if cached else None
    jobs = all_problems[start:] if limit is None else all_problems[start:start+limit]
    if not jobs or workers < 1: raise ValueError('Empty batch or invalid workers')
    t = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(investigate,[(p,saved[p['split']+'/'+p['id']] if saved else None) for p in jobs]))
    write(out/'predictions.json',rows)
    deps = {str(p.relative_to(HERE.parent)):sha(p) for p in HERE.parent.glob('*/run.py')
            if p.parent.name[:3] in ('014','015','016','017','018','021','023','024')}
    write(out/'run.json',{'mode':'guarded','start':start,'count':len(rows),'total':len(all_problems),
        'workers':workers,'seed':None,'seconds':time.perf_counter()-t,'platform':platform.platform(),
        'source_sha256':sha(Path(__file__)),'dependencies':deps,'cached_sha256':sha(cached) if cached else None,
        'problems_sha256':sha(problems),'predictions_sha256':sha(out/'predictions.json')})
    print({'start':start,'tasks':len(rows),'seconds':time.perf_counter()-t},flush=True)


def main():
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest='cmd',required=True)
    q = sub.add_parser('predict'); q.add_argument('--problems',type=Path,required=True); q.add_argument('--cached',type=Path)
    q.add_argument('--out',type=Path,required=True); q.add_argument('--start',type=int,default=0); q.add_argument('--limit',type=int)
    q.add_argument('--workers',type=int,default=12)
    q = sub.add_parser('merge'); q.add_argument('--problems',type=Path,required=True)
    q.add_argument('--batches',nargs='+',type=Path,required=True); q.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    if a.cmd == 'predict': predict(a.problems,a.cached,a.out,a.start,a.limit,a.workers)
    else: e23.merge(a.problems,a.batches,a.out)


if __name__ == '__main__': main()
