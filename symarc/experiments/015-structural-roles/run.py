"""015: finite structural-role hypotheses, with explicit output-action uncertainty.

Reuse 014 for grids, preparation, scoring cells, and raw neighbourhoods.
Train-only coverage, isolated query prediction, and scoring are separate commands.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '014-cross-demonstration-transport'
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
b = load('transport014', BASE / 'run.py')
r14 = load('rect014', BASE / 'case_e88171ec.py')
COPY = 10
LANGUAGES = ('literal', 'copy')
# Order is a declared prior, not an inferred model-complexity ranking.
REPS = [(kind, k) for kind in ('raw', 'canonical') for k in range(4)] + [
    (kind, colour) for kind in ('components_all', 'component_largest', 'rectangle_largest')
    for colour in range(10)]


def components(g, colour):
    h, w = b.shape(g)
    remaining = {(r,c) for r in range(h) for c in range(w) if g[r][c] == colour}
    result = []
    while remaining:
        start = min(remaining); remaining.remove(start); cells = {start}; stack = [start]
        while stack:
            r,c = stack.pop()
            for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if p in remaining:
                    remaining.remove(p); cells.add(p); stack.append(p)
        result.append(cells)
    return result


def mask(g, kind, colour):
    if not any(colour in row for row in g): return None
    if kind == 'rectangle_largest':
        # Reuse the audited maximum-zero-rectangle routine, with a colour mask.
        _, rectangles = r14.largest_zero_rectangles([[0 if v == colour else 1 for v in row] for row in g])
        if len(rectangles) != 1: return None
        top,bottom,left,right = rectangles[0]
        return {(r,c) for r in range(top,bottom+1) for c in range(left,right+1)}
    cc = components(g, colour)
    if kind == 'components_all': return set().union(*cc)
    biggest = max(map(len, cc)); cc = [s for s in cc if len(s) == biggest]
    return cc[0] if len(cc) == 1 else None


def context(g, rep):
    kind,k = rep
    if kind == 'raw': return b.features(g, 'local', k)
    if kind == 'canonical':
        keys = []
        for patch, centre in zip(b.features(g, 'local', k), (v for row in g for v in row)):
            renaming = {centre:0}; values=[]
            for value in patch:
                if value == b.PAD: values.append(b.PAD)
                else:
                    if value not in renaming: renaming[value] = len(renaming)
                    values.append(renaming[value])
            keys.append(bytes(values))
        return keys
    selected = mask(g, kind, k)
    if selected is None: return None
    h,w = b.shape(g)
    def role(r,c):
        if (r,c) not in selected: return 0
        return 2 if all(p in selected for p in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))) else 1
    return [role(r,c) for r in range(h) for c in range(w)]


def fit(pairs, features, language):
    table = {}; universe = (1 << (11 if language == 'copy' else 10))-1
    for pair, keys in zip(pairs, features, strict=True):
        if keys is None: return None
        for key,x,y in zip(keys, (v for row in pair['input'] for v in row),
                          (v for row in pair['output'] for v in row), strict=True):
            allowed = (1 << y) | ((1 << COPY) if language == 'copy' and x == y else 0)
            table[key] = table.get(key, universe) & allowed
            if table[key] == 0: return None
    return table


def apply(g, keys, table, language):
    if keys is None or table is None: return None
    universe = (1 << (11 if language == 'copy' else 10))-1
    values = []
    for key,x in zip(keys, (v for row in g for v in row), strict=True):
        actions = table.get(key, universe)
        labels = {i for i in range(10) if actions & (1 << i)}
        if actions & (1 << COPY): labels.add(x)
        values.append(next(iter(labels)) if len(labels) == 1 else b.UNKNOWN)
    w = len(g[0]); return [values[i:i+w] for i in range(0,len(values),w)]


def cv_score(folds):
    exact = sum(s['exact'] for s in folds)
    correct_fraction = sum((Fraction(s['correct'], s['cells']) for s in folds), Fraction(0))
    return exact, correct_fraction


def consensus(predictions, queries):
    result = []
    for j,g in enumerate(queries):
        h,w = b.shape(g)
        grids = [ps[j] for ps in predictions]
        if not grids or any(grid is None for grid in grids): result.append(None); continue
        result.append([[grids[0][r][c] if all(grid[r][c] == grids[0][r][c] for grid in grids)
                        else b.UNKNOWN for c in range(w)] for r in range(h)])
    return result


def choose(models, queries):
    policies = {}
    for language in LANGUAGES:
        all_models = [m for m in models if m['language'] == language]
        for family in ('local', 'structural', 'all'):
            pool = [m for m in all_models if family == 'all' or m['family'] == family]
            prefix = language + '_' + family
            if not pool:
                for strategy in ('first','cv','consensus'):
                    policies[prefix+'_'+strategy] = {'selected':[], 'predictions':[None]*len(queries)}
                continue
            def score(m): return (m['cv_exact'], Fraction(*m['cv_fraction']))
            best_score = max(map(score,pool)); top = [m for m in pool if score(m) == best_score]
            for strategy, chosen in [('first',pool[:1]), ('cv',top[:1]), ('consensus',top)]:
                ps = consensus([m['predictions'] for m in chosen], queries)
                policies[prefix+'_'+strategy] = {'selected':[m['name'] for m in chosen], 'predictions':ps}
    return policies


def investigate(job):
    problem, training_only = job
    pairs = problem['train']; groups = defaultdict(list)
    for i,pair in enumerate(pairs): groups[b.signature(pair['input'])].append(i)
    eligible = len(groups) >= 2 and all(b.shape(e['input']) == b.shape(e['output']) for e in pairs)
    row = {'id':problem['id'],'split':problem['split'], 'development':problem['id']=='e88171ec',
           'eligible':eligible, 'models':[]}
    if not training_only: row['query_inputs'] = problem['query_inputs']
    if eligible:
        for rep in REPS:
            fs = [context(e['input'],rep) for e in pairs]
            if any(f is None for f in fs): continue
            query_features = None
            for language in LANGUAGES:
                table = fit(pairs,fs,language)
                if table is None: continue
                folds = []
                for indices in groups.values():
                    keep = [i for i in range(len(pairs)) if i not in indices]
                    reduced = fit([pairs[i] for i in keep],[fs[i] for i in keep],language)
                    # Equal-weight distinct inputs. Conflicting duplicate labels cannot fit.
                    i = indices[0]
                    p = apply(pairs[i]['input'],fs[i],reduced,language)
                    score = b.metrics(p,pairs[i]['output'],pairs[i]['input'])
                    score['excluded'] = indices; folds.append(score)
                exact, fraction = cv_score(folds)
                model = {'name':f'{rep[0]}:{rep[1]}:{language}', 'representation':rep,
                         'family':'local' if rep[0] in ('raw','canonical') else 'structural',
                         'language':language,'cv_exact':exact,
                         'cv_fraction':[fraction.numerator,fraction.denominator],
                         'folds':folds, 'keys':len(table)}
                if not training_only:
                    if query_features is None: query_features = [context(g,rep) for g in problem['query_inputs']]
                    model['predictions'] = [apply(g,f,table,language) for g,f in zip(problem['query_inputs'],query_features,strict=True)]
                    if model['family'] == 'structural':
                        model['role_actions'] = {str(k):[('copy' if i==COPY else i) for i in range(11) if v & (1<<i)] for k,v in table.items()}
                row['models'].append(model)
    if not training_only: row['policies'] = choose(row['models'],problem['query_inputs'])
    return row


def run(problems, out, workers, training_only):
    source = json.loads(problems.read_text())
    # Strip queries even from the worker argument for the coverage-only phase.
    if training_only: source = [{k:v for k,v in p.items() if k != 'query_inputs'} for p in source]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(investigate, [(p,training_only) for p in source]))
    filename = 'coverage.json' if training_only else 'predictions.json'
    b.write_json(out/filename, rows)
    summary = {}
    for split in ('training','evaluation'):
        rr = [r for r in rows if r['split']==split and not r['development']]
        summary[split] = {'tasks_excluding_development':len(rr),'eligible':sum(r['eligible'] for r in rr)}
        for language in LANGUAGES:
            for family in ('local','structural'):
                selected = [[m for m in r['models'] if m['family']==family and m['language']==language] for r in rr]
                summary[split][f'{family}_{language}_fits'] = sum(bool(ms) for ms in selected)
                summary[split][f'{family}_{language}_cv_all'] = sum(any(m['cv_exact']==len(m['folds']) for m in ms) for ms in selected)
    b.write_json(out/('coverage-summary.json' if training_only else 'prediction-summary.json'),summary)
    manifest = {'workers':workers,'seconds':time.perf_counter()-start,'training_only':training_only,
                'python':sys.version,'platform':platform.platform(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'problems_sha256':hashlib.sha256(problems.read_bytes()).hexdigest(),
                'output_sha256':hashlib.sha256((out/filename).read_bytes()).hexdigest(),
                'dependencies':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'run.py',BASE/'case_e88171ec.py']}}
    b.write_json(out/('coverage-run.json' if training_only else 'prediction-run.json'),manifest)
    print(json.dumps(summary,indent=2))


def score(predictions, answers, out):
    rows = json.loads(predictions.read_text()); labels = json.loads(answers.read_text())
    summary = defaultdict(Counter); cases = []
    for row in rows:
        ys = labels[row['split']+'/'+row['id']]; xs = row['query_inputs']
        case = {'id':row['id'],'split':row['split'],'development':row['development'], 'policies':{},'models':{}}
        for name,p in row['policies'].items():
            metrics = [b.metrics(g,y,x) for g,y,x in zip(p['predictions'],ys,xs,strict=True)]
            exact = all(s['exact'] for s in metrics); complete=all(s['complete'] for s in metrics)
            case['policies'][name] = {'correct':exact,'complete':complete,'selected':p['selected']}
            if not row['development']:
                c = summary[(row['split'],name)]; c['tasks']+=1; c['eligible']+=row['eligible']; c['selected']+=bool(p['selected'])
                c['complete']+=complete; c['correct']+=exact; c['wrong_complete']+=complete and not exact
                c['any_wrong_known']+=any(s['wrong'] for s in metrics)
                for s in metrics:
                    for k in ('known','wrong','changed','changed_known','changed_correct'):
                        c[k]+=s[k] or 0
        for m in row['models']:
            case['models'][m['name']] = {'correct': all(g==y for g,y in zip(m['predictions'],ys,strict=True)),
                'complete':all(g is not None and all(v != -1 for rr in g for v in rr) for g in m['predictions']),
                'cv_exact':m['cv_exact'],'folds':len(m['folds'])}
        cases.append(case)
    result = {split:{name:dict(c) for (sp,name),c in summary.items() if sp==split} for split in ('training','evaluation')}
    b.write_json(out/'scores.json',cases); b.write_json(out/'summary.json',result)
    b.write_json(out/'score-run.json', {'predictions_sha256':hashlib.sha256(predictions.read_bytes()).hexdigest(),
                                     'answers_sha256':hashlib.sha256(answers.read_bytes()).hexdigest(),
                                     'summary_sha256':hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()})
    for split,policies in result.items():
        for name,c in policies.items(): print(split,name,'complete',c['complete'],'correct',c['correct'],'wrong',c['wrong_complete'])


def main():
    p=argparse.ArgumentParser(__doc__); sub=p.add_subparsers(dest='command',required=True)
    for cmd in ('coverage','predict'):
        q=sub.add_parser(cmd); q.add_argument('--problems',type=Path,required=True); q.add_argument('--out',type=Path,required=True); q.add_argument('--workers',type=int,default=12)
    q=sub.add_parser('score'); q.add_argument('--predictions',type=Path,required=True); q.add_argument('--answers',type=Path,required=True); q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='score': score(a.predictions,a.answers,a.out)
    else: run(a.problems,a.out,a.workers,a.command=='coverage')
if __name__=='__main__': main()
