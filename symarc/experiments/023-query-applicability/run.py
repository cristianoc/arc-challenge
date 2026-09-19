"""023: query-input applicability selection over unchanged 018/021 candidates.

Features, operation inference, training search and internal predictive scores are
inherited unchanged. New policies only select among already-fitting models.
Prediction never reads query answers. Outer folds target only held-out inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import partial
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
from typing import Any

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('defaults021', HERE.parent/'021-default-exceptions/run.py')
e21 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e21)
e18 = e21.e18
POLICIES = ('baseline', 'tie_complete', 'complete_first', 'tie_consensus')
ARMS = tuple(e18.ARMS)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')


def read(path: Path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def evidence(m: dict) -> tuple:
    return (-m['cv_exact'], -Fraction(*m['cv_fraction']))


def rank(m: dict) -> tuple:
    return (*evidence(m), len(m['features']), m['cost'], ARMS.index(m['arm']), tuple(m['features']))


def complete(grids: list) -> bool:
    return bool(grids) and all(g is not None and len(g) > 0 and len(g[0]) > 0
        and all(len(row) == len(g[0]) for row in g)
        and all(type(c) is int and 0 <= c <= 9 for row in g for c in row) for g in grids)


def model_id(m: dict) -> list:
    return [m['arm'], m['features']]


def decide(models: list[dict], count: int, policy: str) -> dict:
    if policy not in POLICIES:
        raise ValueError(policy)
    if any(len(m['predictions']) != count for m in models):
        raise ValueError('Every model must predict the same query set')
    if not models:
        return {'selected': [], 'predictions': [None]*count, 'reason': 'no_fit',
                'eligible_models': 0, 'distinct_complete_answers': 0}
    ordered = sorted(models, key=rank)
    baseline = ordered[0]
    top = [m for m in ordered if evidence(m) == evidence(baseline)]
    allowed = [m for m in (ordered if policy == 'complete_first' else top)
               if complete(m['predictions'])]
    distinct = {json.dumps(m['predictions'], separators=(',', ':')) for m in allowed}
    if policy == 'tie_consensus' and len(distinct) > 1:
        return {'selected': [], 'predictions': [None]*count, 'reason': 'tied_complete_disagreement',
                'eligible_models': len(allowed), 'distinct_complete_answers': len(distinct)}
    choice = baseline if policy == 'baseline' or not allowed else allowed[0]
    return {'selected': model_id(choice), 'predictions': choice['predictions'],
            'reason': 'baseline_retained' if model_id(choice) == model_id(baseline) else 'applicable_reselection',
            'eligible_models': len(allowed), 'distinct_complete_answers': len(distinct)}


def models_at(learner, indices: tuple, grids: list, scenes: list) -> list[dict]:
    models = []
    for arm in ARMS:
        for m in learner.discover(indices, arm)[0]:
            item = {k:m[k] for k in ('arm', 'features', 'names', 'cost', 'cv_exact', 'cv_fraction')}
            item['predictions'] = [learner.predict(indices, m, g, sc) for g, sc in zip(grids, scenes, strict=True)]
            models.append(item)
    return models


def investigate(problem: dict, mode: str) -> dict:
    if mode not in ('base', 'default'):
        raise ValueError(mode)
    train, queries = problem['train'], problem['query_inputs']
    eligible = len({e18.b.signature(p['input']) for p in train}) >= 2 and all(
        e18.b.shape(p['input']) == e18.b.shape(p['output']) for p in train)
    row = {'id': problem['id'], 'split': problem['split'], 'mode': mode,
           'development': problem['id'] in e18.DEVELOPMENT, 'eligible': eligible,
           'queries': queries, 'models': [], 'policies': {}, 'outer': [], 'per_query': []}
    if not eligible:
        row['policies'] = {p:decide([], len(queries), p) for p in POLICIES}
        return row
    learner = (e18.Learner if mode == 'base' else e21.DefaultLearner)(train)
    indices = tuple(range(len(train)))
    models = models_at(learner, indices, queries, [e18.scene(g) for g in queries])
    row['models'] = models
    row['policies'] = {p:decide(models, len(queries), p) for p in POLICIES}
    # Frozen diagnostic only: no fitting/scoring change, apply each rule to one query at a time.
    for i in range(len(queries)):
        subset = [{**m, 'predictions': [m['predictions'][i]]} for m in models]
        row['per_query'].append({p:decide(subset, 1, p) for p in POLICIES})
    for excluded in learner.groups:
        reduced = tuple(i for i in indices if i not in excluded)
        inputs = [train[i]['input'] for i in excluded]
        candidate_pool = models_at(learner, reduced, inputs, [learner.scenes[i] for i in excluded])
        # Choices depend on remaining labels plus held-out INPUTS, not held-out outputs.
        decisions = {p:decide(candidate_pool, len(excluded), p) for p in POLICIES}
        scores = {p:[e18.b.metrics(g, train[i]['output'], train[i]['input'])
                     for g, i in zip(d['predictions'], excluded, strict=True)] for p, d in decisions.items()}
        row['outer'].append({'excluded': list(excluded), 'models': candidate_pool,
                             'policies': decisions, 'scores': scores})
    return row


def predict(problems: Path, out: Path, mode: str, workers: int,
            start: int = 0, limit: int | None = None) -> None:
    all_jobs = read(problems)
    jobs = all_jobs[start:] if limit is None else all_jobs[start:start+limit]
    if not jobs or workers < 1:
        raise ValueError('Empty batch or invalid worker count')
    t = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(partial(investigate, mode=mode), jobs))
    write(out/'predictions.json', rows)
    deps = {}
    for prefix in ('014-', '015-', '016-', '017-', '018-', '021-'):
        for path in HERE.parent.glob(prefix+'*/*.py'):
            deps[str(path.relative_to(HERE.parent))] = sha(path)
    write(out/'run.json', {'mode': mode, 'workers': workers, 'seed': None,
        'seconds': time.perf_counter()-t, 'platform': platform.platform(),
        'start': start, 'count': len(rows), 'total': len(all_jobs),
        'source_sha256': sha(Path(__file__)), 'dependencies': deps,
        'problems_sha256': sha(problems), 'predictions_sha256': sha(out/'predictions.json')})
    print(json.dumps({'mode': mode, 'start': start, 'count': len(rows), 'seconds': time.perf_counter()-t}), flush=True)


def merge(problems: Path, batches: list[Path], out: Path) -> None:
    manifests = sorted(
        [(read(p/'run.json'), p) for p in batches], key=lambda x:x[0]['start'])
    jobs = read(problems); rows = []; offset = 0
    first = manifests[0][0]
    for manifest, path in manifests:
        assert manifest['start'] == offset and manifest['problems_sha256'] == sha(problems)
        assert all(manifest[f] == first[f] for f in ('mode','source_sha256','dependencies','workers','total'))
        assert manifest['predictions_sha256'] == sha(path/'predictions.json')
        part = read(path/'predictions.json'); assert len(part) == manifest['count']
        rows.extend(part); offset += len(part)
    assert [(r['split'],r['id']) for r in rows] == [(p['split'],p['id']) for p in jobs]
    write(out/'predictions.json', rows)
    write(out/'run.json', {**{k:v for k,v in first.items() if k not in ('seconds','start','count','predictions_sha256')},
        'count': len(rows), 'seconds': sum(m['seconds'] for m,p in manifests),
        'batches': [m for m,p in manifests], 'predictions_sha256': sha(out/'predictions.json')})


def score(predictions: Path, answers: Path, out: Path, metadata: Path | None = None,
          stable: Path | None = None) -> None:
    frozen = read(predictions.with_name('run.json'))
    assert sha(predictions) == frozen['predictions_sha256']
    rows, ys = read(predictions), read(answers)
    meta = read(metadata) if metadata else {}
    stable_rows = {r['split']+'/'+r['id']:r for r in map(json.loads, stable.read_text().splitlines())} if stable else {}
    scored = []
    for row in rows:
        key = row['split']+'/'+row['id']
        targets = ys[key] if key in ys else ys[row['id']]
        assert len(targets) == len(row['queries'])
        policies = {}
        baseline_preds = row['policies']['baseline']['predictions']
        for policy, decision in row['policies'].items():
            preds = decision['predictions']
            detail = [e18.b.metrics(p,y,x) for p,y,x in zip(preds, targets, row['queries'], strict=True)]
            policies[policy] = {'selected': decision['selected'], 'reason': decision['reason'],
                'complete': complete(preds), 'correct': preds == targets,
                'wrong_complete': complete(preds) and preds != targets,
                'correct_grids': sum(p == y for p,y in zip(preds,targets,strict=True)),
                'grids': detail,
                'outer_pass': len(row['outer']) >= 2 and all(s['exact'] for f in row['outer'] for s in f['scores'][policy]),
                'changed_from_baseline': preds != baseline_preds,
                'batch_sensitive_grids': sum(d[policy]['predictions'][0] != preds[i] for i,d in enumerate(row['per_query']))}
            if stable:
                s = stable_rows[key]
                use = not s['fitted'] and complete(preds)
                hybrid = preds if use else s['predictions']
                policies[policy]['hybrid'] = {'use_relational': use, 'correct': hybrid == targets,
                    'complete': complete(hybrid), 'correct_grids':sum(p == y for p,y in zip(hybrid, targets, strict=True))}
        oracle = any(m['predictions'] == targets for m in row['models'])
        scored.append({'id': row['id'], 'split': row['split'], 'mode': row['mode'],
            'eligible': row['eligible'], 'development': row['development'], 'model_count':len(row['models']),
            'candidate_oracle': oracle, 'policies': policies})
    write(out/'scores.json', scored)
    summary = []
    if metadata:
        cohorts = [(f'p{int(p)}r{int(r)}_{bank}', [s for s in scored if
                    (meta[s['id']]['palette_diverse'],meta[s['id']]['row_diverse']) == (p,r)], bank)
                    for p,r in ((False,False),(True,False),(False,True),(True,True)) for bank in ('crossed','legacy')]
        for cohort, subset, bank in cohorts:
            for policy in POLICIES:
                outcomes = [s['policies'][policy]['grids'][i] for s in subset
                    for i,q in enumerate(meta[s['id']]['queries']) if q['bank'] == bank]
                summary.append({'cohort':cohort, 'policy':policy, 'grids':len(outcomes),
                    'correct':sum(s['exact'] for s in outcomes),
                    'wrong_complete':sum(s['complete'] and not s['exact'] for s in outcomes),
                    'incomplete':sum(not s['complete'] for s in outcomes)})
    else:
        for exclusion in (False, True):
            for split in ('training','evaluation'):
                subset = [s for s in scored if s['split']==split and (not exclusion or not s['development'])]
                for policy in POLICIES:
                    ps = [s['policies'][policy] for s in subset]
                    base = [s['policies']['baseline'] for s in subset]
                    result = {'cohort':split + ('_nondevelopment' if exclusion else ''),'policy':policy,
                        'tasks':len(subset),'correct':sum(p['correct'] for p in ps),
                        'wrong_complete':sum(p['wrong_complete'] for p in ps),
                        'complete':sum(p['complete'] for p in ps),'correct_grids':sum(p['correct_grids'] for p in ps),
                        'outer_pass':sum(p['outer_pass'] for p in ps),
                        'gated_correct':sum(p['correct'] and p['outer_pass'] for p in ps),
                        'gated_wrong':sum(p['wrong_complete'] and p['outer_pass'] for p in ps),
                        'gains':[s['id'] for s,p,b in zip(subset,ps,base) if p['correct'] and not b['correct']],
                        'losses':[s['id'] for s,p,b in zip(subset,ps,base) if b['correct'] and not p['correct']],
                        'errors':[s['id'] for s,p in zip(subset,ps) if p['wrong_complete']],
                        'oracle':sum(s['candidate_oracle'] for s in subset)}
                    if stable:
                        result['hybrid_correct'] = sum(p['hybrid']['correct'] for p in ps)
                        result['hybrid_correct_grids'] = sum(p['hybrid']['correct_grids'] for p in ps)
                        result['hybrid_gains'] = [s['id'] for s,p,b in zip(subset,ps,base) if p['hybrid']['correct'] and not b['hybrid']['correct']]
                        result['hybrid_losses'] = [s['id'] for s,p,b in zip(subset,ps,base) if b['hybrid']['correct'] and not p['hybrid']['correct']]
                    summary.append(result)
    write(out/'summary.json', summary)
    write(out/'score-run.json', {'predictions_sha256':sha(predictions), 'answers_sha256':sha(answers),
        'metadata_sha256':sha(metadata) if metadata else None, 'stable_sha256':sha(stable) if stable else None,
        'summary_sha256':sha(out/'summary.json'), 'scores_sha256':sha(out/'scores.json')})
    print(json.dumps(summary,indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest='command',required=True)
    a = sub.add_parser('predict'); a.add_argument('--problems',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True); a.add_argument('--mode',choices=('base','default'),required=True)
    a.add_argument('--workers',type=int,default=12); a.add_argument('--start',type=int,default=0); a.add_argument('--limit',type=int)
    a = sub.add_parser('merge'); a.add_argument('--problems',type=Path,required=True)
    a.add_argument('--batches',type=Path,nargs='+',required=True); a.add_argument('--out',type=Path,required=True)
    a = sub.add_parser('score'); a.add_argument('--predictions',type=Path,required=True)
    a.add_argument('--answers',type=Path,required=True); a.add_argument('--out',type=Path,required=True)
    a.add_argument('--metadata',type=Path); a.add_argument('--stable',type=Path)
    args = p.parse_args()
    if args.command == 'predict': predict(args.problems,args.out,args.mode,args.workers,args.start,args.limit)
    elif args.command == 'merge': merge(args.problems,args.batches,args.out)
    else: score(args.predictions,args.answers,args.out,args.metadata,args.stable)

if __name__ == '__main__':
    main()
