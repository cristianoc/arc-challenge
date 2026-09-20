"""021: observed-context defaults plus exceptions, with symmetric operation costs.

Reuse 018's search, selection, nested validation and prediction semantics. This
module substitutes only its table learner within a single process. Worker
parallelism is process-based; the temporary class substitution is not thread-safe.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('order018', HERE.parent/'018-order-guards/run.py')
e18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e18)
BASE = e18.Learner
N_ACTIONS = len(e18.s17.ACTIONS)


def compress(table: dict | None, n_actions: int = N_ACTIONS):
    """All optimal defaults and per-key projections of all minimal programs.

Cost is the number of distinct observed keys requiring an exception. Every
operation has the same cost; no cell counts or arbitrary choice among ties.
Unseen keys remain undefined: the default is guarded by the observed domain.
"""
    if table is None:
        return None, {'defaults': [], 'cost': None}
    if not table:
        return {}, {'defaults': [], 'cost': 0}
    if n_actions <= 0 or any(v <= 0 or v >> n_actions for v in table.values()):
        raise ValueError('Expected nonempty action sets within the vocabulary')
    costs = [sum(not (v & (1 << action)) for v in table.values()) for action in range(n_actions)]
    best = min(costs)
    defaults = [action for action, cost in enumerate(costs) if cost == best]
    out = {}
    for key, allowed in table.items():
        possible = 0
        for default in defaults:
            bit = 1 << default
            possible |= bit if allowed & bit else allowed
        out[key] = possible
    return out, {'defaults': defaults, 'cost': best, 'all_costs': costs}


class DefaultLearner(BASE):
    def __init__(self, pairs):
        super().__init__(pairs)
        self.default_records = {}

    def table(self, indices, selected, arm):
        cache_key = (indices, selected)
        if cache_key not in self.table_cache:
            raw = e18.s17.fit([r for i in indices for r in self.records[i]], selected, e18.MASK)
            restricted, certificate = compress(raw)
            self.table_cache[cache_key] = restricted
            self.default_records[cache_key] = (raw, certificate)
        return self.table_cache[cache_key]


def infer(problem, mode='default'):
    """Invoke the unchanged experiment driver with the selected table learner."""
    if mode not in ('base', 'default'):
        raise ValueError(mode)
    if e18.Learner is not BASE:
        raise RuntimeError('Do not call the substitution driver concurrently in one process')
    instances = []

    class Recorded(DefaultLearner if mode == 'default' else BASE):
        def __init__(self, pairs):
            super().__init__(pairs)
            instances.append(self)

    e18.Learner = Recorded
    try:
        row = e18.investigate(problem)
    finally:
        e18.Learner = BASE
    if mode == 'base':
        return row
    row['default_tables'] = []
    if instances:
        learner = instances[0]
        for (indices, selected), (raw, certificate) in sorted(learner.default_records.items()):
            restricted = learner.table_cache[(indices, selected)]
            row['default_tables'].append({
                'indices': list(indices), 'features': list(selected), **certificate,
                'entries': None if raw is None else [
                    {'key': list(k), 'allowed': v, 'retained': restricted[k]} for k, v in sorted(raw.items())]
            })
    return row


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    e18.b.write_json(path, obj)


def teacher(width, colours, height, farther, kind):
    pair = e18.teacher_grid(width, colours, height, farther)
    mid = height // 2
    if kind == 'literal_fill':
        for c in range(1, width - 1):
            pair['output'][mid][c] = 5 if c == width-1-c else (6, 8)[int((c > width-1-c) != farther)]
    elif kind == 'literal_background':
        for r in range(height):
            if r != mid:
                pair['output'][r] = [6] * width
    elif kind != 'copy':
        raise ValueError(kind)
    return pair


def prepare_control(out):
    problems, answers = [], {}
    for kind in ('copy', 'literal_fill', 'literal_background'):
        for farther in (False, True):
            name = kind + ('_far' if farther else '_near')
            train = [teacher(7, (1, 2), 5, farther, kind), teacher(11, (3, 4), 5, farther, kind)]
            tests = [teacher(w, palette, height, farther, kind)
                     for w in (5, 9, 13, 17, 21, 25, 29)
                     for palette, height in (((6, 8), 5), ((7, 9), 7))]
            problems.append({'id': name, 'split': 'controlled', 'train': train,
                             'query_inputs': [q['input'] for q in tests]})
            answers['controlled/' + name] = [q['output'] for q in tests]
    write(out/'problems.json', problems)
    write(out/'answers.json', answers)


def predict(problems, out, mode, workers):
    jobs = json.loads(problems.read_text())
    start = time.perf_counter()
    # Partial supplies mode without changing any task or table semantics.
    from functools import partial
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(partial(infer, mode=mode), jobs))
    write(out/'predictions.json', rows)
    sources = {str(p.relative_to(HERE.parent)): sha(p) for p in HERE.parent.glob('*/run.py')
               if p.parent.name in ('014-cross-demonstration-transport', '015-structural-roles',
                                    '016-conflict-refinement', '017-split-or-share',
                                    '018-order-guards', HERE.name)}
    write(out/'run.json', {'mode': mode, 'workers': workers, 'seconds': time.perf_counter()-start,
                          'seed': None, 'platform': platform.platform(), 'sources': sources,
                          'problems_sha256': sha(problems), 'predictions_sha256': sha(out/'predictions.json')})


def full(grids):
    return bool(grids) and all(g is not None and bool(g) and bool(g[0])
        and all(len(r) == len(g[0]) for r in g) and all(type(v) is int and 0 <= v <= 9 for r in g for v in r)
        for g in grids)


def score(predictions, answers, out):
    # Predictions are already frozen on disk before this command is invoked.
    rows = json.loads(predictions.read_text())
    ys = json.loads(answers.read_text())
    result = []
    for row in rows:
        target = ys[row['split']+'/'+row['id']]
        policies = {}
        for name, policy in row['policies'].items():
            grids = policy['predictions']
            if len(grids) != len(target):
                raise ValueError('Query count differs')
            is_complete = full(grids)
            policies[name] = {'selected': policy['selected'], 'complete': is_complete,
                'correct': grids == target, 'wrong_complete': is_complete and grids != target,
                'correct_grids': sum(g == y for g, y in zip(grids, target, strict=True)),
                'complete_grids': sum(full([g]) for g in grids), 'query_grids': len(target),
                'outer_pass': len(policy['outer']) >= 2 and all(s['exact'] for f in policy['outer'] for s in f['scores'])}
        result.append({'id': row['id'], 'split': row['split'], 'eligible': row['eligible'],
                       'development': row['development'], 'policies': policies})
    write(out/'scores.json', result)
    write(out/'score-run.json', {'predictions_sha256': sha(predictions), 'answers_sha256': sha(answers),
                                 'scores_sha256': sha(out/'scores.json')})
    for split in sorted({r['split'] for r in result}):
        sr = [r for r in result if r['split'] == split]
        print(split, 'tasks', len(sr))
        for policy in e18.POLICIES:
            print(policy, {field: sum(r['policies'][policy][field] for r in sr)
                          for field in ('correct', 'wrong_complete', 'complete_grids', 'correct_grids', 'outer_pass')})


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare-control'); q.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('predict')
    q.add_argument('--problems', type=Path, required=True); q.add_argument('--out', type=Path, required=True)
    q.add_argument('--mode', choices=('base', 'default'), default='default'); q.add_argument('--workers', type=int, default=12)
    q = sub.add_parser('score')
    q.add_argument('--predictions', type=Path, required=True); q.add_argument('--answers', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'prepare-control': prepare_control(a.out)
    elif a.command == 'predict': predict(a.problems, a.out, a.mode, a.workers)
    else: score(a.predictions, a.answers, a.out)


if __name__ == '__main__':
    main()
