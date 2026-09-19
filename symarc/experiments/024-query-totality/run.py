"""024: exact conditioning of concrete operation programs on query definedness.

Reuse 023 candidate pools/internal evidence and 018 input semantics. Totality is
checked before projecting shared-default branches. No query output is read by
prediction. Cached outer pools were constructed without their held-out labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('applicability023', HERE.parent/'023-query-applicability/run.py')
e23 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e23)
e18, e21 = e23.e18, e23.e21
POLICIES = ('baseline', 'tie_complete', 'total_ranked', 'total_tie_complete')
read, write, sha = e23.read, e23.write, e23.sha


def condition(raw: dict, valid: dict, mode: str, n_actions: int = 19) -> dict:
    """Exact finite-family conditioning; masks encode actions, never colours.

Default branches are original TRAINING-optimal defaults. They are not selected
again after observing query definedness. The returned projection is exact but
not an assertion that independent choices from that projection form a program.
"""
    if mode not in ('base', 'default'):
        raise ValueError(mode)
    full = (1 << n_actions) - 1
    if any(v <= 0 or v & ~full for v in raw.values()):
        raise ValueError('Training table must have nonempty in-vocabulary sets')
    if any(v < 0 or v & ~full for v in valid.values()):
        raise ValueError('Invalid definedness mask')
    prior, cert = e21.compress(raw, n_actions) if mode == 'default' else (dict(raw), {'defaults': []})
    unknown = sorted(set(valid) - set(raw))
    naive = {z: v & valid.get(z, full) for z, v in prior.items()}
    alive = []
    kept = {z: 0 for z in raw}
    if not unknown:
        if mode == 'base':
            if all(naive[z] for z in valid):
                kept = naive.copy()
        elif not raw:  # the unique empty table is total on an empty query domain
            alive = []
        else:
            for d in cert['defaults']:
                bit = 1 << d
                branch = {z: (bit if v & bit else v) & valid.get(z, full) for z, v in raw.items()}
                if all(branch[z] for z in valid):
                    alive.append(d)
                    for z in raw:
                        kept[z] |= branch[z]
    feasible = not unknown and (not raw or (all(kept[z] for z in valid) and any(kept.values())))
    # Empty family: no per-cell answer can be inferred from an empty hypothesis set.
    if not feasible:
        kept = {z: 0 for z in raw}
    naive_feasible = not unknown and all(naive[z] for z in valid)
    return {'feasible': feasible, 'unknown': [list(z) for z in unknown],
            'defaults': cert['defaults'], 'surviving_defaults': alive,
            'naive_feasible': naive_feasible,
            'entries': [[list(z), raw[z], prior[z], valid.get(z, full), kept[z], naive[z]] for z in sorted(raw)]}


def validities(scenes: list, features: tuple) -> dict:
    result = {}
    for scene in scenes:
        for fs, values in scene:
            z = tuple(fs[f] for f in features)
            defined = sum(1 << i for i, value in enumerate(values) if value >= 0)
            result[z] = result.get(z, e18.MASK) & defined
    return result


def conditioned_models(train: list, scenes: list, indices: tuple, grids: list,
                       query_scenes: list, old_models: list, mode: str) -> tuple[list, list]:
    """Reconstruct tables using only indices; cached predictions are checked."""
    records = []
    for i in indices:
        pair = train[i]
        for j, ((fs, values), label) in enumerate(zip(scenes[i], (c for row in pair['output'] for c in row), strict=True)):
            allowed = sum(1 << a for a, v in enumerate(values) if v == label)
            records.append((fs, values, label, (i, j // len(pair['input'][0]), j % len(pair['input'][0])), allowed))
    tables, by_features, models = [], {}, []
    for old in old_models:
        fs = tuple(old['features'])
        if fs not in by_features:
            raw = e18.s17.fit(records, fs, e18.MASK)
            if raw is None:
                raise AssertionError('A cached fitting candidate no longer fits')
            valid = validities(query_scenes, fs)
            cert = condition(raw, valid, mode)
            prior = {tuple(r[0]): r[2] for r in cert['entries']}
            kept = {tuple(r[0]): r[4] for r in cert['entries']}
            ordinary = [e18.s17.apply(g, sc, fs, prior) for g, sc in zip(grids, query_scenes, strict=True)]
            total = ([e18.s17.apply(g, sc, fs, kept) for g, sc in zip(grids, query_scenes, strict=True)]
                     if cert['feasible'] else [None] * len(grids))
            by_features[fs] = len(tables)
            tables.append({'features': list(fs), **cert, 'predictions': ordinary, 'total_predictions': total})
        table_id = by_features[fs]; table = tables[table_id]
        assert table['predictions'] == old['predictions'], 'Cached baseline prediction mismatch'
        models.append({**old, 'table_id': table_id, 'feasible': table['feasible'],
                       'total_predictions': table['total_predictions']})
    return models, tables


def decisions(models: list, count: int) -> dict:
    out = {p: e23.decide(models, count, p) for p in ('baseline', 'tie_complete')}
    survivors = [{**m, 'predictions': m['total_predictions']} for m in models if m['feasible']]
    for p, inner in (('total_ranked', 'baseline'), ('total_tie_complete', 'tie_complete')):
        out[p] = e23.decide(survivors, count, inner)
        out[p]['feasible_models'] = len(survivors)
        if not survivors:
            out[p]['reason'] = 'no_query_total_family'
    return out


def trim_cache(row: dict) -> dict:
    """Keep hypotheses and exclusion indices; strip all cached held-out scores."""
    return {'id': row['id'], 'split': row['split'], 'mode': row['mode'],
            'models': row['models'],
            'outer': [{'excluded': f['excluded'], 'models': f['models']} for f in row['outer']]}


def investigate(job: tuple, mode: str) -> dict:
    problem, saved = job
    if saved is None:
        saved = trim_cache(e23.investigate(problem, mode))
    assert (saved['id'], saved['split'], saved['mode']) == (problem['id'], problem['split'], mode)
    train, queries = problem['train'], problem['query_inputs']
    eligible = len({e18.b.signature(p['input']) for p in train}) >= 2 and all(
        e18.b.shape(p['input']) == e18.b.shape(p['output']) for p in train)
    row = {'id': problem['id'], 'split': problem['split'], 'mode': mode,
           'eligible': eligible, 'development': problem['id'] in e18.DEVELOPMENT,
           'queries': queries, 'models': [], 'tables': [], 'outer': []}
    if not eligible:
        assert not saved['models']
        row['policies'] = decisions([], len(queries)); return row
    scenes = [e18.scene(p['input']) for p in train]
    idx = tuple(range(len(train)))
    row['models'], row['tables'] = conditioned_models(train, scenes, idx, queries,
                                [e18.scene(g) for g in queries], saved['models'], mode)
    row['policies'] = decisions(row['models'], len(queries))
    expected_groups = {}
    for i, p in enumerate(train):
        expected_groups.setdefault(e18.b.signature(p['input']), []).append(i)
    assert [f['excluded'] for f in saved['outer']] == list(expected_groups.values())
    for old_fold in saved['outer']:
        excluded = old_fold['excluded']; reduced = tuple(i for i in idx if i not in excluded)
        inputs = [train[i]['input'] for i in excluded]
        pool, tables = conditioned_models(train, scenes, reduced, inputs,
                            [scenes[i] for i in excluded], old_fold['models'], mode)
        chosen = decisions(pool, len(inputs))
        # Only now are the outer target labels used for scoring.
        scores = {p: [e18.b.metrics(g, train[i]['output'], train[i]['input'])
                       for g, i in zip(d['predictions'], excluded, strict=True)] for p, d in chosen.items()}
        row['outer'].append({'excluded': excluded, 'models': pool, 'tables': tables,
                             'policies': chosen, 'scores': scores})
    return row


def predict(problems: Path, out: Path, mode: str, cached: Path | None,
            workers: int = 12, start: int = 0, limit: int | None = None) -> None:
    all_problems = read(problems)
    old = {r['split']+'/'+r['id']: trim_cache(r) for r in read(cached)} if cached else None
    selected = all_problems[start:] if limit is None else all_problems[start:start+limit]
    if not selected or workers < 1:
        raise ValueError('Empty batch or invalid worker count')
    jobs = [(p, old[p['split']+'/'+p['id']] if old is not None else None) for p in selected]
    t = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(partial(investigate, mode=mode), jobs))
    write(out/'predictions.json', rows)
    deps = {str(p.relative_to(HERE.parent)): sha(p) for prefix in ('014-', '015-', '016-', '017-', '018-', '021-', '023-')
            for p in HERE.parent.glob(prefix+'*/run.py')}
    write(out/'run.json', {'mode': mode, 'workers': workers, 'seed': None,
        'start': start, 'count': len(rows), 'total': len(all_problems),
        'seconds': time.perf_counter()-t, 'platform': platform.platform(),
        'source_sha256': sha(Path(__file__)), 'dependencies': deps,
        'cached_sha256': sha(cached) if cached else None,
        'problems_sha256': sha(problems), 'predictions_sha256': sha(out/'predictions.json')})
    print(json.dumps({'mode': mode, 'start': start, 'count': len(rows), 'seconds': time.perf_counter()-t}), flush=True)


def score(predictions: Path, answers: Path, out: Path, stable: Path | None = None,
          metadata: Path | None = None) -> None:
    manifest = read(predictions.with_name('run.json'))
    assert sha(predictions) == manifest['predictions_sha256']
    rows, ys = read(predictions), read(answers)
    ss = {r['split']+'/'+r['id']: r for r in map(json.loads, stable.read_text().splitlines())} if stable else {}
    meta = read(metadata) if metadata else None
    result = []
    for row in rows:
        key = row['split']+'/'+row['id']; target = ys.get(key, ys.get(row['id']))
        assert target is not None and len(target) == len(row['queries'])
        policies = {}
        for name, choice in row['policies'].items():
            ps = choice['predictions']
            detail = [e18.b.metrics(p, y, x) for p, y, x in zip(ps, target, row['queries'], strict=True)]
            item = {'selected': choice['selected'], 'correct': ps == target, 'complete': e23.complete(ps),
                    'wrong_complete': e23.complete(ps) and ps != target, 'grids': detail,
                    'outer_pass': len(row['outer']) >= 2 and all(g['exact'] for f in row['outer'] for g in f['scores'][name])}
            if stable:
                s = ss[key]; use = not s['fitted'] and e23.complete(ps)
                hp = ps if use else s['predictions']
                item['hybrid'] = {'correct': hp == target, 'use_relational': use,
                                 'correct_grids': sum(p == y for p, y in zip(hp, target, strict=True))}
            policies[name] = item
        result.append({'id': row['id'], 'split': row['split'], 'development': row['development'], 'policies': policies})
    summary = []
    if meta:
        for p, r in ((False, False), (True, False), (False, True), (True, True)):
            group = [s for s in result if (meta[s['id']]['palette_diverse'], meta[s['id']]['row_diverse']) == (p, r)]
            for bank in ('crossed', 'legacy'):
                for policy in POLICIES:
                    gs = [g for s in group for g, q in zip(s['policies'][policy]['grids'], meta[s['id']]['queries'], strict=True) if q['bank'] == bank]
                    summary.append({'cohort': f'p{int(p)}r{int(r)}_{bank}', 'policy': policy,
                        'correct': sum(g['exact'] for g in gs), 'wrong_complete': sum(g['complete'] and not g['exact'] for g in gs),
                        'incomplete': sum(not g['complete'] for g in gs), 'grids': len(gs)})
    else:
        for split in ('training', 'evaluation'):
            for exclude in (False, True):
                group = [s for s in result if s['split'] == split and (not exclude or not s['development'])]
                for policy in POLICIES:
                    qs = [s['policies'][policy] for s in group]; refs = [s['policies']['tie_complete'] for s in group]
                    rec = {'cohort': split + ('_nondevelopment' if exclude else ''), 'policy': policy, 'tasks': len(qs),
                        'correct': sum(q['correct'] for q in qs), 'wrong_complete': sum(q['wrong_complete'] for q in qs),
                        'correct_grids': sum(g['exact'] for q in qs for g in q['grids']),
                        'gated_correct': sum(q['correct'] and q['outer_pass'] for q in qs),
                        'gated_wrong': sum(q['wrong_complete'] and q['outer_pass'] for q in qs),
                        'gains': [s['id'] for s, q, ref in zip(group, qs, refs) if q['correct'] and not ref['correct']],
                        'losses': [s['id'] for s, q, ref in zip(group, qs, refs) if ref['correct'] and not q['correct']],
                        'errors': [s['id'] for s, q in zip(group, qs) if q['wrong_complete']]}
                    if stable:
                        rec.update(hybrid_correct=sum(q['hybrid']['correct'] for q in qs),
                            hybrid_correct_grids=sum(q['hybrid']['correct_grids'] for q in qs),
                            hybrid_gains=[s['id'] for s, q, ref in zip(group, qs, refs) if q['hybrid']['correct'] and not ref['hybrid']['correct']],
                            hybrid_losses=[s['id'] for s, q, ref in zip(group, qs, refs) if ref['hybrid']['correct'] and not q['hybrid']['correct']])
                    summary.append(rec)
    write(out/'scores.json', result); write(out/'summary.json', summary)
    write(out/'score-run.json', {'predictions_sha256': sha(predictions), 'answers_sha256': sha(answers),
        'stable_sha256': sha(stable) if stable else None, 'metadata_sha256': sha(metadata) if metadata else None,
        'scores_sha256': sha(out/'scores.json'), 'summary_sha256': sha(out/'summary.json')})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__); sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('predict'); q.add_argument('--problems', type=Path, required=True); q.add_argument('--cached', type=Path)
    q.add_argument('--mode', choices=('base', 'default'), required=True); q.add_argument('--out', type=Path, required=True)
    q.add_argument('--workers', type=int, default=12); q.add_argument('--start', type=int, default=0); q.add_argument('--limit', type=int)
    q = sub.add_parser('merge'); q.add_argument('--problems', type=Path, required=True); q.add_argument('--batches', nargs='+', type=Path, required=True); q.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('score'); q.add_argument('--predictions', type=Path, required=True); q.add_argument('--answers', type=Path, required=True); q.add_argument('--out', type=Path, required=True)
    q.add_argument('--stable', type=Path); q.add_argument('--metadata', type=Path)
    a = p.parse_args()
    if a.cmd == 'predict': predict(a.problems, a.out, a.mode, a.cached, a.workers, a.start, a.limit)
    elif a.cmd == 'merge': e23.merge(a.problems, a.batches, a.out)
    else: score(a.predictions, a.answers, a.out, a.stable, a.metadata)


if __name__ == '__main__':
    main()
