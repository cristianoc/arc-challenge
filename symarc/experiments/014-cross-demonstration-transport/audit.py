"""Post-run certificate audit; no selection policy is changed.

Independently checks frozen predictions and task scoring, stratifies known local
predictions by distinct demonstration support, and retains one collision per
failing task. Uses labels only after prediction. Cells are not iid samples.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def key(g, r, c, radius):
    h, w = len(g), len(g[0])
    return tuple(g[i][j] if 0 <= i < h and 0 <= j < w else 10
                 for i in range(r-radius, r+radius+1)
                 for j in range(c-radius, c+radius+1))


def first_difference(a, ar, ac, b, br, bc, start):
    for k in range(start+1, 31):
        if key(a, ar, ac, k) != key(b, br, bc, k):
            return k
    return None


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads((a.run/'predictions.json').read_text())
    saved = {(s['split'], s['id']): s for s in json.loads((a.run/'scores.json').read_text())}
    run = json.loads((a.run/'prediction-run.json').read_text())
    assert hashlib.sha256((a.run/'predictions.json').read_bytes()).hexdigest() == run['predictions_sha256']
    hist = defaultdict(Counter)
    observed_tasks = defaultdict(set)
    wrong_tasks = defaultdict(set)
    witnesses = []
    score_checks = 0
    for row in rows:
        task = json.loads((a.data/row['split']/(row['id']+'.json')).read_text())
        for mode in ('local', 'positioned', 'memorise', 'identity'):
            model = row['pipelines'].get(mode)
            for gated in ((False,) if mode == 'identity' else (False, True)):
                name = mode + ('_cv_gate' if gated else '')
                if mode == 'identity':
                    predictions = row['query_inputs'] if row['eligible'] else [None]*len(task['test'])
                elif model is not None and model['compatible'] and (not gated or model['cv_exact']):
                    predictions = model['predictions']
                else:
                    predictions = [None]*len(task['test'])
                complete = all(g is not None and -1 not in [v for rr in g for v in rr] for g in predictions)
                correct = all(g == q['output'] for g, q in zip(predictions, task['test'], strict=True))
                scored = saved[(row['split'], row['id'])]['policies'][name]
                assert complete == scored['complete'] and correct == scored['correct']
                score_checks += 1
        model = row['pipelines'].get('local')
        if model is None or not model['compatible']:
            continue
        radius = model['radius']
        table = defaultdict(list)
        for n, demo in enumerate(task['train']):
            g = demo['input']
            sig = json.dumps(g, separators=(',', ':'))
            for r in range(len(g)):
                for c in range(len(g[0])):
                    table[key(g, r, c, radius)].append((sig, n, r, c, demo['output'][r][c]))
        for origins in table.values():
            assert len({v[-1] for v in origins}) == 1
        failures = []
        for j, (q, pred) in enumerate(zip(task['test'], model['predictions'], strict=True)):
            g, y = q['input'], q['output']
            if len(g) != len(y) or len(g[0]) != len(y[0]):
                continue
            for r in range(len(g)):
                for c in range(len(g[0])):
                    k = key(g, r, c, radius)
                    origins = table.get(k, [])
                    expected = origins[0][-1] if origins else -1
                    assert pred[r][c] == expected
                    if not origins:
                        continue
                    groups = {}
                    for sig, n, rr, cc, label in origins:
                        groups.setdefault(sig, (n, rr, cc, label))
                    support = len(groups)
                    bucket = (row['split'], 'two_or_more' if support >= 2 else 'one')
                    counts = hist[bucket]
                    counts['known_cells'] += 1
                    counts['wrong_cells'] += expected != y[r][c]
                    counts['changed_known_cells'] += g[r][c] != y[r][c]
                    counts['changed_wrong_cells'] += g[r][c] != y[r][c] and expected != y[r][c]
                    observed_tasks[bucket].add(row['id'])
                    if expected != y[r][c]:
                        wrong_tasks[bucket].add(row['id'])
                        first = list(groups.values())
                        distance = [first_difference(task['train'][n]['input'], rr, cc, g, r, c, radius)
                                    for n, rr, cc, label in first]
                        assert all(d is not None and d > radius for d in distance)
                        failures.append(dict(task=row['id'], split=row['split'], radius=radius,
                            distinct_training_inputs=support, query_index=j, query_cell=[r,c],
                            predicted=expected, observed=y[r][c], input_colour=g[r][c],
                            patch=list(k), origins=[dict(demonstration=n,cell=[rr,cc],label=label,
                                first_distinguishing_radius=d) for (n,rr,cc,label),d in zip(first,distance)]))
        if failures:
            # Uniform deterministic choice: greatest support, then query/cell order.
            witnesses.append(min(failures, key=lambda w: (-w['distinct_training_inputs'], w['query_index'], w['query_cell'])))
    strata = []
    for bucket, counts in sorted(hist.items()):
        strata.append(dict(split=bucket[0], support=bucket[1], **dict(counts),
                           tasks_with_predictions=len(observed_tasks[bucket]), tasks_with_errors=len(wrong_tasks[bucket])))
    result = dict(score_checks=score_checks, prediction_hash_verified=True, strata=strata,
                  witness_count=len(witnesses), witnesses=witnesses)
    (a.run/'audit.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'witnesses'}, indent=2))


if __name__ == '__main__':
    main()
