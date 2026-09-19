"""022: fixed-label-budget interventions on demonstration diversity.

No new learner, feature, operation, cost, or tie-break. Inputs and query answers
are projected to separate files before the unchanged 018/021 learners run.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('defaults021', HERE.parent/'021-default-exceptions/run.py')
e21 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e21)
KINDS = ('copy', 'literal_fill', 'literal_background')
DESIGNS = ((False, False), (True, False), (False, True), (True, True))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, obj) -> None:
    e21.write(path, obj)


def example(width, colours, height, active_row, farther, kind):
    if not 1 <= active_row < height - 1:
        raise ValueError('An active row must be strictly inside the canvas')
    if width < 3 or width % 2 == 0 or height < 3:
        raise ValueError('This protocol uses positive odd widths and height >= 3')
    if len(set(colours)) != 2 or any(c not in range(1, 10) for c in colours):
        raise ValueError('Two distinct nonzero endpoint colours are required')
    pair = e21.teacher(width, colours, height, farther, kind)
    middle = height // 2
    for field in ('input', 'output'):
        pair[field][middle], pair[field][active_row] = pair[field][active_row], pair[field][middle]
    return pair


def queries():
    # Fully cross the factors rather than coupling query height with colour.
    result = []
    for width in (5, 13, 21):
        for height in (5, 7):
            for palette in ((6, 8), (7, 9)):
                for row in (1, height//2, height-2):
                    result.append(dict(bank='crossed', width=width, height=height,
                                       palette=list(palette), active_row=row))
    for width in (5, 9, 13, 17, 21, 25, 29):
        for palette, height in (((6, 8), 5), ((7, 9), 7)):
            result.append(dict(bank='legacy', width=width, height=height,
                               palette=list(palette), active_row=height//2))
    return result


def prepare(out: Path) -> None:
    problems, answers, metadata = [], {}, {}
    for kind in KINDS:
        for farther in (False, True):
            for palette_diverse, row_diverse in DESIGNS:
                name = f'{kind}_{int(farther)}_p{int(palette_diverse)}r{int(row_diverse)}'
                opaque_id = hashlib.sha256(name.encode()).hexdigest()[:16]
                train = [example(7, (1, 2), 5, 2, farther, kind),
                         example(11, (3, 4) if palette_diverse else (1, 2),
                                 5, 1 if row_diverse else 2, farther, kind)]
                query_specs = queries()
                tests = [example(q['width'], q['palette'], q['height'],
                                 q['active_row'], farther, kind) for q in query_specs]
                problem = dict(id=opaque_id, split='controlled', train=train,
                               query_inputs=[p['input'] for p in tests])
                problems.append(problem)
                answers[opaque_id] = [p['output'] for p in tests]
                metadata[opaque_id] = dict(name=name, kind=kind, farther=farther,
                    palette_diverse=palette_diverse, row_diverse=row_diverse,
                    demonstration_grids=2, labelled_output_cells=sum(len(p['output'])*len(p['output'][0]) for p in train),
                    queries=query_specs)
    write(out/'problems.json', problems)
    write(out/'answers.json', answers)
    write(out/'metadata.json', metadata)


def infer(problem, mode):
    # Only demonstration labels and query inputs are passed to the learner.
    return e21.infer(problem, mode=mode)


def predict(problems: Path, out: Path, mode: str, workers: int) -> None:
    jobs = json.loads(problems.read_text())
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(partial(infer, mode=mode), jobs))
    write(out/'predictions.json', rows)
    dependencies = {}
    for prefix in ('014-', '015-', '016-', '017-', '018-', '021-'):
        for path in HERE.parent.glob(prefix+'*/*.py'):
            dependencies[str(path.relative_to(HERE.parent))] = sha(path)
    write(out/'prediction-run.json', dict(mode=mode, workers=workers, seed=None,
        seconds=time.perf_counter()-start, source_sha256=sha(Path(__file__)),
        dependencies=dependencies, problems_sha256=sha(problems),
        predictions_sha256=sha(out/'predictions.json')))
    print(json.dumps(dict(mode=mode, jobs=len(jobs), seconds=time.perf_counter()-start)), flush=True)


def score(predictions: Path, answers: Path, metadata: Path, out: Path) -> None:
    frozen = json.loads(predictions.with_name('prediction-run.json').read_text())
    if frozen['predictions_sha256'] != sha(predictions):
        raise ValueError('Frozen predictions changed')
    rows = json.loads(predictions.read_text())
    targets, meta = json.loads(answers.read_text()), json.loads(metadata.read_text())
    scores = []
    for row in rows:
        if row['id'] not in targets or row['id'] not in meta:
            raise ValueError('Missing target or metadata')
        m = meta[row['id']]
        policy = row['policies']['union']
        model = None
        if policy['selected']:
            arm, fs = policy['selected']
            model = next(x for x in row['families'][arm]['models'] if x['features'] == fs)
        bybank = {}
        for bank in ('crossed', 'legacy'):
            indices = [i for i, q in enumerate(m['queries']) if q['bank'] == bank]
            ss = []
            for i in indices:
                pred, target = policy['predictions'][i], targets[row['id']][i]
                complete = e21.full([pred])
                ss.append(dict(query_index=i, correct=pred == target, complete=complete,
                               wrong_complete=complete and pred != target))
            bybank[bank] = dict(grids=len(ss), correct=sum(s['correct'] for s in ss),
                wrong_complete=sum(s['wrong_complete'] for s in ss),
                incomplete=sum(not s['complete'] for s in ss), outcomes=ss)
        available = [x for family in row['families'].values() for x in family['models']]
        candidate_oracle = {bank:sum(any(x['predictions'][i] == targets[row['id']][i] for x in available)
            for i,q in enumerate(m['queries']) if q['bank'] == bank) for bank in ('crossed','legacy')}
        fixed = next((x for x in available if x['arm'] == 'guarded_order' and x['features'] == [24,27,28]), None)
        scores.append(dict(id=row['id'], **{k:v for k,v in m.items() if k!='queries'},
            selected=policy['selected'], names=None if model is None else model['names'],
            cv_exact=None if model is None else model['cv_exact'],
            cv_fraction=None if model is None else model['cv_fraction'],
            outer_pass=len(policy['outer'])>=2 and all(s['exact'] for f in policy['outer'] for s in f['scores']),
            candidate_models=len(available), fixed_endpoint_retained=fixed is not None,
            candidate_oracle=candidate_oracle, banks=bybank))
    summary = []
    for palette_diverse, row_diverse in DESIGNS:
        subset = [s for s in scores if (s['palette_diverse'],s['row_diverse']) == (palette_diverse,row_diverse)]
        summary.append(dict(palette_diverse=palette_diverse, row_diverse=row_diverse,
            teachers=len(subset), teachers_all_crossed_correct=sum(s['banks']['crossed']['correct']==36 for s in subset),
            outer_pass=sum(s['outer_pass'] for s in subset),
            banks={bank:{f:sum(s['banks'][bank][f] for s in subset) for f in ('grids','correct','wrong_complete','incomplete')} for bank in ('crossed','legacy')}))
    write(out/'scores.json', scores)
    write(out/'summary.json', summary)
    write(out/'score-run.json', dict(predictions_sha256=sha(predictions), answers_sha256=sha(answers),
        metadata_sha256=sha(metadata), summary_sha256=sha(out/'summary.json'), scores_sha256=sha(out/'scores.json')))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--out',type=Path,required=True)
    q = sub.add_parser('predict'); q.add_argument('--problems',type=Path,required=True)
    q.add_argument('--out',type=Path,required=True); q.add_argument('--mode',choices=('base','default'),required=True)
    q.add_argument('--workers',type=int,default=12)
    q = sub.add_parser('score'); q.add_argument('--predictions',type=Path,required=True)
    q.add_argument('--answers',type=Path,required=True); q.add_argument('--metadata',type=Path,required=True)
    q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='prepare': prepare(a.out)
    elif a.command=='predict': predict(a.problems,a.out,a.mode,a.workers)
    else: score(a.predictions,a.answers,a.metadata,a.out)

if __name__=='__main__': main()
