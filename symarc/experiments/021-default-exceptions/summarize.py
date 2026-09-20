"""Score frozen relational and no-fit composition policies; never select on answers."""
import argparse
import hashlib
import json
from pathlib import Path


def load(p): return json.loads(p.read_text())
def key(row): return row['split']+'/'+row['id']
def full(gs): return bool(gs) and all(g and all(v in range(10) for r in g for v in r) for g in gs)

def summarize(root):
    old = {key(r): r for r in load(root/'arc019/run/relational/predictions.json')}
    new = {key(r): r for r in load(root/'arc021/arc1/default/predictions.json')}
    stable = {key(r): r for r in map(json.loads, (root/'arc019/run/stable.jsonl').read_text().splitlines())}
    answers = load(root/'arc019/input/answers.json')
    assert set(old) == set(new) == set(stable) == set(answers)
    rows = []
    for k in old:
        row = {'id': old[k]['id'], 'split': old[k]['split'], 'development': old[k]['development'], 'policies': {}}
        for mode, rel in [('base', old[k]), ('default', new[k])]:
            p = rel['policies']['union']; grids = p['predictions']
            gate = len(p['outer']) >= 2 and all(s['exact'] for f in p['outer'] for s in f['scores'])
            use_rel = not stable[k]['fitted'] and full(grids)
            hybrid = grids if use_rel else stable[k]['predictions']
            for name, gs in [(mode+'_relational', grids), (mode+'_hybrid', hybrid),
                             (mode+'_outer_relational', grids if gate else [None]*len(grids))]:
                correct = gs == answers[k]; complete = full(gs)
                row['policies'][name] = {'correct': correct, 'complete': bool(complete),
                    'complete_wrong': bool(complete) and not correct,
                    'correct_grids': sum(x==y for x,y in zip(gs,answers[k],strict=True)),
                    'query_grids':len(gs), 'outer_pass': gate, 'selected':p['selected'],
                    'uses_relational':bool(use_rel) if name.endswith('_hybrid') else True}
        rows.append(row)
    summaries = {}
    for cohort in ('all', 'nondevelopment'):
        summaries[cohort] = {}
        for split in ('training', 'evaluation'):
            selected = [r for r in rows if r['split']==split and (cohort=='all' or not r['development'])]
            summary = {}
            for name in rows[0]['policies']:
                reference = name.replace('default_', 'base_')
                summary[name] = {f:sum(r['policies'][name][f] for r in selected)
                    for f in ('correct','complete','complete_wrong','correct_grids','query_grids','outer_pass')}
                summary[name].update(tasks=len(selected),
                    gains=[r['id'] for r in selected if r['policies'][name]['correct'] and not r['policies'][reference]['correct']],
                    losses=[r['id'] for r in selected if not r['policies'][name]['correct'] and r['policies'][reference]['correct']],
                    wrong_ids=[r['id'] for r in selected if r['policies'][name]['complete_wrong']])
            summaries[cohort][split] = summary
    out=root/'arc021/arc1'
    (out/'paired-scores.json').write_text(json.dumps(rows,sort_keys=True,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summaries,sort_keys=True,indent=2)+'\n')
    for split, s in summaries['all'].items():
        for name, data in s.items():
            print(split,name,{k:v for k,v in data.items() if k != 'wrong_ids'})
        print('Relational errors',s['default_relational']['wrong_ids'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    summarize(p.parse_args().root)
