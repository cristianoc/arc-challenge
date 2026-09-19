"""019 ARC1 transfer: data projection and independent scoring of frozen predictions."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('order018', HERE.parent/'018-order-guards/run.py')
e18 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e18)
DEV = e18.DEVELOPMENT


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p, x): e18.b.write_json(p, x)
def key(r): return r['split']+'/'+r['id']


def complete(ps):
    return bool(ps) and all(g is not None and len(g)>0 and len(g[0])>0
        and all(len(row)==len(g[0]) for row in g)
        and all(type(c) is int and 0<=c<=9 for row in g for c in row) for g in ps)


def outer_pass(p):
    return len(p['outer'])>=2 and all(s['exact'] for f in p['outer'] for s in f['scores'])


def prepare(data, out):
    e18.b.prepare(data, out)
    for split in ('training','evaluation'):
        for path in sorted((data/split).glob('*.json')):
            t=read(path)
            safe={'train':t['train'],'test':[{'input':q['input'],'output':[[0]]} for q in t['test']]}
            put(out/'stable'/split/path.name,safe)
    put(out/'projection.json',{'tasks':len(read(out/'problems.json')),
        'problems_sha256':sha(out/'problems.json'),'answers_sha256':sha(out/'answers.json'),
        'placeholder':'1x1 zero grid; these meaningless scores are never exported'})


def freeze(out):
    # Run this before score; both inputs must have been fully written already.
    put(out/'frozen.json',{'stable_sha256':sha(out/'stable.jsonl'),
        'relational_sha256':sha(out/'relational/predictions.json')})


def choices(s,r):
    rel=r['policies']['union']; ready=complete(rel['predictions']); gate=outer_pass(rel)
    yield 'stable',s['predictions'],'stable'
    for name,p in r['policies'].items(): yield '018_'+name,p['predictions'],'018_'+name
    yield '018_union_outer',rel['predictions'] if gate else [None]*len(rel['predictions']),'018_union_outer'
    for name,condition in [('complete_override',ready),('strict_override',ready and gate),
                           ('nofit_fallback',ready and not s['fitted'])]:
        yield name,rel['predictions'] if condition else s['predictions'],'018_union' if condition else 'stable'


def score(out,answers):
    frozen=read(out/'frozen.json')
    assert sha(out/'stable.jsonl')==frozen['stable_sha256']
    assert sha(out/'relational/predictions.json')==frozen['relational_sha256']
    stable={key(r):r for r in map(json.loads,(out/'stable.jsonl').read_text().splitlines())}
    relational=read(out/'relational/predictions.json'); targets=read(answers)
    assert set(stable)==set(targets)=={key(r) for r in relational}
    scored=[]
    for r in relational:
        k=key(r);s=stable[k];ys=targets[k];policies={}
        for name,ps,source in choices(s,r):
            assert len(ps)==len(ys)
            policies[name]={'source':source,'complete':complete(ps),'correct':ps==ys,
                'correct_grids':sum(p==y for p,y in zip(ps,ys,strict=True))}
        scored.append({'id':r['id'],'split':r['split'],'development':r['id'] in DEV,
            'stable_fitted':s['fitted'],'relational_eligible':r['eligible'],
            'relational_fitted':bool(r['policies']['union']['selected']),
            'relational_outer_pass':outer_pass(r['policies']['union']),
            'queries':len(ys),'policies':policies})
    summary={}
    for cohort in ('all','nondevelopment'):
        summary[cohort]={}
        for split in ('training','evaluation'):
            rows=[r for r in scored if r['split']==split and (cohort=='all' or not r['development'])]
            table={}
            for name in scored[0]['policies']:
                counts=Counter(tasks=len(rows),queries=sum(r['queries'] for r in rows))
                for r in rows:
                    p=r['policies'][name]
                    counts['correct_tasks']+=p['correct'];counts['complete_tasks']+=p['complete']
                    counts['complete_wrong']+=p['complete'] and not p['correct']
                    counts['correct_grids']+=p['correct_grids']
                    counts['relational_chosen']+=p['source']=='018_union'
                table[name]=dict(counts)
                table[name]['gains']=[r['id'] for r in rows if r['policies'][name]['correct'] and not r['policies']['stable']['correct']]
                table[name]['losses']=[r['id'] for r in rows if not r['policies'][name]['correct'] and r['policies']['stable']['correct']]
                table[name]['complete_errors']=[r['id'] for r in rows if r['policies'][name]['complete'] and not r['policies'][name]['correct']]
            table['diagnostic']={'stable_fit':sum(r['stable_fitted'] for r in rows),
                'relational_fit':sum(r['relational_fitted'] for r in rows),
                'outer_pass':sum(r['relational_outer_pass'] for r in rows),
                'correct_either':sum(r['policies']['stable']['correct'] or r['policies']['018_union']['correct'] for r in rows),
                'correct_both':sum(r['policies']['stable']['correct'] and r['policies']['018_union']['correct'] for r in rows)}
            summary[cohort][split]=table
    put(out/'scores.json',scored);put(out/'summary.json',summary)
    put(out/'score-run.json',{**frozen,'answers_sha256':sha(answers),
        'scores_sha256':sha(out/'scores.json'),'summary_sha256':sha(out/'summary.json')})
    for split,table in summary['all'].items():
        for policy,c in table.items():
            print(split,policy,c if policy=='diagnostic' else {k:c[k] for k in ('correct_tasks','complete_wrong','correct_grids','gains','losses')})


def main():
    ap=argparse.ArgumentParser(__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('freeze');p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('score');p.add_argument('--out',type=Path,required=True);p.add_argument('--answers',type=Path,required=True)
    a=ap.parse_args()
    if a.cmd=='prepare':prepare(a.data,a.out)
    elif a.cmd=='freeze':freeze(a.out)
    else:score(a.out,a.answers)
if __name__=='__main__':main()
