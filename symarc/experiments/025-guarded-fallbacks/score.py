"""025: score already-persisted predictions; this command alone reads query labels."""
import argparse
import importlib.util
from pathlib import Path
import json

spec=importlib.util.spec_from_file_location('guard025',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def score(predictions,answers,out,stable=None,metadata=None):
    manifest=m.read(predictions.with_name('run.json'))
    assert m.sha(predictions)==manifest['predictions_sha256']
    rows=m.read(predictions);ys=m.read(answers)
    ss={r['split']+'/'+r['id']:r for r in map(json.loads,stable.read_text().splitlines())} if stable else {}
    meta=m.read(metadata) if metadata else {}
    result=[]
    for row in rows:
        key=row['split']+'/'+row['id'];target=ys.get(key,ys.get(row['id']))
        assert target is not None and len(target)==len(row['queries'])
        policies={}
        for name,choice in row['policies'].items():
            ps=choice['predictions']
            item={'selected':choice['selected'],'correct':ps==target,'complete':m.e23.complete(ps),
                  'wrong_complete':m.e23.complete(ps) and ps!=target,
                  'grids':[m.e18.b.metrics(p,y,x) for p,y,x in zip(ps,target,row['queries'],strict=True)],
                  'outer_pass':len(row['outer'])>=2 and all(g['exact'] for f in row['outer'] for g in f['scores'][name])}
            if stable:
                s=ss[key];use=not s['fitted'] and m.e23.complete(ps)
                hp=ps if use else s['predictions']
                item['hybrid']={'correct':hp==target,'use_relational':use,
                    'correct_grids':sum(g==y for g,y in zip(hp,target,strict=True))}
            policies[name]=item
        result.append({'id':row['id'],'split':row['split'],'development':row['development'],'policies':policies})
    summary=[]
    if meta:
        for p,r in ((0,0),(1,0),(0,1),(1,1)):
            group=[s for s in result if (meta[s['id']]['palette_diverse'],meta[s['id']]['row_diverse'])==(p,r)]
            for bank in ('crossed','legacy'):
                for policy in m.POLICIES:
                    gs=[g for s in group for g,q in zip(s['policies'][policy]['grids'],meta[s['id']]['queries'],strict=True) if q['bank']==bank]
                    summary.append({'cohort':f'p{p}r{r}_{bank}','policy':policy,'grids':len(gs),
                        'correct':sum(g['exact'] for g in gs),'wrong_complete':sum(g['complete'] and not g['exact'] for g in gs),
                        'incomplete':sum(not g['complete'] for g in gs)})
    else:
        for split in ('training','evaluation'):
            for exclude in (False,True):
                group=[s for s in result if s['split']==split and (not exclude or not s['development'])]
                for policy in m.POLICIES:
                    qs=[s['policies'][policy] for s in group];bs=[s['policies']['base'] for s in group]
                    rec={'cohort':split+('_nondevelopment' if exclude else ''),'policy':policy,'tasks':len(qs),
                        'correct':sum(q['correct'] for q in qs),'wrong_complete':sum(q['wrong_complete'] for q in qs),
                        'incomplete':sum(not q['complete'] for q in qs),'correct_grids':sum(g['exact'] for q in qs for g in q['grids']),
                        'gated_correct':sum(q['correct'] and q['outer_pass'] for q in qs),
                        'gated_wrong':sum(q['wrong_complete'] and q['outer_pass'] for q in qs),
                        'gains':[s['id'] for s,q,b in zip(group,qs,bs) if q['correct'] and not b['correct']],
                        'losses':[s['id'] for s,q,b in zip(group,qs,bs) if b['correct'] and not q['correct']],
                        'errors':[s['id'] for s,q in zip(group,qs) if q['wrong_complete']]}
                    if stable:
                        rec.update(hybrid_correct=sum(q['hybrid']['correct'] for q in qs),
                            hybrid_correct_grids=sum(q['hybrid']['correct_grids'] for q in qs),
                            hybrid_gains=[s['id'] for s,q,b in zip(group,qs,bs) if q['hybrid']['correct'] and not b['hybrid']['correct']],
                            hybrid_losses=[s['id'] for s,q,b in zip(group,qs,bs) if b['hybrid']['correct'] and not q['hybrid']['correct']])
                    summary.append(rec)
    m.write(out/'scores.json',result);m.write(out/'summary.json',summary)
    m.write(out/'score-run.json',{'predictions_sha256':m.sha(predictions),'answers_sha256':m.sha(answers),
        'stable_sha256':m.sha(stable) if stable else None,'scores_sha256':m.sha(out/'scores.json'),
        'summary_sha256':m.sha(out/'summary.json')})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for a in ('predictions','answers','out'):p.add_argument('--'+a,type=Path,required=True)
    for a in ('stable','metadata'):p.add_argument('--'+a,type=Path)
    a=p.parse_args();score(a.predictions,a.answers,a.out,a.stable,a.metadata)
