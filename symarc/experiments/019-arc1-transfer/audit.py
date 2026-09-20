"""Independent score/selection audit and ARC1/ARC2 overlap diagnostic."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def k(r):return r['split']+'/'+r['id']
def whole(ps):return len(ps)>0 and all(g and all(len(r)==len(g[0]) for r in g) and all(type(v)==int and v in range(10) for r in g for v in r) for g in ps)

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--arc2',type=Path,required=True)
    a=ap.parse_args();rel=read(a.run/'relational/predictions.json');stable={k(r):r for r in map(json.loads,(a.run/'stable.jsonl').read_text().splitlines())}
    scores={k(r):r for r in read(a.run/'scores.json')};ys=read(a.input/'answers.json')
    freeze=read(a.run/'frozen.json');assert sha(a.run/'stable.jsonl')==freeze['stable_sha256'];assert sha(a.run/'relational/predictions.json')==freeze['relational_sha256']
    old={(r['split']+'/'+r['task']):r for r in csv.DictReader(a.baseline.open(),delimiter='\t') if r['dataset']=='ARC1' and r['arm']=='complete'}
    assert len(old)==len(rel)==len(stable)==len(scores)==800
    decisions=0
    for r in rel:
        name=k(r);s=stable[name];target=ys[name];row=scores[name];p=r['policies']['union']
        gate=len(p['outer'])>=2 and all(z['exact'] for f in p['outer'] for z in f['scores'])
        for policy,sc in row['policies'].items():
            src='stable'
            if policy=='stable':out=s['predictions']
            elif policy=='018_union_outer':out=p['predictions'] if gate else [None]*len(target)
            elif policy.startswith('018_'):out=r['policies'][policy[4:]]['predictions']
            else:
                take=whole(p['predictions']) and (policy=='complete_override' or (policy=='strict_override' and gate) or (policy=='nofit_fallback' and not s['fitted']))
                src='018_union' if take else 'stable';out=p['predictions'] if take else s['predictions']
                assert src==sc['source']
            assert bool(whole(out))==sc['complete'];assert (out==target)==sc['correct']
            assert sum(x==y for x,y in zip(out,target,strict=True))==sc['correct_grids']
            decisions+=1
        assert (s['predictions']==target)==(old[name]['correct']=='true')
        assert sum(x==y for x,y in zip(s['predictions'],target))==int(old[name]['correct_grids'])
        assert s['fitted']==(old[name]['fit']=='true')
    oldproblems={p['id']:p for p in read(a.arc2/'input/problems.json')}
    oldanswers=read(a.arc2/'input/answers.json');oldrows={r['id']:r for r in read(a.arc2/'evidence/predictions.json')}
    newproblems={p['id']:p for p in read(a.input/'problems.json')}
    overlap=[];checks=0;unordered_checks=0
    for r in rel:
        pid=r['id'];new=newproblems[pid];oldp=oldproblems.get(pid)
        entry={'id':pid,'arc1_split':r['split'],'arc2_split':None if oldp is None else oldp['split'],'same_id':oldp is not None}
        if oldp is not None:
            oldy=oldanswers[k(oldp)];newy=ys[k(r)]
            exact=canon(new['train'])==canon(oldp['train']) and canon(new['query_inputs'])==canon(oldp['query_inputs']) and newy==oldy
            sortpairs=lambda ps:sorted(canon(e) for e in ps)
            unord=sortpairs(new['train'])==sortpairs(oldp['train']) and sorted(canon((x,y)) for x,y in zip(new['query_inputs'],newy))==sorted(canon((x,y)) for x,y in zip(oldp['query_inputs'],oldy))
            entry.update(same_ordered_content=exact,same_labelled_content=unord)
            if unord:
                o=oldrows[pid]
                perm=[oldp['query_inputs'].index(x) for x in new['query_inputs']]
                for policy,pred in r['policies'].items():
                    previous=o['policies'][policy]
                    assert pred['selected']==previous['selected'], (pid,policy,'selection')
                    assert pred['predictions']==[previous['predictions'][i] for i in perm], (pid,policy,'prediction')
                    folds={canon([oldp['train'][i]['input'] for i in f['excluded']]): f for f in previous['outer']}
                    for fold in pred['outer']:
                        before=folds[canon([new['train'][i]['input'] for i in fold['excluded']])]
                        assert fold['selected']==before['selected'] and fold['scores']==before['scores'], (pid,policy,'outer')
                unordered_checks+=1
            if exact:
                o=oldrows[pid]
                assert r['families']==o['families'] and r['policies']==o['policies']
                checks+=1
        overlap.append(entry)
    answer={'policy_task_decisions':decisions,'historical_stable_tasks_matched':800,'historical_stable_grid_scores_matched':800,
        'arc2_exact_ordered_prediction_replays':checks,'arc2_permuted_prediction_replays':unordered_checks,'overlap':overlap,'prediction_hashes_verified':True}
    answer['overlap_summary']={split:{'tasks':len([r for r in overlap if r['arc1_split']==split]),
        'same_id':sum(r['same_id'] for r in overlap if r['arc1_split']==split),
        'same_labelled_content':sum(r.get('same_labelled_content',False) for r in overlap if r['arc1_split']==split),
        'same_ordered_content':sum(r.get('same_ordered_content',False) for r in overlap if r['arc1_split']==split)} for split in ['training','evaluation']}
    (a.run/'audit.json').write_text(json.dumps(answer,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in answer.items() if k!='overlap'},indent=2))
if __name__=='__main__':main()
