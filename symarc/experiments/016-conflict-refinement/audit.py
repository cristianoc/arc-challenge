"""Independent post-run certificate and prediction audit, using explicit sets.

The feature reference uses union-find rather than the learner's graph traversal.
Subset sufficiency is exhaustively checked, not conflict-branch searched. This
auditor does not propose or change a prediction policy.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import time


def reference_features(g):
    h,w=len(g),len(g[0]); n=h*w; flat=sum(g,[]); parent=list(range(n))
    def root(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=root(i),root(j)
        if a!=b: parent[b]=a
    near=[[] for _ in range(n)]
    border=[int(i//w in (0,h-1) or i%w in (0,w-1)) for i in range(n)]
    for i in range(n):
        r,c=divmod(i,w)
        for rr,cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if 0<=rr<h and 0<=cc<w and g[rr][cc]==flat[i]: near[i].append(rr*w+cc)
        if r and g[r-1][c]==flat[i]: union(i,i-w)
        if c and g[r][c-1]==flat[i]: union(i,i-1)
    comps=defaultdict(list); peers=defaultdict(list)
    for i in range(n): comps[root(i)].append(i);peers[flat[i]].append(i)
    def agg(ids):
        ds=[len(near[j]) for j in ids]
        return (len(ids),int(any(border[j] for j in ids)),min(ds),max(ds))
    ca={r:agg(ids) for r,ids in comps.items()};pa={c:agg(ids) for c,ids in peers.items()}
    return [(flat[i],border[i],len(near[i]),*agg([i]+near[i]),*ca[root(i)],*pa[flat[i]]) for i in range(n)]


def ops(x,y):return {y}|({'copy'} if x==y else set())


def fit(records,features):
    table={}
    for f,x,y in records:
        k=tuple(f[i] for i in features)
        table[k]=table.get(k,set(range(10))|{'copy'})&ops(x,y)
        if not table[k]:return None
    return table


def apply(g,fs,features,table):
    if table is None:return None
    vals=[]
    for f,x in zip(fs,sum(g,[]),strict=True):
        k=tuple(f[i] for i in features)
        a=table.get(k,set(range(10))|{'copy'})
        ys={x if op=='copy' else op for op in a}
        vals.append(next(iter(ys)) if len(ys)==1 else -1)
    w=len(g[0]);return [vals[i:i+w] for i in range(0,len(vals),w)]


def consensus(ps):
    if not ps or any(p is None for p in ps):return None
    return [[ps[0][r][c] if len({p[r][c] for p in ps})==1 else -1
             for c in range(len(ps[0][0]))] for r in range(len(ps[0]))]


def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('--problems',type=Path,required=True);a.add_argument('--answers',type=Path,required=True);a.add_argument('--run',type=Path,required=True)
    args=a.parse_args();start=time.perf_counter()
    problems=json.loads(args.problems.read_text());answers=json.loads(args.answers.read_text())
    predictions=json.loads((args.run/'predictions.json').read_text());scores=json.loads((args.run/'scores.json').read_text())
    manifest=json.loads((args.run/'prediction-run.json').read_text())
    assert hashlib.sha256((args.run/'predictions.json').read_bytes()).hexdigest()==manifest['predictions_sha256']
    all_vocab=tuple(range(15));no_reach=tuple(i for i in all_vocab if not 7<=i<=10)
    families={'atoms':(all_vocab,1),'refined':(all_vocab,3),'no_reach':(no_reach,3)}
    checks=Counter();failure_modes=defaultdict(Counter);task_rows=[]
    for problem,row,score in zip(problems,predictions,scores,strict=True):
        assert problem['id']==row['id']==score['id']
        ys=answers[row['split']+'/'+row['id']]
        task={'id':row['id'],'split':row['split'],'development':row['development'],'eligible':row['eligible']}
        if row['eligible']:
            pairs=problem['train'];tfs=[reference_features(p['input']) for p in pairs];qfs=[reference_features(g) for g in problem['query_inputs']]
            records=[[(f,x,y) for f,x,y in zip(fs,sum(p['input'],[]),sum(p['output'],[]),strict=True)] for fs,p in zip(tfs,pairs,strict=True)]
            allrecords=list({r for rs in records for r in rs})
            tables={}
            for family,details in row['families'].items():
                vocab,maxf=families[family]
                # A full-vocabulary collision rules out every subset directly.
                if details['diagnostics']['status']=='vocabulary_conflict':
                    a,b=details['diagnostics']['witness'];aa,bb=[] ,[]
                    for witness in [a,b]:
                        d,r,c=witness['origin'];f=tfs[d][r*len(pairs[d]['input'][0])+c]
                        assert list(f)==witness['features']
                        assert pairs[d]['input'][r][c]==witness['input'] and pairs[d]['output'][r][c]==witness['output']
                    assert all(a['features'][i]==b['features'][i] for i in vocab)
                    assert not ops(a['input'],a['output'])&ops(b['input'],b['output'])
                    assert details['models']==[];checks['vocabulary_conflict_certificates']+=1
                    continue
                sufficient=[]
                for n in range(maxf+1):
                    for f in itertools.combinations(vocab,n):
                        if any(set(v).issubset(f) for v in sufficient):continue
                        table=fit(allrecords,f);checks['exhaustive_subset_checks']+=1
                        if table is not None:sufficient.append(f);tables[f]=table
                assert set(sufficient)=={tuple(m['features']) for m in details['models']}
                checks['complete_model_pools']+=1
                for model in details['models']:
                    f=tuple(model['features']);table=tables[f]
                    for reason in model['necessity']:
                        j=reason['necessary_feature'];a,b=reason['witness']
                        for witness in [a,b]:
                            d,r,c=witness['origin'];fv=tfs[d][r*len(pairs[d]['input'][0])+c]
                            assert list(fv)==witness['features']
                            assert pairs[d]['input'][r][c]==witness['input'] and pairs[d]['output'][r][c]==witness['output']
                        assert j in f and a['features'][j]!=b['features'][j]
                        assert all(a['features'][k]==b['features'][k] for k in f if k!=j)
                        assert not ops(a['input'],a['output'])&ops(b['input'],b['output'])
                        checks['necessity_certificates']+=1
                    # Internal validation must infer the action table only from retained grids.
                    val_exact=0;val_fraction=Fraction(0)
                    for fold in model['validation']:
                        reduced=[rec for i,rs in enumerate(records) if i not in fold['excluded'] for rec in rs]
                        tt=fit(reduced,f);oks=[]
                        for i,saved in zip(fold['excluded'],fold['scores'],strict=True):
                            p=apply(pairs[i]['input'],tfs[i],f,tt);target=pairs[i]['output']
                            correct=sum(v==t for pr,tr in zip(p,target) for v,t in zip(pr,tr)) if p is not None else 0
                            assert saved['exact']==(p==target) and saved['correct']==correct
                            oks.append(p==target);checks['internal_fold_predictions']+=1
                        val_exact+=all(oks)
                        i=fold['excluded'][0];s=fold['scores'][0];val_fraction+=Fraction(s['correct'],s['cells'])
                    assert model['cv_exact']==val_exact and model['cv_fraction']==[val_fraction.numerator,val_fraction.denominator]
                    assert model['predictions']==[apply(g,fs,f,table) for g,fs in zip(problem['query_inputs'],qfs,strict=True)]
                    checks['candidate_query_predictions']+=len(qfs)
            for policy,p in row['policies'].items():
                expected=[]
                if policy=='identity': expected=problem['query_inputs']
                elif policy=='coordinates':
                    rs=[((len(pair['input']),len(pair['input'][0]),r,c),pair['input'][r][c],pair['output'][r][c])
                        for pair in pairs for r in range(len(pair['input'])) for c in range(len(pair['input'][0]))]
                    tt=fit(rs,(0,1,2,3))
                    expected=[apply(g,[(len(g),len(g[0]),r,c) for r in range(len(g)) for c in range(len(g[0]))],(0,1,2,3),tt) for g in problem['query_inputs']]
                else:
                    for g,fs in zip(problem['query_inputs'],qfs,strict=True):
                        expected.append(consensus([apply(g,fs,tuple(f),fit(allrecords,tuple(f))) for f in p['selected']]))
                    for fold in p['outer']:
                        reduced=[rec for i,rs in enumerate(records) if i not in fold['excluded'] for rec in rs]
                        for i,saved in zip(fold['excluded'],fold['scores'],strict=True):
                            expected_outer=consensus([apply(pairs[i]['input'],tfs[i],tuple(f),fit(reduced,tuple(f))) for f in fold['selected']])
                            assert saved['exact']==(expected_outer==pairs[i]['output'])
                            checks['outer_fold_predictions']+=1
                        # No duplicate input may be present in the retained set.
                        for i in fold['excluded']:
                            assert all(pairs[i]['input']!=p2['input'] for j,p2 in enumerate(pairs) if j not in fold['excluded'])
                assert p['predictions']==expected;checks['policy_prediction_checks']+=1
            # Describe unresolved query cells for the prespecified refined_cv choice.
            selected=row['policies']['refined_cv']['selected']
            mode=Counter()
            if not selected:mode[row['families']['refined']['diagnostics']['status']]+=1
            else:
                f=tuple(selected[0]);table=fit(allrecords,f)
                for g,fs in zip(problem['query_inputs'],qfs,strict=True):
                    for vec,x in zip(fs,sum(g,[]),strict=True):
                        key=tuple(vec[i] for i in f)
                        if key not in table:mode['unknown_key_cells']+=1
                        elif len({x if a=='copy' else a for a in table[key]})!=1:mode['ambiguous_action_cells']+=1
                        else:mode['determined_cells']+=1
            task['refined_query_diagnostic']=dict(mode)
            if not row['development']:failure_modes[row['split']].update(mode)
        for policy,p in row['policies'].items():
            correct=all(pred==y for pred,y in zip(p['predictions'],ys,strict=True))
            complete=all(g is not None and all(v!=-1 for rr in g for v in rr) for g in p['predictions'])
            assert score['policies'][policy]['correct']==correct and score['policies'][policy]['complete']==complete
            checks['policy_task_scores']+=1
        task_rows.append(task)
    result={'checks':dict(checks),'failure_modes':{s:dict(c) for s,c in failure_modes.items()},
            'seconds':time.perf_counter()-start,'prediction_hash_verified':True,
            'method':'union-find feature reference, explicit action sets, exhaustive bounded subsets; outer predictions reconstructed from retained labels, not independent outer selector reimplementation'}
    (args.run/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (args.run/'failure-modes.json').write_text(json.dumps(task_rows,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
