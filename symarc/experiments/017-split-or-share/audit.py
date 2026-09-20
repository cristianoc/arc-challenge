"""Independent post-run audit: sets and exhaustive subsets, no policy changes."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit016',HERE.parent/'016-conflict-refinement/audit.py')
a16=importlib.util.module_from_spec(spec);spec.loader.exec_module(a16)
ARMS={'base':(tuple(range(15)),11),'split':(tuple(range(19)),11),
      'share':(tuple(range(15)),19),'joint':(tuple(range(19)),19)}


def scene(g):
    """Reference evaluates directional array slices, not the learner's walk."""
    h,w=len(g),len(g[0]);old=a16.reference_features(g);result=[]
    for r in range(h):
        for c in range(w):
            x=g[r][c]
            lines=([g[k][c] for k in range(r-1,-1,-1)],[g[k][c] for k in range(r+1,h)],
                   list(reversed(g[r][:c])),g[r][c+1:])
            counts=[];after=[]
            for line in lines:
                same=len(list(itertools.takewhile(lambda y:y==x,line)))
                counts.append(same+1);after.append(line[same] if same<len(line) else -1)
            values=tuple(range(10))+(x,)+tuple(line[0] if line else -1 for line in lines)+tuple(after)
            result.append((tuple(old[r*w+c])+tuple(counts),values))
    return result


def allowed(record,n):return {i for i in range(n) if record[1][i]==record[2]}

def fit(records,selected,n):
    table={}
    for record in records:
        k=tuple(record[0][i] for i in selected);a=allowed(record,n)
        table[k]=table[k]&a if k in table else a
        if not table[k]:return None
    return table


def apply(g,sc,selected,table):
    if table is None:return None
    ys=[]
    for f,values in sc:
        key=tuple(f[i] for i in selected);acts=table.get(key,set());vals={values[i] for i in acts}
        ys.append(next(iter(vals)) if len(vals)==1 and -1 not in vals else -1)
    w=len(g[0]);return [ys[i:i+w] for i in range(0,len(ys),w)]


def verify_core(witnesses,selected,n,pairs,scenes):
    rs=[]
    for x in witnesses:
        d,r,c=x['origin'];index=r*len(pairs[d]['input'][0])+c;fs,vs=scenes[d][index];y=pairs[d]['output'][r][c]
        assert x['features']==list(fs) and x['action_values']==list(vs)
        assert x['input']==pairs[d]['input'][r][c] and x['output']==y
        assert x['allowed']==sum(1<<i for i in range(19) if vs[i]==y)
        rs.append((fs,vs,y))
    assert len({tuple(r[0][i] for i in selected) for r in rs})==1
    sets=[allowed(r,n) for r in rs]
    assert not set.intersection(*sets)
    assert all(set.intersection(set(range(n)),*(sets[:i]+sets[i+1:])) for i in range(len(sets)))
    return len(sets)


def check_score(pred,target,saved):
    exact=pred==target
    correct=0
    if pred is not None and len(pred)==len(target) and all(len(a)==len(b) for a,b in zip(pred,target)):
        correct=sum(x==y for a,b in zip(pred,target) for x,y in zip(a,b) if x!=-1)
    assert saved['exact']==exact and saved['correct']==correct


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--problems',type=Path,required=True);p.add_argument('--answers',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args();start=time.perf_counter()
    problems=json.loads(a.problems.read_text());rows=json.loads((a.run/'predictions.json').read_text())
    old=json.loads(a.baseline.read_text());scores=json.loads((a.run/'scores.json').read_text());answers=json.loads(a.answers.read_text())
    manifest=json.loads((a.run/'prediction-run.json').read_text());checks=Counter();core_sizes=Counter()
    assert hashlib.sha256((a.run/'predictions.json').read_bytes()).hexdigest()==manifest['predictions_sha256']
    # Input-only/sample eligibility uses predicted training fit, never query labels.
    sample=sorted((r['split']+'/'+r['id'] for r in rows if any(d['models'] for d in r['families'].values())),
                  key=lambda x:hashlib.sha256(x.encode()).digest())[:12]
    for problem,row,previous,scored in zip(problems,rows,old,scores,strict=True):
        assert problem['id']==row['id']==previous['id']==scored['id']
        assert row['eligible']==previous['eligible'];key=row['split']+'/'+row['id'];labels=answers[key]
        checks['baseline_task_eligibility']+=1
        if row['eligible']:
            pairs=problem['train'];ts=[scene(e['input']) for e in pairs];qs=[scene(g) for g in row['query_inputs']]
            records=[[(f,v,y) for (f,v),y in zip(scene_,sum(e['output'],[]),strict=True)] for scene_,e in zip(ts,pairs,strict=True)]
            indices=tuple(range(len(pairs)));table_cache={}
            def table(inds,fs,arm):
                ck=(inds,fs,ARMS[arm][1])
                if ck not in table_cache:table_cache[ck]=fit([rec for i in inds for rec in records[i]],fs,ARMS[arm][1])
                return table_cache[ck]
            oldmodels={tuple(m['features']):m for m in previous['families']['refined']['models']}
            current={tuple(m['features']):m for m in row['families']['base']['models']}
            assert oldmodels.keys()==current.keys();checks['baseline_pools']+=1
            for fs,m in current.items():
                assert m['predictions']==oldmodels[fs]['predictions'] and m['validation']==oldmodels[fs]['validation']
                checks['baseline_candidate_models']+=1
            assert previous['policies']['refined_cv']['predictions']==row['policies']['base']['predictions']
            for x,y in zip(previous['policies']['refined_cv']['outer'],row['policies']['base']['outer'],strict=True):
                assert x['scores']==y['scores'] and x['excluded']==y['excluded']
                assert (x['selected'][0] if x['selected'] else [])==(y['selected'][1] if y['selected'] else [])
                checks['baseline_outer_folds']+=1
            for arm,details in row['families'].items():
                vocab,n=ARMS[arm];diagnostics=details['diagnostics'];models=details['models']
                if 'witness' in diagnostics:
                    size=verify_core(diagnostics['witness'],vocab,n,pairs,ts);core_sizes[str(size)]+=1
                    assert not models;checks['full_vocabulary_certificates']+=1
                elif key in sample:
                    # Enumerate every subset up to three, skipping supersets of fits.
                    minimal=[];unique=list(dict.fromkeys((r[0],tuple(sorted(allowed(r,n)))) for rr in records for r in rr))
                    for size in range(4):
                        for fs in itertools.combinations(vocab,size):
                            if any(set(x).issubset(fs) for x in minimal):continue
                            t={};ok=True
                            for f,acts in unique:
                                k=tuple(f[i] for i in fs);v=set(acts);t[k]=t[k]&v if k in t else v
                                if not t[k]:ok=False;break
                            checks['exhaustive_subsets']+=1
                            if ok:minimal.append(fs)
                    assert set(minimal)=={tuple(m['features']) for m in models},(key,arm)
                    checks['exhaustive_candidate_pools']+=1
                for model in models:
                    fs=tuple(model['features']);t=table(indices,fs,arm);assert t is not None
                    for why in model['necessity']:
                        f=why['necessary_feature'];others=tuple(i for i in fs if i!=f)
                        size=verify_core(why['witness'],others,n,pairs,ts);core_sizes[str(size)]+=1
                        assert len({w['features'][f] for w in why['witness']})>1
                        checks['necessity_certificates']+=1
                    cv_exact=0;cv_fraction=Fraction(0)
                    for fold in model['validation']:
                        inds=tuple(i for i in indices if i not in fold['excluded']);tt=table(inds,fs,arm);oks=[]
                        for i,s in zip(fold['excluded'],fold['scores'],strict=True):
                            pred=apply(pairs[i]['input'],ts[i],fs,tt);check_score(pred,pairs[i]['output'],s);oks.append(pred==pairs[i]['output'])
                            checks['internal_fold_predictions']+=1
                        cv_exact+=all(oks);s=fold['scores'][0];cv_fraction+=Fraction(s['correct'],s['cells'])
                    assert cv_exact==model['cv_exact'] and model['cv_fraction']==[cv_fraction.numerator,cv_fraction.denominator]
                    assert model['predictions']==[apply(g,x,fs,t) for g,x in zip(row['query_inputs'],qs,strict=True)]
                    checks['candidate_query_grids']+=len(qs)
            for policy,detail in row['policies'].items():
                selected=detail['selected']
                if selected:
                    arm,fs=selected;fs=tuple(fs)
                    assert detail['predictions']==[apply(g,x,fs,table(indices,fs,arm)) for g,x in zip(row['query_inputs'],qs,strict=True)]
                else:assert all(p is None for p in detail['predictions'])
                for fold in detail['outer']:
                    selected=fold['selected'];inds=tuple(i for i in indices if i not in fold['excluded'])
                    for i,s in zip(fold['excluded'],fold['scores'],strict=True):
                        if selected:
                            arm,fs=selected;fs=tuple(fs);pred=apply(pairs[i]['input'],ts[i],fs,table(inds,fs,arm))
                        else:pred=None
                        check_score(pred,pairs[i]['output'],s);checks['outer_fold_predictions']+=1
        for policy,d in row['policies'].items():
            exact=all(p==y for p,y in zip(d['predictions'],labels,strict=True))
            complete=all(p is not None and -1 not in sum(p,[]) for p in d['predictions'])
            assert scored['policies'][policy]['correct']==exact and scored['policies'][policy]['complete']==complete
            checks['policy_task_scores']+=1
    result={'checks':dict(checks),'certificate_size_histogram':dict(core_sizes),
            'exhaustive_sample':sample,'prediction_hash_verified':True,'seconds':time.perf_counter()-start,
            'limitations':'Outer predictions are independently reconstructed, but the outer model selector is not independently reimplemented. Synthetic label-isolation checks and source review cover that boundary.'}
    (a.run/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
