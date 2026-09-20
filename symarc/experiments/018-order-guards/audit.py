"""Independent replay using array-slice features, explicit action sets and subsets.

This does not modify selection. Outer predictions are replayed independently;
outer model selection itself is not independently reimplemented on the corpus.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reference017',HERE.parent/'017-split-or-share/audit.py')
a17=importlib.util.module_from_spec(spec);spec.loader.exec_module(a17)
ARMS={'exact':tuple(range(19)), 'order':tuple(range(25)),
      'guarded_exact':tuple(range(19))+tuple(range(25,29)), 'guarded_order':tuple(range(29))}


def scene(g):
    result=[]
    for fs,vs in a17.scene(g):
        comparisons=[]
        for i,j in itertools.combinations(range(15,19),2):
            comparisons.append(0 if fs[i]==fs[j] else (-1 if fs[i]<fs[j] else 1))
        result.append((fs+tuple(comparisons)+tuple(0 if vs[i]==-1 else 1 for i in range(15,19)),vs))
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for arg in ('problems','answers','baseline','run'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();start=time.perf_counter();read=lambda f:json.loads(f.read_text())
    problems=read(a.problems);rows=read(a.run/'predictions.json');old=read(a.baseline)
    scores=read(a.run/'scores.json');answers=read(a.answers);manifest=read(a.run/'prediction-run.json')
    assert hashlib.sha256((a.run/'predictions.json').read_bytes()).hexdigest()==manifest['predictions_sha256']
    counts=Counter();core_sizes=Counter();sample=sorted(
        (r['split']+'/'+r['id'] for r in rows if any(f['models'] for f in r['families'].values())),
        key=lambda s:hashlib.sha256(s.encode()).digest())[:12]
    for problem,row,previous,scored in zip(problems,rows,old,scores,strict=True):
        assert problem['id']==row['id']==previous['id']==scored['id']
        assert row['eligible']==previous['eligible'];counts['baseline_eligibility']+=1
        key=row['split']+'/'+row['id'];labels=answers[key]
        if row['eligible']:
            pairs=problem['train'];ts=[scene(p['input']) for p in pairs];qs=[scene(g) for g in row['query_inputs']]
            records=[[(f,v,y) for (f,v),y in zip(sc,sum(p['output'],[]),strict=True)] for sc,p in zip(ts,pairs,strict=True)]
            inds=tuple(range(len(pairs)));cache={}
            def table(indices,fs):
                k=(indices,fs)
                if k not in cache:cache[k]=a17.fit([r for i in indices for r in records[i]],fs,19)
                return cache[k]
            baseline={tuple(m['features']):m for m in previous['families']['joint']['models']}
            current={tuple(m['features']):m for m in row['families']['exact']['models']}
            assert baseline.keys()==current.keys();counts['baseline_pools']+=1
            for fs,m in current.items():
                assert m['predictions']==baseline[fs]['predictions'] and m['validation']==baseline[fs]['validation']
                counts['baseline_models']+=1
            assert previous['policies']['joint']['predictions']==row['policies']['exact']['predictions']
            for f,g in zip(previous['policies']['joint']['outer'],row['policies']['exact']['outer'],strict=True):
                assert f['scores']==g['scores'] and f['excluded']==g['excluded']
                assert (f['selected'][1:] if f['selected'] else [])==(g['selected'][1:] if g['selected'] else [])
                counts['baseline_outer_folds']+=1
            # Derived comparisons cannot change full-vocabulary compatibility.
            ds=row['families'];assert (ds['exact']['diagnostics']['status']=='vocabulary_conflict')==(ds['order']['diagnostics']['status']=='vocabulary_conflict')
            assert (ds['guarded_exact']['diagnostics']['status']=='vocabulary_conflict')==(ds['guarded_order']['diagnostics']['status']=='vocabulary_conflict')
            counts['derived_feature_boundary']+=2
            for arm,details in ds.items():
                vocab=ARMS[arm];models=details['models'];diag=details['diagnostics']
                if 'witness' in diag:
                    size=a17.verify_core(diag['witness'],vocab,19,pairs,ts)
                    core_sizes[size]+=1;counts['vocabulary_certificates']+=1;assert not models
                elif key in sample:
                    minimal=[];unique=list(dict.fromkeys((r[0],tuple(sorted(a17.allowed(r,19)))) for rs in records for r in rs))
                    for k in range(4):
                        for fs in itertools.combinations(vocab,k):
                            if any(set(f).issubset(fs) for f in minimal):continue
                            d={};ok=True
                            for f,acts in unique:
                                key_=tuple(f[i] for i in fs);acts=set(acts)
                                d[key_]=d[key_]&acts if key_ in d else acts
                                if not d[key_]:ok=False;break
                            counts['exhaustive_subsets']+=1
                            if ok:minimal.append(fs)
                    assert set(minimal)=={tuple(m['features']) for m in models},(key,arm)
                    counts['exhaustive_pools']+=1
                for model in models:
                    fs=tuple(model['features']);t=table(inds,fs);assert t is not None
                    for why in model['necessity']:
                        f=why['necessary_feature'];others=tuple(i for i in fs if i!=f)
                        size=a17.verify_core(why['witness'],others,19,pairs,ts)
                        assert len({w['features'][f] for w in why['witness']})>1
                        counts['necessity_certificates']+=1;core_sizes[size]+=1
                    cv=0;fraction=Fraction(0)
                    for fold in model['validation']:
                        train=tuple(i for i in inds if i not in fold['excluded']);t_=table(train,fs);exact=[]
                        for i,s in zip(fold['excluded'],fold['scores'],strict=True):
                            pred=a17.apply(pairs[i]['input'],ts[i],fs,t_)
                            a17.check_score(pred,pairs[i]['output'],s);counts['inner_predictions']+=1;exact.append(pred==pairs[i]['output'])
                        cv+=all(exact);s=fold['scores'][0];fraction+=Fraction(s['correct'],s['cells'])
                    assert cv==model['cv_exact'] and [fraction.numerator,fraction.denominator]==model['cv_fraction']
                    assert model['predictions']==[a17.apply(g,sc,fs,t) for g,sc in zip(row['query_inputs'],qs,strict=True)]
                    counts['candidate_query_grids']+=len(qs)
            for name,detail in row['policies'].items():
                sel=detail['selected']
                if sel:
                    fs=tuple(sel[1]);assert detail['predictions']==[a17.apply(g,sc,fs,table(inds,fs)) for g,sc in zip(row['query_inputs'],qs,strict=True)]
                else:assert all(g is None for g in detail['predictions'])
                for fold in detail['outer']:
                    train=tuple(i for i in inds if i not in fold['excluded']);sel=fold['selected']
                    for i,s in zip(fold['excluded'],fold['scores'],strict=True):
                        pred=a17.apply(pairs[i]['input'],ts[i],tuple(sel[1]),table(train,tuple(sel[1]))) if sel else None
                        a17.check_score(pred,pairs[i]['output'],s);counts['outer_predictions']+=1
        for name,d in row['policies'].items():
            exact=all(g==y for g,y in zip(d['predictions'],labels,strict=True))
            complete=all(g is not None and -1 not in sum(g,[]) for g in d['predictions'])
            assert scored['policies'][name]['correct']==exact and scored['policies'][name]['complete']==complete
            counts['policy_task_scores']+=1
    result={'checks':dict(counts),'certificate_sizes':dict(core_sizes),'exhaustive_sample':sample,
            'prediction_hash_verified':True,'seconds':time.perf_counter()-start,
            'limitations':'Independent array-slice/union-find features, action-set fits and outer predictions; outer selector not independently reimplemented on corpus.'}
    (a.run/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
