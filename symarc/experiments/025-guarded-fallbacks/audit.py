"""Independent set-based audit of 025; reads labels only after prediction.

Primitive features/actions are reused from 018. Conditional interpretation,
branch support, fitting, totality, ranking and scores are implemented here,
not imported from the scientific 025 implementation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter,defaultdict
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('features018',HERE.parent/'018-order-guards/run.py')
e18=importlib.util.module_from_spec(s);s.loader.exec_module(e18)
PAIRS=tuple((a,b) for a in range(11,19) for b in range(19) if a!=b)
LIBS=('base','all','observed');POLICIES=(*LIBS,'union_observed')


def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bits(xs):return sum(1<<x for x in xs)

@lru_cache(maxsize=8192)
def values(v):
    answer=list(v)
    for primary,fallback in PAIRS:
        answer.append(v[fallback] if v[primary]==-1 else v[primary])
    return tuple(answer)

@lru_cache(maxsize=8192)
def allowed(v,y):return frozenset(i for i,x in enumerate(values(v)) if x==y)


def table(records,fs):
    groups=defaultdict(set)
    for f,v,y,o in records:groups[tuple(f[i] for i in fs)].add((v,y))
    result={}
    for key,rs in groups.items():
        choices=set(range(163))
        for v,y in rs:choices.intersection_update(allowed(v,y))
        yes={a for a in range(11,19) if any(v[a]!=-1 for v,y in rs)}
        no={a for a in range(11,19) if any(v[a]==-1 for v,y in rs)}
        observed={i for i in choices if i<19 or PAIRS[i-19][0] in yes&no}
        result[key]=(frozenset(i for i in choices if i<19),frozenset(choices),frozenset(observed),
                     bits(a-11 for a in yes),bits(a-11 for a in no))
    return result


@lru_cache(maxsize=32768)
def predict_colour(v,options):
    possibles={values(v)[i] for i in options}
    return next(iter(possibles)) if len(possibles)==1 and -1 not in possibles else -1


def project(grid,sc,fs,t):
    flat=[predict_colour(v,t.get(tuple(f[i] for i in fs),frozenset())) for f,v in sc]
    width=len(grid[0]);return [flat[i:i+width] for i in range(0,len(flat),width)]


@lru_cache(maxsize=8192)
def defined(v):return frozenset(i for i,x in enumerate(values(v)) if x!=-1)


def query_domain(qsc,fs):
    good={}
    for sc in qsc:
        for f,v in sc:
            key=tuple(f[i] for i in fs)
            if key not in good:good[key]=set(defined(v))
            else:good[key].intersection_update(defined(v))
    return good


def domain(t,qsc,fs,good=None):
    if good is None:good=query_domain(qsc,fs)
    unknown=sorted(set(good)-set(t))
    kept={key:frozenset(good[key]&t[key]) for key in good if key in t}
    feasible=not unknown and all(t.values()) and all(kept.values())
    return feasible,unknown,good,(kept if feasible else {})


def metric(pred,target,inp):
    h,w=len(target),len(target[0]);same=len(inp)==h and len(inp[0])==w
    changed=sum(inp[r][c]!=target[r][c] for r in range(h) for c in range(w)) if same else None
    complete=pred is not None and all(v!=-1 for row in pred for v in row)
    aligned=pred is not None and len(pred)==h and all(len(row)==w for row in pred)
    known=correct=ck=cc=0
    if aligned:
        for r in range(h):
            for c in range(w):
                if pred[r][c]==-1:continue
                known+=1;correct+=pred[r][c]==target[r][c]
                if same and inp[r][c]!=target[r][c]:
                    ck+=1;cc+=pred[r][c]==target[r][c]
    return dict(complete=complete,exact=pred==target,cells=h*w,known=known,correct=correct,wrong=known-correct,
                changed=changed,changed_known=ck,changed_correct=cc,shape_changed=not same)


def order(model):
    return (-model['cv_exact'],-Fraction(*model['cv_fraction']),len(model['features']),model['cost'],
            LIBS.index(model['library']),list(e18.ARMS).index(model['arm']),tuple(model['features']))


def choose(models,n,policy):
    libraries={'base','observed'} if policy=='union_observed' else {policy}
    ready=sorted((x for x in models if x['library'] in libraries and x['condition']['feasible']),key=order)
    return [] if not ready else [ready[0]['library'],ready[0]['arm'],ready[0]['features']], [None]*n if not ready else ready[0]['predictions']


def audit_row(row,problem,scored,answers,stable):
    counts=Counter(tasks=1);train=problem['train'];idx=tuple(range(len(train)))
    scenes=[e18.scene(p['input']) for p in train];queries=[e18.scene(g) for g in problem['query_inputs']]
    records=[];groups={}
    for i,(p,sc) in enumerate(zip(train,scenes,strict=True)):
        w=len(p['input'][0]);labels=[v for r in p['output'] for v in r]
        if row['eligible']:
            records.append([(f,v,y,(i,j//w,j%w)) for j,((f,v),y) in enumerate(zip(sc,labels,strict=True))])
        else:records.append([])
        groups.setdefault(json.dumps(p['input']),[]).append(i)
    groups=list(groups.values())
    refs=[]
    for recorded in row['tables']:
        fs=tuple(recorded['features']);indices=recorded['indices']
        t=table([r for i in indices for r in records[i]],fs)
        encoded=[[list(k),bits(v[0]),bits(v[1]),bits(v[2]),v[3],v[4]] for k,v in sorted(t.items())]
        assert recorded['entries']==encoded,(row['id'],'training table')
        refs.append(t);counts['tables']+=1;counts['context_action_sets']+=3*len(t)
    pools=[(row,idx,problem['query_inputs'],queries)]+[
        (f,tuple(i for i in idx if i not in f['excluded']),[train[i]['input'] for i in f['excluded']],
         [scenes[i] for i in f['excluded']]) for f in row['outer']]
    if row['eligible']:assert [f['excluded'] for f in row['outer']]==groups
    for pool,indices,xs,qsc in pools:
        projections={};valid_cache={}
        for model in pool['models']:
            fs=tuple(model['features']);lib=LIBS.index(model['library']);tid=model['table_id']
            recorded=row['tables'][tid]
            assert recorded['indices']==list(indices) and recorded['features']==list(fs)
            assert model['cost']==sum(e18.TERMS[f][1] for f in fs)
            t={k:v[lib] for k,v in refs[tid].items()}
            if fs not in valid_cache:valid_cache[fs]=query_domain(qsc,fs)
            feasible,unknown,valid,kept=domain(t,qsc,fs,valid_cache[fs])
            expected={'feasible':feasible,'unknown':[list(k) for k in unknown],
                'valid':[[list(k),bits(v)] for k,v in sorted(valid.items())],
                'kept':[[list(k),bits(v)] for k,v in sorted(kept.items())]}
            assert model['condition']==expected,(row['id'],'domain')
            preds=[project(g,sc,fs,kept) for g,sc in zip(xs,qsc,strict=True)] if feasible else [None]*len(xs)
            assert model['predictions']==preds,(row['id'],'prediction')
            if model['library']=='all':assert feasible==(not unknown),(row['id'],'fallback totality lemma')
            counts['models']+=1;counts['model_query_grids']+=len(xs)
            expected_groups=[g for g in groups if g[0] in indices]
            assert [f['excluded'] for f in model['validation']]==expected_groups
            exact=0;fraction=Fraction()
            for fold in model['validation']:
                excluded=fold['excluded'];ft=row['tables'][fold['table_id']]
                assert ft['indices']==[i for i in indices if i not in excluded] and ft['features']==list(fs)
                tr={k:v[lib] for k,v in refs[fold['table_id']].items()}
                scores=[metric(project(train[i]['input'],scenes[i],fs,tr),train[i]['output'],train[i]['input']) for i in excluded]
                assert scores==fold['scores'],(row['id'],'internal labels')
                exact+=all(s['exact'] for s in scores);fraction+=Fraction(scores[0]['correct'],scores[0]['cells'])
                counts['inner_grids']+=len(scores)
            assert exact==model['cv_exact'] and [fraction.numerator,fraction.denominator]==model['cv_fraction']
            projections[(fs,model['library'])]=(feasible,preds)
        for fs in {k[0] for k in projections}:
            b,bp=projections[(fs,'base')]
            for lib in ('all','observed'):
                f,ps=projections[(fs,lib)]
                if b:
                    assert f
                    for g,h in zip(bp,ps):
                        for rr,ss in zip(g,h):
                            for x,y in zip(rr,ss):
                                if y!=-1:assert x==y
            counts['family_nesting_checks']+=2
        for policy in POLICIES:
            selected,pred=choose(pool['models'],len(xs),policy)
            assert selected==pool['policies'][policy]['selected'] and pred==pool['policies'][policy]['predictions']
            counts['policy_choices']+=1
            if pool is not row:
                expected=[metric(g,train[i]['output'],train[i]['input']) for i,g in zip(pool['excluded'],pred,strict=True)]
                assert pool['scores'][policy]==expected;counts['outer_grids']+=len(expected)
    for witness in row['branch_witnesses']:
        model=next(x for x in row['models'] if [x['library'],x['arm'],x['features']]==witness['selected'])
        fs=tuple(model['features']);kept={tuple(k):set(i for i in range(163) if v&(1<<i)) for k,v in model['condition']['kept']}
        expected=[]
        for key,actions in sorted(kept.items()):
            occurrences=[r for rs in records for r in rs if tuple(r[0][i] for i in fs)==key]
            query_vs=[v for sc in queries for f,v in sc if tuple(f[i] for i in fs)==key]
            for a in range(11,19):
                ops=sorted(i for i in actions if i>=19 and PAIRS[i-19][0]==a)
                if not ops:continue
                yes=next((list(r[3]) for r in occurrences if r[1][a]!=-1),None)
                no=next((list(r[3]) for r in occurrences if r[1][a]==-1),None)
                if model['library']=='observed':assert yes is not None and no is not None
                expected.append({'key':list(key),'primary':a,'operations':ops,'primary_training':yes,'fallback_training':no,
                    'query_primary':sum(v[a]!=-1 for v in query_vs),'query_fallback':sum(v[a]==-1 for v in query_vs)})
                counts['branch_witnesses']+=1;counts['branch_operation_instances']+=len(ops)
        assert witness['branches']==expected,(row['id'],'branch evidence')
    for policy in POLICIES:
        ps=row['policies'][policy]['predictions'];target=answers
        saved=scored['policies'][policy]
        ms=[metric(g,y,x) for g,y,x in zip(ps,target,problem['query_inputs'],strict=True)]
        assert saved['grids']==ms and saved['correct']==(ps==target)
        complete=all(g is not None and all(v in range(10) for r in g for v in r) for g in ps)
        assert saved['complete']==complete and saved['wrong_complete']==(complete and ps!=target)
        gate=len(row['outer'])>=2 and all(g['exact'] for f in row['outer'] for g in f['scores'][policy])
        assert saved['outer_pass']==gate;counts['task_scores']+=1
        if stable:
            use=not stable['fitted'] and complete;hp=ps if use else stable['predictions']
            assert saved['hybrid']==dict(correct=hp==target,use_relational=use,correct_grids=sum(g==y for g,y in zip(hp,target)))
            counts['hybrid_scores']+=1
    # After-score concrete-program oracle; never feeds back into any choice.
    oracle={lib:[] for lib in LIBS}
    if all(len(g)==len(y) and len(g[0])==len(y[0]) for g,y in zip(problem['query_inputs'],answers)):
        for model in row['models']:
            fs=tuple(model['features']);col=LIBS.index(model['library'])
            rem={k:set(v[col]) for k,v in refs[model['table_id']].items()}
            for sc,ys in zip(queries,answers):
                for (f,v),y in zip(sc,[c for r in ys for c in r]):
                    key=tuple(f[i] for i in fs)
                    rem.setdefault(key,set()).intersection_update(allowed(v,y))
            if rem and all(rem.values()):oracle[model['library']].append([model['arm'],model['features']])
            counts['oracle_model_checks']+=1
    return dict(counts),dict(id=row['id'],split=row['split'],correct_models=oracle)


def worker(job):
    return audit_row(*job)


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('problems','predictions','answers','scores','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stable',type=Path);p.add_argument('--start',type=int,default=0);p.add_argument('--limit',type=int)
    p.add_argument('--workers',type=int,default=12)
    a=p.parse_args();all_rows=read(a.predictions);problems={x['id']:x for x in read(a.problems)}
    scores={x['id']:x for x in read(a.scores)};answers=read(a.answers)
    stable={r['id']:r for r in map(json.loads,a.stable.read_text().splitlines())} if a.stable else {}
    rows=all_rows[a.start:] if a.limit is None else all_rows[a.start:a.start+a.limit]
    total=Counter();oracles=[]
    jobs=[]
    for row in rows:
        key=row['split']+'/'+row['id'];target=answers.get(key,answers.get(row['id']))
        jobs.append((row,problems[row['id']],scores[row['id']],target,stable.get(row['id'])))
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for c,o in pool.map(worker,jobs):
            total.update(c);oracles.append(o)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    output={'start':a.start,'ids':[r['id'] for r in rows],'counts':dict(total),'oracle':oracles,'predictions_sha256':sha(a.predictions)}
    a.out.write_text(json.dumps(output,sort_keys=True,separators=(',',':'))+'\n')
    print(dict(total),flush=True)


if __name__=='__main__':main()
