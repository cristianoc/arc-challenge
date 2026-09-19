"""Independent grammar, interpreter, finite search and geometric verification.

Does not import language.py or run.py. Reconstructs minimum complete programs
and checks all label and query predictions from saved files.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def text(x):return json.dumps(x,separators=(',',':'))
def tup(x):return tuple(tup(v) if isinstance(v,list) else v for v in x)
def size(e):return 1 if len(e)==1 or e[0]=='lit' else 1+sum(size(c) for c in e[1:])


def expressions():
    ints={1:{('count',),('span_r',),('span_c',),('lit',0),('lit',1)}}
    bools={1:set()}
    for total in range(2,6):
        ints[total]=set();bools[total]={('not',x) for x in bools[total-1]}
        for i in range(1,total-1):
            j=total-i-1
            for left in ints[i]:
                for right in ints[j]:
                    ordered=tuple(sorted((left,right),key=text))
                    ints[total].update((('add',*ordered),('mul',*ordered),('sub',left,right)))
                    bools[total].update((('eq',*ordered),('lt',left,right)))
            for left in bools[i]:
                for right in bools[j]:bools[total].add(('and',*sorted((left,right),key=text)))
    return sorted(set.union(*bools.values()),key=lambda x:(size(x),text(x)))


def obs(points):
    ps=sorted({tuple(p) for p in points})
    row=sorted(r for r,c in ps);col=sorted(c for r,c in ps)
    return {'count':len(ps),'span_r':row[-1]-row[0]+1,'span_c':col[-1]-col[0]+1}


def interp(expr,environment):
    op=expr[0]
    if op in environment:return environment[op]
    if op=='lit':return expr[1]
    if op=='not':return not interp(expr[1],environment)
    l=interp(expr[1],environment);r=interp(expr[2],environment)
    if op=='add':return l+r
    if op=='mul':return l*r
    if op=='sub':return l-r
    if op=='lt':return l<r
    if op=='eq':return l==r
    if op=='and':return l and r
    raise AssertionError(op)


def rectangle(points):
    # Pointwise membership in the bounding region, not the training teacher's row test.
    ps=set(map(tuple,points));rs=[r for r,c in ps];cs=[c for r,c in ps]
    return all((r,c) in ps for r in range(min(rs),max(rs)+1) for c in range(min(cs),max(cs)+1))


def solve(data,ps):
    xs=[obs(r['points']) for r in data];ys=[r['label'] for r in data]
    colours=sorted(set(ys));answer=[];counts=Counter()
    for p in ps:
        if answer and size(p)>size(answer[0][0]):break
        flags=[interp(p,x) for x in xs]
        counts['predicate_observation_evaluations']+=len(xs);counts['predicates_considered']+=1
        for f,t in itertools.product(colours,repeat=2):
            counts['branch_assignments_considered']+=1
            fits=True
            for b,y in zip(flags,ys):
                counts['label_checks']+=1
                if (t if b else f)!=y:fits=False;break
            if fits:answer.append((p,f,t))
    return answer,dict(counts)


def audit(data:Path,out:Path):
    inp=read(data/'problem.json');answer=read(data/'answers.json');negative=read(data/'frame-control.json')
    construction=read(out/'construction.json');library=read(out/'library.json');prediction=read(out/'predictions.json')
    manifest=read(out/'run.json');scores=read(out/'scores.json')
    assert sha(data/'problem.json')==manifest['problem_sha256']
    assert sha(out/'library.json')==manifest['library_sha256']
    assert sha(out/'predictions.json')==manifest['predictions_sha256']
    ps=expressions();expected_hash=hashlib.sha256(json.dumps(ps,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert expected_hash==construction['candidate_hash'] and len(ps)==construction['candidate_predicates']
    assert len(inp['train'])==511 and all(r['label']==int(rectangle(r['points'])) for r in inp['train'])
    ref,work=solve(inp['train'],ps)
    enc=lambda ms:sorted([(tup(m['predicate']),m['false'],m['true']) for m in ms],key=lambda x:(size(x[0]),text(x[0]),x[1],x[2]))
    ref=sorted(ref,key=lambda x:(size(x[0]),text(x[0]),x[1],x[2]))
    assert ref==enc(construction['minimum_programs'])==enc(library['all_minimal_programs'])
    assert work==construction['direct_work']
    assert enc([construction['selected']])[0]==ref[0]
    for m in construction['minimum_programs']:
        assert m['predicate_cost']==size(tup(m['predicate'])) and m['expanded_program_cost']==size(tup(m['predicate']))+3
    for r in construction['rejections']:
        i,j=r['observations'];a,b=inp['train'][i],inp['train'][j]
        assert a['label']!=b['label']
        assert interp(r['predicate'],obs(a['points']))==interp(r['predicate'],obs(b['points']))
    rejected={text(r['predicate']) for r in construction['rejections']}
    assert all(text(p) in rejected or any(p==q for q,f,t in ref) for p in ps if size(p)<=size(ref[0][0]))
    witness=construction['frame_vocabulary_conflict'];i,j=witness['observations']
    assert obs(negative[i]['points'])==obs(negative[j]['points']) and negative[i]['label']!=negative[j]['label']
    pbody=tup(library['body']);assert size(pbody)==library['body_cost']
    counts={'independent_predicates':len(ps),'minimum_programs':len(ref),'rejection_witnesses':len(rejected),
            'frame_conflict':witness,'banks':{},'replayed_query_outputs':0}
    source_sets={tuple(sorted(map(tuple,e['points']))) for e in inp['train']}
    source_features={tuple(obs(e['points']).values()) for e in inp['train']}
    pair_labels=((2,7),(8,4),(6,3),(9,1))
    for bank,points in inp['queries'].items():
        envs=[obs(s) for s in points];truth=[int(rectangle(s)) for s in points]
        assert truth==answer[bank]
        predictions=[[t if interp(p,x) else f for x in envs] for p,f,t in ref]
        assert prediction[bank]['source']==predictions[0] and prediction[bank]['all_minimal']==predictions
        counts['replayed_query_outputs']+=(len(ref)+1)*len(points)
        pcounts=Counter(truth);counts['banks'][bank]={'positive':pcounts[1],'negative':pcounts[0],
            'positive_correct':sum(y==1 and z==1 for y,z in zip(truth,predictions[0])),
            'negative_correct':sum(y==0 and z==0 for y,z in zip(truth,predictions[0]))}
        seen=[tuple(sorted(map(tuple,s))) in source_sets for s in points]
        counts['banks'][bank].update(exact_source_overlap=sum(seen),
            unseen_sets=sum(not z for z in seen),
            unseen_positive=sum(y==1 and not z for y,z in zip(truth,seen)),
            unseen_correct=sum(y==p and not z for y,p,z in zip(truth,predictions[0],seen)),
            novel_scalar_inputs=sum(tuple(e.values()) not in source_features for e in envs))
        assert scores[bank]['correct']==sum(y==z for y,z in zip(truth,predictions[0]))
        for k,row in enumerate(construction['transfers']):
            target=inp['transfer'][k]['train']
            ref_target,tw=solve(target,sorted({p for p,f,t in ref},key=lambda x:(size(x),text(x))))
            assert enc(row['all_programs'])==sorted(ref_target,key=lambda x:(size(x[0]),text(x[0]),x[1],x[2]))
            assert row['cold_source_work']==work and row['cold_target_work']==tw
            pp,f,t=ref_target[0]
            yp=[t if interp(pp,x) else f for x in envs]
            assert yp==prediction[bank]['transfer'][k]
            assert scores[bank]['transfer_correct'][k]==sum(y==(pair_labels[k][0] if z else pair_labels[k][1]) for y,z in zip(yp,truth))
            counts['replayed_query_outputs']+=len(yp)
    source_checks=construction['constructive_work']['label_checks']
    reuse_checks=sum(t['reuse_target_work']['label_checks'] for t in construction['transfers'])
    cold_checks=work['label_checks']+sum(t['cold_source_work']['label_checks']+t['cold_target_work']['label_checks'] for t in construction['transfers'])
    counts['work_accounting']={'source_plus_four_reuses_label_checks':source_checks+reuse_checks,
        'five_cold_source_solves_plus_target_fits_label_checks':cold_checks,
        'cold_source_label_checks':work['label_checks'],'constructed_source_label_checks':source_checks,
        'reuse_target_label_checks':reuse_checks,
        'predicate_evaluations_source_and_reuse':construction['constructive_work']['predicate_observation_evaluations']+sum(t['reuse_target_work']['predicate_observation_evaluations'] for t in construction['transfers']),
        'predicate_evaluations_cold':work['predicate_observation_evaluations']+sum(t['cold_source_work']['predicate_observation_evaluations']+t['cold_target_work']['predicate_observation_evaluations'] for t in construction['transfers'])}
    write=out/'audit.json';write.write_text(json.dumps(counts,sort_keys=True,indent=2)+'\n')
    print(json.dumps(counts,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();audit(a.input,a.run)
