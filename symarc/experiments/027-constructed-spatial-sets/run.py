"""027: synthesize a spatial set, then learn separate set-valued operations.

prepare / construct / predict / score are separate, hash-linked stages.
No query file is read by construct. No query answers are read by predict.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
import language as L


def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(L.key(x).encode()).hexdigest()
def points(s):return [list(p) for p in sorted(s)]


def teacher_box(s):
    rs=sorted({p[0] for p in s});cs=sorted({p[1] for p in s})
    return rs[0],rs[-1],cs[0],cs[-1]


def teacher_edges(box):
    a,b,c,d=box
    return {(a,j) for j in range(c,d+1)}|{(b,j) for j in range(c,d+1)}|{(i,c) for i in range(a,b+1)}|{(i,d) for i in range(a,b+1)}


def teacher_inside(box):
    a,b,c,d=box
    return {(i,j) for i in range(a+1,b) for j in range(c+1,d)}


def shape(h,w,mask):return [[i//w,i%w] for i in range(h*w) if mask&(1<<i)]
def label(s):return int(set(map(tuple,s))==teacher_edges(teacher_box(s)))


def prepare(out):
    train=[{'points':shape(3,3,m),'label':label(shape(3,3,m))} for m in range(1,512)]
    write(out/'source.json',train);write(out/'sparse.json',[train[254],train[494]])
    primary=[shape(4,4,m) for m in range(1,65536)]
    rng=random.Random(2701);bank=set()
    for h,w in ((5,7),(7,5),(8,8),(11,13)):
        b=(0,h-1,0,w-1);frame=teacher_edges(b);inner=teacher_inside(b)
        examples=[frame]+[frame-{p} for p in sorted(frame)]+[frame|{p} for p in sorted(inner)]
        for _ in range(32):
            remove=set(rng.sample(sorted(frame),rng.randrange(1,min(5,len(frame))+1)))
            add=set(rng.sample(sorted(inner),rng.randrange(1,min(5,len(inner))+1)))
            s=(frame-remove)|add
            if teacher_box(s)!=b:raise AssertionError('Transfer carrier changed')
            examples.append(s)
        for s in examples:
            for dr,dc in ((0,0),(-9,4),(12,-6)):
                t={(r+dr,c+dc) for r,c in s};bank.add(tuple(sorted(t)))
                bank.add(tuple(sorted((c,r) for r,c in t)))
    secondary=[points(s) for s in sorted(bank)]
    queries={'primary':primary,'secondary':secondary}
    write(out/'recognition-inputs.json',queries)
    write(out/'recognition-answers.json',{name:[label(s) for s in ss] for name,ss in queries.items()})
    target_train=[]
    for name in ('repair','interior'):
        examples=[]
        for h,w,dr,dc,noise in ((5,7,0,0,False),(7,5,3,-4,True)):
            b=(dr,dr+h-1,dc,dc+w-1);s=teacher_edges(b)-{(dr,dc+1)}
            if noise:s.add((dr+1,dc+1))
            y=teacher_edges(b) if name=='repair' else teacher_inside(b)
            examples.append({'input':points(s),'output':points(y)})
        target_train.append({'name':name,'train':examples})
    write(out/'target-train.json',target_train);write(out/'target-inputs.json',secondary)
    write(out/'target-answers.json',{name:[points(teacher_edges(teacher_box(s)) if name=='repair' else teacher_inside(teacher_box(s))) for s in secondary] for name in ('repair','interior')})
    b={(r,c) for r in range(5) for c in range(5)}
    controls=[{'points':points(b-{(2,2)}),'label':1},{'points':points(b-{(1,1)}),'label':0}]
    write(out/'negative-control.json',controls)
    write(out/'data.json',{'seed':2701,'source':len(train),'primary':len(primary),'secondary':len(secondary),
        'files':{p.name:sha(p) for p in sorted(out.glob('*.json')) if p.name!='data.json'}})


def synthesize(examples,candidates,direct=False):
    xs=[L.profile(e['points']) for e in examples];ys=[e['label'] for e in examples]
    found=[];rejected=[];work=Counter();best=None
    for c in candidates:
        if best is not None and c['program_cost']>best:break
        sig=[L.matches(c['truth'],x) for x in xs]
        work['candidate_predicates']+=1;work['profile_checks']+=len(xs)
        maps=[]
        if direct:
            for f,t in itertools.product((0,1),repeat=2):
                work['branch_assignments']+=1;fits=True
                for b,y in zip(sig,ys,strict=True):
                    work['label_checks']+=1
                    if (t if b else f)!=y:fits=False;break
                if fits:maps.append((f,t))
        else:
            classes={};conflict=None
            for i,(b,y) in enumerate(zip(sig,ys,strict=True)):
                work['label_checks']+=1
                if b in classes and classes[b][0]!=y:
                    conflict=[classes[b][1],i];break
                classes.setdefault(b,(y,i))
            if conflict is not None:rejected.append({'truth':c['truth'],'witness':conflict})
            else:
                maps=list(itertools.product(*[([classes[b][0]] if b in classes else [0,1]) for b in (False,True)]))
                work['branch_assignments']+=len(maps)
        for f,t in maps:
            best=c['program_cost'];found.append({**c,'false':f,'true':t})
    return found,dict(work),rejected


def construct(source,sparse,out):
    started=time.perf_counter();candidates,stats=L.grammar(7)
    dense=read(source);small=read(sparse)
    models,work,rejections=synthesize(dense,candidates)
    direct,directwork,_=synthesize(dense,candidates,True)
    assert models==direct
    sparsemodels,sparsework,_=synthesize(small,candidates)
    assert sparsemodels==synthesize(small,candidates,True)[0]
    if not models:raise RuntimeError('No dense fit within frozen bound')
    library={'models':models,'source_sha256':sha(source),'sparse_sha256':sha(sparse),'sparse_models':sparsemodels}
    write(out/'library.json',library)
    positive_yes=positive_no=0
    for e in dense:
        if e['label']:
            yes,no=L.profile(e['points']);positive_yes|=yes;positive_no|=no
    checks={'positive_required_true':positive_yes,'positive_required_false':positive_no,
        'all_roles_constrained':positive_yes|positive_no==L.ALL,
        'pointwise_consistent':positive_yes&positive_no==0}
    write(out/'construction.json',{'grammar':stats,'candidates':candidates,'direct_work':directwork,
        'constructive_work':work,'rejections':rejections,'sparse_work':sparsework,'semantic_constraints':checks})
    write(out/'construct-run.json',{'source_sha256':sha(source),'sparse_sha256':sha(sparse),
        'library_sha256':sha(out/'library.json'),'code':{p.name:sha(p) for p in (Path(__file__),Path(L.__file__))},
        'seconds':time.perf_counter()-started,'queries_read':False})
    print(json.dumps({'dense_models':models,'sparse_models':sparsemodels,'grammar':stats,'constraints':checks},indent=2))


def target_models(library,examples):
    good=[];checks=0
    unique={m['truth']:m['expr'] for m in library['models']}
    for sig,e in sorted(unique.items()):
        for op in L.TARGET_OPS:
            fits=True
            for pair in examples:
                checks+=1
                if L.target(op,sig,pair['input'])!=frozenset(map(tuple,pair['output'])):
                    fits=False;break
            if fits:good.append({'truth':sig,'expr':e,'operation':op,'expanded_cost':L.node_count(L.target_ast(op,e))})
    best=min((m['expanded_cost'] for m in good),default=None)
    return [m for m in good if m['expanded_cost']==best],checks


def predict(inputs,out):
    manifest=read(out/'construct-run.json');assert sha(out/'library.json')==manifest['library_sha256']
    library=read(out/'library.json');recognition=read(inputs/'recognition-inputs.json')
    preds={}
    for bank,ss in recognition.items():
        profiles=[L.profile(s) for s in ss]
        preds[bank]={}
        for name,ms in (('dense',library['models']),('sparse',library['sparse_models'])):
            each=[[m['true'] if L.matches(m['truth'],x) else m['false'] for x in profiles] for m in ms]
            consensus=[next(iter(v)) if len(v:=set(xs))==1 else -1 for xs in zip(*each)] if each else [-1]*len(ss)
            preds[bank][name]={'all_models':each,'consensus':consensus}
    targets=[];source=read(inputs/'source.json'); candidates=read(out/'construction.json')['candidates']
    query=read(inputs/'target-inputs.json')
    for task in read(inputs/'target-train.json'):
        models,work=target_models(library,task['train'])
        cold,dm,_=synthesize(source,candidates,True)
        cold_models,coldchecks=target_models({'models':cold},task['train'])
        assert models==cold_models
        predictions=[[points(L.target(m['operation'],m['truth'],s)) for s in query] for m in models]
        targets.append({'name':task['name'],'models':models,'predictions':predictions,
            'target_example_checks':work,'cold_source_work':dm,'cold_target_example_checks':coldchecks})
    write(out/'predictions.json',{'recognition':preds,'targets':targets})
    controls=read(inputs/'negative-control.json')
    profiles=[L.profile(e['points']) for e in controls]
    assert profiles[0]==profiles[1]
    write(out/'negative-certificate.json',{'examples':controls,'profiles':profiles,'same_scalar_observations':[25-1,5,5],
        'different_labels':True,'all_template_observations_identical':True})
    write(out/'predict-run.json',{'library_sha256':sha(out/'library.json'),'predictions_sha256':sha(out/'predictions.json'),
        'query_inputs_sha256':sha(inputs/'recognition-inputs.json'),'target_inputs_sha256':sha(inputs/'target-inputs.json'),
        'target_train_sha256':sha(inputs/'target-train.json'),'query_answers_read':False})


def score(inputs,out):
    run=read(out/'predict-run.json');assert sha(out/'predictions.json')==run['predictions_sha256']
    p=read(out/'predictions.json');answers=read(inputs/'recognition-answers.json');queries=read(inputs/'recognition-inputs.json')
    source={tuple(map(tuple,e['points'])) for e in read(inputs/'source.json')};scores={}
    for bank,y in answers.items():
        overlaps=[tuple(map(tuple,s)) in source for s in queries[bank]]
        scores[bank]={}
        for mode,pred in p['recognition'][bank].items():
            z=pred['consensus'];known=[v!=-1 for v in z]
            scores[bank][mode]={'examples':len(y),'positives':sum(y),'negatives':len(y)-sum(y),
                'correct':sum(a==b for a,b in zip(y,z)),'wrong':sum(k and a!=b for a,b,k in zip(y,z,known)),
                'abstained':sum(not k for k in known),'positive_correct':sum(a==1 and b==1 for a,b in zip(y,z)),
                'negative_correct':sum(a==0 and b==0 for a,b in zip(y,z)),
                'source_overlap':sum(overlaps),'unseen_correct':sum(a==b and not o for a,b,o in zip(y,z,overlaps)),
                'individual_correct':[sum(a==b for a,b in zip(y,z)) for z in pred['all_models']]}
    ty=read(inputs/'target-answers.json');target_input=read(inputs/'target-inputs.json');train=read(inputs/'target-train.json')
    scores['targets']={}
    for t in p['targets']:
        y=ty[t['name']];allpred=t['predictions']
        consensus=[vs[0] if all(x==vs[0] for x in vs) else None for vs in zip(*allpred)] if allpred else [None]*len(y)
        seen={tuple(map(tuple,e['input'])) for x in train if x['name']==t['name'] for e in x['train']}
        scores['targets'][t['name']]={'cases':len(y),'models':t['models'],
            'exact':sum(a==b for a,b in zip(y,consensus)),'abstained':sum(x is None for x in consensus),
            'target_demo_overlap':sum(tuple(map(tuple,s)) in seen for s in target_input),
            'changed_sets':sum(s!=a for s,a in zip(target_input,y)),
            'new_points_written':sum(len(set(map(tuple,a))-set(map(tuple,s))) for s,a in zip(target_input,y))}
    write(out/'scores.json',scores)
    write(out/'score-run.json',{'predictions_sha256':run['predictions_sha256'],
        'recognition_answers_sha256':sha(inputs/'recognition-answers.json'),'target_answers_sha256':sha(inputs/'target-answers.json'),
        'scores_sha256':sha(out/'scores.json')})
    print(json.dumps(scores,indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('construct');q.add_argument('--source',type=Path,required=True);q.add_argument('--sparse',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    for cmd in ('predict','score'):
        q=sub.add_parser(cmd);q.add_argument('--inputs',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='prepare':prepare(a.out)
    elif a.cmd=='construct':construct(a.source,a.sparse,a.out)
    elif a.cmd=='predict':predict(a.inputs,a.out)
    else:score(a.inputs,a.out)
if __name__=='__main__':main()
