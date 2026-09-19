"""020: exact minimum operation palettes, leaving feature contexts distinct."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('order018',HERE.parent/'018-order-guards/run.py')
e18=importlib.util.module_from_spec(spec);spec.loader.exec_module(e18)


def minimum_palettes(constraints):
    """All minimum-cardinality hitting sets. Empty input has the empty palette."""
    cs=sorted(set(constraints),key=lambda x:(x.bit_count(),x))
    if any(x==0 for x in cs):return []
    cs=[x for i,x in enumerate(cs) if not any((y&x)==y for y in cs[:i])]
    union=0
    for c in cs:union|=c
    best=union.bit_count();solutions=set();visited=set()
    def visit(chosen):
        nonlocal best,solutions
        if chosen in visited or chosen.bit_count()>best:return
        visited.add(chosen)
        remaining=[c for c in cs if not c&chosen]
        if not remaining:
            cost=chosen.bit_count()
            if cost<best:best=cost;solutions=set()
            solutions.add(chosen);return
        if chosen.bit_count()>=best:return
        c=remaining[0]
        while c:
            bit=c&-c;c-=bit;visit(chosen|bit)
    visit(0)
    return sorted(solutions)


def restrict_table(table):
    if table is None:return None,[]
    palettes=minimum_palettes(table.values());possible=0
    for p in palettes:possible|=p
    return {k:v&possible for k,v in table.items()},palettes


class Learner(e18.Learner):
    def __init__(self,pairs):
        super().__init__(pairs);self.palette_records={}
    def table(self,indices,selected,arm):
        k=(indices,selected)
        if k not in self.table_cache:
            raw=e18.s17.fit([r for i in indices for r in self.records[i]],selected,e18.MASK)
            table,palettes=restrict_table(raw)
            self.table_cache[k]=table
            self.palette_records[k]={'raw':raw,'palettes':palettes}
        return self.table_cache[k]


def investigate(problem,sharing):
    cls=Learner if sharing else e18.Learner
    learner=cls(problem['train']);idx=tuple(range(len(problem['train'])))
    chosen=learner.selected(idx,'union');outer=[]
    for excluded in learner.groups:
        reduced=tuple(i for i in idx if i not in excluded);model=learner.selected(reduced,'union')
        predictions=[learner.predict(reduced,model,problem['train'][i]['input'],learner.scenes[i]) for i in excluded]
        outer.append({'excluded':list(excluded),'selected':None if model is None else [model['arm'],model['features']],
            'predictions':predictions,'scores':[e18.b.metrics(g,problem['train'][i]['output'],problem['train'][i]['input']) for i,g in zip(excluded,predictions,strict=True)]})
    models=learner.discover(idx,'guarded_order')[0]
    focus=next(m for m in models if m['features']==[24,27,28])
    def export(model):
        fs=tuple(model['features']);table=learner.table(idx,fs,model['arm'])
        predictions=[learner.predict(idx,model,g,e18.scene(g)) for g in problem['query_inputs']]
        one=learner.table((0,),fs,model['arm'])
        def rows(tab):return [{'key':list(k),'actions':[e18.s17.ACTIONS[i] for i in range(19) if mask&(1<<i)]} for k,mask in sorted(tab.items())]
        result={'features':model['features'],'names':model['names'],'arm':model['arm'],
            'cv_exact':model['cv_exact'],'cv_fraction':model['cv_fraction'],
            'predictions':predictions,'full_table':rows(table),'one_demo_table':rows(one)}
        if sharing:
            result['full_palettes']=learner.palette_records[(idx,fs)]['palettes']
            result['one_demo_palettes']=learner.palette_records[((0,),fs)]['palettes']
        return result
    result={'id':problem['id'],'sharing':sharing,'selected':export(chosen),
        'fixed_endpoint_candidate':export(focus),'outer':outer}
    if sharing:
        result['palette_certificates']=[{'indices':list(indices),'features':list(fs),
            'constraints':sorted(set(rec['raw'].values())) if rec['raw'] is not None else [0],
            'palettes':rec['palettes']} for (indices,fs),rec in sorted(learner.palette_records.items())]
    return result


def controlled(out):
    start=time.perf_counter();problems=[];targets={}
    for farther in (False,True):
        task_id='far' if farther else 'near'
        train=[e18.teacher_grid(7,(1,2),farther=farther),e18.teacher_grid(11,(3,4),farther=farther)]
        tests=[e18.teacher_grid(w,cs,h,farther) for w in (5,9,13,17,21,25,29) for cs,h in (((6,8),5),((7,9),7))]
        problems.append({'id':task_id,'train':train,'query_inputs':[t['input'] for t in tests]})
        targets[task_id]=[t['output'] for t in tests]
    # These files are separate; investigate gets no query answers.
    e18.b.write_json(out/'problems.json',problems);e18.b.write_json(out/'answers.json',targets)
    predictions=[investigate(p,sharing) for p in problems for sharing in (False,True)]
    e18.b.write_json(out/'predictions.json',predictions)
    ph=hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest()
    ys=json.loads((out/'answers.json').read_text());scores=[]
    for row in predictions:
        score={'id':row['id'],'sharing':row['sharing']}
        for name in ('selected','fixed_endpoint_candidate'):
            model=row[name];ps=model['predictions'];answer=ys[row['id']]
            complete=[g is not None and all(v>=0 for rr in g for v in rr) for g in ps]
            correct=[p==y for p,y in zip(ps,answer,strict=True)]
            score[name]={'features':model['features'],'cv_exact':model['cv_exact'],'cv_fraction':model['cv_fraction'],
                'correct':sum(correct),'complete_wrong':sum(c and not y for c,y in zip(complete,correct)),
                'incomplete':sum(not c for c in complete)}
        score['outer_pass']=all(s['exact'] for f in row['outer'] for s in f['scores'])
        scores.append(score)
    e18.b.write_json(out/'scores.json',scores)
    e18.b.write_json(out/'manifest.json',{'predictions_sha256':ph,'seconds':time.perf_counter()-start,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':None,
        'scope':'two controlled teachers; serial mechanism check, not runtime comparison'})
    print(json.dumps(scores,indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();controlled(a.out)
if __name__=='__main__':main()
