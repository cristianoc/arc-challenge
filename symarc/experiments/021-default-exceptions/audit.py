"""Independent set-based audit of default optimality and prediction projections.

Uses the previously checked 018 feature/action evaluator, but independently
rebuilds admissible sets from labelled cells and enumerates all 19 defaults.
It does not independently implement feature search or outer model selection.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('audit018',Path(__file__).parent.parent/'018-order-guards/run.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def ops(mask): return {i for i in range(19) if mask & (1 << i)}
def matrix(values,w): return [values[i:i+w] for i in range(0,len(values),w)]


def predict(grid,scene,features,table):
    if table is None:return None
    vals=[]
    for fs,action_values in scene:
        allowed=table.get(tuple(fs[i] for i in features),set())
        outputs={action_values[a] for a in allowed}
        vals.append(next(iter(outputs)) if len(outputs)==1 and -1 not in outputs else -1)
    return matrix(vals,len(grid[0]))


def case(job):
    problem,base,row=job
    assert row['id']==base['id']==problem['id'] and row['eligible']==base['eligible']
    counts=Counter(tasks=1)
    if not row['eligible']:return dict(counts)
    train=problem['train'];scenes=[e.scene(p['input']) for p in train]
    tables={}
    for cert in row['default_tables']:
        counts['table_records']+=1
        indices=tuple(cert['indices']);fs=tuple(cert['features']);grouped={}
        for i in indices:
            for (features,values),label in zip(scenes[i],(v for rr in train[i]['output'] for v in rr),strict=True):
                key=tuple(features[j] for j in fs)
                allowed={a for a,v in enumerate(values) if v==label}
                grouped[key]=grouped.get(key,set(range(19))) & allowed
        if any(not a for a in grouped.values()):
            assert cert['entries'] is None and cert['cost'] is None and not cert['defaults']
            tables[(indices,fs)]=None;counts['incompatible_tables']+=1;continue
        entries={tuple(r['key']):r for r in cert['entries']}
        assert set(entries)==set(grouped)
        if not grouped:
            assert cert['cost']==0 and not cert['defaults'];tables[(indices,fs)]={};counts['empty_tables']+=1;continue
        costs={a:len([v for v in grouped.values() if a not in v]) for a in range(19)}
        defaults={a for a,c in costs.items() if c==min(costs.values())}
        assert cert['defaults']==sorted(defaults) and cert['cost']==min(costs.values())
        assert cert['all_costs']==[costs[a] for a in range(19)]
        result={}
        for key,allowed in grouped.items():
            assert ops(entries[key]['allowed'])==allowed
            possible=set().union(*({d} if d in allowed else allowed for d in defaults))
            assert possible and possible<=allowed and ops(entries[key]['retained'])==possible
            result[key]=possible
            counts['context_projections']+=1
            counts['narrowed_contexts']+=possible!=allowed
        tables[(indices,fs)]=result
        counts['consistent_tables']+=1;counts['defaults_evaluated']+=19
    all_indices=tuple(range(len(train)));qscenes=[e.scene(x) for x in problem['query_inputs']]
    for arm,family in row['families'].items():
        old=base['families'][arm]
        assert family['diagnostics']==old['diagnostics']
        assert [m['features'] for m in family['models']]==[m['features'] for m in old['models']]
        counts['identical_feature_pools']+=1
        for model,before in zip(family['models'],old['models'],strict=True):
            fs=tuple(model['features']);table=tables[(all_indices,fs)]
            preds=[predict(g,scene,fs,table) for g,scene in zip(problem['query_inputs'],qscenes,strict=True)]
            assert preds==model['predictions'];counts['candidate_grids_replayed']+=len(preds)
            for previous,new in zip(before['predictions'],preds,strict=True):
                if previous is None:continue
                for x,y in zip((v for r in previous for v in r),(v for r in new for v in r),strict=True):
                    if x>=0:assert x==y;counts['fixed_model_preserved_cells']+=1
                    elif y>=0:counts['fixed_model_new_cells']+=1
            total=Fraction(0);exact=0
            for fold in model['validation']:
                reduced=tuple(i for i in all_indices if i not in fold['excluded']);table=tables[(reduced,fs)]
                fold_exact=True
                for i,saved in zip(fold['excluded'],fold['scores'],strict=True):
                    p=train[i];pred=predict(p['input'],scenes[i],fs,table)
                    correct=sum(x==y for rr,yy in zip(pred,p['output']) for x,y in zip(rr,yy)) if pred else 0
                    assert correct==saved['correct'] and (pred==p['output'])==saved['exact']
                    fold_exact &= pred==p['output'];counts['internal_grids_replayed']+=1
                first=fold['scores'][0];total+=Fraction(first['correct'],first['cells']);exact+=fold_exact
            assert [total.numerator,total.denominator]==model['cv_fraction'] and exact==model['cv_exact']
    for policy,info in row['policies'].items():
        candidates=[m for arm,f in row['families'].items() if policy=='union' or arm==policy for m in f['models']]
        ranking=lambda m:(-m['cv_exact'],-Fraction(*m['cv_fraction']),len(m['features']),m['cost'],list(e.ARMS).index(m['arm']),tuple(m['features']))
        winner=min(candidates,key=ranking) if candidates else None
        assert info['selected']==([] if winner is None else [winner['arm'],winner['features']])
        assert info['predictions']==([None]*len(qscenes) if winner is None else winner['predictions'])
        counts['full_selection_checks']+=1
        for fold in info['outer']:
            chosen=fold['selected'];idx=tuple(i for i in all_indices if i not in fold['excluded'])
            for i,saved in zip(fold['excluded'],fold['scores'],strict=True):
                p=train[i]
                pred=None if not chosen else predict(p['input'],scenes[i],tuple(chosen[1]),tables[(idx,tuple(chosen[1]))])
                assert (pred==p['output'])==saved['exact']
                correct=sum(x==y for rr,yy in zip(pred,p['output']) for x,y in zip(rr,yy)) if pred else 0
                assert correct==saved['correct'];counts['outer_grids_replayed']+=1
    return dict(counts)


def main():
    p=argparse.ArgumentParser(__doc__)
    for x in ('problems','base','predictions','out'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    load=lambda p:json.loads(p.read_text())
    problems,base,new=load(a.problems),load(a.base),load(a.predictions)
    counts=Counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(case,zip(problems,base,new,strict=True)):counts.update(result)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(counts),indent=2,sort_keys=True)+'\n')
    print(dict(counts))
if __name__=='__main__':main()
