"""Post-score diagnostics; no model selection or primary prediction is changed.

In addition to aggregate outcomes, distinguish an unanimously correct model
from a model merely containing one concrete program consistent with query labels.
The latter is an oracle diagnostic, never evidence supplied to the predictor.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit024',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
e=audit.e
read=audit.read


def oracle_in_table(table,features,scenes,targets):
    raw={tuple(x[0]):{a for a in range(19) if x[1]&(1<<a)} for x in table['entries']}
    domains={}
    for sc,y in zip(scenes,targets,strict=True):
        labels=[v for r in y for v in r]
        if len(labels)!=len(sc):return False
        for (f,values),label in zip(sc,labels,strict=True):
            z=tuple(f[i] for i in features)
            domains[z]=domains.get(z,audit.OPS)&{a for a,v in enumerate(values) if v==label}
    return raw,domains


def case(row,problem,target,stable):
    out={'id':row['id'],'split':row['split'],'mode':row['mode'],'stable_fitted':stable['fitted'],'models':{}}
    scenes=[e.scene(g) for g in problem['query_inputs']]
    for policy in ('tie_complete','total_ranked'):
        decision=row['policies'][policy];identity=decision['selected']
        if not identity:continue
        m=next(m for m in row['models'] if [m['arm'],m['features']]==identity);t=row['tables'][m['table_id']]
        rec={k:m[k] for k in ('arm','features','names','cv_exact','cv_fraction')}
        rec.update(feasible=t['feasible'],unknown_keys=t['unknown'],defaults=t['defaults'],surviving_defaults=t['surviving_defaults'],
            query_exact=[p==y for p,y in zip(decision['predictions'],target,strict=True)])
        new_cells=[]
        for qi,(before,after,y,g,sc) in enumerate(zip(m['predictions'],m['total_predictions'],target,problem['query_inputs'],scenes,strict=True)):
            if before is None or after is None:continue
            for r in range(len(after)):
                for c in range(len(after[0])):
                    if before[r][c]==after[r][c]:continue
                    f,values=sc[r*len(g[0])+c];z=tuple(f[i] for i in m['features'])
                    entry=next(x for x in t['entries'] if tuple(x[0])==z)
                    removed=[]
                    for a in range(19):
                        if entry[2]&(1<<a) and not entry[4]&(1<<a):
                            witness=None
                            for qj,(gg,ss) in enumerate(zip(problem['query_inputs'],scenes)):
                                for j,(ff,vv) in enumerate(ss):
                                    if tuple(ff[i] for i in m['features'])==z and vv[a]<0:
                                        witness=[qj,j//len(gg[0]),j%len(gg[0])];break
                                if witness is not None:break
                            removed.append({'operation':e.s17.ACTIONS[a],'value_here':values[a],'first_undefined_same_key':witness})
                    new_cells.append({'query':qi,'cell':[r,c],'input':g[r][c],'before':before[r][c],'after':after[r][c],
                        'official_output':y[r][c],'key':list(z),
                        'prior_operations':[e.s17.ACTIONS[a] for a in range(19) if entry[2]&(1<<a)],
                        'remaining_operations':[e.s17.ACTIONS[a] for a in range(19) if entry[4]&(1<<a)],'removed':removed})
        rec['newly_determined_cells']=len([x for x in new_cells if x['after']>=0])
        rec['representative_changes']=new_cells[:2]
        out['models'][policy]=rec
    return out


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    problems={(p['split'],p['id']):p for p in read(a.input/'arc1/problems.json')};answers=read(a.input/'arc1/answers.json')
    stable={(s['split'],s['id']):s for s in map(json.loads,(a.input/'arc1/stable.jsonl').read_text().splitlines())}
    result={'arc1':{},'controlled':{},'audits':{},'oracle':{}};cases=[]
    combined=Counter()
    for domain in ('arc1','controlled'):
        result[domain]={}
        for mode in ('base','default'):
            folder=a.root/domain/mode;result[domain][mode]=read(folder/'summary.json')
            result['audits'][domain+'/'+mode]=[]
            for path in sorted(folder.glob('audit*.json')):
                r=read(path);combined.update(r['counts']);result['audits'][domain+'/'+mode].append(r)
    result['audit_counts']=dict(combined)
    for mode in ('base','default'):
        rows=read(a.root/'arc1'/mode/'predictions.json');oracle=[]
        for row in rows:
            key=(row['split'],row['id']);prob=problems[key];target=answers['/'.join(key)]
            sc=[e.scene(g) for g in prob['query_inputs']];good_tables=set()
            # This is deliberately post-score and checks entire query outputs.
            shapes_ok=all(len(g)==len(y) and len(g[0])==len(y[0]) for g,y in zip(prob['query_inputs'],target,strict=True))
            if shapes_ok:
                for i,t in enumerate(row['tables']):
                    raw,domains=oracle_in_table(t,t['features'],sc,target)
                    if audit.set_condition(raw,domains,mode)[0]['feasible']:good_tables.add(i)
            selected=row['policies']['total_ranked']['selected']
            sm=next((m for m in row['models'] if [m['arm'],m['features']]==selected),None)
            oracle.append({'id':row['id'],'split':row['split'],'any_correct_concrete_program':bool(good_tables),
                'any_unanimously_correct_family':any(m['total_predictions']==target for m in row['models']),
                'selected_family_contains_correct_program':sm is not None and sm['table_id'] in good_tables,
                'selected_prediction_exact':row['policies']['total_ranked']['predictions']==target,
                'selected_prediction_complete':audit.full(row['policies']['total_ranked']['predictions'])})
            if row['id'] in ('a699fb00','d5d6de2d','f76d97a5','27a77e38'):
                cases.append(case(row,prob,target,stable[key]))
        result['oracle'][mode]={split:{'any_correct_concrete_program':sum(r['any_correct_concrete_program'] for r in oracle if r['split']==split),
            'any_unanimously_correct_family':sum(r['any_unanimously_correct_family'] for r in oracle if r['split']==split),
            'complete_errors': [r for r in oracle if r['split']==split and r['selected_prediction_complete'] and not r['selected_prediction_exact']]}
            for split in ('training','evaluation')}
        (a.root/'arc1'/mode/'oracle.json').write_text(json.dumps(oracle,indent=2)+'\n')
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (a.out/'case-certificates.json').write_text(json.dumps(cases,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'audit':result['audit_counts'],'oracle':result['oracle']},indent=2))


if __name__=='__main__':main()
