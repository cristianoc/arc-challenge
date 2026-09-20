"""Post-run independent teacher, scoring and selected-action-table audit.

Uses the previously verified input-only primitive evaluator, NOT the learner's
fit/compress/apply procedures. It replays outer choices but does not reconstruct
outer selection independently. All post-run diagnostics leave predictions fixed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction
import importlib.util
import json
from pathlib import Path

s=importlib.util.spec_from_file_location('design022',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
N=19


def expected(grid, kind, farther):
    h,w=len(grid),len(grid[0]); rows=[]
    for r,row in enumerate(grid):
        if row[0]!=0 and row[-1]!=0 and all(v==0 for v in row[1:-1]):rows.append(r)
    assert len(rows)==1
    active=rows[0];left,right=grid[active][0],grid[active][-1]
    assert all(v==0 for r,row in enumerate(grid) if r!=active for v in row)
    y=[row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if r!=active:y[r][c]=6 if kind=='literal_background' else 0
            elif 0<c<w-1:
                if 2*c==w-1:y[r][c]=5
                else:
                    choose_right=(2*c>w-1)^farther
                    y[r][c]=(6 if not choose_right else 8) if kind=='literal_fill' else (right if choose_right else left)
    return y


def make_table(problem, indices, fs, mode, scenes):
    raw={}
    for i in indices:
        targets=[v for row in problem['train'][i]['output'] for v in row]
        for (features,values), y in zip(scenes[i],targets,strict=True):
            k=tuple(features[j] for j in fs)
            possible={a for a in range(N) if values[a]==y}
            if k not in raw:raw[k]=possible
            else:raw[k]&=possible
    if any(not v for v in raw.values()):return None,raw
    if mode=='base' or not raw:return raw,raw
    costs={a:sum(a not in v for v in raw.values()) for a in range(N)}
    best=min(costs.values());ds={a for a,c in costs.items() if c==best}
    projected={}
    for k,allowed in raw.items():
        kept=set()
        for a in ds:kept |= {a} if a in allowed else allowed
        projected[k]=kept
    return projected,raw


def apply(grid, scene, fs, table):
    if table is None:return None
    flat=[]
    for f,values in scene:
        ops=table.get(tuple(f[i] for i in fs))
        outputs={values[a] for a in ops} if ops else set()
        flat.append(next(iter(outputs)) if len(outputs)==1 and -1 not in outputs else -1)
    w=len(grid[0]);return [flat[i:i+w] for i in range(0,len(flat),w)]


def stats(pred,y):
    complete=pred is not None and bool(pred) and all(row and len(row)==len(pred[0]) for row in pred) and all(type(v)==int and 0<=v<=9 for row in pred for v in row)
    known=sum(v>=0 for row in pred for v in row) if pred is not None else 0
    correct=sum(v>=0 and v==b for rowa,rowb in zip(pred,y,strict=True) for v,b in zip(rowa,rowb,strict=True)) if pred is not None else 0
    return dict(complete=complete,exact=pred==y,correct=correct,known=known,wrong=known-correct,cells=sum(map(len,y)))


def conflict_certificate(problem,fs,scenes):
    buckets=defaultdict(list)
    for i,(pair,sc) in enumerate(zip(problem['train'],scenes,strict=True)):
        w=len(pair['input'][0]);labels=[v for row in pair['output'] for v in row]
        for k,((f,vs),y) in enumerate(zip(sc,labels,strict=True)):
            allowed={a for a in range(N) if vs[a]==y}
            buckets[tuple(f[j] for j in fs)].append(dict(demonstration=i,cell=[k//w,k%w],label=y,actions=allowed))
    def intersection(items):
        out=set(range(N))
        for i in items:out&=i['actions']
        return out
    for key,items in sorted(buckets.items()):
        if intersection(items):continue
        # Deletion-minimal witness; every actual training occurrence is retained.
        dedup={tuple(sorted(x['actions'])):x for x in items}
        core=list(dedup.values());i=0
        while i<len(core):
            rest=core[:i]+core[i+1:]
            if rest and not intersection(rest):core=rest
            else:i+=1
        assert not intersection(core)
        assert all(intersection(core[:i]+core[i+1:]) for i in range(len(core)))
        return dict(features=list(fs),key=list(key),points=[{**x,'actions':sorted(x['actions'])} for x in core])
    return None


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root;inputs={p['id']:p for p in json.loads((root/'input/problems.json').read_text())}
    meta=json.loads((root/'input/metadata.json').read_text());answers=json.loads((root/'input/answers.json').read_text())
    assert len(inputs)==len(meta)==len(answers)==24
    teacher_grids=0;checks=dict(query_replays=0,query_scores=0,inner_replays=0,outer_replays=0,full_selection_checks=0)
    for pid,p in inputs.items():
        assert sum(len(x['output'])*len(x['output'][0]) for x in p['train'])==90
        t=meta[pid]
        for pair in p['train']:
            assert expected(pair['input'],t['kind'],t['farther'])==pair['output'];teacher_grids+=1
        for grid,y in zip(p['query_inputs'],answers[pid],strict=True):
            assert expected(grid,t['kind'],t['farther'])==y;teacher_grids+=1
    modes={};witnesses=[];mechanisms=[]
    for mode in ('base','default'):
        rows=json.loads((root/mode/'predictions.json').read_text())
        run=json.loads((root/mode/'prediction-run.json').read_text())
        assert m.sha(root/mode/'predictions.json')==run['predictions_sha256']
        scores={s['id']:s for s in json.loads((root/mode/'scores.json').read_text())}
        assert set(scores)==set(inputs)=={r['id'] for r in rows}
        modes[mode]={}
        for row in rows:
            pid=row['id'];p=inputs[pid];t=meta[pid];score=scores[pid]
            scenes=[m.e21.e18.scene(e['input']) for e in p['train']]
            query_scenes=[m.e21.e18.scene(x) for x in p['query_inputs']]
            all_indices=tuple(range(len(p['train'])))
            models=[x for family in row['families'].values() for x in family['models']]
            chosen=min(models,key=lambda z:(-z['cv_exact'],-Fraction(*z['cv_fraction']),len(z['features']),z['cost'],list(m.e21.e18.ARMS).index(z['arm']),tuple(z['features'])))
            policy=row['policies']['union'];assert policy['selected']==[chosen['arm'],chosen['features']]
            checks['full_selection_checks']+=1
            fs=tuple(chosen['features']);tab,raw=make_table(p,all_indices,fs,mode,scenes)
            assert tab is not None
            for i,(grid,sc) in enumerate(zip(p['query_inputs'],query_scenes,strict=True)):
                predicted=apply(grid,sc,fs,tab)
                assert predicted==policy['predictions'][i]==chosen['predictions'][i]
                checks['query_replays']+=1
                s=stats(predicted,answers[pid][i]);bank=t['queries'][i]['bank']
                scored=next(v for v in score['banks'][bank]['outcomes'] if v['query_index']==i)
                assert s['complete']==scored['complete'] and s['exact']==scored['correct']
                assert (s['complete'] and not s['exact'])==scored['wrong_complete'];checks['query_scores']+=1
            for fold in chosen['validation']:
                idx=tuple(i for i in all_indices if i not in fold['excluded']);table,_=make_table(p,idx,fs,mode,scenes)
                for i,reported in zip(fold['excluded'],fold['scores'],strict=True):
                    s=stats(apply(p['train'][i]['input'],scenes[i],fs,table),p['train'][i]['output'])
                    assert all(s[k]==reported[k] for k in s);checks['inner_replays']+=1
            for fold in policy['outer']:
                idx=tuple(i for i in all_indices if i not in fold['excluded']);selection=fold['selected']
                outerfs=tuple(selection[1]) if selection else ()
                table=make_table(p,idx,outerfs,mode,scenes)[0] if selection else None
                for i,reported in zip(fold['excluded'],fold['scores'],strict=True):
                    s=stats(apply(p['train'][i]['input'],scenes[i],outerfs,table),p['train'][i]['output'])
                    assert all(s[k]==reported[k] for k in s);checks['outer_replays']+=1
            for bank in ('crossed','legacy'):
                ss=score['banks'][bank]
                assert ss['correct']==sum(x['correct'] for x in ss['outcomes'])
                assert ss['wrong_complete']==sum(x['wrong_complete'] for x in ss['outcomes'])
                assert ss['incomplete']==sum(not x['complete'] for x in ss['outcomes'])
            modes[mode][t['name']]={k:score[k] for k in ('selected','cv_exact','cv_fraction','outer_pass','names')}
            if t['palette_diverse'] and t['row_diverse']:
                oldfs=(2,4,24) if t['kind']=='literal_background' else (4,24)
                cert=conflict_certificate(p,oldfs,scenes)
                assert cert is not None
                if mode=='base':witnesses.append(dict(id=pid,name=t['name'],**cert))
                fixedfs=(24,27,28);fixedtab,_=make_table(p,all_indices,fixedfs,mode,scenes)
                assert fixedtab is not None
                fixedpred=[apply(x,sc,fixedfs,fixedtab) for x,sc in zip(p['query_inputs'],query_scenes)]
                fixedcorrect={bank:sum(pred==y for pred,y,q in zip(fixedpred,answers[pid],t['queries'],strict=True) if q['bank']==bank) for bank in ('crossed','legacy')}
                assert fixedcorrect==dict(crossed=36,legacy=14)
                entry=dict(mode=mode,name=t['name'],selected=policy['selected'],fixed_endpoint_correct=fixedcorrect)
                if t['kind']=='literal_background':
                    fixed=next(z for z in models if z['arm']=='guarded_order' and z['features']==list(fixedfs))
                    entry.update(selected_fraction=chosen['cv_fraction'],fixed_fraction=fixed['cv_fraction'],selected_cost=chosen['cost'],fixed_cost=fixed['cost'])
                    entry['crossed_unknown_cells']=sum(v==-1 for g in policy['predictions'][:36] for rr in g for v in rr)
                    entry['crossed_wrong_known_cells']=sum(v>=0 and v!=target for g,y in zip(policy['predictions'][:36],answers[pid][:36]) for rr,yy in zip(g,y) for v,target in zip(rr,yy))
                    assert entry['crossed_unknown_cells']==72 and entry['crossed_wrong_known_cells']==0
                mechanisms.append(entry)
        legacy=[s for pid,s in scores.items() if meta[pid]['palette_diverse'] and not meta[pid]['row_diverse']]
        assert [sum(s['banks']['legacy'][k] for s in legacy) for k in ('correct','wrong_complete','incomplete')]==[42,18,24]
    out=dict(teacher_grids_checked=teacher_grids,training_specifications=24,labelled_cells_per_specification=90,
             **checks,legacy_counts_match_021=True,removed_guard_certificates=witnesses,mechanisms=mechanisms,
             limitation='Shared input-feature/action evaluator; no independent outer search/selection implementation')
    m.write(root/'audit.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ('mechanisms','removed_guard_certificates')},indent=2))

if __name__=='__main__':main()
