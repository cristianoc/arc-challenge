"""Supplementary training-only audit after the prespecified coverage failure.

The colour refinement is an expressivity diagnostic, not a tested selector.
No query input/output is accessed. Direct action enumeration independently
checks the fitting intersections, and rejects produce two-label certificates.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

spec=importlib.util.spec_from_file_location('roles015',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def observations(pairs, fs, refine):
    groups=defaultdict(list)
    for i,(e,keys) in enumerate(zip(pairs,fs,strict=True)):
        w=len(e['input'][0])
        for j,(key,x,y) in enumerate(zip(keys,(v for row in e['input'] for v in row),(v for row in e['output'] for v in row),strict=True)):
            z=(key,x) if refine else key
            groups[z].append((i,j//w,j%w,x,y))
    return groups


def allowed(rows):
    # Independent direct action evaluation (no bitsets or calls to fit).
    return {a for a in range(11) if all((x if a==10 else a)==y for _,_,_,x,y in rows)}


def witness(groups):
    for key,rows in groups.items():
        changing=next((p for p in rows if p[3]!=p[4]),None)
        if changing is not None:
            other=next((p for p in rows if p[4]!=changing[4]),None)
            if other is not None:
                assert not allowed([changing,other])
                return {'role':key,'points':[changing,other],
                        'same_input_colour':changing[3]==other[3],
                        'within_demonstration':changing[0]==other[0]}
    return None


def case(p):
    row={'id':p['id'],'split':p['split'],'development':p['id']=='e88171ec', 'eligible':False,'reps':[]}
    if len({m.b.signature(e['input']) for e in p['train']})<2 or any(m.b.shape(e['input'])!=m.b.shape(e['output']) for e in p['train']): return row
    row['eligible']=True
    for rep in m.REPS[8:]:
        fs=[m.context(e['input'],rep) for e in p['train']]
        if any(f is None for f in fs):
            row['reps'].append({'representation':rep,'domain':False});continue
        groups=observations(p['train'],fs,False)
        ordinary={z:allowed(rows) for z,rows in groups.items()}
        fit=all(ordinary.values())
        actual=m.fit(p['train'],fs,'copy')
        assert (actual is not None)==fit
        if fit:
            assert all({a for a in range(11) if actual[z] & (1<<a)}==acts for z,acts in ordinary.items())
        refined=observations(p['train'],fs,True)
        refined_fit=all(allowed(rows) for rows in refined.values())
        assert not fit or refined_fit
        per_demo=all(all(allowed(rows) for rows in observations([e],[f],False).values()) for e,f in zip(p['train'],fs,strict=True))
        proof=None if fit else witness(groups)
        assert fit or proof is not None
        row['reps'].append({'representation':rep,'domain':True,'fit':fit,'colour_refinement_fits':refined_fit,
                           'fits_each_demo_separately':per_demo,'witness':proof})
    return row


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--problems',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    problems=json.loads(a.problems.read_text())
    problems=[{k:v for k,v in p.items() if k!='query_inputs'} for p in problems]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=12) as pool:rows=list(pool.map(case,problems))
    summary={}
    for split in ('training','evaluation'):
        rr=[r for r in rows if r['split']==split and r['eligible'] and not r['development']]
        c=Counter(eligible=len(rr))
        for r in rr:
            defined=[s for s in r['reps'] if s['domain']]
            c['any_defined_structure']+=bool(defined)
            c['original_fit']+=any(s['fit'] for s in defined)
            c['colour_refinement_fit']+=any(s['colour_refinement_fits'] for s in defined)
            c['any_candidate_fits_each_demo']+=any(s['fits_each_demo_separately'] for s in defined)
            for s in defined:
                c['defined_candidates']+=1;c['fitting_candidates']+=s['fit']
                if not s['fit']:
                    c['conflict_candidates']+=1
                    c['conflict_only_across_demos']+=s['fits_each_demo_separately']
                    c['conflict_already_within_demo']+=not s['fits_each_demo_separately']
        summary[split]=dict(c)
    m.b.write_json(a.out/'training-audit.json',rows)
    m.b.write_json(a.out/'training-audit-summary.json',summary)
    m.b.write_json(a.out/'training-audit-run.json',{'workers':12,'seconds':time.perf_counter()-start,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'audit_sha256':hashlib.sha256((a.out/'training-audit.json').read_bytes()).hexdigest()})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
