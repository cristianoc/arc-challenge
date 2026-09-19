"""Post-primary, preregistered training-only shared-operation sensitivity."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import time

spec=importlib.util.spec_from_file_location('audit017',Path(__file__).with_name('audit.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
DEV={'00d62c1b','e88171ec'}


def majority(g,r,c):
    neighbours=[g[i][j] for i,j in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                if 0<=i<len(g) and 0<=j<len(g[0])]
    counts=Counter(neighbours)
    winners=[v for v,k in counts.items() if 2*k>len(neighbours)]
    return winners[0] if len(winners)==1 else -1


def case(p):
    pairs=p['train']
    if p['id'] in DEV or len({json.dumps(e['input']) for e in pairs})<2 or any(
        (len(e['input']),len(e['input'][0]))!=(len(e['output']),len(e['output'][0])) for e in pairs):return None
    old={};new={};origins={};representatives={};known=[]
    for d,e in enumerate(pairs):
        g=e['input'];w=len(g[0]);sc=audit.scene(g)
        for i,((f,values),y) in enumerate(zip(sc,sum(e['output'],[]),strict=True)):
            r,c=divmod(i,w);a={j for j,v in enumerate(values) if v==y};v=majority(g,r,c);b=a|({19} if v==y else set())
            old[f]=old.get(f,set(range(19)))&a;new[f]=new.get(f,set(range(20)))&b
            origins.setdefault(f,[]).append([d,r,c]);representatives.setdefault(f,(g[r][c],y))
            if p['id']=='7e0986d6' and d==0 and (r,c) in ((2,10),(5,5),(8,12),(11,12)):
                known.append({'origin':[d,r,c],'input':g[r][c],'output':y,'neighbours':list(values[11:15]),
                              'majority':v,'old_allowed':sorted(a),'features':list(f)})
    old_ok=all(old.values());new_ok=all(new.values())
    assert new_ok or not old_ok
    result={'id':p['id'],'split':p['split'],'old_full_fit':old_ok,'extended_full_fit':new_ok,
            'classes':len(old),'old_conflicting_classes':sum(not s for s in old.values()),
            'remaining_conflicting_classes':sum(not s for s in new.values())}
    if new_ok and not old_ok:
        # All classes that the old language could not explain need the new action.
        added=[f for f in old if not old[f]]
        assert all(new[f]=={19} for f in added)
        result['resolved_classes']=[{'features':list(f),'occurrences':len(origins[f]),
                                    'first_origin':origins[f][0],'common_actions':sorted(new[f])} for f in added]
    if known:
        assert len(known)==4 and len({tuple(k['features']) for k in known})==1
        sets=[set(k['old_allowed']) for k in known]
        assert not set.intersection(*sets)
        assert all(set.intersection(*(sets[:i]+sets[i+1:])) for i in range(4))
        assert all(k['majority']==k['output'] for k in known)
        result['four_point_case']=known
        if not new_ok:
            f=next(f for f,a in new.items() if not a)
            result['remaining_witness_class']={'features':list(f),'origins':origins[f]}
    return result


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--training',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();problems=json.loads(a.training.read_text());assert all('query_inputs' not in p and 'test' not in p for p in problems)
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=12) as pool:tasks=[r for r in pool.map(case,problems) if r is not None]
    summary={}
    for split in ('training','evaluation'):
        rs=[r for r in tasks if r['split']==split]
        summary[split]={'eligible':len(rs),'old_full_fits':sum(r['old_full_fit'] for r in rs),
                        'extended_full_fits':sum(r['extended_full_fit'] for r in rs),
                        'new_fits':[r['id'] for r in rs if r['extended_full_fit'] and not r['old_full_fit']]}
    result={'summary':summary,'tasks':tasks,'workers':12,'seconds':time.perf_counter()-start,
            'query_scored':False,'protocol_commit':'bb8bf2e2dec73e9831f39627cd79a189bc195b9d',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training_sha256':hashlib.sha256(a.training.read_bytes()).hexdigest()}
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
    print('known_case',json.dumps(next(r for r in tasks if r['id']=='7e0986d6'),indent=2))

if __name__=='__main__':main()
