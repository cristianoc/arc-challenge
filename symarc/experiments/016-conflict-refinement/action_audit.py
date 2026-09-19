"""Registered, training-only sensitivity to adding neighbour-copy operations.

The script uses only demonstrations and the frozen training-conflict coordinates.
It never scores query outputs or changes the primary learner.
"""
import argparse
from collections import defaultdict,Counter
import importlib.util
import json
from pathlib import Path
import time
spec=importlib.util.spec_from_file_location('independent016',Path(__file__).with_name('audit.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
OFFSETS=[('C',0,0),('N',-1,0),('S',1,0),('W',0,-1),('E',0,1)]


def allowed(g,r,c,y,extended):
    values={f'constant:{y}'}
    for name,dr,dc in OFFSETS if extended else OFFSETS[:1]:
        rr,cc=r+dr,c+dc
        if 0<=rr<len(g) and 0<=cc<len(g[0]) and g[rr][cc]==y:values.add('copy:'+name)
    return values


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--training',type=Path,required=True);p.add_argument('--certificates',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();start=time.perf_counter()
    training=json.loads(args.training.read_text());certs=json.loads(args.certificates.read_text());results=[];summary=defaultdict(Counter)
    for problem in training:
        key=problem['split']+'/'+problem['id'];pairs=problem['train']
        if problem['id'] in ('00d62c1b','e88171ec'):continue
        if len({json.dumps(e['input']) for e in pairs})<2 or any((len(e['input']),len(e['input'][0]))!=(len(e['output']),len(e['output'][0])) for e in pairs):continue
        tables=[{},{}]
        for e in pairs:
            g=e['input'];fs=audit.reference_features(g);w=len(g[0])
            for i,f in enumerate(fs):
                r,c=divmod(i,w);y=e['output'][r][c]
                for ext,table in enumerate(tables):
                    v=allowed(g,r,c,y,bool(ext));table[f]=table[f]&v if f in table else v
        old=all(tables[0].values());new=all(tables[1].values());assert new or not old
        item={'id':problem['id'],'split':problem['split'],'old_full_partition_fit':old,'extended_full_partition_fit':new}
        counts=summary[problem['split']];counts['eligible']+=1;counts['old_full_partition_fits']+=old;counts['extended_full_partition_fits']+=new
        if key in certs:
            endpoints=certs[key];old_common=None;new_common=None
            for endpoint in endpoints:
                d,r,c=endpoint['origin'];e=pairs[d];g=e['input'];y=e['output'][r][c]
                a,b=allowed(g,r,c,y,False),allowed(g,r,c,y,True)
                old_common=a if old_common is None else old_common&a
                new_common=b if new_common is None else new_common&b
            assert not old_common and not old
            item['pair_resolved']=bool(new_common);item['shared_actions']=sorted(new_common)
            counts['primary_conflict_pairs']+=1;counts['pairs_resolved']+=bool(new_common)
            if new_common:
                item['endpoints']=endpoints
                counts['pair_resolved_and_whole_task_fit']+=new
        results.append(item)
    result={'summary':{s:dict(c) for s,c in summary.items()},'tasks':results,'seconds':time.perf_counter()-start,
            'query_scored':False,'protocol_commit':'8bc397aa20e2707233c6b722afdde3db647d39a3'}
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
