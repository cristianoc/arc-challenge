#!/usr/bin/env python3
"""Fixed-pair diagnostic; symmetry contracts are assumptions, not inferred labels."""
import concurrent.futures, copy, hashlib, importlib.util, itertools, json, os
import platform, signal, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CORPUS=Path(os.environ.get('ARC_REPAIR_CORPUS',ROOT/'out/experiments/006-repair-audit/corpus.json'))
os.environ['ARC_REPAIR_CORPUS']=str(CORPUS)
AUDIT=ROOT/'experiments/006-repair-audit'
spec=importlib.util.spec_from_file_location('repairs',AUDIT/'repair_probe.py')
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
manifest=json.loads((AUDIT/'inputs.json').read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
for task in repair.REPAIRS:
    data=repair.C[task]
    assert sha(data['source'].encode())==manifest['tasks'][task]['source_sha256']
    assert sha(json.dumps(data['data'],sort_keys=True,separators=(',',':')).encode())==manifest['tasks'][task]['data_canonical_sha256']
# Only training pairs remain accessible to the diagnostic workers.
repair.C={k:{'source':v['source'],'data':{'train':v['data']['train']}} for k,v in repair.C.items() if k in repair.REPAIRS}

def alarm(*_):raise TimeoutError('5 second candidate limit')
def call(f,x):
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,5)
    try:return {'output':f(copy.deepcopy(x))}
    except Exception as e:return {'error':type(e).__name__+': '+str(e)}
    finally:signal.setitimer(signal.ITIMER_REAL,0)

def transform(g,kind,args):
    if kind=='colour':
        a,b=args;return [[b if v==a else a if v==b else v for v in row] for row in g]
    if kind=='reflection':return [row[::-1] for row in g]
    if kind=='transpose':return [list(row) for row in zip(*g)]
    return copy.deepcopy(g)

def controls():
    g=[[0,1,2],[2,1,0]]
    assert transform(g,'identity',[])==g
    assert transform(g,'colour',[1,2])==[[0,2,1],[1,2,0]]
    for kind,args in [('colour',[1,2]),('reflection',[]),('transpose',[])]:
        assert transform(transform(g,kind,args),kind,args)==g
    # Both histogram and path commute with colour renaming, but can disagree.
    x=[[0,0,0,0,0],[0,1,2,0,0],[0,0,3,1,0],[0,0,0,0,0]]
    def hist(g):
        vs=[v for row in g for v in row if v];return [[v] for v in dict.fromkeys(vs) for _ in range(vs.count(v))]
    assert hist(x)!=repair.path(x)
    for a,b in itertools.combinations(range(1,10),2):
        for f in [hist,repair.path]:assert f(transform(x,'colour',[a,b]))==transform(f(x),'colour',[a,b])

def worker(task):
    source=repair.ns(task);fs={'original':source['p'],'repair':repair.REPAIRS[task]}
    fixed={0,5} if task=='e3721c99' else {0,3,4} if task=='221dfab4' else {0}
    ops=[('identity',[])]+[('colour',list(p)) for p in itertools.combinations(sorted(set(range(10))-fixed),2)]
    if task=='1ae2feb7':ops.append(('reflection',[]))
    if task in ['135a2760','221dfab4']:ops.append(('transpose',[]))
    counts={name:{} for name in fs};witnesses=[];seen={};identities=0;total=0
    for i,e in enumerate(repair.C[task]['data']['train']):
        for kind,args in ops:
            x,y=transform(e['input'],kind,args),transform(e['output'],kind,args)
            key=json.dumps([x,y],separators=(',',':'));seen[key]=1
            if kind!='identity' and x==e['input'] and y==e['output']:identities+=1
            total+=1
            for name,f in fs.items():
                result=call(f,x);ok='output' in result and result['output']==y
                co=counts[name].setdefault(kind,{'passed':0,'total':0,'errors':0})
                co['passed']+=int(ok);co['total']+=1;co['errors']+=int('error' in result)
                if not ok and not any(w['candidate']==name and w['kind']==kind for w in witnesses):
                    witnesses.append(dict(task=task,candidate=name,kind=kind,args=args,training_index=i,input=x,expected=y,actual=result))
    survive={name:all(v['passed']==v['total'] for v in cs.values()) for name,cs in counts.items()}
    outcome='both' if all(survive.values()) else 'repair-only' if survive['repair'] else 'original-only' if survive['original'] else 'neither'
    assert all(cs['identity']['passed']==cs['identity']['total'] for cs in counts.values())
    return dict(task=task,fixed_colours=sorted(fixed),counts=counts,outcome=outcome,probe_count=total,distinct_labelled_probes=len(seen),nonidentity_ops_acting_identically=identities,witnesses=witnesses)

def main():
    controls();start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:rows=list(pool.map(worker,sorted(repair.REPAIRS)))
    dest=Path(sys.argv[1]);dest.mkdir(parents=True,exist_ok=True)
    (dest/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# 008 fixed repair-pair constraint audit','','| Task | Original colour | Repair colour | Original geometry | Repair geometry | Survivors |','|---|---:|---:|---|---|---|']
    def fmt(cs,kind):
        s=cs.get(kind);return f"{s['passed']}/{s['total']}" if s else '—'
    for r in rows:
        a,b=r['counts']['original'],r['counts']['repair'];geo='reflection' if 'reflection' in a else 'transpose'
        lines.append(f"| {r['task']} | {fmt(a,'colour')} | {fmt(b,'colour')} | {fmt(a,geo)} | {fmt(b,geo)} | {r['outcome']} |")
    lines+=['',f"{sum(r['probe_count'] for r in rows)} probes per candidate family; {sum(r['distinct_labelled_probes'] for r in rows)} distinct labelled probes summed across tasks; {sum(r['nonidentity_ops_acting_identically'] for r in rows)} nonidentity operations act identically.", '', 'Contracts are inherited retrospective assumptions. No official test output was used by workers. Survival is consistency, not proof of intended semantics.']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    record={'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'changes':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),'workers':12,'elapsed_seconds':time.monotonic()-start,'python':sys.version,'platform':platform.platform(),'source_sha256':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [Path(__file__),AUDIT/'repair_probe.py',AUDIT/'inputs.json']},'inputs':{t:manifest['tasks'][t] for t in sorted(repair.REPAIRS)},'controls':'passed','artifacts':{p.name:sha(p.read_bytes()) for p in dest.iterdir() if p.is_file()}}
    (dest/'run.json').write_text(json.dumps(record,indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
