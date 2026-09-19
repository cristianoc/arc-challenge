#!/usr/bin/env python3
import concurrent.futures, hashlib, itertools, json, platform, random, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LAWS=['colour','geometry','product']
def geom(g,k):
    a=[row[:] for row in g]
    if k>=4:a=[row[::-1] for row in a]
    for _ in range(k%4):a=[list(row) for row in zip(*a[::-1])]
    return a
def rename(g,m):return [[m[v] for v in row] for row in g]
def transport(a,b,n):
    if len(a)!=len(b) or len(a[0])!=len(b[0]):return None
    m={};inv={}
    for r,s in zip(a,b):
        for x,y in zip(r,s):
            if (x in m and m[x]!=y) or (y in inv and inv[y]!=x):return None
            m[x]=y;inv[y]=x
    return m
def check(es,law,n=10):
    witness=None;cross=set();self_transports=0
    for i,e in enumerate(es):
        for j,f in enumerate(es):
            for k in ([0] if law=='colour' else range(8)):
                x,y=geom(e['input'],k),geom(e['output'],k)
                if law=='geometry':
                    if x!=f['input']:continue
                    partial={c:c for c in range(n)}
                else:
                    partial=transport(x,f['input'],n)
                    if partial is None:continue
                if i!=j:cross.add(tuple(sorted([i,j])))
                else:self_transports+=1
                unused=sorted(set(range(n))-partial.keys());targets=sorted(set(range(n))-set(partial.values()))
                m=dict(partial);m.update(zip(unused,targets));candidates=[m]
                # If the base completion works, every possible discrepancy must
                # involve an output colour absent from the input. One swapped
                # completion witnessing such a discrepancy suffices.
                unknown=sorted({v for row in y for v in row}&set(unused))
                if len(unused)>1 and unknown:
                    a=unknown[0];b=next(v for v in unused if v!=a);alt=m.copy();alt[a],alt[b]=alt[b],alt[a];candidates.append(alt)
                for m in candidates:
                    predicted=rename(y,m)
                    if predicted!=f['output'] and witness is None:
                        witness=dict(source=i,target=j,geometry=k,colour_map=[m[c] for c in range(n)],transformed_output=predicted,expected=f['output'])
                        assert rename(x,m)==f['input']
    return dict(compatible=witness is None,cross_example_pairs=[list(p) for p in sorted(cross)],self_geometry_transports=self_transports,witness=witness)
def brute(es,law,n=3):
    for e in es:
        for f in es:
            for k in ([0] if law=='colour' else range(8)):
                for perm in ([tuple(range(n))] if law=='geometry' else itertools.permutations(range(n))):
                    if rename(geom(e['input'],k),perm)==f['input'] and rename(geom(e['output'],k),perm)!=f['output']:return False
    return True
def tests():
    assert check([{'input':[[0]],'output':[[0]]}],'product')['compatible']
    assert not check([{'input':[[0]],'output':[[1]]}],'colour')['compatible']
    assert not check([{'input':[[0]],'output':[[0,1]]}],'geometry')['compatible']
    assert not check([{'input':[[0]],'output':[[0]]},{'input':[[1]],'output':[[2]]}],'colour')['compatible']
    rng=random.Random(9)
    for _ in range(100):
        es=[]
        for __ in range(rng.randint(1,3)):
            def grid():
                h,w=rng.randint(1,2),rng.randint(1,2);return [[rng.randrange(3) for _ in range(w)] for _ in range(h)]
            es.append(dict(input=grid(),output=grid()))
        for law in LAWS:assert check(es,law,3)['compatible']==brute(es,law)
    return '4 targeted checks and 300 symbolic/brute-force comparisons passed'
def worker(item):
    task,data=item;train={law:check(data['train'],law) for law in LAWS}
    # Only after all training results are fixed, read test pairs for diagnosis.
    extended={law:check(data['train']+data['test'],law) for law in LAWS}
    return dict(task=task,train_count=len(data['train']),train=train,extended=extended)
def main():
    control=tests()
    if sys.argv[1]=='--test':print(control);return
    start=time.monotonic();corpus=Path(sys.argv[1]);dest=Path(sys.argv[2]);dest.mkdir(parents=True,exist_ok=True)
    src=json.loads(corpus.read_text());manifest=json.loads((ROOT/'experiments/006-repair-audit/inputs.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
    assert len(manifest['tasks'])==120
    items=[]
    for task,m in sorted(manifest['tasks'].items()):
        data=src[task]['data'];assert sha(json.dumps(data,sort_keys=True,separators=(',',':')).encode())==m['data_canonical_sha256'];items.append((task,data))
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:rows=list(pool.map(worker,items))
    (dest/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# 009: compatibility of uniformly supplied laws','','| Law | Training incompatible | Training compatible | Compatible with cross-example transports | Compatible then contradicted by test pairs |','|---|---:|---:|---:|---:|']
    for law in LAWS:
        bad=sum(not r['train'][law]['compatible'] for r in rows);cross=sum(r['train'][law]['compatible'] and bool(r['train'][law]['cross_example_pairs']) for r in rows);new=sum(r['train'][law]['compatible'] and not r['extended'][law]['compatible'] for r in rows)
        lines.append(f'| {law} | {bad} | {120-bad} | {cross} | {new} |')
    lines+=['','Compatibility is existence of a fitting equivariant function, not evidence that the intended function obeys the law. Laws are identical across tasks; no task-specific exemptions. Known test answers provide a retrospective diagnostic only.']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    rec=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),changes=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),workers=12,elapsed_seconds=time.monotonic()-start,python=sys.version,platform=platform.platform(),command=sys.argv,controls=control,source_sha256=sha(Path(__file__).read_bytes()),input_hashes={k:v['data_canonical_sha256'] for k,v in manifest['tasks'].items()},artifacts={p.name:sha(p.read_bytes()) for p in dest.iterdir() if p.is_file()})
    (dest/'run.json').write_text(json.dumps(rec,indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
