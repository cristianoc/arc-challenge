#!/usr/bin/env python3
import concurrent.futures, hashlib, json, subprocess, sys, time, platform
from pathlib import Path
NAMES=['identity','rotate90','rotate180','rotate270','reflect_lr','reflect_antidiagonal','reflect_ud','reflect_diagonal']
def geo(g,k):
    a=[r[:] for r in g]
    if k>=4:a=[r[::-1] for r in a]
    for _ in range(k%4):a=[list(r) for r in zip(*a[::-1])]
    return a
def key(g):return tuple(tuple(r) for r in g)
def informative(g,k):
    h,w=len(g),len(g[0]);coords=[[r*w+c for c in range(w)] for r in range(h)]
    return geo(coords,k)!=coords
def apply(g,op):
    kind,arg=op
    if kind=='geometry':return geo(g,arg)
    a,b=arg;return [[b if v==a else a if v==b else v for v in row] for row in g]
def order(op):return 4 if op[0]=='geometry' and op[1] in [1,3] else 2
def compatible(es,op):
    lookup={}
    for i,e in enumerate(es):lookup.setdefault(key(e['input']),[]).append(i)
    self_count=0;cross=set();changed_cross=set();witness=None
    for i,e in enumerate(es):
        x,y=e['input'],e['output']
        for power in range(order(op)):
            for j in lookup.get(key(x),[]):
                if power:
                    if i==j:self_count+=1
                    else:
                        cross.add((i,j,power))
                        if x!=e['input']:changed_cross.add((i,j,power))
                if y!=es[j]['output'] and witness is None:witness=dict(source=i,target=j,power=power)
            x,y=apply(x,op),apply(y,op)
    return dict(compatible=witness is None,self_transports=self_count,cross_transports=len(cross),changed_input_cross_transports=len(changed_cross),witness=witness)
def audit(data):
    train=data['train'];patterns=[];laws=[]
    for k in range(1,8):
        xs=[i for i,e in enumerate(train) if informative(e['input'],k) and geo(e['input'],k)==e['input']]
        preserves=bool(xs) and all(geo(train[i]['output'],k)==train[i]['output'] for i in xs)
        outputs=len(train)>=2 and all(informative(e['output'],k) and geo(e['output'],k)==e['output'] for e in train)
        if outputs or (preserves and len(xs)>=2):patterns.append(dict(k=k,output=outputs,preservation=preserves and len(xs)>=2,support=xs,all_training=len(xs)==len(train),monochrome_outputs=all(len({v for r in e['output'] for v in r})==1 for e in train)))
    ops=[('geometry',k) for k in range(1,8)]+[('colour',[a,b]) for a in range(10) for b in range(a+1,10)]
    for op in ops:laws.append((op,compatible(train,op)))
    # Candidate pattern/law results fixed before inspecting any test output.
    flags=[];law_flags=[]
    for p in patterns:
        k=p['k']
        for j,e in enumerate(data['test']):
            ybad=geo(e['output'],k)!=e['output'];xfixed=geo(e['input'],k)==e['input']
            if p['output'] and ybad and informative(e['output'],k):flags.append(dict(kind='output',test=j,transform=NAMES[k],**p,input_still_symmetric=xfixed))
            if p['preservation'] and xfixed and informative(e['input'],k) and ybad:flags.append(dict(kind='preservation',test=j,transform=NAMES[k],**p,input_still_symmetric=True))
    for op,tr in laws:
        if tr['compatible']:
            ext=compatible(train+data['test'],op)
            if not ext['compatible']:law_flags.append(dict(operation=op,training=tr,extended=ext))
    return dict(training_count=len(train),test_count=len(data['test']),patterns=patterns,compatible_laws=sum(t['compatible'] for _,t in laws),flags=flags,law_flags=law_flags)
def controls():
    sym=[[1,0,1],[0,2,0]];bad=[[1,0,0],[0,2,0]]
    d=dict(train=[dict(input=sym,output=sym)]*2,test=[dict(input=sym,output=bad)])
    assert any(f['kind']=='preservation' and f['k']==4 for f in audit(d)['flags'])
    d['test'][0]['input']=bad
    assert any(f['kind']=='output' and f['k']==4 and not f['input_still_symmetric'] for f in audit(d)['flags'])
    assert not any(f['kind']=='preservation' and f['k']==4 for f in audit(d)['flags'])
    assert not informative([[1,2]],6)
    es=[dict(input=[[1]],output=[[1]]),dict(input=[[2]],output=[[3]])]
    assert not compatible(es,('colour',[1,2]))['compatible']
    assert all(compatible([dict(input=sym,output=sym)],('geometry',k))['compatible'] for k in range(1,8))
    return '5 targeted controls passed'
def worker(p):
    data=json.loads(p.read_text());return dict(task=p.stem,split=p.parent.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**audit(data))
def main():
    control=controls()
    if sys.argv[1]=='--test':print(control);return
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();assert revision=='f3283f727488ad98fe575ea6a5ac981e4a188e49'
    ps=sorted((root/'data').glob('*/*.json'));assert len(ps)==1120
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:rows=list(pool.map(worker,ps))
    (out/'results.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    leads=[dict(task=r['task'],split=r['split'],training_count=r['training_count'],**f) for r in rows for f in r['flags'] if f['kind']=='preservation']
    leads.sort(key=lambda r:(not r['all_training'],-len(r['support']),r['task'],r['k'],r['test']))
    selected=[];seen=set()
    for r in leads:
        if r['task'] not in seen:selected.append(r);seen.add(r['task'])
        if len(selected)==6:break
    (out/'inspection-selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    lines=['# 010: symmetry patterns broken by intended outputs','','| Corpus split | Tasks | Output-pattern flags (tasks) | Preservation flags (tasks) | All-training preservation (tasks) | Newly contradicted cyclic laws (tasks) |','|---|---:|---:|---:|---:|---:|']
    for split in ['training','evaluation']:
        rs=[r for r in rows if r['split']==split];n=lambda pred:sum(any(pred(f) for f in r['flags']) for r in rs)
        lines.append(f"| {split} | {len(rs)} | {n(lambda f:f['kind']=='output')} | {n(lambda f:f['kind']=='preservation')} | {n(lambda f:f['kind']=='preservation' and f['all_training'])} | {sum(bool(r['law_flags']) for r in rs)} |")
    lines+=['','Counts overlap. Flags are counterexamples to selected observed patterns, not certified task defects. See results for every pattern, support count and flagged query.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    manifest=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),changes=subprocess.check_output(['git','status','--porcelain'],text=True).splitlines(),data_revision=revision,command=sys.argv,workers=12,elapsed_seconds=time.monotonic()-start,python=sys.version,platform=platform.platform(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),controls=control,artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    (out/'run.json').write_text(json.dumps(manifest,indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
