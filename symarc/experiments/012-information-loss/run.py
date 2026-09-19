"""Retrospective mechanism audit. Synthetic predictions are deliberately unlabelled."""
import argparse, copy, hashlib, importlib.util, itertools, json, os, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
IDS = ['7b5033c1', '8f215267', '97d7923e']

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def load():
    spec = importlib.util.spec_from_file_location('repair_probe', ROOT / 'experiments/006-repair-audit/repair_probe.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

def features(m, task, g, y=None):
    n = m.ns(task)
    if task == IDS[0]:
        seq = [row[0] for row in m.path(g)]
        runs = [c for c, _ in itertools.groupby(seq)]
        hist = n['orderColoursByFirstSeen'](n['tallyColours'](g), n['findFirstPosition'](g))
        return dict(path_runs=runs, single_run_per_colour=len(runs)==len(set(runs)),
                    histogram_order=[c for c,k in hist], path_equals_histogram=m.path(g)==n['p'](g),
                    output_equals_path=None if y is None else y==m.path(g))
    if task == IDS[1]:
        fs=n['extractFrames'](g); bg=n['_most_common_color'](g); right=max(f[-1] for f in fs)
        gg=[[0 if c<=right or v==bg else v for c,v in enumerate(row)] for row in g]
        counts=Counter(g[co[0][0]][co[0][1]] for co in m.components(gg))
        out=[]
        for f in fs:
            color,r0,r1,c0,c1=f
            local=n['lookupStripeCount'](n['sliceInstructionPatch'](g,f))
            cap=len(n['_candidate_positions'](c1-c0-1))
            observed=None if y is None else sum(y[(r0+r1)//2][c0+1+k]==color for k in n['_candidate_positions'](c1-c0-1))
            out.append(dict(frame=list(f), local_count=local, global_count=counts[color],capacity=cap,
                            observed_stripes=observed, counts_agree_after_clipping=min(cap,local)==min(cap,counts[color])))
        return out
    cols=n['parseColumnRuns'](g); ranks={rs[0].color:rs[0].length for rs in cols.values() if rs[0].color!=0}
    groups={}
    for c,rs in cols.items():
        p=n['detectCapPattern'](rs)
        if p: groups.setdefault(p[0].color,[]).append((c,p))
    old=n['p'](g); out=[]
    for color,items in sorted(groups.items()):
        items.sort(key=lambda it:it[1][1].length,reverse=True)
        def filled(a,c,p):
            mid=p[1];return all(a[r][c]==color for r in range(mid.start,mid.start+mid.length))
        out.append(dict(color=color,marker_length=ranks[color],bars_descending=[dict(column=c,length=p[1].length) for c,p in items],
                        original_selected_ranks=[i+1 for i,(c,p) in enumerate(items) if filled(old,c,p)],
                        observed_selected_ranks=None if y is None else [i+1 for i,(c,p) in enumerate(items) if filled(y,c,p)],
                        ties=len({p[1].length for c,p in items})!=len(items)))
    return out

def witness(m,task):
    n=m.ns(task); q=m.REPAIRS[task]
    if task==IDS[0]:
        def grid(seq): return [[0]*7,[0]+list(seq)+[0]*(6-len(seq)),[0]*7]
        a,b=grid([1,1,2]),grid([1,2,1])
        assert n['p'](a)==n['p'](b) and q(a)!=q(b)
        return dict(kind='same histogram and colour order, different path order',input_a=a,input_b=b,
                    original_a=n['p'](a),original_b=n['p'](b),repair_a=q(a),repair_b=q(b),ground_truth=None)
    for i,e in enumerate(m.C[task]['data']['train']):
        a=e['input']
        if task==IDS[1]:
            fs=n['extractFrames'](a);right=max(f[-1] for f in fs);bg=n['_most_common_color'](a)
            for f in fs:
                color,r0,r1,c0,c1=f
                for r in range(len(a)):
                    for c in range(right+1,len(a[0])):
                        if r0<=r<=r1 or a[r][c]!=bg:continue
                        if any(a[rr][cc]!=bg for rr,cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)] if 0<=rr<len(a) and 0<=cc<len(a[0])):continue
                        b=copy.deepcopy(a);b[r][c]=color
                        if n['extractFrames'](b)!=fs or n['_most_common_color'](b)!=bg:continue
                        pa,pb=n['p'](a),n['p'](b);qa,qb=q(a),q(b)
                        lo,hi=r0,r1+1
                        if pa[lo:hi]==pb[lo:hi] and qa[lo:hi]!=qb[lo:hi]:
                            assert n['sliceInstructionPatch'](a,f)==n['sliceInstructionPatch'](b,f)
                            return dict(kind='same target local patch, different global count',training_index=i,target_frame=f,added_cell=[r,c,color],input_a=a,input_b=b,original_a=pa,original_b=pb,repair_a=qa,repair_b=qb,ground_truth=None)
        else:
            for c,d in itertools.combinations(range(len(a[0])),2):
                def swap(g):
                    b=copy.deepcopy(g)
                    for row in b:row[c],row[d]=row[d],row[c]
                    return b
                b=swap(a)
                try: qb=q(b)
                except (KeyError,IndexError,AssertionError):continue
                if qb!=swap(q(a)):continue
                if n['p'](b)!=qb:
                    return dict(kind='column swap preserves marker/rank relation, breaks positional guards',training_index=i,swapped_columns=[c,d],input_a=a,input_b=b,original_a=n['p'](a),original_b=n['p'](b),repair_a=q(a),repair_b=qb,ground_truth=None)
    return None

def case(task):
    m=load();n=m.ns(task);q=m.REPAIRS[task];d=m.C[task]['data']
    expected=json.loads((ROOT/'experiments/006-repair-audit/inputs.json').read_text())['tasks'][task]
    assert hashlib.sha256(m.C[task]['source'].encode()).hexdigest()==expected['source_sha256']
    assert digest(d)==expected['data_canonical_sha256']
    rows={split:[dict(index=i,original_correct=n['p'](e['input'])==e['output'],repair_correct=q(e['input'])==e['output'],features=features(m,task,e['input'],e['output'])) for i,e in enumerate(d[split])] for split in ['train','test']}
    w=witness(m,task);assert w is not None
    return task,dict(provenance=expected,examples=rows,witness=w)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corpus',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    os.environ['ARC_REPAIR_CORPUS']=str(Path(args.corpus).resolve());start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=12) as pool: results=dict(pool.map(case,IDS))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'run.json').write_text(json.dumps(dict(workers=12,seconds=time.perf_counter()-start,seed=None,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest()),indent=2)+'\n')
    for task,r in results.items():
        print(task,json.dumps(r['examples']),r['witness']['kind'])
