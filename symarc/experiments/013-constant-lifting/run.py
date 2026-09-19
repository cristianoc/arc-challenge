"""Uniform single-literal repairs. `predict` never receives query labels."""
import argparse, ast, copy, hashlib, json, signal, subprocess, time, multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE=Path(__file__).resolve().parent
MANIFEST=HERE.parent/'006-repair-audit/inputs.json'
FEATURES=['height','width','min_dim','max_dim','height_minus_1','width_minus_1','colours','foreground_colours','min_population','max_population','components','max_component_area']
BINDING='__arc013_input_feature'

class BudgetExpired(Exception):pass

def alarm(signum,frame):raise BudgetExpired()

def bounded(fn,budget):
    signal.signal(signal.SIGPROF,alarm);signal.setitimer(signal.ITIMER_PROF,budget)
    try:return fn(),None
    except BudgetExpired:return None,'cpu_timeout'
    except Exception as e:return None,type(e).__name__
    finally:signal.setitimer(signal.ITIMER_PROF,0)

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def features(g):
    h,w=len(g),len(g[0]);counts=Counter(v for row in g for v in row)
    bg=min(counts,key=lambda c:(-counts[c],c));pop=[n for c,n in counts.items() if c!=bg]
    cells={(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v!=bg};areas=[]
    while cells:
        root=min(cells);cells.remove(root);stack=[root];area=0
        while stack:
            r,c=stack.pop();area+=1
            for rr,cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if (rr,cc) in cells and g[rr][cc]==g[r][c]:cells.remove((rr,cc));stack.append((rr,cc))
        areas.append(area)
    return dict(zip(FEATURES,[h,w,min(h,w),max(h,w),h-1,w-1,len(counts),len(pop),min(pop,default=0),max(pop,default=0),len(areas),max(areas,default=0)]))

def sites(source):
    tree=ast.parse(source);found={}
    def collect(node):
        if isinstance(node,(ast.Lambda,ast.FunctionDef,ast.AsyncFunctionDef,ast.ListComp,ast.SetComp,ast.DictComp,ast.GeneratorExp)):return
        if isinstance(node,ast.Constant) and type(node.value) is int and 2<=node.value<=30:
            found[(node.lineno,node.col_offset)]=node.value
        for child in ast.iter_child_nodes(node):collect(child)
    for node in ast.walk(tree):
        if isinstance(node,ast.Compare):
            for child in [node.left]+node.comparators:collect(child)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='range':
            for child in node.args:collect(child)
    return [dict(line=l,column=c,value=v) for (l,c),v in sorted(found.items())][:16]

class Replace(ast.NodeTransformer):
    def __init__(self,site):self.site=site
    def visit_Constant(self,node):
        if (getattr(node,'lineno',-1),getattr(node,'col_offset',-1))==(self.site['line'],self.site['column']):
            return ast.copy_location(ast.Name(id=BINDING,ctx=ast.Load()),node)
        return node

def code_for(source,site):
    if site is None:return compile(source,'original','exec')
    return compile(ast.fix_missing_locations(Replace(site).visit(ast.parse(source))),'mutant','exec')

def run_grid(code,g,value):
    # Fresh namespace and input per call; compile once, initialise feature before module execution.
    namespace={BINDING:value};exec(code,namespace)
    ans=namespace['p'](copy.deepcopy(g))
    if not isinstance(ans,list) or not ans or not all(isinstance(row,list) and row for row in ans):raise ValueError('invalid grid')
    if len({len(row) for row in ans})!=1 or not all(type(v) is int and 0<=v<=9 for row in ans for v in row):raise ValueError('invalid grid')
    return ans

def assess(source,site,feature,train,queries,train_features,query_features):
    budget=5 if site is None else .25
    def fit():
        code=code_for(source,site)
        for ex,fs in zip(train,train_features):
            if run_grid(code,ex['input'],None if feature is None else fs[feature])!=ex['output']:return False
        return True
    fitted,error=bounded(fit,budget)
    row=dict(site=site,feature=feature,fit=bool(fitted),fit_error=error)
    if site is not None:row['same_literal_on_all_train']=all(fs[feature]==site['value'] for fs in train_features)
    if not fitted:return row
    def predict():
        code=code_for(source,site)
        return [run_grid(code,g,None if feature is None else fs[feature]) for g,fs in zip(queries,query_features)]
    predictions,error=bounded(predict,budget);row.update(predictions=predictions,query_error=error)
    return row

def consensus(rows):
    fitted=[r for r in rows if r['fit']]
    if not fitted:return 'no_fit',None
    if any(r['query_error'] is not None for r in fitted):return 'undefined',None
    answers={canonical(r['predictions']) for r in fitted}
    if len(answers)!=1:return 'disagreement',None
    return 'unanimous',fitted[0]['predictions']

def predict(job):
    task,source,train,queries,provenance=job
    assert BINDING not in source
    tf=list(map(features,[e['input'] for e in train]));qf=list(map(features,queries));ss=sites(source)
    rows=[assess(source,None,None,train,queries,tf,qf)]
    for site in ss:
        for f in FEATURES:rows.append(assess(source,site,f,train,queries,tf,qf))
    status,answer=consensus(rows)
    return task,dict(provenance=provenance,sites=ss,status=status,answer=answer,candidates=rows,
                     input_features=dict(train=tf,query=qf))

def controls():
    source='def p(g):\n return [[2 if len(g[0]) == 3 else 1]]\n'
    train=[dict(input=[[0]*4],output=[[2]])];qs=[[[0]*5]]
    ss=sites(source);s=next(s for s in ss if s['value']==3)
    r=assess(source,s,'width',train,qs,list(map(features,[e['input'] for e in train])),list(map(features,qs)))
    assert r['fit'] and r['predictions']==[[[2]]]
    train=[dict(input=[[0]*3],output=[[2]])]
    tf=[features(e['input']) for e in train];qf=list(map(features,qs))
    original=assess(source,None,None,train,qs,tf,qf);repair=assess(source,s,'width',train,qs,tf,qf)
    assert consensus([original,repair])[0]=='disagreement'
    undefined=dict(repair,query_error='IndexError',predictions=None)
    assert consensus([repair,undefined])[0]=='undefined'
    assert features([[0,1,1],[0,0,2]])['components']==2
    assert sites('def p(g):\n return [g[i] for i in range(3)]\n')[0]['value']==3
    return dict(literal_to_width=True,ambiguity=True,undefined_blocks=True,component_control=True,site_control=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['controls','predict','score']);ap.add_argument('--corpus');ap.add_argument('--out');args=ap.parse_args()
    if args.mode=='controls':print(json.dumps(controls()));return
    corpus=json.loads(Path(args.corpus).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST.read_text());expected=manifest['tasks']
    if args.mode=='predict':
        checks=controls();jobs=[]
        for task in sorted(expected):
            item=corpus[task];source=item['source'];data=item['data'];p=expected[task]
            assert sha(source)==p['source_sha256'] and sha(canonical(data))==p['data_canonical_sha256']
            # Worker payload excludes every test label; no worker sees `corpus` via arguments.
            jobs.append((task,source,data['train'],[e['input'] for e in data['test']],p))
        start=time.perf_counter();results={}
        with ProcessPoolExecutor(max_workers=12, mp_context=multiprocessing.get_context("spawn")) as pool:
            futs=[pool.submit(predict,job) for job in jobs]
            for fut in as_completed(futs):
                task,r=fut.result();results[task]=r
                (out/'predictions.json').write_text(json.dumps(results,sort_keys=True)+'\n')
                if len(results)%10==0:print('completed',len(results),flush=True)
        (out/'run.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,workers=12,seed=None,controls=checks,source_sha256=sha(Path(__file__).read_text()),prediction_sha256=sha((out/'predictions.json').read_text()),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()),indent=2)+'\n')
        return
    raw=(out/'predictions.json').read_text();run=json.loads((out/'run.json').read_text());assert sha(raw)==run['prediction_sha256']
    preds=json.loads(raw);summary=Counter();tasks={};repairs=[]
    for task,r in sorted(preds.items()):
        intended=[e['output'] for e in corpus[task]['data']['test']]
        assert sha(canonical(corpus[task]['data']))==expected[task]['data_canonical_sha256']
        cs=r['candidates'];original=cs[0];fits=[c for c in cs if c['fit']];mutfits=[c for c in cs[1:] if c['fit']]
        good=[c for c in mutfits if c['query_error'] is None and c['predictions']==intended]
        oc=original['fit'] and original['query_error'] is None and original['predictions']==intended
        # Re-evaluate a non-fitting original only for baseline test accuracy, not for membership.
        if not original['fit']:
            source=corpus[task]['source'];qs=[e['input'] for e in corpus[task]['data']['test']]
            baseline,error=bounded(lambda:[run_grid(code_for(source,None),g,None) for g in qs],5)
            baseline_correct=error is None and baseline==intended
        else:baseline_correct=oc
        unanimous_correct=r['status']=='unanimous' and r['answer']==intended
        counts=dict(candidates=len(cs)-1,fitting_mutants=len(mutfits),distinct_defined_answers=len({canonical(c['predictions']) for c in fits if c['query_error'] is None}),fit_timeouts=sum(c['fit_error']=='cpu_timeout' for c in cs),query_timeouts=sum(c.get('query_error')=='cpu_timeout' for c in fits),fitting_original=original['fit'],original_test_correct=baseline_correct,correct_mutants=len(good),status=r['status'],unanimous_correct=unanimous_correct,inherited_fit_mutants=sum(c['same_literal_on_all_train'] for c in mutfits))
        tasks[task]=counts;summary['tasks']+=1
        for key in ['candidates','fitting_mutants','fit_timeouts','query_timeouts','inherited_fit_mutants']:summary[key]+=counts[key]
        summary['status_'+r['status']]+=1;summary['original_fits']+=original['fit'];summary['original_test_correct']+=baseline_correct
        summary['tasks_with_correct_mutant']+=bool(good);summary['unanimous_correct']+=unanimous_correct
        summary['training_fitting_wrong_original_with_correct_mutant']+=bool(original['fit'] and not baseline_correct and good)
        summary['training_failure_with_correct_mutant']+=bool(not original['fit'] and good)
        summary['changed_correct_unanimity']+=bool(unanimous_correct and not baseline_correct)
        if good and not baseline_correct:
            repairs.append(dict(task=task,original_fits=original['fit'],status=r['status'],correct_mutants=[dict(site=c['site'],feature=c['feature'],same_literal_on_all_train=c['same_literal_on_all_train']) for c in good]))
    (out/'scores.json').write_text(json.dumps(dict(summary=dict(summary),tasks=tasks,oracle_repairs=repairs,prediction_sha256=sha(raw)),indent=2)+'\n')
    print(json.dumps(dict(summary=dict(summary),oracle_repairs=repairs),indent=2))

if __name__=='__main__':main()
