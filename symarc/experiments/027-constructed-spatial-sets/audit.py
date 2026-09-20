"""Independent raw-syntax, geometry, prediction and teaching-certificate audit.

Imports neither scientific module. The reference grammar keeps every ordered
syntax tree, including redundant trees; no semantic quotient is used in search.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import itertools
import json
from pathlib import Path

ATOMS=('rmin','rmax','cmin','cmax')
OPS=('input','generated','complement','union','intersection','input_minus','generated_minus','xor')


def read(p):return json.loads(p.read_text())
def code(x):return json.dumps(x,separators=(',',':'))
def write(p,x):p.write_text(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def canon(e):
    if e[0] in ATOMS or e[0]=='bool':return tuple(e)
    children=[canon(x) for x in e[1:]]
    if e[0] in ('and','or'):children.sort(key=code)
    return (e[0],*children)


def eval_bool(e,v):
    if e[0] in ATOMS:return bool(v & (1<<ATOMS.index(e[0])))
    if e[0]=='bool':return e[1]
    if e[0]=='not':return not eval_bool(e[1],v)
    if e[0]=='and':return eval_bool(e[1],v) and eval_bool(e[2],v)
    if e[0]=='or':return eval_bool(e[1],v) or eval_bool(e[2],v)
    raise ValueError(e)


def reference_grammar(bound=7):
    # Enumerate ALL ordered trees, without memoizing by their functions.
    levels={1:[((a,),tuple(bool(v&(1<<i)) for v in range(16)),6) for i,a in enumerate(ATOMS)]
              +[(('bool',False),(False,)*16,1),(('bool',True),(True,)*16,1)]}
    for n in range(2,bound+1):
        level=[(('not',e),tuple(not x for x in ys),c+1) for e,ys,c in levels[n-1]]
        for i in range(1,n-1):
            for a,xs,ca in levels[i]:
                for b,ys,cb in levels[n-1-i]:
                    level.append((('and',a,b),tuple(x and y for x,y in zip(xs,ys)),ca+cb+1))
                    level.append((('or',a,b),tuple(x or y for x,y in zip(xs,ys)),ca+cb+1))
        levels[n]=level
    minima={}
    for level in levels.values():
        for e,vals,cost in level:
            sig=sum((1<<i) for i,v in enumerate(vals) if v);e=canon(e)
            if sig not in minima or (cost,code(e))<(minima[sig][1],code(minima[sig][0])):minima[sig]=(e,cost)
    return minima,{str(k):len(v) for k,v in levels.items()}


def carrier(s):
    s=set(map(tuple,s));rs,cs=zip(*s)
    return {(r,c) for r in range(min(rs),max(rs)+1) for c in range(min(cs),max(cs)+1)}


def geometry(s):
    """Independent row interpreter for perimeter and interior."""
    s=set(map(tuple,s));rs,cs=zip(*s);a,b,c,d=min(rs),max(rs),min(cs),max(cs)
    frame=set();inside=set()
    for r in range(a,b+1):
        row=set(range(c,d+1)) if r==a or r==b else {c,d}
        frame.update((r,x) for x in row)
        if a<r<b:inside.update((r,x) for x in range(c+1,d))
    return frame,inside


def profile(s):
    ps=set(map(tuple,s));rs,cs=zip(*ps);a,b,c,d=min(rs),max(rs),min(cs),max(cs)
    counts=Counter()
    for r,c0 in carrier(s):
        bits=int(r==a)+2*int(r==b)+4*int(c0==c)+8*int(c0==d)
        counts[(bits,(r,c0) in ps)]+=1
    return counts


def signature_match(sig,prof):
    return all(bool(sig&(1<<role))==present for role,present in prof)


def reference_fit(examples,minima):
    profiles=[profile(e['points']) for e in examples];labels=[e['label'] for e in examples];out=[];best=None
    for sig,(e,cost) in minima.items():
        states=[signature_match(sig,p) for p in profiles]
        for f,t in itertools.product((0,1),repeat=2):
            if all((t if b else f)==y for b,y in zip(states,labels)):
                if best is None or cost<best:best=cost;out=[]
                if cost==best:out.append((sig,f,t,cost+22))
    return sorted(out)


def generated(sig,s):
    ps=set(map(tuple,s));rs,cs=zip(*ps)
    def role(p):
        r,c=p
        return sum(int(t)<<i for i,t in enumerate((r==min(rs),r==max(rs),c==min(cs),c==max(cs))))
    return {p for p in carrier(ps) if sig&(1<<role(p))}


def operation(name,s,g):
    s=set(map(tuple,s));universe=carrier(s)
    return {'input':s,'generated':g,'complement':universe-g,'union':s|g,'intersection':s&g,
            'input_minus':s-g,'generated_minus':g-s,'xor':s.symmetric_difference(g)}[name]


def teaching_certificate(source):
    # Post-score diagnostic: deterministic greedy coverage, not a new learner.
    positive=[(i,profile(e['points'])) for i,e in enumerate(source) if e['label']==1]
    selected=[];remaining=set(range(16))
    while remaining:
        i,p=max(positive,key=lambda x:(len({r for r,b in x[1]}&remaining),-x[0]))
        covered={r for r,b in p}&remaining
        if not covered:raise AssertionError('Roles not covered')
        selected.append(i);remaining-=covered
    groups=defaultdict(list)
    for i,e in enumerate(source):
        if not e['label']:groups[tuple(sorted(carrier(e['points'])))].append(i)
    samebox=next(v[:2] for k,v in sorted(groups.items()) if len(v)>=2)
    selected+=samebox
    # Enumeration of every Boolean truth table, not just <=7-node formulas.
    mini=[source[i] for i in selected];profiles=[profile(e['points']) for e in mini]
    labels=[e['label'] for e in mini];fits=[]
    for sig in range(65536):
        bs=[signature_match(sig,p) for p in profiles]
        for f,t in itertools.product((0,1),repeat=2):
            if all((t if b else f)==y for b,y in zip(bs,labels)):fits.append([sig,f,t])
    assert fits==[[65534,0,1]],fits
    return {'post_score_diagnostic':True,'source_indices':selected,'examples':mini,'labels':len(mini),
            'all_boolean_tables_checked':65536,'unique_semantic_program':fits[0],
            'selection_is_target_informed_not_a_passive_learning_result':True}


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    a=ap.parse_args();lib=read(a.run/'library.json');construction=read(a.run/'construction.json');pred=read(a.run/'predictions.json');scores=read(a.run/'scores.json')
    assert sha(a.run/'library.json')==read(a.run/'construct-run.json')['library_sha256']
    assert sha(a.run/'predictions.json')==read(a.run/'predict-run.json')['predictions_sha256']
    minima,raw=reference_grammar();expected={r['truth']:(canon(r['expr']),r['predicate_cost']) for r in construction['candidates']}
    assert minima==expected
    source=read(a.input/'source.json');sparse=read(a.input/'sparse.json')
    def models(ms):return sorted((m['truth'],m['false'],m['true'],m['program_cost']) for m in ms)
    assert reference_fit(source,minima)==models(lib['models'])
    assert reference_fit(sparse,minima)==models(lib['sparse_models'])
    for r in construction['rejections']:
        i,j=r['witness'];assert source[i]['label']!=source[j]['label']
        assert signature_match(r['truth'],profile(source[i]['points']))==signature_match(r['truth'],profile(source[j]['points']))
    for e in source:assert e['label']==int(set(map(tuple,e['points']))==geometry(e['points'])[0])
    # Same three scalars, different actual spatial observations.
    pair=[source[254],source[494]]
    scalars=lambda s:(len(s),len({p[0] for p in carrier(s)}),len({p[1] for p in carrier(s)}))
    assert scalars(pair[0]['points'])==scalars(pair[1]['points']) and pair[0]['label']!=pair[1]['label']
    assert profile(pair[0]['points'])!=profile(pair[1]['points'])
    totals=Counter();recognition_inputs=read(a.input/'recognition-inputs.json');answers=read(a.input/'recognition-answers.json')
    actual_overlap={}
    for bank,ss in recognition_inputs.items():
        observed={name:set(tuple(map(tuple,e['points'])) for e in data) for name,data in (('dense',source),('sparse',sparse))}
        actual_overlap[bank]={mode:sum(tuple(map(tuple,s)) in keys for s in ss) for mode,keys in observed.items()}
        for j,s in enumerate(ss):
            truthlabel=int(set(map(tuple,s))==geometry(s)[0]);assert truthlabel==answers[bank][j]
            pr=profile(s)
            for mode,ms in (('dense',lib['models']),('sparse',lib['sparse_models'])):
                vals=[m['true'] if signature_match(m['truth'],pr) else m['false'] for m in ms]
                for i,v in enumerate(vals):assert v==pred['recognition'][bank][mode]['all_models'][i][j]
                unanimous=vals[0] if vals and all(x==vals[0] for x in vals) else -1
                assert unanimous==pred['recognition'][bank][mode]['consensus'][j]
                totals['recognition_program_predictions']+=len(vals)
            totals['recognition_teacher_cases']+=1
        for mode in ('dense','sparse'):
            ys=answers[bank];ps=pred['recognition'][bank][mode]['consensus'];s=scores[bank][mode]
            assert s['correct']==sum(y==p for y,p in zip(ys,ps))
            assert s['wrong']==sum(y!=p and p!=-1 for y,p in zip(ys,ps))
            assert s['positive_correct']==sum(y==p==1 for y,p in zip(ys,ps))
            assert s['negative_correct']==sum(y==p==0 for y,p in zip(ys,ps))
    target_inputs=read(a.input/'target-inputs.json');target_answers=read(a.input/'target-answers.json');target_train=read(a.input/'target-train.json')
    for t in pred['targets']:
        task=next(x for x in target_train if x['name']==t['name'])
        choices=[]
        for m in lib['models']:
            for op in OPS:
                if all(operation(op,p['input'],generated(m['truth'],p['input']))==set(map(tuple,p['output'])) for p in task['train']):choices.append((m['truth'],op))
        assert choices==[(m['truth'],m['operation']) for m in t['models']],choices
        for j,s in enumerate(target_inputs):
            y=geometry(s)[0 if t['name']=='repair' else 1]
            assert sorted(y)==[tuple(p) for p in target_answers[t['name']][j]]
            for i,m in enumerate(t['models']):
                value=operation(m['operation'],s,generated(m['truth'],s))
                assert sorted(value)==[tuple(p) for p in t['predictions'][i][j]]
                totals['set_valued_predictions']+=1
        assert scores['targets'][t['name']]['exact']==sum(t['predictions'][0][j]==target_answers[t['name']][j] for j in range(len(target_inputs)))
    collision=read(a.input/'negative-control.json')
    assert scalars(collision[0]['points'])==scalars(collision[1]['points'])
    assert profile(collision[0]['points'])==profile(collision[1]['points'])
    for sig in range(65536):assert signature_match(sig,profile(collision[0]['points']))==signature_match(sig,profile(collision[1]['points']))
    cert=teaching_certificate(source)
    write(a.run/'teaching-certificate.json',cert)
    write(a.run/'audit.json',{'raw_syntax_counts':raw,'raw_syntax_trees':sum(raw.values()),'semantic_predicates':len(minima),
        'rejection_witnesses':len(construction['rejections']),'counts':dict(totals),'actual_fitting_source_overlaps':actual_overlap,
        'scorer_overlap_note':'scores.json source_overlap always references the common 511-example source bank; actual sparse fitting overlap is given here.',
        'teaching_certificate_labels':cert['labels'],'teaching_tables_checked':65536,'thickness_two_collision_checked_tables':65536,
        'scientific_module_imports':False,'prediction_hash':sha(a.run/'predictions.json')})
    print(json.dumps(read(a.run/'audit.json'),indent=2))

if __name__=='__main__':main()
