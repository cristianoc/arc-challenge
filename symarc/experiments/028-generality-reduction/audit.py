"""Independent 028 semantic/selection/scoring audit; imports no scientific module.

Uses explicit classifier and input-class sets. Rechecks the raw Boolean grammar,
finite role quotient, all request decisions, every exclusion and score.
"""
from collections import Counter
from functools import lru_cache
import argparse
import hashlib
import itertools
import json
from pathlib import Path

ATOMS=('rmin','rmax','cmin','cmax')


def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jkey(x):return json.dumps(x,separators=(',',':'))
def maskset(h):return frozenset(i for i in range(h.bit_length()) if h&(1<<i))
def ashex(xs):return hex(sum(1<<i for i in xs))


def point_profile(points):
    s=set(map(tuple,points));rs=[r for r,c in s];cs=[c for r,c in s]
    a,b,c,d=min(rs),max(rs),min(cs),max(cs);states={}
    for r in range(a,b+1):
        for col in range(c,d+1):
            flags=(r==a,r==b,col==c,col==d)
            role=sum(2**i for i,v in enumerate(flags) if v)
            states.setdefault(role,set()).add((r,col) in s)
    if any(len(v)>1 for v in states.values()):return (-1,-1)
    return (sum(1<<i for i,v in states.items() if True in v),
            sum(1<<i for i,v in states.items() if False in v))


def truth_of(e,v):
    op=e[0]
    if op in ATOMS:return bool(v & (1<<ATOMS.index(op)))
    if op=='bool':return bool(e[1])
    if op=='not':return not truth_of(e[1],v)
    if op=='and':return truth_of(e[1],v) and truth_of(e[2],v)
    if op=='or':return truth_of(e[1],v) or truth_of(e[2],v)
    raise ValueError(op)


def raw_grammar():
    # Raw ordered trees, unlike the scientific semantic dynamic programme.
    layers={1:[(a,) for a in ATOMS]+[('bool',False),('bool',True)]}
    for n in range(2,8):
        out=[('not',a) for a in layers[n-1]]
        for i in range(1,n-1):
            for a in layers[i]:
                for b in layers[n-1-i]:
                    out.extend((('and',a,b),('or',a,b)))
        layers[n]=out
    minima={};count=0
    def cost(e):return 6 if e[0] in ATOMS else 1+sum(cost(a) for a in e[1:] if isinstance(a,tuple))
    for trees in layers.values():
        for e in trees:
            t=sum(1<<v for v in range(16) if truth_of(e,v));c=cost(e)+22
            minima[t]=min(c,minima.get(t,c));count+=1
    return minima,count


def evaluate(table,p):
    y,n=p
    if y<0:return False
    return all((not y&(1<<v) or table&(1<<v)) and (not n&(1<<v) or not table&(1<<v)) for v in range(16))


def geometric(points,kind,invert=False):
    s=set(map(tuple,points));r0=min(r for r,c in s);r1=max(r for r,c in s)
    c0=min(c for r,c in s);c1=max(c for r,c in s)
    selected=[]
    for r in range(r0,r1+1):
        for c in range(c0,c1+1):
            if kind=='filled':inside=True
            elif kind=='perimeter':inside=r==r0 or r==r1 or c==c0 or c==c1
            elif kind=='missing_corner':inside=not(r==r1 and c==c1)
            elif kind=='corners':inside=(r==r0 or r==r1) and (c==c0 or c==c1)
            else:raise ValueError(kind)
            if inside:selected.append((r,c))
    return int((s==set(selected)) ^ bool(invert))


def audit(predictions,query_inputs,scores):
    data=load(predictions);xs=load(query_inputs);scored=load(scores)
    counts=Counter();domain=data['domain'];models=data['models'];pool=data['pool']
    profiles=[point_profile(d['points']) for d in domain]
    assert profiles==[tuple(d['profile']) for d in domain]
    assert len(set(profiles))==len(profiles)
    index={p:i for i,p in enumerate(profiles)}
    source=set()
    for p in pool:
        assert point_profile(p['points'])==tuple(p['profile'])==profiles[p['index']]
        source.add(tuple(map(tuple,p['points'])))
    assert set(index)=={point_profile(p['points']) for p in pool}|{(-1,-1)}
    # All classifier extensions and cheapest costs rederived from raw syntax.
    forms,trees=raw_grammar();counts['ordered_syntax_trees']=trees;counts['point_truth_tables']=len(forms)
    positive={t:frozenset(i for i,p in enumerate(profiles) if evaluate(t,p)) for t in forms}
    allpoints=frozenset(range(len(profiles)));expected={}
    for t,c in forms.items():
        for f,b in itertools.product((0,1),repeat=2):
            extension=(positive[t] if b else frozenset())|((allpoints-positive[t]) if f else frozenset())
            expected[extension]=min(c,expected.get(extension,c))
    ext=[]
    for m in models:
        assert sum(1<<v for v in range(16) if truth_of(m['expr'],v))==m['truth']
        extension=maskset(int(m['extent'],16));assert expected[extension]==m['cost']
        computed=frozenset(i for i,p in enumerate(profiles) if (m['true'] if evaluate(m['truth'],p) else m['false']))
        assert computed==extension;ext.append(extension)
    assert len(ext)==len(set(ext)) and set(ext)==set(expected)
    counts['distinct_classifier_extensions']=len(ext)
    allmodels=frozenset(range(len(models)));bymask={p['mask']:p for p in pool}

    @lru_cache(maxsize=None)
    def bounds(v):
        if not v:return None
        parts=[ext[i] for i in v]
        return frozenset.intersection(*parts),frozenset.union(*parts)

    def update(v,i,y):return frozenset(h for h in v if int(i in ext[h])==y)

    def choose(v,seen,policy):
        lo,up=bounds(v);proposals=[]
        for row in pool:
            if row['mask'] in seen:continue
            a,b=update(v,row['index'],0),update(v,row['index'],1)
            if not a or not b:continue
            l0,u0=bounds(a);l1,u1=bounds(b)
            n=[len(a),len(b)];w=[len(u0-l0),len(u1-l1)];removed=len(up-u0)
            quality={'first':0,'halving':max(n),'negative_reduction':-removed,'scope':max(w)}[policy]
            proposals.append((quality,row['mask'],{'mask':row['mask'],'index':row['index'],
                'quality':quality,'negative_exclusion':removed,'remaining_models':n,'remaining_width':w}))
        return min(proposals,key=lambda x:x[:2])[2] if proposals else None

    for trace in data['traces']:
        v=allmodels;seen=set()
        for i,state in enumerate(trace['records']):
            if i==0:
                assert [s['mask'] for s in state['labels']]==[255,495]
                additions=state['labels']
            else:
                q=choose(v,seen,trace['policy']);assert q==state['query']
                assert q['mask']==state['labels'][-1]['mask'];additions=[state['labels'][-1]]
                old=v;old_bounds=bounds(v);counts['acquisition_decisions']+=1
            for label in additions:
                row=bymask[label['mask']]
                assert label['index']==row['index']
                assert label['label']==geometric(row['points'],trace['kind'],trace['invert'])
                v=update(v,row['index'],label['label']);seen.add(row['mask'])
                counts['teacher_labels_checked']+=1
            assert v==maskset(int(state['version'],16)) and len(v)==state['models'] and v
            lo,up=bounds(v)
            assert state['lower']==ashex(lo) and state['upper']==ashex(up) and state['width']==len(up-lo)
            assert state['shortest']==min(v)
            actual=frozenset(i for i,d in enumerate(domain) if geometric(d['points'],trace['kind'],trace['invert']))
            assert lo<=actual<=up and actual in [ext[h] for h in v]
            if i:
                assert state['excluded_models']==ashex(old-v)
                assert state['removed_positive']==ashex(old_bounds[1]-up)
                assert state['removed_negative']==ashex(lo-old_bounds[0])
                assert old_bounds[0]<=lo<=up<=old_bounds[1]
                a=ext[trace['records'][i-1]['shortest']];b=ext[state['shortest']]
                kind=('equal' if a==b else 'specialisation' if b<a else 'generalisation' if a<b else 'incomparable')
                assert state['shortest_change']==kind
                counts['elimination_witnesses']+=len(old-v)
            counts['trace_states']+=1
        assert len(trace['records'])==9 or choose(v,seen,trace['policy']) is None
    for kind in ('perimeter','filled','missing_corner','corners'):
        for policy in ('first','halving','scope'):
            pair=[t for t in data['traces'] if t['kind']==kind and t['policy']==policy]
            assert [[s['labels'][-1]['mask'] for s in t['records']] for t in pair][0]==[[s['labels'][-1]['mask'] for s in t['records']] for t in pair][1]
            for a,b in zip(pair[0]['records'],pair[1]['records'],strict=True):
                assert a['models']==b['models'] and a['width']==b['width']
                assert maskset(int(a['lower'],16))==allpoints-maskset(int(b['upper'],16))
            counts['complement_trace_pairs']+=1
    # Reconstruct the no-syntax-bound mechanism directly from truth-table bits.
    negative=bymask[255]['index'];positive_seed=bymask[495]['index']
    raw=[]
    for t in range(65536):
        n,p=evaluate(t,profiles[negative]),evaluate(t,profiles[positive_seed])
        if n!=p:raw.append((t,int(n)))
    for name,state in data['fixed_mechanism'].items():
        rs=raw
        if state['extra_input'] is not None:
            p=point_profile(state['extra_input']);y=state['label']
            rs=[(t,b) for t,b in raw if (evaluate(t,p)^bool(b))==bool(y)]
        alternatives={frozenset(i for i,p in enumerate(profiles) if evaluate(t,p)^bool(b)) for t,b in rs}
        lo=frozenset.intersection(*alternatives);up=frozenset.union(*alternatives)
        assert len(rs)==state['point_table_programs'] and len(alternatives)==state['semantic_classifiers']
        assert ashex(lo)==state['lower'] and ashex(up)==state['upper']
        counts['unbounded_mechanism_states']+=1
    assert data['misspecification']['inconsistent'] and data['misspecification']['width'] is None
    assert point_profile(data['misspecification']['inputs'][0])==point_profile(data['misspecification']['inputs'][1])

    # Independent concrete geometry, exact quotient completeness and all scores.
    hist={}
    for name,bank in xs.items():
        hist[name]={k:Counter() for k in ('perimeter','filled','missing_corner','corners')}
        overlap=0
        for points in bank:
            p=point_profile(points);assert p in index;i=index[p]
            overlap+=tuple(map(tuple,points)) in source
            for kind in hist[name]:hist[name][kind][(i,geometric(points,kind))]+=1
            counts['concrete_geometries']+=1
        assert scored['exact_source_catalogue_overlap'][name]==overlap
    for trace,saved in zip(data['traces'],scored['traces'],strict=True):
        for s,ss in zip(trace['records'],saved['records'],strict=True):
            lo=maskset(int(s['lower'],16));up=maskset(int(s['upper'],16));ch=ext[s['shortest']]
            for name in xs:
                c=Counter()
                for (i,y),n in hist[name][trace['kind']].items():
                    y=bool(y)^trace['invert'];a=i in lo;b=i in up;p=i in ch
                    c.update({'cases':n,'positive':int(y)*n,'negative':int(not y)*n,
                        'determined':int(a==b)*n,'determined_wrong':int(a==b and a!=y)*n,
                        'may_false_positive':int(b and not y)*n,'must_false_negative':int(not a and y)*n,
                        'shortest_correct':int(p==y)*n,'shortest_false_positive':int(p and not y)*n,
                        'shortest_false_negative':int(not p and y)*n})
                assert dict(c)==ss['scores'][name];assert not c['determined_wrong']
                counts['bank_state_scores']+=1
    for name,state in data['fixed_mechanism'].items():
        lo=maskset(int(state['lower'],16));up=maskset(int(state['upper'],16))
        for bank in xs:
            items=hist[bank]['perimeter']
            vals={'possible_positive':sum(n for (i,y),n in items.items() if i in up),
                'necessary_positive':sum(n for (i,y),n in items.items() if i in lo),
                'ambiguous':sum(n for (i,y),n in items.items() if i in up-lo),
                'may_false_positive':sum(n for (i,y),n in items.items() if i in up and not y)}
            assert vals==scored['fixed_mechanism'][name][bank]
    return {'checks':dict(counts),'all_passed':True,'predictions_sha256':digest(predictions),
            'scores_sha256':digest(scores),'boundary':'Exact within the fixed grammar; independent geometry, requests and scores, not a proof of the Python programme.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('predictions','query-inputs','scores','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=audit(a.predictions,a.query_inputs,a.scores)
    a.out.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n');print(json.dumps(result,indent=2))
