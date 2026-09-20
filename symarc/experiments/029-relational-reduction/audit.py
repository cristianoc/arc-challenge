"""Independent column-intersection audit. Imports neither scientific module.

The geometric family is checked against retained 028 data and independently
interpreted on all finite semantic representatives. Pair scores are computed
from column intersections, not the scientific row-to-pair lifting algorithm.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def geometry_profile(points):
    ps={tuple(p) for p in points};rs=[p[0] for p in ps];cs=[p[1] for p in ps]
    a,b,c,d=min(rs),max(rs),min(cs),max(cs)
    yes=set();no=set()
    for r in range(a,b+1):
        for col in range(c,d+1):
            # Existing language order: rmin,rmax,cmin,cmax; first is bit zero.
            role=int(r==a)+2*int(r==b)+4*int(col==c)+8*int(col==d)
            (yes if (r,col) in ps else no).add(role)
    return yes,no


class Reference:
    def __init__(self,data):
        self.extents=[int(x['extent'],16) for x in data['models']]
        self.n=len(data['domain']);self.universe=(1<<self.n)-1
        self.full=(1<<len(self.extents))-1
        self.queries=sorted(data['pool'],key=lambda p:p['mask'])
        self.columns=[sum(1<<h for h,e in enumerate(self.extents) if (e>>i)&1) for i in range(self.n)]
        self.subset_checks=0;self.choice_checks=0

    @staticmethod
    def count_patterns(v,a,b):
        na=v^a;nb=v^b
        return int(bool(na&nb))+int(bool(na&b))+int(bool(a&nb))+int(bool(a&b))

    @staticmethod
    def patterns(v,a,b):
        na=v^a;nb=v^b
        return {i for i,p in enumerate((na&nb,na&b,a&nb,a&b)) if p}

    @lru_cache(maxsize=None)
    def stats(self,v):
        if not v:return None
        grouped=Counter(c&v for c in self.columns)
        width=sum(count for c,count in grouped.items() if c not in (0,v))
        joint=0;items=sorted(grouped.items())
        for i,(a,weight) in enumerate(items):
            # Equal columns permit one output pair if fixed, two if ambiguous.
            joint+=weight*(weight-1)//2*(int(a not in (0,v)))
            for b,other in items[i+1:]:
                joint+=weight*other*(self.count_patterns(v,a,b)-1)
        self.subset_checks+=1
        return width,joint,len(grouped)

    def snapshot(self,v):
        w,p,n=self.stats(v)
        return dict(version=hex(v),hypotheses=v.bit_count(),inconsistent=False,
                    unary=w,pair=p,cartesian_gap=(self.n-1)*w+w*(w-1)//2-p,equality_classes=n)

    @lru_cache(maxsize=None)
    def choice(self,v,policy):
        best=None
        for q in self.queries:
            yes=self.columns[q['index']]&v;no=v^yes
            if not yes or not no:continue
            if policy=='halving':quality=max(yes.bit_count(),no.bit_count())
            else:
                pos=0 if policy=='marginal' else 1
                quality=max(self.stats(yes)[pos],self.stats(no)[pos])
            rank=(quality,q['mask'])
            if best is None or rank<best[0]:best=rank,q,no,yes
        self.choice_checks+=1
        if best is None:return None
        rank,q,no,yes=best
        return dict(mask=q['mask'],index=q['index'],score=rank[0],children=[hex(no),hex(yes)],
                    remaining_models=[no.bit_count(),yes.bit_count()],
                    remaining_unary=[self.stats(no)[0],self.stats(yes)[0]],
                    remaining_pair=[self.stats(no)[1],self.stats(yes)[1]])

    @lru_cache(maxsize=None)
    def relational_gain(self,before,after):
        # Group inputs by both before/after columns; group cardinalities account
        # for every unordered input pair without lifting any row into a bitset.
        groups=Counter((c&before,c&after) for c in self.columns if c&after not in (0,after))
        answer=0;items=sorted(groups.items())
        for i,((a,aa),weight) in enumerate(items):
            for (b,bb),other in items[i+1:]:
                old=self.patterns(before,a,b);new=self.patterns(after,aa,bb)
                assert new<=old
                answer+=weight*other*len(old-new)
        return answer


def audit(data,baseline):
    assert data['models']==baseline['models']
    assert data['domain']==baseline['domain'] and data['pool']==baseline['pool']
    models=data['models'];domain=data['domain'];n=len(domain)
    profiles=[geometry_profile(d['points']) for d in domain]
    geometry_checks=0
    for model in models:
        truth=model['truth'];extent=int(model['extent'],16)
        for i,(yes,no) in enumerate(profiles):
            match=all(truth&(1<<r) for r in yes) and all(not truth&(1<<r) for r in no)
            expected=model['true'] if match else model['false']
            assert expected==((extent>>i)&1)
            geometry_checks+=1
    r=Reference(data);states=0;transitions=0;witnesses=0;strict_quotient_edges=set()
    choices={(d['policy'],int(d['version'],16)):d['choice'] for d in data['decisions']}
    for (p,v),c in choices.items():assert r.choice(v,p)==c,(p,v,'choice')
    traces={(t['target'],t['policy']):t for t in data['traces']}
    assert len(traces)==260*3
    source={q['mask']:q for q in data['pool']}
    unique_witness_edges={};decoded=[]
    for (target,policy),trace in traces.items():
        extent=r.extents[target];v=r.full
        for mask,label in trace['seed_labels']:
            index=source[mask]['index'];assert label==((extent>>index)&1)
            v &= r.columns[index] if label else (r.full^r.columns[index])
        assert trace['states'][0]==r.snapshot(v)
        for k,step in enumerate(trace['steps']):
            expected=r.choice(v,policy);assert expected is not None
            assert step['query']==expected['mask'] and step['index']==expected['index']
            label=(extent>>step['index'])&1;assert label==step['label']
            new=v&(r.columns[step['index']] if label else r.full^r.columns[step['index']])
            assert new and (new&(1<<target)) and not new&~v
            assert trace['states'][k+1]==r.snapshot(new)
            assert r.stats(new)[0]<=r.stats(v)[0] and r.stats(new)[1]<=r.stats(v)[1]
            assert r.stats(new)[2]<=r.stats(v)[2]
            if r.stats(new)[2]<r.stats(v)[2]:strict_quotient_edges.add((v,new))
            gain=r.relational_gain(v,new)
            assert gain==step['relational_only_exclusions']
            witness=step['first_witness']
            assert (witness is not None)==(gain>0)
            if witness:
                i,j=witness['indices'];a,b=r.columns[i]&v,r.columns[j]&v
                aa,bb=r.columns[i]&new,r.columns[j]&new
                assert aa not in (0,new) and bb not in (0,new)
                old=r.patterns(v,a,b);remaining=r.patterns(new,aa,bb)
                forbidden=2*witness['forbidden'][0]+witness['forbidden'][1]
                assert old==set(witness['before']) and remaining==set(witness['after'])
                assert forbidden in old-remaining
                # Verify lexicographically first witness directly from columns.
                found=None
                for x in range(n):
                    if r.columns[x]&new in (0,new):continue
                    for y in range(x+1,n):
                        if r.columns[y]&new in (0,new):continue
                        previous=r.patterns(v,r.columns[x]&v,r.columns[y]&v)
                        latest=r.patterns(new,r.columns[x]&new,r.columns[y]&new)
                        if previous-latest:
                            found=([x,y],min(previous-latest));break
                    if found:break
                assert found==([i,j],forbidden)
                edge=(v,new,step['query'],label)
                unique_witness_edges[edge]=step
                witnesses+=1
            v=new;transitions+=1
        if len(trace['steps'])<8:assert r.choice(v,policy) is None
        states+=len(trace['states'])
    # Exact reproduction of all original geometric scope/halving paths.
    historical_checks=0
    for h in data['historical']:
        for old_name,new_name in [('halving','halving'),('scope','marginal')]:
            old=next(t for t in baseline['traces'] if t['kind']==h['kind'] and t['invert']==h['invert'] and t['policy']==old_name)
            new=traces[(h['target'],new_name)]
            assert [s['version'] for s in old['records']]==[s['version'] for s in new['states']]
            assert [s['query']['mask'] for s in old['records'][1:]]==[s['query'] for s in new['steps']]
            historical_checks+=1
    # Target-label complementation is checked throughout all policy trees.
    complements=0
    index={e:i for i,e in enumerate(r.extents)}
    for target,e in enumerate(r.extents):
        partner=index[r.universe^e]
        if target>=partner:continue
        for policy in ('halving','marginal','pair'):
            a,b=traces[(target,policy)],traces[(partner,policy)]
            assert [x['query'] for x in a['steps']]==[x['query'] for x in b['steps']]
            assert [x['label'] for x in a['steps']]==[1-x['label'] for x in b['steps']]
            assert [x['unary'] for x in a['states']]==[x['unary'] for x in b['states']]
            assert [x['pair'] for x in a['states']]==[x['pair'] for x in b['states']]
            complements+=1
    # Decode deterministic examples; these are observed edges, not extra runs.
    for (v,new,q,label),step in sorted(unique_witness_edges.items(),key=lambda kv:(kv[0][2],kv[0][0],kv[0][3]))[:5]:
        i,j=step['first_witness']['indices']
        decoded.append({'before_version':hex(v),'after_version':hex(new),'query_mask':q,'query_label':label,
            'query_points':source[q]['points'],'input_indices':[i,j],'input_sets':[domain[i]['points'],domain[j]['points']],
            'before':r.snapshot(v),'after':r.snapshot(new),'pair_before':step['first_witness']['before'],
            'pair_after':step['first_witness']['after'],'joint_exclusions_with_ambiguous_coordinates':step['relational_only_exclusions']})
    # Independently derived counts, with explicit censoring at eight queries.
    policy_scores={}
    for policy in ('halving','marginal','pair'):
        subset=[t for (h,p),t in traces.items() if p==policy]
        policy_scores[policy]=[sum(t['states'][min(b,len(t['states'])-1)]['hypotheses']==1 for t in subset) for b in range(9)]
    return {'geometry_model_input_checks':geometry_checks,'independent_subset_measures':r.subset_checks,
            'independent_choices':r.choice_checks,'recorded_decisions':len(choices),'trace_states':states,
            'acquired_labels':transitions,'relational_witness_instances':witnesses,
            'unique_relational_edges':len(unique_witness_edges),'strict_quotient_edges':len(strict_quotient_edges),
            'historical_trace_matches':historical_checks,'label_complement_trace_pairs':complements,
            'identification_by_budget':policy_scores,'decoded_witnesses':decoded,
            'pair_and_marginal_paths_identical':all(traces[(h,'pair')]['steps']==traces[(h,'marginal')]['steps'] for h in range(260)),
            'boundary':'Primitive family/grammar inherited from 028; new projections and selections independently reconstructed.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=audit(read(a.predictions),read(a.baseline))
    result['predictions_sha256']=sha(a.predictions);result['baseline_sha256']=sha(a.baseline)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='decoded_witnesses'},indent=2))
