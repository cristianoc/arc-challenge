"""029: exact unary/pair consequence reduction, with no new geometry.

The family is imported from 028. Teachers expose only one requested source label
at a time. All targets in that family are examined, not a favorable subset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('reduction028', HERE.parent/'028-generality-reduction/run.py')
e28 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(e28)
POLICIES = ('halving', 'marginal', 'pair')


def members(mask: int) -> Iterable[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length()-1
        mask -= bit


def lifts(extent: int, n: int) -> tuple[int, int, int, int]:
    """Four bitsets of lexicographically ordered unordered input pairs.

    Bit 2*a+b records the output pair (a,b). This is the scientific algorithm;
    the auditor instead intersects columns of hypothesis IDs.
    """
    result = [0, 0, 0, 0]
    offset = 0
    for i in range(n-1):
        width = n-i-1
        universe = (1 << width)-1
        right = (extent >> (i+1)) & universe
        label = (extent >> i) & 1
        result[2*label] |= (universe ^ right) << offset
        result[2*label+1] |= right << offset
        offset += width
    return tuple(result)


def pair_at(index: int, n: int) -> tuple[int, int]:
    if not 0 <= index < n*(n-1)//2:
        raise ValueError('Pair index out of range')
    for i in range(n-1):
        width = n-i-1
        if index < width:
            return i, i+1+index
        index -= width
    raise AssertionError('Unreachable pair index')


class Engine:
    def __init__(self, extents: list[int], n: int, queries: list[dict]):
        if n < 2 or not extents or len(set(extents)) != len(extents):
            raise ValueError('Need at least two inputs and distinct classifier functions')
        self.extents = extents
        self.n = n
        self.n_pairs = n*(n-1)//2
        self.universe = (1 << n)-1
        if any(e < 0 or e & ~self.universe for e in extents):
            raise ValueError('Classifier extension outside the semantic domain')
        self.all_models = (1 << len(extents))-1
        self.queries = sorted(queries, key=lambda q:q['mask'])
        self.rows = {q['mask']:q for q in self.queries}
        self.ones = [sum(1 << h for h,e in enumerate(extents) if e & (1 << i)) for i in range(n)]
        self.pair_bits = [lifts(e,n) for e in extents]
        self.decisions: dict[tuple[str,int],dict | None] = {}

    def observe(self, version: int, index: int, label: int) -> int:
        if label not in (0,1) or not 0 <= index < self.n:
            raise ValueError('Invalid observation')
        yes = self.ones[index]
        return version & (yes if label else self.all_models ^ yes)

    @lru_cache(maxsize=131072)
    def unary(self, version: int) -> tuple[int,int] | None:
        if version == 0:
            return None
        lower, upper = self.universe, 0
        for h in members(version):
            lower &= self.extents[h]
            upper |= self.extents[h]
        return lower,upper

    def joint(self, version: int) -> tuple[int,int,int,int] | None:
        if version == 0:
            return None
        result = [0,0,0,0]
        for h in members(version):
            bits = self.pair_bits[h]
            for j in range(4):
                result[j] |= bits[j]
        return tuple(result)

    @lru_cache(maxsize=131072)
    def pair_width(self, version: int) -> int:
        if not version:
            raise ValueError('Empty family is an inconsistency, not zero uncertainty')
        if version.bit_count() == 1:
            return 0
        if version.bit_count() == 2:
            a,b = members(version)
            equal_positions = self.n-(self.extents[a]^self.extents[b]).bit_count()
            return self.n_pairs-equal_positions*(equal_positions-1)//2
        return sum(mask.bit_count() for mask in self.joint(version))-self.n_pairs

    def measure(self, version: int, policy: str) -> int:
        if not version:
            raise ValueError('Cannot measure an inconsistent family')
        if policy == 'halving':
            return version.bit_count()
        if policy == 'marginal':
            low,up = self.unary(version)
            return (low^up).bit_count()
        if policy == 'pair':
            return self.pair_width(version)
        raise ValueError(policy)

    def choose(self, version: int, policy: str) -> dict | None:
        cache_key = (policy,version)
        if cache_key in self.decisions:
            return self.decisions[cache_key]
        if not version:
            self.decisions[cache_key] = None
            return None
        best = None
        for row in self.queries:
            yes = version & self.ones[row['index']]
            no = version ^ yes
            if not yes or not no:
                continue
            quality = max(self.measure(no,policy),self.measure(yes,policy))
            candidate = (quality,row['mask'])
            if best is None or candidate < best[0]:
                best = candidate,row,no,yes
        answer = None
        if best is not None:
            _,row,no,yes = best
            answer = {'mask':row['mask'],'index':row['index'],'score':best[0][0],
                      'children':[hex(no),hex(yes)],
                      'remaining_models':[no.bit_count(),yes.bit_count()],
                      'remaining_unary':[self.measure(no,'marginal'),self.measure(yes,'marginal')],
                      'remaining_pair':[self.pair_width(no),self.pair_width(yes)]}
        self.decisions[cache_key] = answer
        return answer

    @lru_cache(maxsize=16384)
    def snapshot(self, version: int) -> dict:
        if not version:
            return dict(version='0x0',hypotheses=0,inconsistent=True,unary=None,pair=None,
                        cartesian_gap=None,equality_classes=None)
        lo,up = self.unary(version)
        w = (lo^up).bit_count()
        p = self.pair_width(version)
        # Distinct surviving column vectors induce exact equality of outputs.
        signatures = {v & version for v in self.ones}
        return dict(version=hex(version),hypotheses=version.bit_count(),inconsistent=False,
                    unary=w,pair=p,cartesian_gap=(self.n-1)*w+w*(w-1)//2-p,
                    equality_classes=len(signatures))

    @lru_cache(maxsize=16384)
    def relation_gain(self, before: int, after: int) -> dict:
        if not after or after & ~before:
            raise ValueError('Need a nonempty restricted family')
        lo,up = self.unary(after)
        both_ambiguous = lifts(lo^up,self.n)[3]
        old,new = self.joint(before),self.joint(after)
        newly_forbidden = tuple((a & ~b) & both_ambiguous for a,b in zip(old,new))
        witness = None
        candidates = [(next(members(mask)),value) for value,mask in enumerate(newly_forbidden) if mask]
        if candidates:
            pair_index,value = min(candidates)
            i,j = pair_at(pair_index,self.n)
            witness = {'indices':[i,j],'forbidden':[value//2,value%2],
                       'before':[v for v in range(4) if old[v] & (1 << pair_index)],
                       'after':[v for v in range(4) if new[v] & (1 << pair_index)]}
        return {'relational_only_exclusions':sum(x.bit_count() for x in newly_forbidden),
                'first_witness':witness}

    def simulate(self, oracle, policy: str, budget: int=8) -> dict:
        version = self.all_models
        seed_labels = []
        for mask in e28.SEEDS:
            index = self.rows[mask]['index']
            label = oracle(index)
            version = self.observe(version,index,label)
            seed_labels.append([mask,label])
        if not version:
            return {'seed_labels':seed_labels,'states':[self.snapshot(0)],'steps':[]}
        states = [self.snapshot(version)]
        steps = []
        for _ in range(budget):
            choice = self.choose(version,policy)
            if choice is None:
                break
            label = oracle(choice['index'])
            after = self.observe(version,choice['index'],label)
            if not after:
                raise ValueError('Oracle answer refutes the entire supplied family')
            gain = self.relation_gain(version,after)
            steps.append({'query':choice['mask'],'index':choice['index'],'label':label,**gain})
            version = after
            states.append(self.snapshot(version))
        return {'seed_labels':seed_labels,'states':states,'steps':steps}


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(out: Path) -> None:
    domain,pool = e28.finite_domain()
    models,grammar = e28.model_pool(domain)
    assert len(domain)==401 and len(models)==260
    extents = [m['extent'] for m in models]
    engine = Engine(extents,len(domain),pool)
    traces = []
    for target,extent in enumerate(extents):
        for policy in POLICIES:
            # The selector sees only version spaces; the target exists solely in oracle.
            oracle = lambda index, e=extent:(e >> index)&1
            traces.append({'target':target,'policy':policy,**engine.simulate(oracle,policy)})
    historical = []
    for kind in e28.KINDS:
        for invert in (False,True):
            extent = sum(e28.teacher(d['points'],kind,invert) << i for i,d in enumerate(domain))
            target = extents.index(extent)
            historical.append({'kind':kind,'invert':invert,'target':target})
    decisions = [{'policy':p,'version':hex(v),'choice':c}
                 for (p,v),c in sorted(engine.decisions.items())]
    data = {'baseline':'2f534a92f728bde2446b368cc03f68a8ad14cd34',
            'domain':domain,'pool':pool,'models':[dict(m,extent=hex(m['extent'])) for m in models],
            'grammar':grammar,'budget':8,'traces':traces,'decisions':decisions,'historical':historical}
    write(out/'predictions.json',data)
    write(out/'prediction-run.json',{
        'predictions_sha256':digest(out/'predictions.json'),
        'source_sha256':digest(Path(__file__)),
        'dependencies':{str(Path(m.__file__).parent.name)+'/run.py' if m is e28 else
                        str(Path(m.__file__).parent.name)+'/language.py':digest(Path(m.__file__))
                        for m in (e28,e28.L)},
        'query_budget':8,'teachers':len(models),'oracle':'supplied noiseless semantic target',
        'new_arc_query_outputs':False})
    print(json.dumps({'targets':len(models),'traces':len(traces),'decision_states':len(decisions)}))


def score(out: Path) -> None:
    data = json.loads((out/'predictions.json').read_text())
    manifest = json.loads((out/'prediction-run.json').read_text())
    if digest(out/'predictions.json') != manifest['predictions_sha256']:
        raise ValueError('Prediction hash changed')
    aggregates = {}
    for policy in POLICIES:
        traces = [t for t in data['traces'] if t['policy']==policy]
        rows = []
        for b in range(9):
            states = [t['states'][min(b,len(t['states'])-1)] for t in traces]
            rows.append({'budget':b,'identified':sum(s['hypotheses']==1 for s in states),
                         'sum_unary':sum(s['unary'] for s in states),
                         'sum_pair':sum(s['pair'] for s in states),
                         'sum_hypotheses':sum(s['hypotheses'] for s in states)})
        required = [next((i for i,s in enumerate(t['states']) if s['hypotheses']==1),None) for t in traces]
        aggregates[policy] = {'budgets':rows,'identified_by_eight':sum(x is not None for x in required),
                              'labels_to_identification_or_eight':sum(x if x is not None else 8 for x in required),
                              'depth_histogram':dict(Counter(str(x) for x in required)),
                              'relational_gain_steps':sum(s['relational_only_exclusions']>0 for t in traces for s in t['steps'])}
    by_target = {(t['target'],t['policy']):t for t in data['traces']}
    historical = []
    for h in data['historical']:
        historical.append({**h,'depths':{p:next((i for i,s in enumerate(by_target[(h['target'],p)]['states'])
                            if s['hypotheses']==1),None) for p in POLICIES}})
    unique_edges = {}
    for trace in data['traces']:
        for i,step in enumerate(trace['steps']):
            if step['first_witness'] is None:
                continue
            edge = (trace['states'][i]['version'],trace['states'][i+1]['version'],step['query'],step['label'])
            unique_edges.setdefault(edge,{'before':edge[0],'after':edge[1],**step})
    # Deterministic examples: smallest observation mask, then version as integer.
    examples = sorted(unique_edges.values(),key=lambda e:(e['query'],int(e['before'],16),e['label']))
    output = {'targets':260,'input_classes':401,'unordered_pairs':80200,'policies':aggregates,
              'historical':historical,'unique_relational_gain_edges':len(unique_edges),'first_relational_examples':examples[:5]}
    write(out/'summary.json',output)
    write(out/'score-run.json',{'predictions_sha256':digest(out/'predictions.json'),
                               'summary_sha256':digest(out/'summary.json')})
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=('predict','score'))
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='predict':predict(args.out)
    else:score(args.out)
