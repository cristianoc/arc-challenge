"""028: reduce semantic generality without conflating it with hypothesis counts.

`predict` simulates explicitly requested source labels and freezes its trace.
`score` evaluates the already-frozen trace on the separate geometric test banks.
Only the standard library is required. The 027 grammar is reused unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('spatial027', HERE.parent/'027-constructed-spatial-sets/language.py')
L = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(L)
POLICIES = ('first', 'halving', 'negative_reduction', 'scope')
KINDS = ('perimeter', 'filled', 'missing_corner', 'corners')
SEEDS = (255, 495)


def put(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, separators=(',', ':'))+'\n')


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(h: int, w: int, mask: int) -> list[list[int]]:
    return [[i//w, i%w] for i in range(h*w) if mask & (1 << i)]


def profile(points) -> tuple[int, int]:
    yes, no = L.profile(points)
    # Every contradictory role has exactly the same template-match semantics.
    return (-1, -1) if yes & no else (yes, no)


def finite_domain():
    pool = [{'mask':m, 'points':shape(3, 3, m), 'profile':profile(shape(3, 3, m))}
            for m in range(1, 512)]
    unique = {}
    for row in pool:
        unique.setdefault(row['profile'], row['points'])
    mixed = [[r,c] for r in range(4) for c in range(4) if (r,c)!=(1,1)]
    unique[profile(mixed)] = mixed
    domain = [{'profile':list(p), 'points':unique[p]} for p in sorted(unique)]
    indices = {tuple(x['profile']):i for i,x in enumerate(domain)}
    for row in pool:
        row['index'] = indices[row['profile']]
        row['profile'] = list(row['profile'])
    return domain, pool


def match(table: int, p) -> bool:
    yes, no = p
    return yes >= 0 and table & yes == yes and table & no == 0


def extent(table: int, false_label: int, true_label: int, domain) -> int:
    return sum(1 << i for i,d in enumerate(domain)
               if (true_label if match(table, d['profile']) else false_label))


def deduplicate(rows):
    by_extension = {}
    for m in rows:
        rank = (m['cost'], L.key(m['expr']), m['false'], m['true'], m['truth'])
        key = m['extent']
        previous = by_extension.get(key)
        if previous is None or rank < previous[0]:
            by_extension[key] = (rank, m)
    return [m for _,m in sorted(by_extension.values(), key=lambda v:v[0])]


def model_pool(domain):
    predicates, stats = L.grammar(7)
    rows = []
    for p in predicates:
        for f in (0,1):
            for t in (0,1):
                rows.append({'truth':p['truth'], 'expr':p['expr'], 'cost':p['program_cost'],
                             'false':f, 'true':t, 'extent':extent(p['truth'],f,t,domain)})
    return deduplicate(rows), dict(stats, classifier_spellings=len(rows))


def teacher(points, kind: str, invert: bool=False) -> int:
    """Independent geometric teacher, called only at requested source points."""
    ps = set(map(tuple, points))
    rs, cs = [p[0] for p in ps], [p[1] for p in ps]
    a,b,c,d = min(rs),max(rs),min(cs),max(cs)
    box = {(r,s) for r in range(a,b+1) for s in range(c,d+1)}
    if kind == 'perimeter':
        expected = {(r,s) for r,s in box if r in (a,b) or s in (c,d)}
    elif kind == 'filled': expected = box
    elif kind == 'missing_corner': expected = box-{(b,d)}
    elif kind == 'corners': expected = {(a,c),(a,d),(b,c),(b,d)}
    else: raise ValueError(kind)
    return int((ps == expected) != invert)


class Engine:
    def __init__(self, models, domain, pool):
        self.models, self.domain, self.pool = models, domain, pool
        self.all_points = (1 << len(domain))-1
        self.all_models = (1 << len(models))-1
        self.extents = [m['extent'] for m in models]
        self.ones = [sum(1 << j for j,e in enumerate(self.extents) if e & (1 << i))
                     for i in range(len(domain))]
        self.rows = {r['mask']:r for r in pool}

    @lru_cache(maxsize=None)
    def bounds(self, version: int):
        if version == 0: return None
        lower, upper, remaining = self.all_points, 0, version
        while remaining:
            bit = remaining & -remaining; remaining -= bit
            e = self.extents[bit.bit_length()-1]
            lower &= e; upper |= e
        return lower, upper

    def observe(self, version: int, index: int, label: int) -> int:
        if label not in (0,1): raise ValueError('Binary labels required')
        positives = self.ones[index]
        return version & (positives if label else self.all_models ^ positives)

    def select(self, version: int, seen: set[int], policy: str):
        if policy not in POLICIES: raise ValueError(policy)
        bounds = self.bounds(version)
        if bounds is None: return None
        lo, up = bounds; options = []
        for row in self.pool:
            if row['mask'] in seen: continue
            yes = self.observe(version, row['index'], 1)
            no = self.observe(version, row['index'], 0)
            if not yes or not no: continue
            b0, b1 = self.bounds(no), self.bounds(yes)
            assert b0 is not None and b1 is not None
            n0,n1 = no.bit_count(),yes.bit_count()
            w0,w1 = (b0[1]^b0[0]).bit_count(),(b1[1]^b1[0]).bit_count()
            neg = (up & ~b0[1]).bit_count()
            quality = {'first':0, 'halving':max(n0,n1),
                       'negative_reduction':-neg, 'scope':max(w0,w1)}[policy]
            options.append(((quality,row['mask']), {'mask':row['mask'], 'index':row['index'],
                'remaining_models':[n0,n1], 'remaining_width':[w0,w1],
                'negative_exclusion':neg, 'quality':quality}))
        return min(options, key=lambda x:x[0])[1] if options else None

    def snapshot(self, version: int):
        bounds = self.bounds(version)
        if bounds is None:
            return {'version':'0x0','models':0,'inconsistent':True,
                    'lower':None,'upper':None,'width':None,'shortest':None}
        lo,up = bounds
        chosen = (version & -version).bit_length()-1
        return {'version':hex(version),'models':version.bit_count(),'inconsistent':False,
                'lower':hex(lo),'upper':hex(up),'width':(up^lo).bit_count(),
                'lower_count':lo.bit_count(),'upper_count':up.bit_count(),'shortest':chosen}

    def simulate(self, oracle, policy: str, budget: int=8):
        version = self.all_models; seen = set(); labels = []
        for mask in SEEDS:
            row = self.rows[mask]; label = oracle(row['points'])
            version = self.observe(version,row['index'],label); seen.add(mask)
            labels.append({'mask':mask,'index':row['index'],'label':label})
        records = [dict(self.snapshot(version), acquired=0, labels=list(labels))]
        for step in range(1,budget+1):
            choice = self.select(version,seen,policy)
            if choice is None: break
            mask = choice['mask']; label = oracle(self.rows[mask]['points'])
            before = version; version = self.observe(version,choice['index'],label)
            old,new = self.bounds(before),self.bounds(version)
            assert old is not None and new is not None
            seen.add(mask); labels.append({'mask':mask,'index':choice['index'],'label':label})
            snap = self.snapshot(version)
            old_extent = self.extents[records[-1]['shortest']]
            new_extent = self.extents[snap['shortest']]
            relation = ('equal' if old_extent==new_extent else
                        'specialisation' if not new_extent & ~old_extent else
                        'generalisation' if not old_extent & ~new_extent else 'incomparable')
            records.append(dict(snap, acquired=step, labels=list(labels), query=choice,
                excluded_models=hex(before & ~version),
                removed_positive=hex(old[1] & ~new[1]),
                removed_negative=hex(new[0] & ~old[0]), shortest_change=relation))
        return records


def fixed_mechanism(domain):
    """Prespecified contrast; no cost bound, all point predicates and orientations."""
    inputs = [shape(3,3,255),shape(3,3,495)]
    profiles = [profile(p) for p in inputs]
    raw = []
    for table in range(65536):
        for invert in (0,1):
            if [int(match(table,p)) ^ invert for p in profiles] == [0,1]:
                raw.append({'truth':table,'invert':invert,
                            'extent':extent(table,invert,1-invert,domain)})
    result = {}
    for name,extra,label in [('seed',None,None),('positive_singleton',[[0,0]],1),
                             ('negative_solid',shape(3,3,511),0)]:
        rs = raw if extra is None else [m for m in raw if int(match(m['truth'],profile(extra)))^m['invert']==label]
        ext = sorted({r['extent'] for r in rs})
        lo,up = (1<<len(domain))-1,0
        for e in ext:lo &= e;up |= e
        result[name] = {'point_table_programs':len(rs),'semantic_classifiers':len(ext),
                        'lower':hex(lo),'upper':hex(up),'width':(up^lo).bit_count(),
                        'extra_input':extra,'label':label}
    return result


def predict(out: Path):
    domain,pool = finite_domain(); models,grammar = model_pool(domain)
    engine = Engine(models,domain,pool); traces = []
    for kind in KINDS:
        for invert in (False,True):
            for policy in POLICIES:
                call = lambda ps,k=kind,i=invert:teacher(ps,k,i)
                traces.append({'kind':kind,'invert':invert,'policy':policy,
                               'records':engine.simulate(call,policy)})
    # Explicit out-of-language observation: equal abstraction, contradictory labels.
    first = [[r,c] for r in range(5) for c in range(5) if (r,c)!=(2,2)]
    second = [[r,c] for r in range(5) for c in range(5) if (r,c)!=(1,1)]
    assert profile(first)==profile(second)==(-1,-1)
    mixed_index = next(i for i,d in enumerate(domain) if tuple(d['profile'])==(-1,-1))
    version = engine.observe(engine.observe(engine.all_models,mixed_index,1),mixed_index,0)
    assert not version
    payload = {'baseline':'ed4b76bd202202976dfeee82dccd1cd6d32d7bc7',
        'grammar':grammar,'domain':domain,'pool':pool,
        'models':[dict(m,extent=hex(m['extent'])) for m in models],
        'traces':traces,'fixed_mechanism':fixed_mechanism(domain),
        'misspecification':{'inputs':[first,second],'labels':[1,0],**engine.snapshot(version)}}
    put(out/'predictions.json',payload)
    put(out/'prediction-run.json',{'source_sha256':sha(Path(__file__)),
        'language_sha256':sha(Path(L.__file__)), 'predictions_sha256':sha(out/'predictions.json'),
        'label_budget_after_two_seeds':8,'selection_reads_unrequested_labels':False,
        'test_inputs_used_for_selection':False})
    print('Frozen traces:',len(traces),'semantic models:',len(models),'scope classes:',len(domain))


def score(predictions: Path, query_inputs: Path, out: Path):
    frozen = read(predictions.with_name('prediction-run.json'))
    assert frozen['predictions_sha256']==sha(predictions)
    data = read(predictions); queries = read(query_inputs)
    pidx = {tuple(d['profile']):i for i,d in enumerate(data['domain'])}
    counts = {}; uniform = {}
    for kind in KINDS:
        counts[kind] = {}
        for name,xs in queries.items():
            hist = Counter()
            for ps in xs:
                i = pidx[profile(ps)]; y = teacher(ps,kind)
                if (kind,i) in uniform: assert uniform[(kind,i)]==y
                uniform[(kind,i)] = y; hist[(i,y)] += 1
            counts[kind][name] = [(i,y,n) for (i,y),n in sorted(hist.items())]
    summaries = []
    for trace in data['traces']:
        records=[]
        for state in trace['records']:
            lo,up = int(state['lower'],16),int(state['upper'],16)
            chosen = int(data['models'][state['shortest']]['extent'],16)
            banks = {}
            for name,hist in counts[trace['kind']].items():
                c = Counter()
                for i,y,n in hist:
                    y ^= trace['invert']; bit=1<<i
                    must,may,decision = bool(lo&bit),bool(up&bit),bool(chosen&bit)
                    c['cases']+=n;c['positive']+=y*n;c['negative']+=(1-y)*n
                    c['determined']+=(must==may)*n
                    c['determined_wrong']+=((must==may) and must!=y)*n
                    c['may_false_positive']+=(may and not y)*n
                    c['must_false_negative']+=(not must and y)*n
                    c['shortest_correct']+=(decision==y)*n
                    c['shortest_false_positive']+=(decision and not y)*n
                    c['shortest_false_negative']+=(not decision and y)*n
                banks[name]=dict(c)
            records.append({'acquired':state['acquired'],'models':state['models'],
                            'width':state['width'],'scores':banks})
        summaries.append({k:trace[k] for k in ('kind','invert','policy')}|{'records':records})
    mechanism={}
    for name,state in data['fixed_mechanism'].items():
        lo,up=int(state['lower'],16),int(state['upper'],16)
        metrics={}
        for bank,hist in counts['perimeter'].items():
            metrics[bank]={'possible_positive':sum(n for i,y,n in hist if up&(1<<i)),
                'necessary_positive':sum(n for i,y,n in hist if lo&(1<<i)),
                'ambiguous':sum(n for i,y,n in hist if (up^lo)&(1<<i)),
                'may_false_positive':sum(n for i,y,n in hist if not y and up&(1<<i))}
        mechanism[name]=metrics
    source={tuple(map(tuple,r['points'])) for r in data['pool']}
    overlaps={name:sum(tuple(map(tuple,ps)) in source for ps in xs) for name,xs in queries.items()}
    put(out/'scores.json',{'traces':summaries,'fixed_mechanism':mechanism,
        'exact_source_catalogue_overlap':overlaps,
        'scope_weights':'uniform semantic input classes, not uniform grids',
        'actual_fitting_sets':'two seed objects plus each trace\'s requested objects'})
    put(out/'score-run.json',{'predictions_sha256':sha(predictions),'query_inputs_sha256':sha(query_inputs),
                             'scores_sha256':sha(out/'scores.json')})
    print('Scored',len(summaries),'traces')


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('predict');q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('score');q.add_argument('--predictions',type=Path,required=True)
    q.add_argument('--query-inputs',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='predict':predict(a.out)
    else:score(a.predictions,a.query_inputs,a.out)

if __name__=='__main__':main()
