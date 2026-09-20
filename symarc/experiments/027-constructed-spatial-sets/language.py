"""Generic Boolean point predicates and bounded set comprehension.

Equivalence is an exact 16-row truth table, never equality on labelled examples.
The four compact atoms expand to coordinate/projection/extremum/equality syntax.
"""
from __future__ import annotations
from functools import lru_cache
import json

ATOMS = ('rmin', 'rmax', 'cmin', 'cmax')
ALL = (1 << 16)-1
TARGET_OPS = ('input','generated','complement','union','intersection','input_minus','generated_minus','xor')


def key(x): return json.dumps(x, separators=(',', ':'))
def node_count(e): return 1 + sum(node_count(x) for x in e[1:] if isinstance(x, (tuple,list)))


def expand(e, aliases=None, active=frozenset()):
    tag=e[0]
    if tag=='alias':
        if not aliases or e[1] not in aliases or e[1] in active: raise ValueError('Unknown/cyclic alias')
        return expand(aliases[e[1]],aliases,active|{e[1]})
    if tag in ATOMS:
        axis=tag[0]; extreme=tag[1:]
        return ('eq',('coord',axis,('var','p')),(extreme,('project',axis,('var','S'))))
    if tag=='bool': return ('bool',bool(e[1]))
    if tag not in ('not','and','or'): raise ValueError(tag)
    if len(e)!=(2 if tag=='not' else 3): raise ValueError('Arity')
    return (tag,*(expand(x,aliases,active) for x in e[1:]))


def carrier_ast():
    def bounds(axis):
        return ('range',('min',('project',axis,('var','S'))),('max',('project',axis,('var','S'))))
    return ('product',bounds('r'),bounds('c'))


def set_ast(e): return ('filter',carrier_ast(),('lambda','p',expand(e)))
def recognition_cost(e): return node_count(('if',('set_eq',('var','S'),set_ast(e)),('label',0),('label',1)))
def cost(e): return node_count(expand(e))


def truth(e):
    if e[0] in ATOMS:
        bit=ATOMS.index(e[0]); return sum(1<<v for v in range(16) if v&(1<<bit))
    if e[0]=='bool':return ALL if e[1] else 0
    if e[0]=='not':return ALL^truth(e[1])
    if e[0]=='and':return truth(e[1])&truth(e[2])
    if e[0]=='or':return truth(e[1])|truth(e[2])
    raise ValueError(e)


def canonical(op,a,b=None):
    if b is None:return (op,a)
    a,b=sorted((a,b),key=key);return (op,a,b)


def grammar(bound=7, reverse=False):
    """Exact best expanded-cost representative per (Boolean size, truth table).

All operations are congruent for the complete truth-table semantics. Keeping
one least-cost child therefore preserves minimum achievable expanded costs.
"""
    levels={k:{} for k in range(1,bound+1)}; transitions=0
    def offer(level,signature,e):
        nonlocal transitions
        transitions+=1;previous=level.get(signature)
        if previous is None or (cost(e),key(e))<(cost(previous),key(previous)):level[signature]=e
    leaves=[(a,) for a in ATOMS]+[('bool',False),('bool',True)]
    for e in reversed(leaves) if reverse else leaves:offer(levels[1],truth(e),e)
    for n in range(2,bound+1):
        for sig,e in sorted(levels[n-1].items(),reverse=reverse):offer(levels[n],ALL^sig,('not',e))
        for i in range(1,n-1):
            j=n-1-i
            for x,a in sorted(levels[i].items(),reverse=reverse):
                for y,b in sorted(levels[j].items(),reverse=reverse):
                    for op,sig in (('and',x&y),('or',x|y)):
                        offer(levels[n],sig,canonical(op,a,b))
    all_forms={}
    for n in levels:
        for sig,e in levels[n].items():offer(all_forms,sig,e)
    candidates=[{'truth':sig,'expr':e,'predicate_cost':cost(e),'program_cost':recognition_cost(e)} for sig,e in all_forms.items()]
    candidates.sort(key=lambda r:(r['program_cost'],key(r['expr']),r['truth']))
    return candidates,{'states_per_size':{str(n):len(s) for n,s in levels.items()},'semantic_predicates':len(candidates),'transitions':transitions}


def normalize(points):
    ps=set()
    for p in points:
        if len(p)!=2 or any(type(v)is not int for v in p):raise ValueError('Integer coordinate pairs required')
        ps.add(tuple(p))
    if not ps:raise ValueError('Nonempty sets required')
    return frozenset(ps)


def bounds(points):
    ps=normalize(points);rs=[r for r,c in ps];cs=[c for r,c in ps]
    return min(rs),max(rs),min(cs),max(cs)


def roles(points):
    low,high,left,right=bounds(points)
    return [(r,c,int(r==low)+2*int(r==high)+4*int(c==left)+8*int(c==right))
            for r in range(low,high+1) for c in range(left,right+1)]


def profile(points):
    ps=normalize(points);yes=no=0
    for r,c,role in roles(ps):
        if (r,c) in ps:yes|=1<<role
        else:no|=1<<role
    return yes,no


def matches(signature,observations):
    yes,no=observations
    return signature&yes==yes and signature&no==0


def generate(signature,points):return frozenset((r,c) for r,c,role in roles(points) if signature&(1<<role))


def target(op,signature,points):
    ps=normalize(points);generated=generate(signature,ps)
    if op=='input':return ps
    if op=='generated':return generated
    if op=='complement':return frozenset((r,c) for r,c,k in roles(ps))-generated
    if op=='union':return ps|generated
    if op=='intersection':return ps&generated
    if op=='input_minus':return ps-generated
    if op=='generated_minus':return generated-ps
    if op=='xor':return ps^generated
    raise ValueError(op)


def target_ast(op,e):
    if op=='input':return ('var','S')
    g=set_ast(e)
    if op=='generated':return g
    if op=='complement':return ('difference',carrier_ast(),g)
    if op=='generated_minus':return ('difference',g,('var','S'))
    return ({'input_minus':'difference'}.get(op,op),('var','S'),g)
