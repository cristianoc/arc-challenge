"""Synthetic controls, independent expression enumeration, no ARC corpus input."""
import importlib.util
import itertools
from pathlib import Path
import random
import unittest

spec=importlib.util.spec_from_file_location('guards025',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def primitive(centre=2, north=-1, south=-1, west=-1, east=-1):
    return tuple(range(10))+(centre,north,south,west,east,-1,-1,-1,-1)


def independent_eval(i,values):
    if i<19:return values[i]
    a,b=m.PAIRS[i-19]
    return values[b] if values[a]==-1 else values[a]


def explicit_fit(rows,features,lib):
    groups={}
    for row in rows:groups.setdefault(tuple(row[0][f] for f in features),[]).append(row)
    answer={}
    for key,rs in groups.items():
        good=[]
        for i in range(19 if lib=='base' else 163):
            if not all(independent_eval(i,r[1])==r[2] for r in rs):continue
            if lib=='observed' and i>=19:
                a,b=m.PAIRS[i-19]
                if not {r[1][a]>=0 for r in rs}=={True,False}:continue
            good.append(i)
        answer[key]=sum(1<<i for i in good)
    return answer


class Controls(unittest.TestCase):
    def test_grammar_and_piecewise(self):
        self.assertEqual((len(m.PAIRS),len(m.ACTIONS)),(144,163))
        for a,b in m.PAIRS:self.assertNotEqual(a,b)
        v=primitive(north=0,east=7)
        self.assertEqual(m.outputs(v),tuple(independent_eval(i,v) for i in range(163)))
        i=19+m.PAIRS.index((11,14));self.assertEqual(m.outputs(v)[i],0)

    def test_positive_requires_guard(self):
        rs=[((),primitive(north=1),1,(0,0,0)),((),primitive(),3,(1,0,0))]
        out=m.fit(rs,())[()];i=19+m.PAIRS.index((11,3))
        self.assertEqual(out[0],0);self.assertTrue(out[2]&(1<<i))
        self.assertEqual(m.apply([[2]],[((),primitive())],(),{():1<<i}),[[3]])

    def test_unobserved_fallback_is_not_identified(self):
        rs=[((),primitive(north=1),1,(0,0,0))]
        state=m.fit(rs,())[()]
        fallbacks=sum(1<<(19+m.PAIRS.index((11,c))) for c in range(10))
        self.assertEqual(state[1]&fallbacks,fallbacks)
        self.assertEqual(state[2]&fallbacks,0)
        query=[[((),primitive())]]
        d=m.condition({():fallbacks},query,())
        self.assertTrue(d['feasible'])
        self.assertEqual(m.apply([[2]],query[0],(),{tuple(k):v for k,v in d['kept']}),[[-1]])

    def test_zero_is_defined(self):
        a=19+m.PAIRS.index((11,7));v=primitive(north=0)
        self.assertEqual(m.outputs(v)[a],0)
        self.assertTrue(m.masks(v)[2]&(1<<a))

    def test_both_branches_per_key_not_task(self):
        rs=[((0,),primitive(north=1),1,(0,0,0)),((1,),primitive(),3,(0,0,1))]
        table=m.fit(rs,(0,));op=1<<(19+m.PAIRS.index((11,3)))
        self.assertTrue(all(v[1]&op for v in table.values()))
        self.assertTrue(all(not v[2]&op for v in table.values()))
        self.assertTrue(m.fit(rs,())[()][2]&op)

    def test_fit_against_enumeration(self):
        rng=random.Random(250)
        for trial in range(600):
            rs=[]
            for j in range(rng.randrange(1,8)):
                v=tuple(range(10))+(rng.randrange(3),)+tuple(rng.randrange(-1,3) for _ in range(8))
                rs.append(((rng.randrange(3),),v,rng.randrange(3),(0,0,j)))
            table=m.fit(rs,(0,))
            for column,lib in enumerate(m.LIBRARIES):
                self.assertEqual({k:v[column] for k,v in table.items()},explicit_fit(rs,(0,),lib))

    def test_one_operation_for_all_occurrences(self):
        raw={(): (1<<11)|(1<<12)}
        scenes=[[((),primitive(north=1)),((),primitive(south=1))]]
        d=m.condition(raw,scenes,());self.assertFalse(d['feasible']);self.assertEqual(d['kept'],[])

    def test_unknown_and_empty(self):
        self.assertTrue(m.condition({},[],())['feasible'])
        self.assertFalse(m.condition({},[[((),primitive())]],())['feasible'])
        self.assertFalse(m.condition({():0},[[((),primitive())]],())['feasible'])
        self.assertEqual(m.apply([[2]],[((),primitive())],(),{}),[[-1]])

    def test_totality_against_concrete_programs(self):
        rng=random.Random(251)
        for trial in range(200):
            actions=rng.sample(range(163),5)
            raw={(k,):sum(1<<a for a in actions if rng.randrange(2)) for k in range(2)}
            scenes=[[((rng.randrange(2),),tuple(range(10))+(rng.randrange(3),)+tuple(rng.randrange(-1,3) for _ in range(8))) for _ in range(4)]]
            choices=[[i for i in range(163) if v&(1<<i)] for k,v in raw.items()]
            programs=[p for p in itertools.product(*choices)
                      if all(independent_eval(p[fs[0]],vs)>=0 for fs,vs in scenes[0])]
            d=m.condition(raw,scenes,(0,));self.assertEqual(bool(programs),d['feasible'])
            if programs:
                for key,kept in d['kept']:
                    self.assertEqual({i for i in range(163) if kept&(1<<i)},{p[key[0]] for p in programs})

    def test_duplicate_support(self):
        rs=[((),primitive(north=1),1,(0,0,0)),((),primitive(),3,(1,0,0))]
        self.assertEqual(m.fit(rs,()),m.fit(rs*3,()))

    def test_fold_label_isolation(self):
        a={'input':[[1,2]],'output':[[1,2]]};b={'input':[[3,4]],'output':[[3,4]]}
        evil={'input':b['input'],'output':[[9,9]]}
        one=m.Engine([a,b]);two=m.Engine([a,evil])
        old=[{'arm':'exact','features':[]}]
        self.assertEqual(one.pool((0,),[b['input']],[one.scenes[1]],old),
                         two.pool((0,),[b['input']],[two.scenes[1]],old))
        self.assertEqual(m.Engine([a,b,a]).groups,[[0,2],[1]])

    def test_rank_prefers_base_only_on_tie(self):
        def model(lib,exact):
            return {'library':lib,'arm':'exact','features':[0],'cost':1,'cv_exact':exact,
                    'cv_fraction':[0,1],'condition':{'feasible':True},'predictions':[[[2]]]}
        base=model('base',0);obs=model('observed',0)
        self.assertEqual(m.decide([obs,base],1,'union_observed')['selected'][0],'base')
        obs['cv_exact']=1
        self.assertEqual(m.decide([obs,base],1,'union_observed')['selected'][0],'observed')
        self.assertEqual(m.decide([],1,'all')['predictions'],[None])


if __name__=='__main__':unittest.main()
