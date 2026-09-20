"""Synthetic controls; corpus labels and 028 traces are not loaded here."""
import importlib.util
import itertools
from pathlib import Path
import random
import unittest

s=importlib.util.spec_from_file_location('rel029',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def reference(extents,n,version):
    hs=[h for h in range(len(extents)) if version & (1<<h)]
    if not hs:return None
    unary=[{(extents[h]>>i)&1 for h in hs} for i in range(n)]
    pairs=[{((extents[h]>>i)&1,(extents[h]>>j)&1) for h in hs}
           for i in range(n) for j in range(i+1,n)]
    return sum(len(x)-1 for x in unary),sum(len(x)-1 for x in pairs)


class Tests(unittest.TestCase):
    def test_lifts_enumeration(self):
        for n in range(2,7):
            for e in range(1<<n):
                lifts=m.lifts(e,n)
                for k,(i,j) in enumerate(itertools.combinations(range(n),2)):
                    value=2*((e>>i)&1)+((e>>j)&1)
                    self.assertEqual(m.pair_at(k,n),(i,j))
                    self.assertEqual([v for v,x in enumerate(lifts) if x&(1<<k)],[value])

    def test_small_family_reference(self):
        rng=random.Random(2901)
        for _ in range(300):
            n=rng.randrange(2,8);hs=rng.sample(range(1<<n),rng.randrange(1,min(20,1<<n)+1))
            e=m.Engine(hs,n,[{'mask':i+1,'index':i} for i in range(n)])
            v=rng.randrange(1,1<<len(hs))
            expected=reference(hs,n,v)
            self.assertEqual((e.measure(v,'marginal'),e.measure(v,'pair')),expected)

    def test_information_without_individual_answers(self):
        e=m.Engine([0,1,2,3],2,[])
        before=15;after=(1<<0)|(1<<3)
        self.assertEqual(e.unary(before),e.unary(after))
        self.assertEqual(e.measure(before,'pair'),3)
        self.assertEqual(e.measure(after,'pair'),1)
        result=e.relation_gain(before,after)
        self.assertEqual(result['relational_only_exclusions'],2)
        self.assertEqual(e.snapshot(before)['equality_classes'],2)
        self.assertEqual(e.snapshot(after)['equality_classes'],1)

    def test_pairwise_is_not_full_joint(self):
        e=m.Engine(list(range(8)),3,[])
        parity=sum(1<<h for h in range(8) if h.bit_count()%2==0)
        self.assertEqual(e.unary(255),e.unary(parity))
        self.assertEqual(e.joint(255),e.joint(parity))
        self.assertEqual(parity.bit_count(),4)
        self.assertEqual(e.relation_gain(255,parity)['relational_only_exclusions'],0)

    def test_nonbinary_relation(self):
        full=list(itertools.product(range(3),repeat=2));diagonal=[(x,x) for x in range(3)]
        self.assertEqual([{x[i] for x in full} for i in range(2)],
                         [{x[i] for x in diagonal} for i in range(2)])
        self.assertEqual((len(set(full)),len(set(diagonal))),(9,3))

    def test_empty_is_inconsistent(self):
        e=m.Engine([0,3],2,[])
        self.assertIsNone(e.unary(0));self.assertIsNone(e.joint(0))
        self.assertTrue(e.snapshot(0)['inconsistent'])
        with self.assertRaises(ValueError):e.pair_width(0)

    def test_query_choice_by_reference(self):
        hs=[0,3,5,10,15];n=4;queries=[{'index':i,'mask':i+1} for i in range(n)]
        e=m.Engine(hs,n,queries);v=31
        for p in m.POLICIES:
            candidates=[]
            for q in queries:
                children=[sum(1<<h for h,x in enumerate(hs) if ((x>>q['index'])&1)==y) for y in range(2)]
                scores=[x.bit_count() if p=='halving' else reference(hs,n,x)[p=='pair'] for x in children]
                candidates.append((max(scores),q['mask']))
            self.assertEqual((e.choose(v,p)['score'],e.choose(v,p)['mask']),min(candidates))

    def test_label_complement_invariant(self):
        hs=[0,1,3,6,12,15];queries=[{'index':i,'mask':i+1} for i in range(4)]
        a=m.Engine(hs,4,queries);b=m.Engine([15^h for h in hs],4,queries)
        for v in range(1,64):
            for policy in m.POLICIES:
                x,y=a.choose(v,policy),b.choose(v,policy)
                self.assertEqual(None if x is None else x['mask'],None if y is None else y['mask'])

    def test_duplicate_spellings_are_not_votes(self):
        # Production input is required to be extensional. Duplicated syntax is
        # quotiented first, exactly as model_pool does; do not quietly count it.
        hs=[0,1,3,6];duplicated=hs+[1,1,6]
        self.assertEqual(list(dict.fromkeys(duplicated)),hs)
        with self.assertRaises(ValueError):m.Engine(duplicated,3,[])

    def test_repeated_observation(self):
        e=m.Engine(list(range(8)),3,[])
        after=e.observe(255,1,0)
        self.assertEqual(after,e.observe(after,1,0))
        self.assertEqual(e.observe(after,1,1),0)

    def test_marginal_gap_identity(self):
        e=m.Engine([0,3,5,6,7],3,[])
        for v in range(1,32):
            snap=e.snapshot(v);hs=list(m.members(v));gap=0
            for i,j in itertools.combinations(range(3),2):
                left={(e.extents[h]>>i)&1 for h in hs};right={(e.extents[h]>>j)&1 for h in hs}
                actual={((e.extents[h]>>i)&1,(e.extents[h]>>j)&1) for h in hs}
                gap+=len(left)*len(right)-len(actual)
            self.assertEqual(snap['cartesian_gap'],gap)

    def test_forced_equalities_survive_filtering(self):
        hs=[0,3,4,7];e=m.Engine(hs,3,[])
        for v in range(1,16):
            for w in range(1,16):
                if w & ~v:continue
                for i,j in itertools.combinations(range(3),2):
                    if all(((hs[h]>>i)&1)==((hs[h]>>j)&1) for h in m.members(v)):
                        self.assertTrue(all(((hs[h]>>i)&1)==((hs[h]>>j)&1) for h in m.members(w)))


if __name__=='__main__':unittest.main()
