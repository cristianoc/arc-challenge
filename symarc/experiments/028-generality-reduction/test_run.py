"""Pre-run semantic controls; no teacher experiment or benchmark score."""
import importlib.util
from pathlib import Path
import random
import unittest

s=importlib.util.spec_from_file_location('reduction028',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def engine(extents,n):
    models=[{'extent':e} for e in extents]
    domain=[{} for _ in range(n)]
    pool=[{'mask':i+1,'index':i,'points':[[0,i]]} for i in range(n)]
    return m.Engine(models,domain,pool)


class Checks(unittest.TestCase):
    def test_empty_is_inconsistent(self):
        e=engine([0,7],3)
        self.assertIsNone(e.bounds(0));self.assertTrue(e.snapshot(0)['inconsistent'])
        self.assertIsNone(e.select(0,set(),'scope'))

    def test_bounds_against_sets(self):
        rng=random.Random(2801)
        for _ in range(300):
            vals=[rng.randrange(256) for _ in range(6)];e=engine(vals,8)
            ids=rng.randrange(1,64)
            sets=[{i for i in range(8) if v&(1<<i)} for j,v in enumerate(vals) if ids&(1<<j)]
            lo=set.intersection(*sets);up=set.union(*sets)
            self.assertEqual(e.bounds(ids),(sum(1<<i for i in lo),sum(1<<i for i in up)))

    def test_monotone_envelopes(self):
        e=engine([0,1,3,7,15],4)
        v=e.all_models
        for index,label in [(0,1),(2,0)]:
            old=e.bounds(v);v=e.observe(v,index,label);new=e.bounds(v)
            self.assertFalse(old[0]&~new[0]);self.assertFalse(new[1]&~old[1])

    def test_duplicate_observation(self):
        e=engine([1,2,3],2)
        v=e.observe(e.all_models,0,1)
        self.assertEqual(v,e.observe(v,0,1))

    def test_duplicate_hypotheses(self):
        rows=[dict(truth=i,expr=['bool',bool(i)],cost=1,false=0,true=1,extent=i) for i in (0,1)]
        self.assertEqual(m.deduplicate(rows),m.deduplicate(rows*7))

    def test_specialisation_can_destroy_positives(self):
        # Fewer accepted inputs is not necessarily more accurate.
        target=7;specialised=1
        self.assertFalse(specialised & ~target)
        self.assertEqual((target & ~specialised).bit_count(),2)

    def test_count_and_scope_are_distinct(self):
        e=engine([1,3,13,15],4)
        all_lo,all_up=e.bounds(e.all_models)
        self.assertEqual(e.observe(e.all_models,1,0).bit_count(),2)
        self.assertEqual(e.observe(e.all_models,2,0).bit_count(),2)
        widths=[]
        for q in (1,2):
            b=e.bounds(e.observe(e.all_models,q,0));widths.append((b[1]^b[0]).bit_count())
        self.assertNotEqual(widths[0],widths[1])

    def test_label_symmetric_policies(self):
        vals=[0,1,3,7,15,31];e=engine(vals,5);f=engine([31^v for v in vals],5)
        for policy in ('scope','halving','first'):
            self.assertEqual(e.select(e.all_models,set(),policy)['mask'],f.select(f.all_models,set(),policy)['mask'])

    def test_only_informative_queries(self):
        e=engine([1,3],3)
        for p in m.POLICIES:
            self.assertEqual(e.select(e.all_models,set(),p)['index'],1)
            self.assertIsNone(e.select(e.all_models,{2},p))

    def test_coordinate_translation(self):
        ps=[[0,0],[0,1],[2,2]]
        self.assertEqual(m.profile(ps),m.profile([[r-9,c+7] for r,c in ps]))

    def test_nonuniform_role_collapse(self):
        box=[[r,c] for r in range(5) for c in range(5)]
        a=[p for p in box if p!=[2,2]];b=[p for p in box if p!=[1,1]]
        self.assertEqual(m.profile(a),m.profile(b))
        self.assertEqual(m.profile(a),(-1,-1))
        for t in (0,1,65534,65535):self.assertFalse(m.match(t,m.profile(a)))

    def test_selection_has_no_teacher_access(self):
        import inspect
        signature=inspect.signature(m.Engine.select)
        self.assertEqual(list(signature.parameters),['self','version','seen','policy'])
        e=engine([0,7],3)
        with self.assertRaises(ValueError):e.observe(e.all_models,0,2)


if __name__=='__main__':unittest.main()
