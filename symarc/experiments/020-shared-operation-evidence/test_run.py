"""Synthetic tests only; independent exhaustive minimum-cover reference."""
import importlib.util,itertools,random,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('palette020',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def exhaustive(cs,n):
    good=[mask for mask in range(1<<n) if all(mask&c for c in cs)]
    return [mask for mask in good if mask.bit_count()==min(map(int.bit_count,good))] if good else []

class Tests(unittest.TestCase):
    def test_exhaustive(self):
        rng=random.Random(0)
        for n in range(1,7):
            for _ in range(100):
                cs=[rng.randrange(1<<n) for _ in range(rng.randrange(8))]
                self.assertEqual(m.minimum_palettes(cs),exhaustive(cs,n))
    def test_unchanged_colours(self):
        self.assertEqual(m.minimum_palettes([(1<<1)|(1<<10),(1<<2)|(1<<10)]),[1<<10])
    def test_one_colour_ambiguity(self):
        self.assertEqual(m.minimum_palettes([(1<<1)|(1<<10)]),[1<<1,1<<10])
    def test_distinct_contexts(self):
        table={(1,):(1<<1)|(1<<10),(2,):(1<<2)|(1<<10)}
        restricted,ps=m.restrict_table(table)
        self.assertEqual(set(restricted),set(table));self.assertNotIn((3,),restricted)
        self.assertEqual(set(restricted.values()),{1<<10})
    def test_mandatory_constants(self):
        self.assertEqual(m.minimum_palettes([1<<2,(1<<2)|(1<<10)]),[1<<2])
    def test_empty(self):
        self.assertEqual(m.restrict_table({}),({},[0]));self.assertEqual(m.restrict_table(None),(None,[]))
    def test_subset_isolation(self):
        a=m.e18.teacher_grid(7,(1,2));b=m.e18.teacher_grid(11,(3,4))
        other={'input':b['input'],'output':[[9]*11 for _ in b['input']]}
        l=m.Learner([a,b]);l2=m.Learner([a,other])
        for fs in ((24,27,28),(4,24)):
            self.assertEqual(l.table((0,),fs,'guarded_order'),l2.table((0,),fs,'guarded_order'))
    def test_undefined_not_removed(self):
        grid=[[0]];vals=tuple([-1]*19)
        self.assertEqual(m.e18.s17.apply(grid,[(tuple(range(29)),vals)],(),{():1<<15}),[[-1]])
if __name__=='__main__':unittest.main()
