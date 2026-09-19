"""Synthetic pre-run tests. No ARC corpus is read."""
import importlib.util
import itertools
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('roles015',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_constant_vs_copy(self):
        es=[{'input':[[1,2]],'output':[[1,2]]}]
        self.assertIsNone(m.fit(es,[[0,0]],'literal'))
        t=m.fit(es,[[0,0]],'copy');self.assertEqual(t,{0:1<<10})
        self.assertEqual(m.apply([[7]],[0],t,'copy'),[[7]])
    def test_action_ambiguity_not_hidden_copy_preference(self):
        es=[{'input':[[1]],'output':[[1]]}];t=m.fit(es,[[0]],'copy')
        self.assertEqual(t[0],(1<<1)|(1<<10))
        self.assertEqual(m.apply([[2]],[0],t,'copy'),[[-1]])
        self.assertEqual(m.apply([[1]],[0],t,'copy'),[[1]])
    def test_unseen_role_not_default_copy(self):
        self.assertEqual(m.apply([[5]],[2],{1:1<<10},'copy'),[[-1]])
    def test_absent_domain_and_tied_largest(self):
        self.assertIsNone(m.mask([[1]],'rectangle_largest',0))
        self.assertIsNone(m.mask([[0,1,0]],'rectangle_largest',0))
        self.assertIsNone(m.mask([[0,1,0]],'component_largest',0))
    def test_four_connectivity(self):
        self.assertEqual(sorted(map(len,m.components([[1,0],[0,1]],1))),[1,1])
    def test_roles(self):
        self.assertEqual(m.context([[0]*3 for _ in range(3)],('rectangle_largest',0)),[1,1,1,1,2,1,1,1,1])
        self.assertEqual(m.context([[1,0,1]],('rectangle_largest',0)),[0,1,0])
    def test_rectangle_vs_component_scope(self):
        g=[[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,0,1,1]]
        a=m.context(g,('rectangle_largest',0));b=m.context(g,('components_all',0))
        self.assertEqual((a[12],b[12]),(1,2))
    def test_rectangle_bruteforce(self):
        for values in itertools.product((0,1),repeat=6):
            g=[list(values[:3]),list(values[3:])]
            for colour in (0,1):
                rectangles=[]
                for t in range(2):
                    for b in range(t,2):
                        for l in range(3):
                            for r in range(l,3):
                                s={(i,j) for i in range(t,b+1) for j in range(l,r+1)}
                                if all(g[i][j]==colour for i,j in s):rectangles.append(s)
                biggest=max(map(len,rectangles),default=0)
                best=[s for s in rectangles if len(s)==biggest]
                expected=best[0] if len(best)==1 else None
                self.assertEqual(m.mask(g,'rectangle_largest',colour),expected)
    def test_canonical_colour_and_boundary(self):
        g=[[1,2],[2,1]];gg=[[7,9],[9,7]]
        self.assertEqual(m.context(g,('canonical',1)),m.context(gg,('canonical',1)))
        self.assertNotEqual(m.context(g,('raw',1)),m.context(gg,('raw',1)))
        self.assertIn(10,m.context([[1]],('canonical',1))[0])
    def test_duplicate_group_and_training_only(self):
        a={'input':[[1,2]],'output':[[1,2]]};bb={'input':[[3,4]],'output':[[3,4]]}
        p={'id':'toy','split':'training','train':[a,bb,a]}
        r=m.investigate((p,True));self.assertNotIn('query_inputs',r)
        model=next(x for x in r['models'] if x['name']=='canonical:0:copy')
        self.assertEqual(model['cv_exact'],2)
        self.assertEqual(model['folds'][0]['excluded'],[0,2])
    def test_query_labels_not_read(self):
        class Explodes(dict):
            def __getitem__(self,k):
                if k=='output':raise AssertionError('query label read')
                return super().__getitem__(k)
        data={'train':[{'input':[[1]],'output':[[2]]},{'input':[[1,1]],'output':[[2,2]]}],
              'test':[Explodes(input=[[1]])]}
        r=m.investigate((m.b.project(data,'toy','training'),False))
        self.assertTrue(r['eligible'])
    def test_undefined_blocks_consensus(self):
        self.assertEqual(m.consensus([[[[1]]],[None]],[[[1]]]),[None])
        self.assertEqual(m.consensus([[[[1]]],[[[2]]]],[[[1]]]),[[[-1]]])
    def test_literal_fits_equal_seen_labels(self):
        es=[{'input':[[1,1]],'output':[[3,3]]}]
        t=m.fit(es,[[0,0]],'literal');self.assertEqual(m.apply([[2]],[0],t,'literal'),[[3]])
    def test_shape_change_ineligible(self):
        p={'id':'toy','split':'training','train':[{'input':[[0]],'output':[[0,0]]},{'input':[[1]],'output':[[1]]}]}
        self.assertFalse(m.investigate((p,True))['eligible'])
if __name__=='__main__':unittest.main()
