"""Synthetic controls; no ARC data read. Exhaustive reference checks are small."""
import importlib.util
import itertools
from pathlib import Path
import random
import unittest
spec=importlib.util.spec_from_file_location('refine016',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def reference(records,selected):
    table={}
    for f,x,y,_ in records:
        k=tuple(f[i] for i in selected)
        table[k]=table.get(k,2047)&m.allowed(x,y)
    return all(table.values())


class Checks(unittest.TestCase):
    def test_two_cell_certificate_complete(self):
        xy=list(itertools.product(range(3),repeat=2))
        for n in range(1,4):
            for xs in itertools.product(xy,repeat=n):
                rs=[((0,),x,y,(i,0,0)) for i,(x,y) in enumerate(xs)]
                p=m.conflict(rs,())
                self.assertEqual(p is None,reference(rs,()))
                if p: self.assertEqual(m.allowed(*p[0][1:3])&m.allowed(*p[1][1:3]),0)

    def test_complete_search_vs_exhaustive(self):
        rng=random.Random(19)
        for trial in range(120):
            rs=[(tuple(rng.randrange(3) for _ in range(5)),rng.randrange(3),rng.randrange(3),(i,0,0)) for i in range(6)]
            found,_=m.search(rs,tuple(range(5)),3)
            fits=[s for n in range(4) for s in itertools.combinations(range(5),n) if reference(rs,s)]
            minimal=[s for s in fits if not any(set(t)<set(s) for t in fits)]
            self.assertEqual(set(found),set(minimal))

    def test_essential_conjunction(self):
        rs=[((a,b),0,a^b,(i,0,0)) for i,(a,b) in enumerate(itertools.product(range(2),repeat=2))]
        self.assertEqual(m.search(rs,(0,1),2)[0],[(0,1)])
        self.assertEqual(m.search(rs,(0,1),1)[1]['status'],'feature_bound')

    def test_identical_features_different_required_action(self):
        rs=[((1,2),0,1,(0,0,0)),((1,2),0,2,(0,0,1))]
        self.assertEqual(m.search(rs,(0,1),3)[1]['status'],'vocabulary_conflict')

    def test_different_outputs_can_share_copy(self):
        rs=[((0,),1,1,(0,0,0)),((0,),2,2,(0,0,1))]
        self.assertEqual(m.search(rs,(0,),1)[0],[()])

    def test_scope_composition(self):
        g=[[0]*5,[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0]*5]
        fs=m.feature_rows(g)
        self.assertEqual(fs[12][7:11],(1,0,0,0))
        self.assertEqual(fs[0][8],1)
        self.assertEqual(fs[12][12],1)  # same colour elsewhere touches border
        self.assertEqual(fs[12][11],17)
        self.assertEqual(fs[12][4],0)

    def test_reach_has_unbounded_path_length(self):
        g=[[0,1,1,1,1],[0,0,0,0,1],[1,1,1,0,1],[1,1,1,0,1],[1,1,1,1,1]]
        fs=m.feature_rows(g)
        self.assertEqual(fs[18][4],0)  # no near neighbour on border
        self.assertEqual(fs[18][8],1)  # but connected to canvas border

    def test_raw_colour_vs_copy_action_uncertainty(self):
        pairs=[{'input':[[1]],'output':[[1]]}]
        learner=m.Learner(pairs)
        table=learner.table((0,),())
        out=m.s15.apply([[2]],[()],table,'copy')
        self.assertEqual(out,[[-1]])  # constant 1 and copy now disagree

    def test_duplicate_groups(self):
        p={'input':[[1,0]],'output':[[2,0]]}
        l=m.Learner([p,p,{'input':[[0,1]],'output':[[0,2]]}])
        self.assertEqual(l.groups,((0,1),(2,)))

    def test_refinement_relearned_on_subset(self):
        pairs=[{'input':[[1,0]],'output':[[2,0]]},{'input':[[1,1,3]],'output':[[4,4,3]]}]
        l=m.Learner(pairs)
        full,_=l.discover((0,1),'refined'); reduced,_=l.discover((0,),'refined')
        self.assertTrue(full and reduced)
        self.assertNotEqual([x['features'] for x in full],[x['features'] for x in reduced])
        self.assertTrue(all(v['origin'][0]==0 for mm in reduced for w in mm['necessity'] for v in w['witness']))

    def test_query_answer_not_accessed(self):
        class Problem(dict):
            def __getitem__(self,k):
                if k in ('test','answers','outputs'): raise AssertionError('answer access')
                return super().__getitem__(k)
        p=Problem(id='synthetic',split='training',train=[{'input':[[1,0]],'output':[[2,0]]},{'input':[[0,1]],'output':[[0,2]]}],query_inputs=[[[1,1,0]]])
        r=m.investigate(p)
        self.assertEqual(r['policies']['refined_cv']['predictions'],[[[2,2,0]]])

    def test_training_shape_eligibility(self):
        p={'id':'s','split':'training','train':[{'input':[[0]],'output':[[0,0]]},{'input':[[1]],'output':[[1]]}],'query_inputs':[[[0]]]}
        self.assertFalse(m.investigate(p)['eligible'])

    def test_necessity_witnesses(self):
        ps=[{'input':[[1,0]],'output':[[2,0]]},{'input':[[0,1]],'output':[[0,2]]}]
        l=m.Learner(ps); models,_=l.discover((0,1),'refined')
        for model in models:
            for reason in model['necessity']:
                a,b=reason['witness']; f=reason['necessary_feature']
                self.assertNotEqual(a['features'][f],b['features'][f])
                for g in model['features']:
                    if g!=f:self.assertEqual(a['features'][g],b['features'][g])

if __name__=='__main__':unittest.main()
