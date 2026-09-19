"""Pre-corpus controls. No ARC corpus or query answer files are read."""
import copy
import importlib.util
import itertools
from pathlib import Path
import random
import unittest
spec=importlib.util.spec_from_file_location('exp018',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Controls(unittest.TestCase):
    def test_order_and_validity_values(self):
        grid=[[3,0,0,0,7]];row=m.scene(grid)[1];f,v=row
        self.assertEqual(f[24],-1) # W compared with E
        self.assertEqual(f[25:29],(0,0,1,1))
        self.assertEqual((v[17],v[18]),(3,7))
        self.assertEqual(m.scene(grid)[2][0][24],0)
        self.assertEqual(m.scene(grid)[3][0][24],1)

    def test_guards_do_not_treat_zero_as_undefined(self):
        f,v=m.scene([[0,1,0]])[1]
        self.assertEqual((f[27],f[28]),(1,1))
        self.assertEqual((v[17],v[18]),(0,0))

    def test_blank_row_has_no_horizontal_endpoint(self):
        self.assertTrue(all(f[27:29]==(0,0) for f,v in m.scene([[0]*7])))

    def test_order_preserved_under_increasing_relabelling_of_lengths(self):
        for a,b in itertools.product(range(1,9),repeat=2):
            f=(0,)*15+(1,1,a,b);v=tuple(range(10))+(0,)*9
            before=m.extend([(f,v)])[0][0][24]
            f2=f[:17]+(3*a+2,3*b+2)
            self.assertEqual(before,m.extend([(f2,v)])[0][0][24])

    def test_fixed_relational_key_can_transfer_to_unseen_lengths(self):
        pairs=[m.teacher_grid(7,(1,2)),m.teacher_grid(11,(3,4))]
        learner=m.Learner(pairs);key=(24,27,28)
        table=learner.table((0,1),key,'guarded_order')
        self.assertIsNotNone(table)
        e=m.teacher_grid(15,(6,8))
        pred=m.s17.apply(e['input'],m.scene(e['input']),key,table)
        self.assertEqual(pred,e['output'])

    def test_guarded_rule_rejects_nonexistent_endpoint_actions(self):
        pairs=[m.teacher_grid(7,(1,2)),m.teacher_grid(11,(3,4))]
        l=m.Learner(pairs);fs=(24,27,28);table=l.table((0,1),fs,'guarded_order')
        self.assertEqual(m.s17.apply([[0]*13],m.scene([[0]*13]),fs,table),[[0]*13])

    def test_unknown_key_abstains(self):
        rows=m.scene([[0,1]])
        self.assertEqual(m.s17.apply([[0,1]],rows,(0,),{(0,):1}),[[0,-1]])

    def test_undefined_survivor_not_filtered(self):
        self.assertEqual(m.s17.apply([[0]],m.scene([[0]]),(),{():1|(1<<15)}),[[-1]])

    def test_guided_search_matches_exhaustive(self):
        rng=random.Random(1800)
        for _ in range(150):
            n=5;rs=[]
            for j in range(10):
                fs=tuple(rng.randrange(3) for _ in range(n))
                allow=rng.randrange(1,16)
                rs.append((fs,(0,)*19,0,(0,0,j),allow))
            got,_=m.search(rs,tuple(range(n)),3)
            fits=[s for k in range(4) for s in itertools.combinations(range(n),k) if m.s17.fit(rs,s,m.MASK) is not None]
            expected=[s for s in fits if not any(set(t)<set(s) for t in fits)]
            self.assertEqual(set(got),set(expected))

    def test_three_way_conflict(self):
        rs=[((0,), (0,)*19,0,(0,0,i),mask) for i,mask in enumerate((3,5,6))]
        core=m.s17.conflict(rs,(0,),m.MASK)
        self.assertEqual(len(core),3)
        self.assertEqual(m.search(rs,(0,))[1]['status'],'vocabulary_conflict')

    def test_exact_arm_matches_017(self):
        pairs=[m.teacher_grid(5,(1,2)),m.teacher_grid(7,(3,4))]
        a,b=m.Learner(pairs),m.s17.Learner(pairs)
        ma,da=a.discover((0,1),'exact');mb,db=b.discover((0,1),'joint')
        self.assertEqual([x['features'] for x in ma],[x['features'] for x in mb])
        for x,y in zip(ma,mb):self.assertEqual((x['cv_exact'],x['cv_fraction']),(y['cv_exact'],y['cv_fraction']))
        x,y=a.selected((0,1),'exact'),b.selected((0,1),'joint')
        self.assertEqual(None if x is None else x['features'],None if y is None else y['features'])

    def test_outer_label_isolation(self):
        pairs=[m.teacher_grid(5,(1,2)),m.teacher_grid(7,(3,4)),m.teacher_grid(9,(6,8))]
        changed=copy.deepcopy(pairs);changed[2]['output']=[[9]*9 for _ in range(5)]
        a,b=m.Learner(pairs),m.Learner(changed)
        self.assertEqual(a.selected((0,1),'union'),b.selected((0,1),'union'))
        self.assertEqual(a.selected((0,1),'union'),m.Learner(pairs[:2]).selected((0,1),'union'))

    def test_duplicate_inputs_withheld_together(self):
        pairs=[m.teacher_grid(5,(1,2)),m.teacher_grid(7,(3,4))]
        pairs.append(copy.deepcopy(pairs[0]));l=m.Learner(pairs)
        self.assertEqual(l.groups,((0,2),(1,)))

    def test_query_labels_are_not_read(self):
        class Trap(dict):
            def __getitem__(self,k):
                if k=='output':raise AssertionError('query label read')
                return super().__getitem__(k)
        task={'train':[m.teacher_grid(5,(1,2)),m.teacher_grid(7,(3,4))],
              'test':[Trap(input=[[1,0,2]])]}
        projected=m.b.project(task,'synthetic','training')
        self.assertEqual(projected['query_inputs'],[[[1,0,2]]])

    def test_prediction_eligibility_uses_demonstrations(self):
        p={'id':'x','split':'training','train':[{'input':[[0]],'output':[[0,0]]},{'input':[[1]],'output':[[1,1]]}],
           'query_inputs':[[[0]]]}
        self.assertFalse(m.investigate(p)['eligible'])

if __name__=='__main__':unittest.main()
