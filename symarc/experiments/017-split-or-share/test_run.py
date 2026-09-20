"""Synthetic pre-corpus controls; no ARC files or query answers are opened."""
import copy
import importlib.util
import itertools
from pathlib import Path
import random
import unittest

spec=importlib.util.spec_from_file_location('split017',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def rec(features,mask,i=0):return (tuple(features),tuple(range(19)),0,(0,0,i),mask)

def problem():
    def pair(g):return dict(input=g,output=[[2 if v==1 else v for v in row] for row in g])
    return dict(id='synthetic',split='training',train=[pair([[1,0],[0,1]]),pair([[1,0,1]])],query_inputs=[[[0,1]]])


class Controls(unittest.TestCase):
    def test_three_way_conflict_not_pair(self):
        rs=[rec([0],0b011,0),rec([0],0b101,1),rec([0],0b110,2)]
        self.assertTrue(all(m.intersection(p,7) for p in itertools.combinations(rs,2)))
        core=m.conflict(rs,(0,),7);self.assertEqual(len(core),3)
        self.assertEqual(m.intersection(core,7),0)

    def test_minimal_core_discards_redundancy(self):
        rs=[rec([0],7,0),rec([0],3,1),rec([0],5,2),rec([0],6,3)]
        core=m.minimal_core(rs,7);self.assertEqual(len(core),3)
        self.assertTrue(all(m.intersection(core[:i]+core[i+1:],7) for i in range(3)))

    def test_split_resolves_triple(self):
        rs=[rec([0,0],3,0),rec([0,1],5,1),rec([1,0],6,2)]
        self.assertEqual(set(m.search(rs,(0,1),7,2)[0]),{(0,),(1,)})

    def test_new_shared_operation_resolves_without_features(self):
        rs=[rec([0],3|8,0),rec([0],5|8,1),rec([0],6|8,2)]
        self.assertEqual(m.search(rs,(0,),7)[1]['status'],'vocabulary_conflict')
        self.assertEqual(m.search(rs,(0,),15)[0],[()])

    def test_random_search_equals_exhaustive(self):
        rng=random.Random(1701)
        for case in range(150):
            n=4;rs=[rec([rng.randrange(3) for _ in range(n)],rng.randrange(1,16),i) for i in range(8)]
            expected=[]
            for size in range(4):
                for fs in itertools.combinations(range(n),size):
                    table={}
                    for r in rs:
                        k=tuple(r[0][i] for i in fs);table[k]=table.get(k,15)&r[4]
                    if all(table.values()) and not any(set(x)<set(fs) for x in expected):expected.append(fs)
            self.assertEqual(set(m.search(rs,tuple(range(n)),15)[0]),set(expected))

    def test_base_action_property_exhaustive(self):
        labels=list(itertools.product(range(3),repeat=2))
        for xy in itertools.product(labels,repeat=3):
            rs=[rec([0],(1<<y)|((1<<10) if x==y else 0),i) for i,(x,y) in enumerate(xy)]
            old=[((0,),x,y,(0,0,i)) for i,(x,y) in enumerate(xy)]
            self.assertEqual(m.conflict(rs,(0,),m.BASE_MASK) is None,m.s16.conflict(old,(0,)) is None)

    def test_directional_run_and_terminal(self):
        g=[[3,3,4,4],[3,2,2,4],[1,2,2,4]]
        rows=m.scene(g)
        f,a=rows[5] # (1,1), colour 2
        self.assertEqual(f[15:],(1,2,1,2))
        self.assertEqual(a[11:15],(3,2,3,2))
        self.assertEqual(a[15:],(3,-1,3,4))

    def test_run_at_edge(self):
        f,a=m.scene([[1,1,1]])[1]
        self.assertEqual(f[15:],(1,1,2,2))
        self.assertEqual(a[15:],(-1,-1,-1,-1))

    def test_undefined_surviving_action_abstains(self):
        grid=[[1]];rows=m.scene(grid)
        self.assertEqual(m.apply(grid,rows,(),{(): (1<<1)|(1<<11)}),[[-1]])
        self.assertEqual(m.apply(grid,rows,(),{(): (1<<1)|(1<<10)}),[[1]])

    def test_disagreeing_actions_abstain(self):
        grid=[[1,2]];rows=m.scene(grid)
        self.assertEqual(m.apply(grid,rows,(),{(): (1<<10)|(1<<14)}),[[-1,-1]])

    def test_unknown_key_abstains(self):
        grid=[[1]];self.assertEqual(m.apply(grid,m.scene(grid),(0,),{(2,):1<<2}),[[-1]])

    def test_query_projection_does_not_read_outputs(self):
        class NoAnswer(dict):
            def __getitem__(self,k):
                if k=='output':raise AssertionError('read test answer')
                return super().__getitem__(k)
        p=problem();t={'train':p['train'],'test':[NoAnswer(input=p['query_inputs'][0])]}
        self.assertEqual(m.b.project(t,p['id'],p['split']),p)

    def test_duplicate_inputs_grouped(self):
        p=problem();p['train'].append(copy.deepcopy(p['train'][0]))
        learner=m.Learner(p['train']);self.assertEqual(learner.groups[0],(0,2))

    def test_outer_label_not_used_in_construction(self):
        p=problem();q=copy.deepcopy(p);q['train'][0]['output']=[[9,9],[9,9]]
        a=m.Learner(p['train']);b=m.Learner(q['train'])
        for arm in m.POLICIES:
            ma=a.selected((1,),arm);mb=b.selected((1,),arm)
            self.assertEqual(ma,mb)
            self.assertEqual(a.predict((1,),ma,p['train'][0]['input'],a.scenes[0]),
                             b.predict((1,),mb,p['train'][0]['input'],b.scenes[0]))

    def test_base_matches_previous_learner_on_synthetic(self):
        p=problem();a=m.investigate(p);b=m.s16.investigate(p)
        self.assertEqual(a['policies']['base']['predictions'],b['policies']['refined_cv']['predictions'])
        self.assertEqual({tuple(x['features']) for x in a['families']['base']['models']},
                         {tuple(x['features']) for x in b['families']['refined']['models']})

    def test_training_shape_eligibility(self):
        p=problem();p['train'][0]['output']=[[1]]
        self.assertFalse(m.investigate(p)['eligible'])

if __name__=='__main__':unittest.main()
