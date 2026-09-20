"""Pre-corpus synthetic checks for the query-aware selection policies."""
import copy
import importlib.util
from pathlib import Path
import random
import unittest

spec=importlib.util.spec_from_file_location('app023',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def candidate(i,preds,exact=1,fraction=(1,1),cost=None):
    return dict(arm='exact',features=[i],names=[str(i)],cost=i if cost is None else cost,
                cv_exact=exact,cv_fraction=list(fraction),predictions=preds)


class Tests(unittest.TestCase):
    def test_empty(self):
        for policy in m.POLICIES:
            self.assertEqual(m.decide([],2,policy)['predictions'],[None,None])

    def test_complete_semantics(self):
        for grids in ([],[None],[[[]]],[[[True]]],[[[-1]]],[[[10]]],[[[1],[1,2]]]):
            self.assertFalse(m.complete(grids),grids)
        self.assertTrue(m.complete([[[0,1]]]))

    def test_tie_uses_applicable_model(self):
        pool=[candidate(0,[[[-1]]]),candidate(1,[[[2]]])]
        self.assertEqual(m.decide(pool,1,'baseline')['selected'],['exact',[0]])
        self.assertEqual(m.decide(pool,1,'tie_complete')['selected'],['exact',[1]])

    def test_evidence_is_not_overridden_in_tie_arm(self):
        pool=[candidate(0,[[[-1]]],exact=2),candidate(1,[[[2]]],exact=1)]
        self.assertEqual(m.decide(pool,1,'tie_complete')['selected'],['exact',[0]])
        self.assertEqual(m.decide(pool,1,'complete_first')['selected'],['exact',[1]])

    def test_equal_rationals_are_ties(self):
        pool=[candidate(0,[[[-1]]],fraction=(2,4)),candidate(1,[[[3]]],fraction=(1,2))]
        self.assertEqual(m.decide(pool,1,'tie_complete')['selected'],['exact',[1]])

    def test_complete_baseline_preserved(self):
        rng=random.Random(23)
        for _ in range(100):
            pool=[candidate(i,[[[rng.choice([-1,0,1,2])]]],exact=rng.randrange(3)) for i in range(6)]
            old=m.decide(pool,1,'baseline')
            if m.complete(old['predictions']):
                for policy in ('tie_complete','complete_first'):
                    self.assertEqual(m.decide(pool,1,policy)['predictions'],old['predictions'])

    def test_no_complete_candidates_keep_original(self):
        pool=[candidate(0,[[[-1]]]),candidate(1,[None])]
        for policy in m.POLICIES:
            self.assertEqual(m.decide(pool,1,policy)['predictions'],[[[-1]]])

    def test_consensus_disagreement(self):
        pool=[candidate(0,[[[1]]]),candidate(1,[[[2]]])]
        self.assertEqual(m.decide(pool,1,'tie_consensus')['predictions'],[None])
        self.assertEqual(m.decide(pool,1,'tie_consensus')['distinct_complete_answers'],2)

    def test_aliases_do_not_change_consensus(self):
        pool=[candidate(0,[[[1]]]),candidate(1,[[[1]]])]
        self.assertEqual(m.decide(pool,1,'tie_consensus')['predictions'],[[[1]]])

    def test_complete_is_not_correct(self):
        # The observed specification need not constrain this query label.
        pool=[candidate(0,[[[-1]]]),candidate(1,[[[7]]])]
        for policy in ('tie_complete','complete_first','tie_consensus'):
            self.assertNotEqual(m.decide(pool,1,policy)['predictions'],[[[8]]])

    def test_selection_can_depend_on_query_batch(self):
        pool=[candidate(0,[[[1]],[[-1]]]),candidate(1,[[[2]],[[3]]])]
        one=[{**p,'predictions':p['predictions'][:1]} for p in pool]
        self.assertEqual(m.decide(one,1,'tie_complete')['predictions'],[[[1]]])
        self.assertEqual(m.decide(pool,2,'tie_complete')['predictions'][0],[[2]])

    def test_order_invariant_and_nonmutating(self):
        pool=[candidate(0,[[[-1]]]),candidate(1,[[[2]]])];old=copy.deepcopy(pool)
        for policy in m.POLICIES:
            self.assertEqual(m.decide(pool,1,policy),m.decide(pool[::-1],1,policy))
        self.assertEqual(pool,old)

    def test_outer_labels_do_not_choose_their_model(self):
        train=[{'input':[[0,1]],'output':[[0,2]]},{'input':[[1,0]],'output':[[2,0]]}]
        p={'id':'isolation','split':'synthetic','train':train,'query_inputs':[[[1,1]]]}
        q=copy.deepcopy(p);q['train'][0]['output']=[[9,9]]
        for mode in ('base','default'):
            a=m.investigate(p,mode);b=m.investigate(q,mode)
            fa=next(f for f in a['outer'] if f['excluded']==[0]);fb=next(f for f in b['outer'] if f['excluded']==[0])
            self.assertEqual(fa['models'],fb['models']);self.assertEqual(fa['policies'],fb['policies'])
            self.assertEqual(a['policies']['baseline']['selected'],a['models'][0] and
                m.model_id(min(a['models'],key=m.rank)))

    def test_duplicate_inputs_are_held_together(self):
        pair={'input':[[0,1]],'output':[[0,2]]}
        p={'id':'duplicates','split':'synthetic','train':[pair,copy.deepcopy(pair),
             {'input':[[1,0]],'output':[[2,0]]}],'query_inputs':[[[1]]]}
        row=m.investigate(p,'base')
        self.assertEqual([f['excluded'] for f in row['outer']],[[0,1],[2]])

    def test_extra_query_answer_field_is_ignored(self):
        p={'id':'projection','split':'synthetic','train':[
            {'input':[[0,1]],'output':[[0,2]]},{'input':[[1,0]],'output':[[2,0]]}],
           'query_inputs':[[[1,0]]]}
        q={**p,'test_outputs':[[[9]]],'answers':object()}
        self.assertEqual(m.investigate(p,'base'),m.investigate(q,'base'))

if __name__=='__main__': unittest.main()
