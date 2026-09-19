"""Synthetic tests only. Enumerate concrete programs independently of conditioner."""
import importlib.util
import itertools
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location('total024', Path(__file__).with_name('run.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def enumeration(raw, valid, mode, n):
    keys = sorted(raw); allowed = {z: [a for a in range(n) if raw[z] & (1 << a)] for z in keys}
    if mode == 'base':
        branches = [(None, allowed)]
    elif not keys:
        branches = [(None, {})]
    else:
        cost = {a: sum(a not in allowed[z] for z in keys) for a in range(n)}
        ds = [a for a in range(n) if cost[a] == min(cost.values())]
        branches = [(d, {z: [d] if d in allowed[z] else allowed[z] for z in keys}) for d in ds]
    programmes = []; alive = set()
    for d, choices in branches:
        for values in itertools.product(*(choices[z] for z in keys)):
            function = dict(zip(keys, values))
            if all(z in function and valid[z] & (1 << function[z]) for z in valid):
                programmes.append(function)
                if d is not None: alive.add(d)
    projection = {z: sum(1 << a for a in {f[z] for f in programmes}) for z in keys}
    return bool(programmes), projection, sorted(alive)


class Tests(unittest.TestCase):
    def test_exhaustive_concrete_programs(self):
        rng = random.Random(24)
        for _ in range(600):
            n = rng.randrange(1, 6); k = rng.randrange(0, 5)
            raw = {(i,): rng.randrange(1, 1 << n) for i in range(k)}
            valid = {(i,): rng.randrange(1 << n) for i in range(k + rng.randrange(2)) if rng.randrange(3)}
            for mode in ('base', 'default'):
                actual = m.condition(raw, valid, mode, n)
                expected = enumeration(raw, valid, mode, n)
                self.assertEqual((actual['feasible'], {tuple(r[0]): r[4] for r in actual['entries']},
                                  actual['surviving_defaults']), expected)

    def test_existential_not_agreement(self):
        result = m.condition({(0,): 3}, {(0,): 3}, 'base', 2)
        self.assertTrue(result['feasible']); self.assertEqual(result['entries'][0][4], 3)

    def test_filter_defined_operation(self):
        result = m.condition({(0,): 3}, {(0,): 1}, 'base', 2)
        self.assertEqual(result['entries'][0][4], 1)

    def test_shared_key_requires_one_operation_everywhere(self):
        vals1 = (1, -1); vals2 = (-1, 2)
        q = [[((0,), vals1)], [((0,), vals2)]]
        valid = m.validities(q, (0,))
        self.assertEqual(valid, {(0,): 0})
        self.assertFalse(m.condition({(0,): 3}, valid, 'base', 2)['feasible'])

    def test_shared_default_cannot_be_projected_first(self):
        raw = {(0,): 3, (1,): 3}; valid = {(0,): 1, (1,): 2}
        r = m.condition(raw, valid, 'default', 2)
        self.assertTrue(r['naive_feasible']); self.assertFalse(r['feasible'])
        self.assertEqual(r['surviving_defaults'], [])

    def test_default_branch_propagates_between_contexts(self):
        r = m.condition({(0,): 7, (1,): 3}, {(0,): 5, (1,): 3}, 'default', 3)
        self.assertEqual(r['surviving_defaults'], [0])
        self.assertEqual(r['entries'][1][4:], [1, 3])

    def test_do_not_reoptimize_after_conditioning(self):
        raw = {(0,): 3, (1,): 5}; valid = {(0,): 2, (1,): 4}
        r = m.condition(raw, valid, 'default', 3)
        self.assertEqual(r['defaults'], [0])
        self.assertFalse(r['feasible'])
        # A more expensive default would work; conditioning must not adopt it.
        self.assertTrue(m.condition(raw, valid, 'base', 3)['feasible'])

    def test_unknown_key_rejects_family_not_only_cell(self):
        r = m.condition({(0,): 1}, {(0,): 1, (1,): 1}, 'base', 1)
        self.assertFalse(r['feasible']); self.assertEqual(r['entries'][0][4], 0)

    def test_zero_is_defined(self):
        self.assertEqual(m.validities([[((0,), (0, -1))]], (0,)), {(0,): 1})

    def test_empty_family_is_not_vacuous_certainty(self):
        r = m.condition({(0,): 1}, {(0,): 0}, 'base', 1)
        self.assertFalse(r['feasible']); self.assertEqual(r['entries'][0][4], 0)

    def test_tied_defaults_retained(self):
        r = m.condition({(0,): 3}, {(0,): 3}, 'default', 2)
        self.assertEqual(r['surviving_defaults'], [0,1])

    def test_domain_condition_order_independent(self):
        a = [[((0,), (1,-1,2))]]; b = [[((0,), (-1,3,2))]]
        self.assertEqual(m.validities(a+b, (0,)), m.validities(b+a, (0,)))
        self.assertEqual(m.validities(a+b, (0,)), {(0,): 4})

    def test_filter_does_not_read_excluded_labels(self):
        train = [{'input': [[0,1]], 'output': [[0,1]]}, {'input': [[1,0]], 'output': [[1,0]]}]
        scenes = [m.e18.scene(p['input']) for p in train]; fs=(0,)
        table = m.e18.s17.fit([(f,v,y,(0,0,i),sum(1<<a for a,x in enumerate(v) if x==y))
                     for i,((f,v),y) in enumerate(zip(scenes[0],train[0]['output'][0]))], fs, m.e18.MASK)
        expected = m.e18.s17.apply(train[1]['input'],scenes[1],fs,table)
        model = dict(arm='exact',features=[0],names=['colour'],cost=1,cv_exact=0,cv_fraction=[0,1],predictions=[expected])
        x = m.conditioned_models(train,scenes,(0,),[train[1]['input']],[scenes[1]],[model],'base')
        train[1]['output'] = [[8,9]]
        y = m.conditioned_models(train,scenes,(0,),[train[1]['input']],[scenes[1]],[model],'base')
        self.assertEqual(x,y)

    def test_decision_may_keep_feasible_ambiguity(self):
        first = dict(arm='exact',features=[0],cost=1,cv_exact=1,cv_fraction=[1,1],predictions=[[[-1]]],
                     feasible=True,total_predictions=[[[-1]]])
        second = dict(arm='exact',features=[1],cost=2,cv_exact=1,cv_fraction=[1,1],predictions=[[[3]]],
                      feasible=True,total_predictions=[[[3]]])
        ds=m.decisions([first,second],1)
        self.assertEqual(ds['total_ranked']['predictions'],[[[-1]]])
        self.assertEqual(ds['total_tie_complete']['predictions'],[[[3]]])

    def test_complete_baseline_is_preserved(self):
        model = dict(arm='exact',features=[],cost=0,cv_exact=0,cv_fraction=[0,1],predictions=[[[2]]],
                     feasible=True,total_predictions=[[[2]]])
        ds=m.decisions([model],1)
        self.assertTrue(all(d['predictions']==[[[2]]] for d in ds.values()))

    def test_cache_strips_heldout_scores(self):
        c={'id':'a','split':'t','mode':'base','models':[], 'outer':[{'excluded':[0],'models':[], 'scores': 'secret'}]}
        self.assertNotIn('scores',m.trim_cache(c)['outer'][0])


if __name__ == '__main__': unittest.main()
