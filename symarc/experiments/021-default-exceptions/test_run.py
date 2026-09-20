"""Synthetic controls; no ARC corpus or controlled query answers are read."""
import copy
import importlib.util
import itertools
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location('defaults021', Path(__file__).with_name('run.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def exhaustive(masks, n):
    choices = [[a for a in range(n) if v & (1 << a)] for v in masks]
    programs = [(sum(d != a for a in assignment), d, assignment)
                for d in range(n) for assignment in itertools.product(*choices)]
    cost = min(p[0] for p in programs)
    best = [p for p in programs if p[0] == cost]
    return cost, sorted({p[1] for p in best}), [sum(1 << a for a in {p[2][i] for p in best}) for i in range(len(masks))]


class Controls(unittest.TestCase):
    def test_exhaustive_programs(self):
        rng = random.Random(21)
        for n in range(1, 5):
            for _ in range(75):
                values = [rng.randrange(1, 1 << n) for _ in range(rng.randrange(1, 5))]
                raw = {(i,): v for i, v in enumerate(values)}
                got, cert = m.compress(raw, n)
                cost, defaults, retained = exhaustive(values, n)
                self.assertEqual((cert['cost'], cert['defaults'], list(got.values())), (cost, defaults, retained))

    def test_copy_not_hardcoded(self):
        # A literal (operation 6) wins over centre-copy (operation 10).
        raw = {(0,): 1 << 6, (1,): (1 << 6) | (1 << 10), (2,): (1 << 6) | (1 << 10)}
        tab, cert = m.compress(raw)
        self.assertEqual(cert['defaults'], [6])
        self.assertEqual(set(tab.values()), {1 << 6})

    def test_copy_identified_across_colours(self):
        raw = {(1,): (1 << 1) | (1 << 10), (2,): (1 << 2) | (1 << 10)}
        tab, cert = m.compress(raw)
        self.assertEqual(cert['defaults'], [10])
        self.assertEqual(set(tab.values()), {1 << 10})

    def test_all_optima_preserved(self):
        tab, cert = m.compress({(0,): (1 << 1) | (1 << 10)})
        self.assertEqual(cert['defaults'], [1, 10])
        self.assertEqual(tab[(0,)], (1 << 1) | (1 << 10))

    def test_exceptions_stay_ambiguous(self):
        raw = {(0,): 1, (1,): 1, (2,): 6}
        tab, cert = m.compress(raw, 3)
        self.assertEqual(cert['defaults'], [0]); self.assertEqual(cert['cost'], 1)
        self.assertEqual(tab[(2,)], 6)

    def test_unknown_not_defaulted(self):
        table, _ = m.compress({(0,): 1})
        self.assertNotIn((1,), table)
        values = tuple(range(10)) + (0,) * 9
        self.assertEqual(m.e18.s17.apply([[1]], [((1,), values)], (0,), table), [[-1]])

    def test_undefined_survivor_remains(self):
        table, _ = m.compress({(): 1 << 15})
        self.assertEqual(m.e18.s17.apply([[0]], [((), (-1,) * 19)], (), table), [[-1]])

    def test_empty_and_invalid(self):
        self.assertEqual(m.compress(None)[0], None)
        self.assertEqual(m.compress({})[0], {})
        with self.assertRaises(ValueError): m.compress({(): 0})
        with self.assertRaises(ValueError): m.compress({(): 1 << 19})

    def test_label_subset_isolation(self):
        a = m.e18.teacher_grid(7, (1, 2)); b = m.e18.teacher_grid(11, (3, 4))
        altered = copy.deepcopy(b); altered['output'] = [[9] * 11 for _ in altered['input']]
        one, two = m.DefaultLearner([a, b]), m.DefaultLearner([a, altered])
        for fs in ((24, 27, 28), (4, 24)):
            self.assertEqual(one.table((0,), fs, 'guarded_order'), two.table((0,), fs, 'guarded_order'))

    def test_duplicate_observations_do_not_vote(self):
        pairs = [m.e18.teacher_grid(7, (1, 2)), m.e18.teacher_grid(11, (3, 4))]
        a = m.DefaultLearner(pairs); b = m.DefaultLearner(pairs + [copy.deepcopy(pairs[0])])
        fs = (24, 27, 28)
        self.assertEqual(a.table((0, 1), fs, 'guarded_order'), b.table((0, 1, 2), fs, 'guarded_order'))
        self.assertEqual(b.groups[0], (0, 2))

    def test_context_refinement_changes_prior(self):
        # Deliberate limitation: splitting one context changes this prior's vote.
        self.assertEqual(m.compress({(0,): 1, (1,): 2}, 2)[1]['defaults'], [0, 1])
        self.assertEqual(m.compress({(0,): 1, (1,): 2, (2,): 1}, 2)[1]['defaults'], [0])

    def test_driver_restores_original(self):
        problem = {'id': 'fixture', 'split': 'controlled', 'train': [m.e18.teacher_grid(7, (1, 2))], 'query_inputs': [[[0]]]}
        m.infer(problem, 'default')
        self.assertIs(m.e18.Learner, m.BASE)
        self.assertEqual(m.infer(problem, 'base'), m.e18.investigate(problem))


if __name__ == '__main__':
    unittest.main()
