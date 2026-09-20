"""Synthetic controls run before the construction experiment."""
import random
import unittest
import language as L
import run as R

class Tests(unittest.TestCase):
    def test_observation_sets(self):
        self.assertEqual(L.observations([(0,0),(0,2),(0,2)]),(2,1,3))
        self.assertEqual(L.observations([(-7,5),(-5,5)]),(2,3,1))
        with self.assertRaises(ValueError): L.observations([])
        with self.assertRaises(ValueError): L.observations([(True,0)])

    def test_types(self):
        with self.assertRaises(ValueError): L.canon(("eq",("eq",("lit",0),("lit",1)),("count",)))
        with self.assertRaises(ValueError): L.canon(("and",("count",),("count",)))
        with self.assertRaises(ValueError): L.canon(("lit",2))

    def test_commutativity_and_noncommutativity(self):
        x,y=("count",),("span_r",)
        self.assertEqual(L.canon(("mul",x,y)),L.canon(("mul",y,x)))
        self.assertNotEqual(L.canon(("sub",x,y)),L.canon(("sub",y,x)))

    def test_macro_cost(self):
        x=L.canon(("eq",("count",),("mul",("span_r",),("span_c",))))
        aliases={"P":x,"P2":("alias","P"),"n":("count",)}
        self.assertEqual(L.canon(("alias","P2"),aliases),x)
        self.assertEqual(L.cost(x),5)
        self.assertEqual(L.grammar(5),L.grammar(5,aliases))
        with self.assertRaises(ValueError):L.canon(("alias","loop"),{"loop":("alias","loop")})

    def test_exhaustive_direct_agreement(self):
        rng=random.Random(2600); ps=L.grammar(3)
        for _ in range(120):
            xs=[tuple(rng.randrange(1,5) for _ in range(3)) for _ in range(6)]
            labels=[rng.randrange(3) for _ in xs]
            direct,_=R.direct(xs,labels,ps);built,_,rejected=R.construct(xs,labels,ps)
            self.assertEqual(R.sorted_programs(direct),R.sorted_programs(built))
            for row in rejected:
                i,j=row['observations']
                self.assertEqual(L.evaluate(row['predicate'],xs[i]),L.evaluate(row['predicate'],xs[j]))
                self.assertNotEqual(labels[i],labels[j])

    def test_conflict_with_identical_observations(self):
        examples=[{'points':[(0,0),(1,1)],'label':0},{'points':[(0,1),(1,0)],'label':1}]
        self.assertEqual(R.vocabulary_conflict(examples)['scalar_values'],[2,2,2])

    def test_not_observational_deduplication(self):
        a=('count',);b=('span_r',)
        self.assertNotEqual(L.canon(a),L.canon(b))
        self.assertEqual(L.evaluate(a,(2,2,3)),L.evaluate(b,(2,2,3)))
        self.assertNotEqual(L.evaluate(a,(4,2,3)),L.evaluate(b,(4,2,3)))

    def test_constants_not_copy_prior(self):
        xs=[(1,1,1),(2,1,2)];ps=L.grammar(3)
        models,_,_=R.construct(xs,[7,4],ps)
        self.assertTrue(models)
        self.assertTrue(all({m['false'],m['true']}=={4,7} for m in models))

    def test_absent_branch_keeps_all_consistent_assignments(self):
        p=L.canon(('eq',('lit',0),('lit',0)))
        models,_,_=R.construct([(1,1,1)],[6],[p])
        self.assertEqual(models[0]['true'],6)
        self.assertEqual(R.direct([(1,1,1)],[6],[p])[0],models)

    def test_teacher_interpretations(self):
        rect=[(r,c) for r in range(3) for c in range(3)]
        frame=[p for p in rect if p!=(1,1)]
        self.assertTrue(R.teacher_rectangle(rect));self.assertFalse(R.teacher_rectangle(frame))
        self.assertTrue(R.teacher_frame(frame));self.assertFalse(R.teacher_frame(rect))
        self.assertTrue(R.teacher_rectangle([(0,0)]))

    def test_expanded_program_cost(self):
        xs=[(1,1,1),(2,1,2)]
        models,_,_=R.construct(xs,[0,1],L.grammar(3))
        for m in models:self.assertEqual(m['expanded_program_cost'],L.cost(m['predicate'])+3)

    def test_coordinate_translation(self):
        ps=[(0,0),(0,1),(1,0)]
        shifted=[(r-100,c+57) for r,c in ps]
        self.assertEqual(L.observations(ps),L.observations(shifted))
        self.assertEqual(R.teacher_rectangle(ps),R.teacher_rectangle(shifted))

if __name__=='__main__':unittest.main()
