"""Synthetic controls; no corpus source or query file is read here."""
import itertools
import random
import unittest
import language as L
import run as R


class Tests(unittest.TestCase):
    def test_atom_truth(self):
        for k,name in enumerate(L.ATOMS):
            for v in range(16):self.assertEqual(bool(L.truth((name,))&(1<<v)),bool(v&(1<<k)))
    def test_operator_truth(self):
        for a,b in itertools.product(L.ATOMS,repeat=2):
            x,y=L.truth((a,)),L.truth((b,))
            self.assertEqual(L.truth(('and',(a,),(b,))),x&y)
            self.assertEqual(L.truth(('or',(a,),(b,))),x|y)
            self.assertEqual(L.truth(('not',(a,))),65535^x)
    def test_aliases_charge_expanded_body(self):
        body=('or',('rmin',),('cmax',));aliases={'A':body,'B':('alias','A')}
        self.assertEqual(L.expand(('alias','B'),aliases),L.expand(body))
        self.assertEqual(L.node_count(L.expand(('alias','B'),aliases)),L.cost(body))
        with self.assertRaises(ValueError):L.expand(('alias','x'),{'x':('alias','x')})
    def test_unlabelled_enumerator_invariant(self):
        a,stats=L.grammar();b,other=L.grammar(reverse=True)
        self.assertEqual(a,b);self.assertEqual(stats,other)
        self.assertEqual(len({c['truth'] for c in a}),len(a))
    def test_profile_not_scalar(self):
        x=[[0,0],[0,1],[1,0]];y=[[0,0],[0,1],[1,1]]
        self.assertEqual(len(x),len(y));self.assertEqual(L.bounds(x),L.bounds(y))
        self.assertNotEqual(L.profile(x),L.profile(y))
    def test_pointwise_set_semantics(self):
        s=[[0,0],[1,2]];phi=('or',('rmin',),('cmax',));sig=L.truth(phi)
        expected={(0,0),(0,1),(0,2),(1,2)}
        self.assertEqual(L.generate(sig,s),expected)
        self.assertTrue(L.matches(sig,L.profile(expected)))
    def test_generic_new_points(self):
        s=[[0,0],[1,2]];sig=L.truth(('rmin',))
        self.assertIn((0,1),L.target('generated',sig,s))
        self.assertNotIn((0,1),set(map(tuple,s)))
        self.assertEqual(L.target('xor',sig,s),L.target('union',sig,s)-L.target('intersection',sig,s))
    def test_degenerate_domain(self):
        self.assertEqual(L.roles([[7,-3]]),[(7,-3,15)])
        self.assertEqual(L.profile([[7,-3]]),(1<<15,0))
        with self.assertRaises(ValueError):L.normalize([])
        with self.assertRaises(ValueError):L.normalize([[True,1]])
    def test_input_set_invariance(self):
        a=[[0,0],[1,2]];b=list(reversed(a))+a
        self.assertEqual(L.profile(a),L.profile(b))
        for sig in (0,123,65535):self.assertEqual(L.generate(sig,a),L.generate(sig,b))
    def test_translation(self):
        s=[[0,0],[1,2],[2,2]];t=[[r-9,c+7] for r,c in s]
        self.assertEqual(L.profile(s),L.profile(t))
        for sig in (0,43525,65535):
            self.assertEqual(L.generate(sig,t),{(r-9,c+7) for r,c in L.generate(sig,s)})
    def test_synthesis_reference(self):
        rng=random.Random(2702);candidates=L.grammar(4)[0]
        for _ in range(40):
            examples=[{'points':R.shape(2,2,m),'label':rng.randrange(2)} for m in rng.sample(range(1,16),4)]
            self.assertEqual(R.synthesize(examples,candidates)[0],R.synthesize(examples,candidates,True)[0])
    def test_mixed_role_refutes_every_template(self):
        s=[[0,0],[0,3],[3,0],[3,3],[1,1]]
        yes,no=L.profile(s);self.assertTrue(yes&no)
        for sig in range(65536):self.assertFalse(L.matches(sig,(yes,no)))
    def test_total_constant_targets(self):
        library={'models':[{'truth':0,'expr':['bool',False]}]}
        train=[{'input':[[0,0],[1,1]],'output':[]}]
        ms,_=R.target_models(library,train)
        self.assertTrue(ms);self.assertTrue(all(L.target(m['operation'],m['truth'],train[0]['input'])==set() for m in ms))


if __name__=='__main__':unittest.main()
