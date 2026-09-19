"""Pre-run controls for the data design; no corpus or scored query file."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('design022',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def independent(width, colours, height, row, farther, kind):
    x=[[0]*width for _ in range(height)]
    x[row][0], x[row][-1]=colours
    y=[r[:] for r in x]
    for r in range(height):
        for c in range(width):
            if r!=row:
                y[r][c]=6 if kind=='literal_background' else 0
            elif 0<c<width-1:
                if c==width-1-c:y[r][c]=5
                else:
                    right=(width-1-c<c)
                    if farther:right=not right
                    y[r][c]=(6,8)[int(right)] if kind=='literal_fill' else colours[int(right)]
    return dict(input=x,output=y)


class Controls(unittest.TestCase):
    def test_teacher_against_independent_loop(self):
        for kind in m.KINDS:
            for farther in (False,True):
                for height in (3,5,7):
                    for row in range(1,height-1):
                        for width in (3,7,9):
                            self.assertEqual(m.example(width,(1,4),height,row,farther,kind),
                                             independent(width,(1,4),height,row,farther,kind))

    def test_matches_old_teacher_when_centred(self):
        for kind in m.KINDS:
            for farther in (False,True):
                self.assertEqual(m.example(7,(1,2),5,2,farther,kind),m.e21.teacher(7,(1,2),5,farther,kind))

    def test_budget_and_projection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m.prepare(root)
            ps=json.loads((root/'problems.json').read_text())
            meta=json.loads((root/'metadata.json').read_text())
            self.assertEqual(len(ps),24)
            for p in ps:
                self.assertEqual(len(p['train']),2)
                self.assertEqual(sum(len(e['output'])*len(e['output'][0]) for e in p['train']),90)
                self.assertEqual(meta[p['id']]['labelled_output_cells'],90)
                self.assertNotIn('output',p['query_inputs'])
                self.assertEqual(len(p['query_inputs']),50)
                for e in p['train']:
                    self.assertEqual(sum(v!=0 for row in e['input'] for v in row),2)

    def test_position_intervention_preserves_palette_and_colour_counts(self):
        for kind in m.KINDS:
            a=m.example(11,(3,4),5,2,False,kind)
            b=m.example(11,(3,4),5,1,False,kind)
            for field in ('input','output'):
                self.assertEqual(sorted(v for r in a[field] for v in r),sorted(v for r in b[field] for v in r))

    def test_queries_fully_cross_factors(self):
        qs=[q for q in m.queries() if q['bank']=='crossed']
        self.assertEqual(len(qs),36)
        self.assertEqual(len({(q['width'],q['height'],tuple(q['palette']),q['active_row']) for q in qs}),36)
        self.assertTrue(all(q['width'] not in (7,11) for q in qs))
        for h in (5,7):
            for colours in ((6,8),(7,9)):
                self.assertEqual(sum(q['height']==h and tuple(q['palette'])==colours for q in qs),9)

    def test_invalid_domains(self):
        for width, colours, height, row in ((2,(1,2),5,2),(7,(0,2),5,2),(7,(1,1),5,2),(7,(1,2),5,0)):
            with self.assertRaises(ValueError):m.example(width,colours,height,row,False,'copy')

    def test_input_labels_only_subsets(self):
        pairs=[m.example(7,(1,2),5,2,False,'copy'),m.example(11,(3,4),5,1,False,'copy')]
        alt=copy.deepcopy(pairs);alt[1]['output']=[[9]*11 for _ in range(5)]
        a,b=m.e21.DefaultLearner(pairs),m.e21.DefaultLearner(alt)
        for fs in ((24,27,28),(4,24)):
            self.assertEqual(a.table((0,),fs,'guarded_order'),b.table((0,),fs,'guarded_order'))

    def test_unknown_context_is_not_defaulted(self):
        table,_=m.e21.compress({(0,):1<<10})
        self.assertNotIn((1,),table)

if __name__=='__main__':unittest.main()
