"""Reproduce ONLY the previously inspected e88171ec development fixture.

Predictions are persisted before its known test answer is read. This is not a
blind result: the rectangle hypothesis was supplied during experiment 014.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
spec=importlib.util.spec_from_file_location('roles015',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--problems',type=Path,required=True)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    problem=next(p for p in json.loads(a.problems.read_text()) if p['id']=='e88171ec')
    r=m.investigate((problem,False))
    m.b.write_json(a.out/'development-predictions.json',r)
    frozen=hashlib.sha256((a.out/'development-predictions.json').read_bytes()).hexdigest()
    m.b.write_json(a.out/'development-freeze.json',{'predictions_sha256':frozen,'fixture':'e88171ec','test_answer_previously_inspected':True})
    raw=a.task.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='3cfcc30c08268f7ca52c83f23a0c1de674ffd8df147312fd334ab51445345127'
    y=json.loads(raw)['test'][0]['output']
    results={name:{'selected':q['selected'],'exact':q['predictions'][0]==y,
             'complete':q['predictions'][0] is not None and all(v!=-1 for row in q['predictions'][0] for v in row)}
             for name,q in r['policies'].items()}
    m.b.write_json(a.out/'development-results.json',{'fixture':'e88171ec','test_answer_previously_inspected':True,
                                                   'predictions_sha256':frozen,'policies':results})
    print(json.dumps(results['copy_all_cv'],indent=2))
if __name__=='__main__':main()
