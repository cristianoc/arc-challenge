"""Reproduce 028 without network access; uses the retained 027 input generator."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=HERE.parent/'evidence/028-generality-reduction/scientific-hashes.json')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    def call(*args):subprocess.run([sys.executable,*map(str,args)],check=True)
    call(HERE/'test_run.py')
    call(HERE.parent/'027-constructed-spatial-sets/run.py','prepare','--out',a.out/'input')
    call(HERE/'run.py','predict','--out',a.out/'run')
    call(HERE/'run.py','score','--predictions',a.out/'run/predictions.json',
         '--query-inputs',a.out/'input/recognition-inputs.json','--out',a.out/'run')
    call(HERE/'audit.py','--predictions',a.out/'run/predictions.json',
         '--query-inputs',a.out/'input/recognition-inputs.json',
         '--scores',a.out/'run/scores.json','--out',a.out/'run/audit.json')
    expected=json.loads(a.reference.read_text())
    for name,value in expected.items():
        actual=sha(a.out/'run'/name)
        if actual!=value:raise AssertionError((name,actual,value))
    print('All scientific hashes match.')

if __name__=='__main__':main()
