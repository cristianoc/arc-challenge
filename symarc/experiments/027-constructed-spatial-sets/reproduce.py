"""Reproduce 027 from source and verify its scientific outputs, not wall times."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SCIENTIFIC=('library.json','construction.json','predictions.json','scores.json','negative-certificate.json','teaching-certificate.json','audit.json')


def run(out,reference=None):
    def call(script,*args):subprocess.run([sys.executable,str(HERE/script),*map(str,args)],cwd=HERE,check=True)
    call('test_run.py')
    call('run.py','prepare','--out',out/'input')
    call('run.py','construct','--source',out/'input/source.json','--sparse',out/'input/sparse.json','--out',out/'run')
    call('run.py','predict','--inputs',out/'input','--out',out/'run')
    call('run.py','score','--inputs',out/'input','--out',out/'run')
    call('audit.py','--input',out/'input','--run',out/'run')
    hashes={name:hashlib.sha256((out/'run'/name).read_bytes()).hexdigest() for name in SCIENTIFIC}
    if reference:
        expected=json.loads(reference.read_text())
        assert hashes==expected,(hashes,expected)
    (out/'scientific-hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    print('All scientific outputs verified.' if reference else 'Scientific hashes recorded.')

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args();run(a.out.resolve(),a.reference.resolve() if a.reference else None)
