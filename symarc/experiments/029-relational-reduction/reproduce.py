"""Rebuild the inherited family, all targets, choices and independent audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
EXPECTED={
 'baseline/predictions.json':'5c0f09731eb2ba6a190b760f9827832433039d84b2d10572247a1aa631ff7108',
 'result/predictions.json':'4ea7649492f5272a6c8bcee4daffb72ba928f4d9dc27e47dacbb49aae6d711f5',
 'result/summary.json':'9385a76aee14c85bd97cc702b275c65f012538b1ef0950955c2f7c0315016a81',
 'result/audit.json':'5ab410ad5db4881daccb63ff9a755a20871fd67a264ebe7c53bbdbe82c915769'}


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result/predictions.json').exists():
        raise ValueError('Use a new output directory for an independent reproduction')
    def call(file,*args):subprocess.run([sys.executable,str(file),*map(str,args)],check=True)
    call(HERE/'test_run.py')
    call(HERE.parent/'028-generality-reduction/run.py','predict','--out',out/'baseline')
    call(HERE/'run.py','predict','--out',out/'result')
    call(HERE/'run.py','score','--out',out/'result')
    call(HERE/'audit.py','--predictions',out/'result/predictions.json',
         '--baseline',out/'baseline/predictions.json','--out',out/'result/audit.json')
    observed={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in EXPECTED}
    if observed!=EXPECTED:raise AssertionError({'expected':EXPECTED,'observed':observed})
    (out/'verified.json').write_text(json.dumps(observed,indent=2)+'\n')
    print('All four semantic output hashes reproduced exactly.')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
