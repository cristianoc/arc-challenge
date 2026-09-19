"""Combine saved predictions without loading any query answer file."""
import argparse
import importlib.util
import json
from pathlib import Path
spec=importlib.util.spec_from_file_location('transfer019',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--stable',type=Path,required=True)
    p.add_argument('--relational',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--policy',choices=('nofit_fallback','complete_override','strict_override'),default='nofit_fallback')
    a=p.parse_args()
    stable={m.key(r):r for r in map(json.loads,a.stable.read_text().splitlines())}
    relational=json.loads(a.relational.read_text())
    if set(stable)!={m.key(r) for r in relational}:raise ValueError('Task sets differ')
    result=[]
    for r in relational:
        choice=next((grids,source) for name,grids,source in m.choices(stable[m.key(r)],r) if name==a.policy)
        result.append({'id':r['id'],'split':r['split'],'source':choice[1],'predictions':choice[0]})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in result))
if __name__=='__main__':main()
