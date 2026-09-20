"""Verify every recorded palette family by independent exhaustive enumeration."""
import argparse,itertools,json
from pathlib import Path

def audit(path):
    rows=json.loads(path.read_text());tables=0;subsets=0;narrowed=0;changed_tables=0
    for row in rows:
        for cert in row.get('palette_certificates',[]):
            constraints=cert['constraints'];observed=cert['palettes'];universe=0
            for c in constraints:universe|=c
            bits=[1<<i for i in range(19) if universe&(1<<i)]
            if not constraints:expected=[0]
            elif 0 in constraints:expected=[]
            else:
                expected=[]
                for size in range(len(bits)+1):
                    for choice in itertools.combinations(bits,size):
                        candidate=sum(choice);subsets+=1
                        if all(candidate&c for c in constraints):expected.append(candidate)
                    if expected:break
            assert sorted(expected)==observed,cert
            allowed=0
            for p in observed:allowed|=p
            differences=sum(c!=(c&allowed) for c in constraints)
            narrowed+=differences;changed_tables+=bool(differences);tables+=1
    return dict(palettes_independently_verified=tables,candidate_subsets_checked=subsets,
                narrowed_action_sets=narrowed,tables_narrowed=changed_tables)
if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit(a.predictions);a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)
