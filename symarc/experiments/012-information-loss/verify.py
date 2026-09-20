"""Independent checks of the three stored witness mechanisms (no new labels)."""
import json
from collections import Counter
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'evidence/012-information-loss'
r=json.loads((p/'results.json').read_text())
w=r['7b5033c1']['witness'];a,b=w['input_a'],w['input_b']
assert Counter(sum(a,[]))==Counter(sum(b,[]))
assert list(dict.fromkeys(v for v in sum(a,[]) if v))==list(dict.fromkeys(v for v in sum(b,[]) if v))
assert w['original_a']==w['original_b'] and w['repair_a']!=w['repair_b']
assert [v for v in a[1] if v]==sum(w['repair_a'],[])
assert [v for v in b[1] if v]==sum(w['repair_b'],[])
w=r['8f215267']['witness'];a,b=w['input_a'],w['input_b'];color,r0,r1,c0,c1=w['target_frame'];rr,cc,v=w['added_cell']
assert sum(x!=y for ra,rb in zip(a,b) for x,y in zip(ra,rb))==1
assert a[r0:r1+1]==b[r0:r1+1] and not r0<=rr<=r1 and cc>c1 and v==color
bg=Counter(sum(a,[])).most_common(1)[0][0]
assert all(a[x][y]==bg for x,y in [(rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)] if 0<=x<len(a) and 0<=y<len(a[0]))
assert w['original_a']==w['original_b']
assert sum(x!=y for ra,rb in zip(w['repair_a'],w['repair_b']) for x,y in zip(ra,rb))==1
w=r['97d7923e']['witness'];c,d=w['swapped_columns']
def swap(g):
    out=[row[:] for row in g]
    for row in out: row[c],row[d]=row[d],row[c]
    return out
assert swap(w['input_a'])==w['input_b']
assert swap(w['repair_a'])==w['repair_b']
assert swap(w['original_a'])!=w['original_b']
assert w['original_b']!=w['repair_b']
assert all(x['witness']['ground_truth'] is None for x in r.values())
print('PASS: histogram collision, local-patch collision, positional sensitivity; no synthetic ground-truth labels')
