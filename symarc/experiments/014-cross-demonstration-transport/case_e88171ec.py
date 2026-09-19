"""Post-hoc interpretation of one automatically flagged locality collision.

The rectangle hypothesis was manually supplied after inspecting the test.
This is not part of the frozen learner and is not a blind repair result.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def largest_zero_rectangles(grid):
    h, w = len(grid), len(grid[0])
    best = 0
    rectangles = []
    for top in range(h):
        allowed = [True]*w
        for bottom in range(top, h):
            allowed = [a and grid[bottom][c] == 0 for c, a in enumerate(allowed)]
            left = 0
            while left < w:
                if not allowed[left]:
                    left += 1
                    continue
                right = left
                while right+1 < w and allowed[right+1]:
                    right += 1
                area = (bottom-top+1)*(right-left+1)
                if area > best:
                    best, rectangles = area, []
                if area == best:
                    rectangles.append([top, bottom, left, right])
                left = right+1
    return best, rectangles


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--task', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    raw=a.task.read_bytes(); task=json.loads(raw)
    colours={y for e in task['train'] for ri,ro in zip(e['input'],e['output'])
             for x,y in zip(ri,ro) if x != y}
    assert len(colours)==1
    fill=next(iter(colours))
    result={'retrospective':True,'structural_hypothesis':'fill strict interior of unique maximum-area all-zero rectangle; copy elsewhere',
            'fill_colour_inferred_from_training':fill,'source_sha256':hashlib.sha256(raw).hexdigest(),'examples':[]}
    for split in ('train','test'):
        for i,e in enumerate(task[split]):
            area,rs=largest_zero_rectangles(e['input'])
            assert len(rs)==1, 'No tie rule is supplied'
            top,bottom,left,right=rs[0]
            pred=copy.deepcopy(e['input'])
            for r in range(top+1,bottom):
                for c in range(left+1,right):
                    pred[r][c]=fill
            result['examples'].append(dict(split=split,index=i,rectangle=rs[0],area=area,
                unique=True,exact=pred==e['output']))
    assert all(e['exact'] for e in result['examples'])
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
