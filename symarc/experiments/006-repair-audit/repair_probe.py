"""Retrospective witnesses; not a SymArc solver or blind synthesis experiment."""
import json,copy
from collections import Counter
from pathlib import Path
import os
CORPUS_PATH = Path(os.environ.get('ARC_REPAIR_CORPUS', Path(__file__).resolve().parents[2] / 'out/experiments/006-repair-audit/corpus.json'))
C=json.loads(CORPUS_PATH.read_text())
def ns(id):
 n={};exec(C[id]['source'],n);return n

def barrier(g):
 h,w=len(g),len(g[0])
 def vertical_run(c):
  best=run=0;last=None
  for row in g:
   v=row[c];run=run+1 if v and v==last else int(v!=0);last=v;best=max(best,run)
  return best
 scores=[vertical_run(c) for c in range(w)];b=max(range(w),key=lambda c:scores[c]);assert scores.count(scores[b])==1
 left=sum(v!=0 for row in g for v in row[:b]);right=sum(v!=0 for row in g for v in row[b+1:])
 if right>left:return [row[::-1] for row in barrier([row[::-1] for row in g])]
 out=copy.deepcopy(g)
 for r,row in enumerate(g):
  seg=[];i=0
  while i<b:
   if row[i]==0:i+=1;continue
   j=i+1
   while j<b and row[j]==row[i]:j+=1
   seg.append((row[i],j-i));i=j
  for color,length in reversed(seg):
   for c in range(b+1,w,length):
    if out[r][c]==0:out[r][c]=color
 return out

def path(g):
 bg=Counter(v for row in g for v in row).most_common(1)[0][0];cells={(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v!=bg}
 def adj(p):
  r,c=p;return [(r+dr,c+dc) for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)] if (r+dr,c+dc) in cells]
 ends=sorted(p for p in cells if len(adj(p))==1);assert len(ends)==2
 cur=ends[0];prev=None;out=[]
 while cur is not None:
  r,c=cur;out.append([g[r][c]]);n=[q for q in adj(cur) if q!=prev];prev,cur=cur,n[0] if n else None
 assert len(out)==len(cells)
 return out

def stripe(g):
 h,w=len(g),len(g[0]);yellow=[(r,c) for r in range(h) for c in range(w) if g[r][c]==4]
 if len({r for r,c in yellow})>1:return list(map(list,zip(*stripe(list(map(list,zip(*g)))))))
 sr=yellow[0][0];cols={c for r,c in yellow};bg=Counter(v for row in g for v in row).most_common(1)[0][0];out=copy.deepcopy(g)
 for r in range(h):
  phase=abs(r-sr)%6
  for c in range(w):
   if c in cols:out[r][c]=3 if phase==4 else 4 if phase in (0,2) else bg
   elif phase==4 and g[r][c]!=bg:out[r][c]=3
 return out

def periodic(g):
 n=ns('135a2760');T=lambda x:list(map(list,zip(*x)))
 # Detect frame orientation from counts of non-background runs in central lines.
 bg=g[0][0];h,w=len(g),len(g[0]);border=next(v for row in g for v in row if v!=bg)
 # Frame interiors: components excluding global background and frame colour aren't enough;
 # select orientation by longest straight runs of frame colour.
 hr=max((len(run) for row in g for run in ''.join('1' if v==border else '0' for v in row).split('0')),default=0)
 vr=max((len(run) for row in T(g) for run in ''.join('1' if v==border else '0' for v in row).split('0')),default=0)
 return n['p'](g) if hr>=vr else T(n['p'](T(g)))

REPAIRS={'1ae2feb7':barrier,'7b5033c1':path,'221dfab4':stripe,'135a2760':periodic}

def components(g, diagonal=False, multicolor=False):
 cells={(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v!=0};out=[]
 dirs=[(dr,dc) for dr in [-1,0,1] for dc in [-1,0,1] if (dr or dc) and (diagonal or abs(dr)+abs(dc)==1)]
 while cells:
  start=min(cells);cells.remove(start);stack=[start];comp=[start]
  while stack:
   r,c=stack.pop()
   for dr,dc in dirs:
    q=(r+dr,c+dc)
    if q in cells and (multicolor or g[q[0]][q[1]]==g[r][c]):cells.remove(q);comp.append(q);stack.append(q)
  out.append(comp)
 return out

def legend_partner(g):
 comps=components(g,multicolor=True);legend=[co for co in comps if len({g[r][c] for r,c in co})>1];assert len(legend)==1
 co=legend[0];r0,r1=min(r for r,c in co),max(r for r,c in co);c0,c1=min(c for r,c in co),max(c for r,c in co)
 pairs=[(g[r0][c],g[r1][c]) for c in range(c0,c1+1)] if r1-r0==1 else [(g[r][c0],g[r][c1]) for r in range(r0,r1+1)]
 pairs=[(b,a) for a,b in pairs] if (r1==len(g)-1 and r1-r0==1) or (c1==len(g[0])-1 and c1-c0==1 and r1-r0!=1) else pairs;mapping=dict(pairs);n=ns('dbff022c');out=copy.deepcopy(g)
 for co in n['enumerateZeroCavities'](g):
  if not co['touches_border'] and co['color'] in mapping:
   for r,c in co['cells']:out[r][c]=mapping[co['color']]
 return out

def holes_legend(g):
 n=ns('e3721c99');comps=components(g,diagonal=True);mapping={};targets=[]
 for co in comps:
  color=g[co[0][0]][co[0][1]];r0,r1=min(r for r,c in co),max(r for r,c in co);c0,c1=min(c for r,c in co),max(c for r,c in co)
  mask=[[int((r,c) in co) for c in range(c0,c1+1)] for r in range(r0,r1+1)];holes=n['_count_internal_holes'](mask)
  if color==5:targets.append((co,holes))
  elif r1-r0>=2 and c1-c0>=2:mapping[holes]=color
 out=copy.deepcopy(g)
 for co,k in targets:
  for r,c in co:out[r][c]=mapping.get(k,0)
 return out
REPAIRS.update({'dbff022c':legend_partner,'e3721c99':holes_legend})

def global_counts(g):
 n=ns('8f215267');frames=n['extractFrames'](g);bg=n['_most_common_color'](g);right=max(f[-1] for f in frames)
 gg=[[0 if c<=right or v==bg else v for c,v in enumerate(row)] for row in g]
 counts=Counter(g[co[0][0]][co[0][1]] for co in components(gg));out=copy.deepcopy(g)
 for fr in frames:out=n['clearAndPaintStripes'](out,fr,counts[fr[0]])
 return n['clearNoise'](out,frames)
REPAIRS['8f215267']=global_counts

def ranked_bars(g):
 n=ns('97d7923e');cols=n['parseColumnRuns'](g);ranks={runs[0].color:runs[0].length for runs in cols.values() if runs[0].color!=0};groups={}
 for c,runs in cols.items():
  p=n['detectCapPattern'](runs)
  if p:groups.setdefault(p[0].color,[]).append((c,p))
 out=copy.deepcopy(g)
 for color,items in groups.items():
  items.sort(key=lambda it:it[1][1].length,reverse=True);c,(_,mid,_)=items[ranks[color]-1];out=n['paintColumnRun'](out,c,mid,color)
 return out
REPAIRS['97d7923e']=ranked_bars

def symmetry(g):
 n=ns('0934a4d8');h,w=len(g),len(g[0]);r0,r1,c0,c1=n['bbox'](g)
 def axis_sums(axis):
  size=h if axis==0 else w;candidates=[]
  for s in range(size//2,3*size//2):
   good=bad=0
   for r in range(h):
    for c in range(w):
     rr,cc=(s-r,c) if axis==0 else (r,s-c)
     if 0<=rr<h and 0<=cc<w and (rr,cc)!=(r,c) and g[r][c]!=8 and g[rr][cc]!=8:
      if g[r][c]==g[rr][cc]:good+=1
      else:bad+=1
   if good and bad==0:candidates.append((good,s))
  return max(candidates)[1] if candidates else None
 rs,cs=axis_sums(0),axis_sums(1);out=[]
 for r in range(r0,r1):
  row=[]
  for c in range(c0,c1):
   vals={g[rr][cc] for rr in [r,rs-r] if 0<=rr<h for cc in [c,cs-c] if 0<=cc<w and g[rr][cc]!=8};assert len(vals)==1;row.append(vals.pop())
  out.append(row)
 return out
# Partial repair: this function deliberately rejects unresolved symmetry orbits.

def periodic_full(g):
 n=ns('135a2760');T=lambda x:list(map(list,zip(*x)));bg=g[0][0];border=next(v for row in g for v in row if v!=bg)
 def longest(rows):return max((len(run) for row in rows for run in ''.join('1' if v==border else '0' for v in row).split('0')),default=0)
 trans=longest(T(g))>longest(g);a=T(g) if trans else g;out=[]
 for row in a:
  ix=[i for i,v in enumerate(row) if v==border]
  if len(ix)>=2 and any(v not in (bg,border) for v in row):
   lo,hi=min(ix)+1,max(ix);seg=row[lo:hi];pat,score=n['selectBestPattern'](n['enumeratePatterns'](seg));out.append(row[:lo]+[pat[i%len(pat)] for i in range(len(seg))]+row[hi:])
  else:out.append(row[:])
 return T(out) if trans else out
REPAIRS['135a2760']=periodic_full

def stamp(g):
 h,w=len(g),len(g[0]);rects=[]
 for r0 in range(h-3):
  for r1 in range(r0+3,h):
   for c0 in range(w-3):
    v=g[r0][c0]
    if not all(g[r][c0]==v for r in range(r0,r1+1)):continue
    for c1 in range(c0+3,w):
     if not all(g[r0][c]==g[r1][c]==v for c in range(c0,c1+1)):continue
     if not all(g[r][c1]==v for r in range(r0,r1+1)):continue
     rects.append((r0,r1,c0,c1))
 rects.sort(key=lambda q:(q[1]-q[0]+1)*(q[3]-q[2]+1),reverse=True)
 selected=[]
 for q in rects:
  if all(q[1]<p[0] or p[1]<q[0] or q[3]<p[2] or p[3]<q[2] for p in selected):selected.append(q)
 rects=selected
 assert len(rects)==2,rects
 panels=[]
 for r0,r1,c0,c1 in rects:
  panel=[row[c0:c1+1] for row in g[r0:r1+1]];bg=Counter(v for row in panel[1:-1] for v in row[1:-1]).most_common(1)[0][0];objs=[[0 if r in(0,len(panel)-1) or c in(0,len(row)-1) or v==bg else v for c,v in enumerate(row)] for r,row in enumerate(panel)];cos=components(objs,multicolor=True)
  panels.append((panel,cos))
 target=next((panel,cos) for panel,cos in panels if all(len(co)==1 for co in cos));source=next((panel,cos) for panel,cos in panels if any(len(co)>1 for co in cos));out=copy.deepcopy(target[0]);templates={}
 for co in source[1]:
  counts=Counter(source[0][r][c] for r,c in co);anchors=[(r,c) for r,c in co if counts[source[0][r][c]]==1];assert len(anchors)==1
  ar,ac=anchors[0];color=source[0][ar][ac];temp={(r-ar,c-ac):source[0][r][c] for r,c in co}
  if color in templates:assert templates[color]==temp
  templates[color]=temp
 for co in target[1]:
  ar,ac=co[0]
  for (dr,dc),v in templates[out[ar][ac]].items():out[ar+dr][ac+dc]=v
 return out
REPAIRS['a251c730']=stamp

def rotate_instructions(g):
 cells={(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v};groups=[]
 while cells:
  p=min(cells);cells.remove(p);co=[p];stack=[p]
  while stack:
   r,c=stack.pop()
   for dr in range(-2,3):
    for dc in range(-2,3):
     q=r+dr,c+dc
     if q in cells:cells.remove(q);co.append(q);stack.append(q)
  groups.append(co)
 co=max(groups,key=len);r0,r1=min(r for r,c in co),max(r for r,c in co);c0,c1=min(c for r,c in co),max(c for r,c in co)
 block=[row[c0:c1+1] for row in g[r0:r1+1]];assert len(block)==len(block[0]);size=len(block)
 commands=Counter(v for r,row in enumerate(g) for c,v in enumerate(row) if v and not(r0<=r<=r1 and c0<=c<=c1));out=[[0]*size for _ in range(size)]
 for color in {v for row in block for v in row}-{0}:
  mask=[[v==color for v in row] for row in block]
  for _ in range(commands[color]%4):mask=[list(row) for row in zip(*mask[::-1])]
  for r in range(size):
   for c in range(size):
    if mask[r][c]:assert out[r][c]==0;out[r][c]=color
 return out
REPAIRS['6ffbe589']=rotate_instructions
