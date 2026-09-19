#!/usr/bin/env python3
"""Illustrate a scored witness from an existing run; performs no search."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

root=Path(__file__).resolve().parents[2]
run=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve()
task=json.loads((root.parent/'data/training/88a62173.json').read_text())
pool=[json.loads(s) for s in (run/'pools/88a62173-objects.jsonl').read_text().splitlines()]
pred=pool[0]['predictions'][0]
pred=np.array(pred['cells']).reshape(pred['h'],pred['w']).tolist()
assert pred==task['test'][0]['output']
assert all(p['predictions']==pool[0]['predictions'] for p in pool)
colours=['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25']
fig,axes=plt.subplots(2,3,figsize=(10,6.4),facecolor='white')
fig.suptitle('An object explanation: keep the shape that occurs only once',fontsize=15,y=.98)
fig.text(.5,.924,'88a62173 · Same-colour connected objects, including diagonal neighbours',ha='center',fontsize=11,color='#444444')
examples=[task['train'][0],task['train'][-1],{'input':task['test'][0]['input'],'output':pred}]
for col,(example,title) in enumerate(zip(examples,['Training example 1','Reserved training example','Test example'])):
    for row,key in enumerate(['input','output']):
        ax=axes[row,col];g=np.array(example[key]);h,w=g.shape
        left=(5-w)/2;top=(5-h)/2
        ax.imshow(g,cmap=ListedColormap(colours),vmin=0,vmax=9,interpolation='none',extent=(left,left+w,top+h,top))
        for x in range(w+1):ax.plot([left+x,left+x],[top,top+h],color='#666666',lw=.5)
        for y in range(h+1):ax.plot([left,left+w],[top+y,top+y],color='#666666',lw=.5)
        ax.set_xlim(-.15,5.15);ax.set_ylim(5.15,-.15);ax.set_aspect('equal');ax.axis('off')
        ax.set_title(title+' — input' if row==0 else ('Correct prediction' if col==2 else 'Target output'),fontsize=11)
fig.text(.5,.065,f'After fitting all three training examples, {len(pool)} object programs agree on the correct test answer.\n{len(set(p["segment"] for p in pool))} named segmentation interpretations remain: answer certainty does not require a unique interpretation.',ha='center',fontsize=10,color='#333333')
fig.subplots_adjust(top=.85,bottom=.13,hspace=.22,wspace=.15)
out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=160,facecolor='white')
