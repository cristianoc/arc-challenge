"""Extract the finite quotient universally respected by a saved program family.

This reads predictions only, not an oracle or an answer file. It is a post-run
representation of already-recorded consequences, not a selection policy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def extract(data: dict, version: int) -> dict:
    extents=[int(m['extent'],16) for m in data['models']]
    if not version or version & ~((1<<len(extents))-1):
        raise ValueError('Specify a nonempty subfamily of the saved candidates')
    hs=[i for i in range(len(extents)) if version&(1<<i)]
    signatures={};classes=[];mapping=[]
    for x in range(len(data['domain'])):
        column=tuple((extents[h]>>x)&1 for h in hs)
        if column not in signatures:
            signatures[column]=len(classes)
            classes.append({'representative':x,'members':[],
                            'possible_labels':sorted(set(column)),
                            'column':list(column)})
        index=signatures[column];mapping.append(index);classes[index]['members'].append(x)
    selectors={str(h):[(extents[h]>>c['representative'])&1 for c in classes] for h in hs}
    for h in hs:
        assert all(selectors[str(h)][mapping[x]]==((extents[h]>>x)&1) for x in range(len(mapping)))
    assert len({tuple(c['column']) for c in classes})==len(classes)
    return {'version':hex(version),'hypotheses':hs,'input_to_class':mapping,
            'classes':classes,'fitting_selectors':selectors}


def compare(data: dict, before: int, after: int) -> dict:
    if after & ~before:raise ValueError('The after family must be a subset')
    old,new=extract(data,before),extract(data,after)
    transport=[]
    for c in old['classes']:
        target={new['input_to_class'][x] for x in c['members']}
        assert len(target)==1
        transport.append(next(iter(target)))
    assert [transport[i] for i in old['input_to_class']]==new['input_to_class']
    return {'before':old,'after':new,'old_class_to_new_class':transport,
            'scope':'Exact finite observational quotient, conditional on this candidate family'}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--before',required=True);p.add_argument('--after',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.predictions.read_text())
    result=compare(data,int(a.before,0),int(a.after,0))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n')
