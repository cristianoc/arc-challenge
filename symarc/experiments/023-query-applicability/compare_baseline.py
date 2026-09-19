"""Compare every retained original full-data model and baseline outer decision.

Streams the large historical JSON/gzip arrays using the standard library.
This checks replication, not independence of the inherited search implementation.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def items(path):
    decoder=json.JSONDecoder()
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        buf='';started=False;eof=False
        while True:
            if not eof and len(buf)<1048576:
                part=f.read(1048576);eof=not part;buf+=part
            buf=buf.lstrip()
            if not started:
                if not buf and not eof:continue
                if not buf.startswith('['):raise ValueError('Expected JSON array')
                buf=buf[1:];started=True
            buf=buf.lstrip()
            if buf.startswith(','):buf=buf[1:].lstrip()
            if buf.startswith(']'):return
            try:
                obj,end=decoder.raw_decode(buf)
            except json.JSONDecodeError:
                if eof:raise
                part=f.read(1048576);eof=not part;buf+=part;continue
            yield obj;buf=buf[end:]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--historical',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();new={(r['split'],r['id']):r for r in items(a.predictions)};counts=Counter();seen=set()
    for old in items(a.historical):
        key=(old['split'],old['id']);row=new[key];seen.add(key)
        assert row['eligible']==old['eligible']
        oldmodels=[m for f in old['families'].values() for m in f['models']]
        assert len(oldmodels)==len(row['models'])
        lookup={(m['arm'],tuple(m['features'])):m for m in row['models']}
        for m in oldmodels:
            n=lookup[(m['arm'],tuple(m['features']))]
            for field in ('names','cost','cv_exact','cv_fraction','predictions'):
                assert m[field]==n[field],(key,m['features'],field)
            counts['full_candidate_models']+=1
        baseline=row['policies']['baseline'];original=old['policies']['union']
        assert baseline['selected']==original['selected'] and baseline['predictions']==original['predictions']
        assert len(original['outer'])==len(row['outer'])
        for f,g in zip(original['outer'],row['outer'],strict=True):
            assert f['excluded']==g['excluded'] and f['selected']==g['policies']['baseline']['selected']
            assert f['scores']==g['scores']['baseline']
            counts['baseline_outer_folds']+=1
        counts['tasks']+=1
    assert seen==set(new)
    a.out.write_text(json.dumps(dict(counts),indent=2)+'\n');print(dict(counts))

if __name__=='__main__':main()
