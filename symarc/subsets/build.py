#!/usr/bin/env python3
"""Rebuild the stratified quick subset from full runs.

    python3 subsets/build.py out/full.txt

Writes subsets/quick.txt, subsets/full.txt and subsets/strata.json.
Strata (see README.md): A1 fitted edge cases (all included), A2 fitted,
B unfit with a test input inside the closure, C unfit with some candidate
generator rejected by functionality, D unfit with nothing rejected.
Sampling is deterministic (seed 7).
"""
import json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data')
LINE = re.compile(r'([0-9a-f]{8})\s+func\[(.*?)\] real\[(.*?)\] \|C\|=(\d+)(\+?) .*?progsD=(\d+) progs=(\d+) fitD=([\d.]+) fitC=([\d.]+) .*?sym=(\d+)/(\d+)\(ok (\d+)\) equiv=(\S+) det=(\S+)\s+(\S+)\s+:: (.*)$')
SIZES = {'A2_fitted': 13, 'B_unfit_symreach': 5, 'C_unfit_funcrejected': 9, 'D_unfit_nothing': 4}

def family(prog):
    p = prog.split(' ; ')[-1].split()[0]
    if p in ('rot90', 'rot180', 'rot270', 'flipH', 'flipV', 'transpose'): return 'geometric'
    if p.startswith(('hconcat', 'vconcat', 'mirror4', 'tile', 'scale')): return 'tiling'
    if p.startswith(('crop', 'leftHalf', 'rightHalf', 'topHalf', 'bottomHalf', 'dedup')): return 'crop'
    if p.startswith(('recolour', 'keepColour', 'removeColour', 'majority')): return 'colour'
    if p.startswith('split'): return 'split'
    return 'painting'

def load(path):
    rows = {}
    for line in open(path):
        m = LINE.match(line)
        if not m:
            continue
        tid = m.group(1)
        splits = [s for s in ('training', 'evaluation') if os.path.exists(os.path.join(DATA, s, tid + '.json'))]
        if len(splits) != 1:
            raise ValueError(f'{tid}: expected exactly one data split, got {splits}')
        split = splits[0]
        t = json.load(open(os.path.join(DATA, split, tid + '.json')))
        grids = [g for p in t['train'] for g in (p['input'], p['output'])] + [p['input'] for p in t['test']]
        pal = {c for g in grids for r in g for c in r if c}
        maxH = max(len(g) for g in grids); maxW = max(len(g[0]) for g in grids)
        cand = {'dihedral': 3, 'colours': (len(pal) + 1) * len(pal) // 2 if len(pal) < 9 else 36,
                'cyclic': 2, 'rows': maxH - 1, 'cols': maxW - 1}
        fc = dict((k, int(v)) for k, v in (x.split(':') for x in m.group(2).split())) if m.group(2) else {}
        rows[tid] = dict(split=split, fitted=m.group(8) == '1.00', status=m.group(15), prog=m.group(16),
                         sym=int(m.group(10)) > 0, detNO='NO' in m.group(14),
                         funcRejected=any(fc.get(k, 0) < v for k, v in cand.items()))
    return rows

def main():
    random.seed(7)
    R = {}
    for p in sys.argv[1:]:
        R.update(load(p))
    expected = {os.path.splitext(f)[0] for split in ('training', 'evaluation')
                for f in os.listdir(os.path.join(DATA, split)) if f.endswith('.json')}
    if set(R) != expected:
        raise ValueError(f'need a full run: missing {len(expected - set(R))} tasks; unexpected {len(set(R) - expected)}')
    # Canonical order makes sampling independent of how full runs are split into files.
    R = dict(sorted(R.items(), key=lambda item: (item[1]['split'] != 'training', item[0])))
    strata = {}
    for tid, r in R.items():
        if r['fitted']:
            s = 'A1_fitted_edge' if (r['status'] == 'fit-only' or r['detNO']) else 'A2_fitted'
        elif r['sym']:
            s = 'B_unfit_symreach'
        elif r['funcRejected']:
            s = 'C_unfit_funcrejected'
        else:
            s = 'D_unfit_nothing'
        strata.setdefault(s, []).append(tid)
    sample = {'A1_fitted_edge': sorted(strata.get('A1_fitted_edge', []))}
    # A2: cover each (program family, split) cell, round-robin
    cells = {}
    for t in strata['A2_fitted']:
        cells.setdefault((family(R[t]['prog']), R[t]['split']), []).append(t)
    for l in cells.values():
        random.shuffle(l)
    pick = []
    for rnd in range(4):
        for k, l in sorted(cells.items()):
            if len(l) > rnd and len(pick) < SIZES['A2_fitted'] and l[rnd] not in pick:
                pick.append(l[rnd])
    sample['A2_fitted'] = sorted(pick)
    for s in ('B_unfit_symreach', 'C_unfit_funcrejected', 'D_unfit_nothing'):
        l = sorted(strata.get(s, [])); random.shuffle(l); sample[s] = sorted(l[:SIZES[s]])
    with open(os.path.join(HERE, 'quick.txt'), 'w') as f:
        for s in sorted(sample):
            for t in sample[s]:
                f.write(f"{t} {R[t]['split']} {s}\n")
    with open(os.path.join(HERE, 'full.txt'), 'w') as f:
        for t in sorted(R):
            f.write(f"{t} {R[t]['split']}\n")
    json.dump({s: {'size': len(strata[s]), 'sample': len(sample.get(s, [])), 'members': sorted(strata[s])}
               for s in sorted(strata)}, open(os.path.join(HERE, 'strata.json'), 'w'), indent=1)
    print({s: (len(strata[s]), len(sample.get(s, []))) for s in sorted(strata)})

if __name__ == '__main__':
    main()
