#!/usr/bin/env python3
"""Reweight a quick-subset run to full-set estimates, and compare with full runs.

    python3 subsets/estimate.py out/quick.txt                # estimates only
    python3 subsets/estimate.py out/quick.txt out/training.txt out/evaluation.txt

The quick subset is a stratified sample (see subsets/strata.json). Each task
in stratum s carries weight size(s) / sample(s). A count over the full set is
estimated as the weighted count over the sample. Standard errors use the stratified random-sampling formula. Because A2 is
family-balanced, these are approximate diagnostics, not calibrated intervals.
"""
import json, math, re, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
STRATA = json.load(open(os.path.join(HERE, 'strata.json')))
STRATUM_OF = {t: s for s, d in STRATA.items() for t in d['members']}
LINE = re.compile(r'([0-9a-f]{8})\s+func\[(.*?)\] real\[(.*?)\] \|C\|=(\d+)(\+?) .*?progsD=(\d+) progs=(\d+) fitD=([\d.]+) fitC=([\d.]+) .*?sym=(\d+)/(\d+)\(ok (\d+)\) equiv=(\S+) det=(\S+)\s+(\S+)\s+:: (.*)$')

def parse(path):
    rows = {}
    for line in open(path):
        m = LINE.match(line)
        if not m:
            continue
        rows[m.group(1)] = dict(
            func=m.group(2), real=m.group(3), capped=m.group(5) == '+',
            fitted=m.group(8) == '1.00', solved=m.group(15) == 'SOLVED',
            rejected=m.group(3) not in ('n/a', m.group(2)),
            sym=int(m.group(10)), symok=int(m.group(12)),
            equivNO=m.group(13).count('NO'), detNO=m.group(14).count('NO'),
            ntests=int(m.group(11)))
    return rows

METRICS = {
    'fits data': lambda r: r['fitted'],
    'fits data and solves all tests': lambda r: r['fitted'] and r['solved'],
    'fits data, wrong on a test': lambda r: r['fitted'] and not r['solved'],
    'realizability rejected a functional generator': lambda r: r['rejected'],
    'test inputs inside the closure': lambda r: r['sym'],
    'of which correct': lambda r: r['symok'],
    'equivariance failures at a test input': lambda r: r['equivNO'],
    'test inputs where survivors disagree': lambda r: r['detNO'],
    'closure capped': lambda r: r['capped'],
}

def estimate(quick):
    expected = {line.split()[0] for line in open(os.path.join(HERE, 'quick.txt'))
                if line.strip() and not line.lstrip().startswith('#')}
    if set(quick) != expected:
        raise ValueError(f'need the complete quick set: missing {len(expected - set(quick))} tasks; unexpected {len(set(quick) - expected)}')
    out = {}
    for name, f in METRICS.items():
        est = 0.0; var = 0.0
        for s, d in STRATA.items():
            xs = [float(f(quick[t])) for t in d['members'] if t in quick]
            n = len(xs)
            if n == 0:
                continue
            N = d['size']; mean = sum(xs) / n
            est += N * mean
            if n > 1:
                sv = sum((x - mean) ** 2 for x in xs) / (n - 1)
                var += N * N * (1 - n / N) * sv / n
        out[name] = (est, math.sqrt(var))
    return out

def main():
    quick = parse(sys.argv[1])
    missing = [t for t in quick if t not in STRATUM_OF]
    if missing:
        print('warning: tasks not in strata:', missing)
    est = estimate(quick)
    full = {}
    for p in sys.argv[2:]:
        full.update(parse(p))
    print(f"{'metric':48s} {'estimate':>10s} {'± s.e.':>8s} {'full':>6s}")
    for name, (e, se) in est.items():
        actual = sum(float(METRICS[name](r)) for r in full.values()) if full else None
        a = f"{actual:6.0f}" if actual is not None else '     -'
        print(f"{name:48s} {e:10.1f} {se:8.1f} {a}")

if __name__ == '__main__':
    main()
