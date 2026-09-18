#!/usr/bin/env python3
"""Check current output contracts, seeded search, CLI errors and worker independence.
Run after cargo build --release, from any directory. Uses the repository ARC data.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'target/release/symarc'
FIXTURES = ROOT / 'tests/fixtures'


def run(args):
    return subprocess.run([str(BIN), *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def main():
    for name, args in json.loads((FIXTURES / 'cases.json').read_text()).items():
        actual = run(args)
        expected = (FIXTURES / f'{name}.txt').read_text()
        assert actual == expected, f'{name}: output differs from expected fixture'
        print(f'{name}: passed')
    args = ['--root', '../data', '--tasks-file', 'subsets/quick.txt', '--show']
    serial = run(args)
    assert serial == run([*args, '--threads', '4']), 'worker count changed results'
    assert len([s for s in serial.splitlines() if '  func[' in s]) == 40
    sys.path.insert(0, str(ROOT))
    from subsets import estimate
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'quick.txt'
        path.write_text(serial)
        rows = estimate.parse(path)
        metrics = estimate.estimate(rows)
        assert metrics['fits data'][0] == 77.0
        rows.pop(next(iter(rows)))
        try:
            estimate.estimate(rows)
        except ValueError:
            pass
        else:
            raise AssertionError('estimator accepted an incomplete quick run')
    for args in [[], ['--wat'], ['--threads', '0'], ['--seed', 'bad'],
                 ['--tasks-file', 'subsets/quick.txt'], ['--data']]:
        p = subprocess.run([str(BIN), *args], cwd=ROOT, capture_output=True)
        assert p.returncode == 2, f'invalid CLI accepted: {args}'
    print('quick set, worker independence, CLI errors: passed')


if __name__ == '__main__':
    main()
