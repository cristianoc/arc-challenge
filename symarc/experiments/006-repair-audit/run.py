"""Serial reproduction of the retrospective audit, with provenance and assertions."""
import argparse
import copy
import hashlib
import json
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from fetch_inputs import DEST, HERE, validate

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', action='store_true', help='also reproduce all 120 original solvers (requires NumPy)')
args = parser.parse_args()
corpus = json.loads(DEST.read_text())
validate(corpus)
start = time.monotonic()
subprocess.run([sys.executable, str(HERE / 'verify_repairs.py')], check=True)
artifacts = [DEST.parent / 'repair_results.json']
if args.baseline:
    if not hasattr(signal, 'SIGALRM'):
        raise RuntimeError('Baseline time limits require POSIX SIGALRM')
    def timeout(*_):
        raise TimeoutError('3-second per-example limit')
    signal.signal(signal.SIGALRM, timeout)
    results = {}
    for task, item in corpus.items():
        namespace = {}; result = {}
        try:
            exec(item['source'], namespace)
            solver = namespace.get('p') or namespace.get('solve_' + task)
            for split, examples in item['data'].items():
                result[split] = []
                for example in examples:
                    signal.alarm(3)
                    try:
                        result[split].append(solver(copy.deepcopy(example['input'])) == example['output'])
                    except Exception as error:
                        result[split].append(type(error).__name__ + ': ' + str(error))
                    finally:
                        signal.alarm(0)
        except Exception as error:
            result['error'] = str(error)
        results[task] = result
    output = DEST.parent / 'baseline.json'
    output.write_text(json.dumps(results, indent=2) + '\n')
    artifacts.append(output)
    expected = json.loads((HERE.parent / 'evidence/006-repair-audit/baseline.json').read_text())
    assert results == expected, 'Baseline differs from retained evidence; inspect output'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    'utc': datetime.now(timezone.utc).isoformat(),
    'command': sys.argv,
    'python': sys.version,
    'platform': platform.platform(),
    'elapsed_seconds': time.monotonic() - start,
    'timing_interpretation': 'reproduction duration only; not a benchmark comparison',
    'method': 'Retrospective witnesses; official test answers inspected during development',
    'corpus_sha256': sha(DEST),
    'source_sha256': {p.name: sha(p) for p in sorted(HERE.glob('*.py'))},
    'inputs_manifest_sha256': sha(HERE / 'inputs.json'),
    'artifacts_sha256': {p.name: sha(p) for p in artifacts},
}
(DEST.parent / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
print('Retained evidence reproduced. Manifest:', DEST.parent / 'run.json')
