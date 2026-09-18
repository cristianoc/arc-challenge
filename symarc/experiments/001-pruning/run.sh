#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python3 - "$@" <<'PY'
import datetime
import hashlib
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

root = Path.cwd()
exp = Path('experiments/001-pruning')
extra = sys.argv[1:]
if '--data' in extra or '--threads' in extra:
    sys.exit('this protocol uses public training tasks and 12 workers; do not override data or workers')
out = Path('out')
out.mkdir(exist_ok=True)
lock = out / '.bench-lock'
try:
    lock.mkdir()
except FileExistsError:
    sys.exit(f'another benchmark owns {lock}; timing runs must be serial')
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
try:
    (lock / 'pid').write_text(str(__import__('os').getpid()) + '\n')
    subprocess.run(['cargo', 'build', '--release', '--manifest-path', str(exp / 'Cargo.toml')], check=True)
    command = [str(exp / 'target/release/symarc-exp-001-pruning'), '--data', '../data/training', '--threads', '12', *extra]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest = out / 'experiments/001-pruning' / stamp
    dest.mkdir(parents=True)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    core = [*sorted(Path('src').rglob('*.rs')), Path('Cargo.toml'), Path('Cargo.lock')]
    core_hashes = {str(p): sha(p) for p in core}
    sources = [*sorted((exp / 'src').rglob('*.rs')), exp/'Cargo.toml', exp/'Cargo.lock', exp/'README.md', exp/'run.sh']
    metadata = dict(command=command, cwd=str(root), utc=stamp,
                    git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    git_changes=subprocess.check_output(['git','status','--porcelain','--','src','Cargo.toml','Cargo.lock',str(exp)],text=True).splitlines(),
                    core_sha256=hashlib.sha256(json.dumps(core_hashes,sort_keys=True).encode()).hexdigest(),
                    core_sources=core_hashes, experiment_sources={str(p):sha(p) for p in sources},
                    data_sha256={str(p):sha(p) for p in sorted(Path('../data/training').glob('*.json'))},
                    rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),
                    platform=platform.platform(), results_file='report.md')
    (dest/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Running: {dest}',flush=True)
    start=time.monotonic()
    with (dest/'report.md').open('w') as stdout, (dest/'stderr.txt').open('w') as stderr:
        result=subprocess.run(command,stdout=stdout,stderr=stderr)
    metadata.update(elapsed_seconds=time.monotonic()-start,exit_code=result.returncode,results_sha256=sha(dest/'report.md'))
    (dest/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'{metadata["elapsed_seconds"]:.2f}s; report: {dest / "report.md"}',flush=True)
    if result.returncode:
        sys.exit(f'experiment failed ({result.returncode}); see {dest / "stderr.txt"}')
finally:
    (lock/'pid').unlink(missing_ok=True)
    lock.rmdir()
PY
