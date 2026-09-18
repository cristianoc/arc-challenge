#!/bin/bash
# Serial experiment harness. Complete pipeline unless --bench is passed.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 - "$@" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

root = Path.cwd()
mode = sys.argv[1] if len(sys.argv) > 1 else 'quick'
if mode not in ('quick', 'full'):
    sys.exit('usage: ./bench.sh quick|full [solver options]')
extra = sys.argv[2:]
if any(a in extra for a in ('--root', '--data', '--tasks-file', '--demo', '--threads')):
    sys.exit('select the task set with quick|full and workers with THREADS=N')
try:
    threads = int(os.environ.get('THREADS', '12'))
    reps = int(os.environ.get('REPS', '1'))
    if threads < 1 or reps < 1:
        raise ValueError()
except ValueError:
    sys.exit('THREADS and REPS must be positive integers')
out = root / 'out'
out.mkdir(exist_ok=True)
lock = out / '.bench-lock'
try:
    lock.mkdir()
except FileExistsError:
    sys.exit(f'another harness run owns {lock}; if it was killed, remove the stale lock')
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
try:
    (lock / 'pid').write_text(str(os.getpid()) + '\n')
    subprocess.run(['cargo', 'build', '--release'], check=True)
    task_list = root / 'subsets' / f'{mode}.txt'
    argv = ['./target/release/symarc', '--root', '../data', '--tasks-file',
            f'subsets/{mode}.txt', '--threads', str(threads), *extra]
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = {}
    for line in task_list.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            tid, split, *_ = line.split()
            path = Path('../data') / split / f'{tid}.json'
            inputs[str(path)] = digest(path)
    sources = [*sorted(Path('src').rglob('*.rs')), Path('Cargo.toml'), Path('Cargo.lock'), Path('bench.sh')]
    core_sources = [*sorted(Path('src').rglob('*.rs')), Path('Cargo.toml'), Path('Cargo.lock')]
    core_hashes = {str(p): digest(p) for p in core_sources}
    core_sha256 = hashlib.sha256(json.dumps(core_hashes, sort_keys=True).encode()).hexdigest()
    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    git_changes = subprocess.check_output(
        ['git', 'status', '--porcelain', '--', 'src', 'Cargo.toml', 'Cargo.lock',
         'bench.sh', 'subsets'], text=True).splitlines()
    for rep in range(reps):
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        dest = out / f'{stamp}-{mode}'
        dest.mkdir()
        metadata = dict(git_revision=git_revision, git_changes=git_changes, core_sha256=core_sha256, command=argv, cwd=str(root), utc=stamp, repetition=rep+1,
                        mode='bench' if '--bench' in extra else 'solver',
                        machine=platform.platform(), cpu_count=os.cpu_count(),
                        rustc=subprocess.check_output(['rustc', '--version'], text=True).strip(),
                        task_list=str(task_list.relative_to(root)), task_list_sha256=digest(task_list),
                        source_sha256={str(p): digest(p) for p in sources},
                        binary_sha256=digest(Path('target/release/symarc')),
                        data_sha256=inputs)
        result_name = 'bench.txt' if '--bench' in extra else 'report.md'
        metadata['results_file'] = result_name
        manifest = dest / 'run.json'
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        start = time.monotonic()
        with (dest / result_name).open('w') as stdout, (dest / 'stderr.txt').open('w') as stderr:
            p = subprocess.run(argv, stdout=stdout, stderr=stderr)
        metadata.update(elapsed_seconds=time.monotonic()-start, exit_code=p.returncode,
                        results_sha256=digest(dest / result_name))
        manifest.write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'{mode}: {metadata["elapsed_seconds"]:.2f} s; report: {(dest / result_name).relative_to(root)}', flush=True)
        if p.returncode:
            sys.exit(f'solver failed ({p.returncode}); see {dest / "stderr.txt"}')
finally:
    (lock / 'pid').unlink(missing_ok=True)
    lock.rmdir()
PY
