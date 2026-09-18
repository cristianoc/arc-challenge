#!/usr/bin/env python3
"""Serial, bounded, reproducible depth experiment; solver emits report directly."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.chdir(Path(__file__).resolve().parents[2])
mode = sys.argv[1]
assert mode in ('pilot', 'selected')
exp = Path('experiments/002-depth')
Path('out').mkdir(exist_ok=True)
lock = Path('out/.bench-lock')
lock.mkdir()  # fail if another benchmark owns the machine
try:
    subprocess.run(['cargo', 'build', '--release', '--manifest-path', str(exp/'Cargo.toml')], check=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest = Path('out/experiments/002-depth') / (stamp+'-'+mode)
    dest.mkdir(parents=True)
    binary = exp/'target/release/symarc-exp-002-depth'
    command = ['/usr/bin/time', '-l', str(binary), mode, str(dest)]
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    core = sorted(Path('src').glob('*.rs'))+[Path('Cargo.toml'), Path('Cargo.lock')]
    sources = core+sorted(p for p in exp.rglob('*') if p.is_file() and 'target' not in p.parts)
    core_hashes = {str(p): digest(p) for p in core}
    meta = dict(utc=stamp, mode=mode, workers=12, timeout_seconds=600,
                command=command, cwd=str(Path.cwd()),
                git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                git_changes=subprocess.check_output(['git','status','--porcelain','--','src',str(exp)],text=True).splitlines(),
                core_sha256=hashlib.sha256(json.dumps(core_hashes,sort_keys=True).encode()).hexdigest(),
                source_sha256={str(p):digest(p) for p in sources},
                data_sha256={str(p):digest(p) for p in sorted(Path('../data/training').glob('*.json'))},
                binary_sha256=digest(binary), platform=platform.platform(),
                rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),results_file='report.md')
    manifest=dest/'run.json'
    manifest.write_text(json.dumps(meta,indent=2)+'\n')
    start=time.monotonic()
    with (dest/'report.md').open('w') as stdout,(dest/'stderr.txt').open('w') as stderr:
        proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True)
        try:
            code=proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
            code=124
    meta.update(elapsed_seconds=time.monotonic()-start,exit_code=code,
                results_sha256=digest(dest/'report.md'),
                artifact_sha256={str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='run.json'})
    manifest.write_text(json.dumps(meta,indent=2)+'\n')
    print(f'{mode}: {meta["elapsed_seconds"]:.2f}s; {dest}',flush=True)
    sys.exit(code)
finally:
    lock.rmdir()
