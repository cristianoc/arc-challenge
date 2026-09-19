#!/usr/bin/env python3
"""Serial benchmark wrapper; independent experiment using stable libraries."""
from pathlib import Path
import datetime, hashlib, json, os, platform, shutil, signal, subprocess, sys, time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.chdir(ROOT)
mode=sys.argv[1]
assert mode in ('pilot','full','development')
if mode=='full':
    previous=sorted((ROOT/'out/experiments/007-path-order').glob('*-pilot/run.json'))
    assert previous, 'Run a pilot first'
    pilot=json.loads(previous[-1].read_text())
    assert pilot['exit_code']==0 and pilot['elapsed_seconds']<120 and pilot['peak_rss_kib']<4*1024*1024, 'Pilot gate failed'
cargo=shutil.which('cargo') or str(Path.home()/'.cargo/bin/cargo')
rustc=shutil.which('rustc') or str(Path.home()/'.cargo/bin/rustc')
(ROOT/'out').mkdir(exist_ok=True)
lock=ROOT/'out/.bench-lock'
lock.mkdir()
try:
    subprocess.run([cargo,'build','--release','--manifest-path',str(HERE/'Cargo.toml')],check=True)
    dest=ROOT/'out/experiments/007-path-order'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+mode)
    dest.mkdir(parents=True)
    binary=HERE/'target/release/symarc-exp-007-path-order'
    command=[sys.executable,str(HERE/'measure.py'),str(dest/'resources.txt'),str(binary),mode,str(dest)]
    if mode=='development': command.append(str(Path(sys.argv[2]).resolve()))
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    sources=sorted((ROOT/'src').glob('*.rs'))+[ROOT/'Cargo.toml',ROOT/'Cargo.lock']
    core={str(p.relative_to(ROOT)):sha(p) for p in sources}
    sourcefiles=sources+sorted(p for folder in [HERE,ROOT/'experiments/003-objects'] for p in folder.rglob('*') if p.is_file() and 'target' not in p.parts and '__pycache__' not in p.parts)
    datafiles=[Path(command[-1])] if mode=='development' else sorted((ROOT.parent/'data/training').glob('*.json'))
    record=dict(mode=mode,workers=12,command=command,cwd=str(ROOT),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_changes=subprocess.check_output(['git','status','--porcelain','--','symarc'],cwd=ROOT.parent,text=True).splitlines(),core_sha256=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sourcefiles},data_sha256={str(p):sha(p) for p in datafiles},binary_sha256=sha(binary),platform=platform.platform(),rustc=subprocess.check_output([rustc,'--version'],text=True).strip(),timeout_seconds=600)
    start=time.monotonic()
    with (dest/'report.md').open('w') as out,(dest/'stderr.txt').open('w') as err:
        proc=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
        try:code=proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
    elapsed=time.monotonic()-start
    import re
    resource=(dest/'resources.txt').read_text()
    m=re.search(r'peak_rss_kib=(\d+)',resource)
    record.update(exit_code=code,elapsed_seconds=elapsed,peak_rss_kib=int(m.group(1)) if m else None,artifact_sha256={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()})
    (dest/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(dest,flush=True)
    print(f'exit={code}; {elapsed:.3f}s; peak RSS={record["peak_rss_kib"]} KiB',flush=True)
    sys.exit(code)
finally:lock.rmdir()
