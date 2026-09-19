from pathlib import Path
import subprocess,sys,json,time,hashlib,platform,csv
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2];ROOT=REPO/'symarc';MODE=sys.argv[1];ARC2=Path(sys.argv[2]).resolve();OUT=ROOT/'out/experiments/011-arc2-transfer'/MODE
assert MODE in ['pilot','full']
if MODE=='full':
 p=json.loads((OUT.parent/'pilot/run.json').read_text());assert p['exit_code']==0 and p['elapsed_seconds']<120
OUT.mkdir(parents=True,exist_ok=True);lock=ROOT/'out/.bench-lock';lock.mkdir()
try:
 cargo=str(Path.home()/'.cargo/bin/cargo');subprocess.run([cargo,'build','--release','--manifest-path',str(HERE/'Cargo.toml')],check=True)
 binary=HERE/'target/release/symarc-exp-011-arc2-transfer';cmd=[str(binary),MODE,str(REPO/'data'),str(ARC2/'data')];start=time.monotonic()
 with (OUT/'tasks.tsv').open('w') as f,(OUT/'stderr.txt').open('w') as err:r=subprocess.run(cmd,stdout=f,stderr=err,timeout=600)
 elapsed=time.monotonic()-start;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 sources=[p for base in [ROOT/'src',ROOT/'experiments/003-objects/src',HERE] for p in base.rglob('*') if p.is_file() and 'target' not in p.parts and '__pycache__' not in p.parts]
 m=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),changes=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).splitlines(),command=cmd,workers=12,elapsed_seconds=elapsed,exit_code=r.returncode,platform=platform.platform(),python=sys.version,rustc=subprocess.check_output([str(Path.home()/'.cargo/bin/rustc'),'--version'],text=True).strip(),data_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ARC2,text=True).strip(),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},binary_sha256=sha(binary),data_sha256={str(p):sha(p) for d in [REPO/'data',ARC2/'data'] for p in sorted(d.glob('*/*.json'))},report_sha256=sha(OUT/'tasks.tsv'))
 (OUT/'run.json').write_text(json.dumps(m,indent=2)+'\n');assert r.returncode==0
 rows=list(csv.DictReader((OUT/'tasks.tsv').open(),delimiter='\t'));lines=['# 011: unchanged solver transfer','','| Dataset | Split | Arm | Fitting tasks | Correct tasks | Correct grids |','|---|---|---|---:|---:|---:|'];stats={}
 for d in ['ARC1','ARC2']:
  for s in ['training','evaluation']:
   for arm in ['complete','grid_d2','objects','grid_objects']:
    rs=[r for r in rows if (r['dataset'],r['split'],r['arm'])==(d,s,arm)];n=len(rs);v=sum(r['correct']=='true' for r in rs);stats[d,s,arm]=v;lines.append(f"| {d} | {s} | {arm} | {sum(r['fit']=='true' for r in rs)}/{n} | {v}/{n} | {sum(int(r['correct_grids']) for r in rs)}/{sum(int(r['test_grids']) for r in rs)} |")
 if MODE=='full':assert stats['ARC1','training','complete']==57 and stats['ARC1','evaluation','complete']==23 and stats['ARC1','training','grid_objects']==60
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));print('elapsed',elapsed)
finally:lock.rmdir()
