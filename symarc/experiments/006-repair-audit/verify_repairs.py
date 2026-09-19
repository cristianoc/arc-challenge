from repair_probe import *
import random
rng=random.Random(0);results={}
for id,f in REPAIRS.items():
 result={}
 for split,es in C[id]['data'].items():
  result[split]=[f(copy.deepcopy(e['input']))==e['output'] for e in es]
 fixed={0,5} if id=='e3721c99' else {0,3,4} if id=='221dfab4' else {0}
 colors=[i for i in range(10) if i not in fixed];checks=[]
 for _ in range(10):
  shuffled=colors[:];rng.shuffle(shuffled);mapping=dict(zip(colors,shuffled));mapping.update({v:v for v in fixed});rename=lambda g:[[mapping[v] for v in row] for row in g]
  for es in C[id]['data'].values():
   for e in es:checks.append(f(rename(e['input']))==rename(e['output']))
 result['colour_permutation_checks']={'passed':sum(checks),'total':len(checks),'fixed_colours':sorted(fixed)}
 if id in ['1ae2feb7','135a2760','221dfab4']:
  transform=(lambda g:[row[::-1] for row in g]) if id=='1ae2feb7' else (lambda g:list(map(list,zip(*g))))
  checks=[f(transform(e['input']))==transform(e['output']) for es in C[id]['data'].values() for e in es]
  result['reflection' if id=='1ae2feb7' else 'transpose']={'passed':sum(checks),'total':len(checks)}
 results[id]=result
 print(id,result)
output = CORPUS_PATH.parent / 'repair_results.json'
output.write_text(json.dumps(results, indent=2)+'\n')
expected_path = Path(__file__).resolve().parents[1] / 'evidence/006-repair-audit/repair_results.json'
assert results == json.loads(expected_path.read_text()), 'Results differ from retained evidence'
assert all(all(v[split]) for v in results.values() for split in ('train','test'))
print('Verified: 25 training, 14 test, 403 metamorphic checks; retained results match.')
