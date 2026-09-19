"""Fetch the external audit corpus at exact revisions; verify before executing it."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
DEST = Path(os.environ.get('ARC_REPAIR_CORPUS', HERE.parents[1] / 'out/experiments/006-repair-audit/corpus.json'))
manifest = json.loads((HERE / 'inputs.json').read_text())

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def validate(corpus):
    assert set(corpus) == set(manifest['tasks']), 'Task inventory mismatch'
    for task, expected in manifest['tasks'].items():
        item = corpus[task]
        assert digest(item['source']) == expected['source_sha256'], f'{task}: source mismatch'
        canonical = json.dumps(item['data'], sort_keys=True, separators=(',', ':'))
        assert digest(canonical) == expected['data_canonical_sha256'], f'{task}: data mismatch'

if __name__ == '__main__':
    if DEST.exists():
        validate(json.loads(DEST.read_text()))
        print(f'Verified cached corpus: {DEST}')
    else:
        corpus = {}
        for task in manifest['tasks']:
            paths = {
                'source': (manifest['solver_repository'], manifest['solver_revision'], f'tasks/{task}/solution.py'),
                'data': (manifest['data_repository'], manifest['data_revision'], f'data/evaluation/{task}.json'),
            }
            item = {}
            for kind, (repo, revision, path) in paths.items():
                url = f'https://raw.githubusercontent.com/{repo}/{revision}/{path}'
                with urllib.request.urlopen(url, timeout=30) as response:
                    content = response.read().decode('utf-8')
                item[kind] = json.loads(content) if kind == 'data' else content
            corpus[task] = item
        validate(corpus)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        temp = DEST.with_suffix('.tmp')
        temp.write_text(json.dumps(corpus))
        temp.replace(DEST)
        print(f'Fetched and verified corpus: {DEST}')
