"""Compare reconstructed primitive models and selections with frozen 024/base."""
import argparse
import gzip
import json
from pathlib import Path


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as file:
        return json.load(file)


def compare(predictions, baseline):
    old = {row['id']: row for row in read(baseline)}
    rows = read(predictions)
    if set(old) != {row['id'] for row in rows}:
        raise ValueError('Baseline and prediction task sets differ')
    counts = dict(tasks=0, model_occurrences=0, outer_choices=0)
    for row in rows:
        prior = old[row['id']]
        assert row['eligible'] == prior['eligible']

        def pool(current, previous):
            by_features = {tuple(m['features']): m for m in current if m['library'] == 'base'}
            for reference in previous:
                model = by_features[tuple(reference['features'])]
                for field in ('features', 'cost', 'cv_exact', 'cv_fraction'):
                    assert model[field] == reference[field], (row['id'], field)
                assert model['predictions'] == reference['total_predictions'], row['id']
                assert model['condition']['feasible'] == reference['feasible'], row['id']
                counts['model_occurrences'] += 1

        def choice(current, previous):
            selected = current['selected']
            assert (selected[1:] if selected else []) == previous['selected'], row['id']
            assert current['predictions'] == previous['predictions'], row['id']

        pool(row['models'], prior['models'])
        choice(row['policies']['base'], prior['policies']['total_ranked'])
        for current, previous in zip(row['outer'], prior['outer'], strict=True):
            assert current['excluded'] == previous['excluded']
            pool(current['models'], previous['models'])
            choice(current['policies']['base'], previous['policies']['total_ranked'])
            assert current['scores']['base'] == previous['scores']['total_ranked']
            counts['outer_choices'] += 1
        counts['tasks'] += 1
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('predictions', 'baseline', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.predictions, args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(result)
