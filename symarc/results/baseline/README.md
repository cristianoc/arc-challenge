# Current stable baseline

Source commit: `3a251a5c5ebde96ab0d77ff08b7e05eef911fe87`.
Measured 2026-09-18 from a clean solver/harness/task-list state. The full and
quick runs have the same core fingerprint and both use **12 workers**.
These are reference results for experiments, not an experimental treatment.

| Measurement | Quick | Full |
|---|---:|---:|
| Tasks | 40 | 800 |
| Test inputs | 45 | 835 |
| Fits training data | 22 | 77 |
| Solves every test | 19 | 80 |
| Fits training and solves every test | 19 | 74 |
| Fits training but misses a test | 3 | 3 |
| Tasks with a realizability rejection | 21 | 73 |
| Tasks with a successful repair | 1 | 1 |
| Capped final closures | 14 | 584 |
| Test inputs reached by symmetry / correct symmetry answers | 5 / 3 | 28 / 14 |
| Test inputs with survivor disagreement | 6 | 6 |
| Wall time, 12 workers | 2.06 s | 15.49 s |

The full set contains 400 public training and 400 public evaluation tasks.
The quick set is deliberately enriched in fitted and ambiguous tasks; its
raw counts must not be extrapolated uniformly. Six full-set tasks solve all
tests without fitting all training examples, so `solves every test` and
`fits training and solves every test` are separate metrics.

## Configuration and reproduction

Default full pipeline: enumeration depth 2, hill-climbing length threshold 3,
seed 0, functionality cap 1000, sample cap 300, final cap 20000, 64 extra sample
attempts, 16 restarts × 200 steps, 150 repair steps, realizability enabled.
There is no `--bench` shortcut in these measurements.

From `symarc/` at the source revision above:

```sh
THREADS=12 ./bench.sh quick
THREADS=12 ./bench.sh full
python3 subsets/estimate.py results/baseline/quick.txt results/baseline/full.txt
```

Run the two commands serially. The current harness defaults to 12 workers;
the explicit environment variable also reproduces this setting at the source
commit. Keep worker count fixed for ordinary experiments. These are single
wall-time observations on the machine recorded in the manifests, not a
speedup claim or a precision timing study.

## Retained evidence

- [quick.txt](quick.txt): every quick-set task and aggregate output.
- [full.txt](full.txt): every full-set task and aggregate output.
- [quick.run.json](quick.run.json), [full.run.json](full.run.json): commands,
  source revision, clean-state check, core/source/binary/data hashes, toolchain,
  machine, UTC timestamp, timing, exit code, and result-file hash.

All 40 quick-set task lines agree exactly with their full-set counterparts.
The retained output files match the hashes in their manifests. Raw runs under
`out/` are disposable; these files and this summary are the committed evidence.

Capped closure gains are explored-coverage measurements, and realizability
uses sampling and possible repair. Program-count ratios must not be treated
as exact hypothesis entropy reductions without controlling those mechanisms.
See [the current mathematics](../../MATH.md) for these boundaries.
