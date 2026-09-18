# SymArc run report

Selection: task list subsets/quick.txt.

Workers: 12. Seed: 0. Loading + search wall time: 1.85 s (excludes report I/O and compilation).

## Predictive performance

One selected prediction per test input. **Task exact match** requires correct shape and every cell on every test output in a task; **test-grid exact match** scores individual test outputs. Counts and denominators are explicit.

| Metric | Public training tasks | Public evaluation tasks | All selected tasks |
|---|---:|---:|---:|
| Tasks | 24 | 16 | 40 |
| Test grids | 28 | 17 | 45 |
| **Task exact-match accuracy (primary)** | 12/24 (50.00%) | 7/16 (43.75%) | 19/40 (47.50%) |
| Test-grid exact-match accuracy | 14/28 (50.00%) | 8/17 (47.06%) | 22/45 (48.89%) |
| Prediction coverage (test grids) | 28/28 (100.00%) | 17/17 (100.00%) | 45/45 (100.00%) |
| Fits all within-task training examples | 15/24 (62.50%) | 7/16 (43.75%) | 22/40 (55.00%) |
| Fits training and solves every test | 12/24 (50.00%) | 7/16 (43.75%) | 19/40 (47.50%) |
| Task accuracy conditional on training fit | 12/15 (80.00%) | 7/7 (100.00%) | 19/22 (86.36%) |

Split labels describe task datasets, not a model trained across tasks. Each task is fitted independently. Test inputs participate in label-free constraints; test output labels are used only for scoring. Conditional accuracy applies only to the training-fitting subset.

Rates describe the selected tasks. A stratified quick set is not representative without reweighting. Public evaluation data used during development is not an untouched hidden test set.

## Symmetry diagnostics

| Metric | Public training tasks | Public evaluation tasks | All selected tasks |
|---|---:|---:|---:|
| Realizability rejected a generator (tasks) | 14/15 (93.33%) | 7/7 (100.00%) | 21/22 (95.45%) |
| Successful repair (tasks) | 1/15 (6.67%) | 0/7 (0.00%) | 1/22 (4.55%) |
| Capped final closure (tasks) | 6/24 (25.00%) | 8/16 (50.00%) | 14/40 (35.00%) |
| Median reported coverage gain (bits) | 8.91 | 11.97 | 10.84 |
| Test inputs reached by symmetry | 3/28 (10.71%) | 2/17 (11.76%) | 5/45 (11.11%) |
| Symmetry-answer accuracy on reached inputs | 1/3 (33.33%) | 2/2 (100.00%) | 3/5 (60.00%) |
| Program undefined at test input | 0 | 0 | 0 |
| Test-input equivariance: yes / no / unassessed | 28 / 0 / 0 | 16 / 1 / 0 | 44 / 1 / 0 |
| Survivor agreement: yes / no / unassessed | 10 / 5 / 13 | 3 / 1 / 13 | 13 / 6 / 26 |
| Tasks retaining dihedral: functional / realizable | 23 / 14 | 16 / 7 | 39 / 21 |
| Tasks retaining colours: functional / realizable | 24 / 13 | 16 / 7 | 40 / 20 |
| Tasks retaining cyclic: functional / realizable | 18 / 7 | 14 / 3 | 32 / 10 |
| Tasks retaining rows: functional / realizable | 20 / 12 | 15 / 5 | 35 / 17 |
| Tasks retaining cols: functional / realizable | 22 / 13 | 16 / 4 | 38 / 17 |

Rejection and repair denominators are tasks with a fitting program. Capped gains measure explored coverage; they are not exact closure entropies. Realizability uses sampling and may replace programs through repair. Program-count ratios are not automatically hypothesis-entropy reductions.

## Configuration

Enumeration depth: 2. Mutation length threshold: 3. Realizability filter: true.

Closure caps: functionality 1000, sampling 300, final 20000. Extra sample attempts: 64.

Hill climbing: 16 restarts × 200 steps; repair: 150 steps.

This is one run at one seed. Seed variation, cell accuracy, probability calibration, and a causal benefit from symmetry are not measured by this report. For harness runs, run.json records the exact command, revision, hashes, platform, and end-to-end subprocess timing.

## Task details

```text
0520fde7  func[dihedral:3 colours:3 cols:4] real[dihedral:2 colours:1 cols:4] |C|=20 gain=2.74b outs=7  progsD=36 progs=33 fitD=1.00 fitC=1.00 evals=1038 sym=0/1(ok 0) equiv=yes det=NO  SOLVED  :: splitH and 2
3428a4f5  func[dihedral:3 colours:3 cyclic:2 rows:11 cols:4] real[dihedral:2 colours:1 cyclic:1 rows:7 cols:4] |C|=1920 gain=8.91b outs=960  progsD=35 progs=32 fitD=1.00 fitC=1.00 evals=2458 sym=0/2(ok 0) equiv=yes,yes det=yes,NO  SOLVED  :: splitV xor 3
5d2a5c43  func[dihedral:3 colours:3 cyclic:2 rows:4 cols:6] real[dihedral:2 colours:1 cyclic:1 rows:4 cols:5] |C|=14400 gain=11.49b outs=2520  progsD=34 progs=29 fitD=1.00 fitC=1.00 evals=1705 sym=0/2(ok 0) equiv=yes,yes det=NO,yes  SOLVED  :: splitH or 8
72ca375d  func[dihedral:3 colours:36 cyclic:2 rows:8 cols:9] real[dihedral:2 colours:22 rows:7 cols:5] |C|=5488 gain=10.84b outs=63  progsD=1 progs=1 fitD=1.00 fitC=0.89 evals=4195 sym=0/1(ok 0) equiv=yes det=-  fit-only  :: recolour 2 0 ; removeColour 8 ; cropLargest
c59eb873  func[dihedral:3 colours:28 rows:5 cols:5] real[dihedral:3 colours:28 rows:4 cols:4] |C|=15456 gain=12.33b outs=15456  progsD=71 progs=2 fitD=1.00 fitC=1.00 evals=2214 sym=0/1(ok 0) equiv=yes det=yes  fit-only  :: scale 2 ; cropSmallest
ce4f8723  func[dihedral:3 colours:6 cyclic:2 rows:6 cols:3] real[dihedral:2 colours:3 cyclic:1 rows:5 cols:3] |C|=1152 gain=8.17b outs=84  progsD=49 progs=46 fitD=1.00 fitC=1.00 evals=2510 sym=0/1(ok 0) equiv=yes det=NO  SOLVED  :: splitV or 3
d9fac9be  func[dihedral:3 colours:15 cyclic:2 rows:11 cols:11] real[dihedral:1 colours:15 rows:11 cols:11] |C|=240 gain=5.91b outs=6  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=763 sym=0/1(ok 0) equiv=yes det=-  fit-only  :: shiftRight ; topHalf ; cropSmallest
ed36ccf7  func[dihedral:2 colours:10] real[colours:10] |C|=20 gain=2.32b outs=20  progsD=25 progs=15 fitD=1.00 fitC=1.00 evals=694 sym=0/1(ok 0) equiv=yes det=NO  SOLVED  :: rot270
f2829549  func[dihedral:3 colours:6 cyclic:2 rows:2 cols:4] real[dihedral:2 colours:3 cyclic:1 rows:2 cols:3] |C|=1296 gain=8.02b outs=132  progsD=50 progs=47 fitD=1.00 fitC=1.00 evals=2264 sym=0/1(ok 0) equiv=yes det=NO  SOLVED  :: splitH nor 3
32e9702f  func[dihedral:1 colours:10 cyclic:1 rows:9 cols:6] real[dihedral:1 colours:10 cyclic:1 rows:9] |C|=1345 gain=8.81b outs=1345  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=513 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: shiftLeft ; recolour 0 5
66f2d22f  func[dihedral:3 colours:3 cyclic:2 rows:3 cols:10] real[dihedral:2 colours:1 cyclic:2 rows:3 cols:7] |C|=5376 gain=10.39b outs=1092  progsD=31 progs=27 fitD=1.00 fitC=1.00 evals=1932 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitH nor 5
67e8384a  func[dihedral:3 colours:30 cyclic:2 rows:3 cols:3] real[dihedral:1 colours:30 rows:3 cols:3] |C|=19152 gain=12.23b outs=19152  progsD=70 progs=68 fitD=1.00 fitC=1.00 evals=5922 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: mirror4
68b16354  func[dihedral:3 colours:21 cyclic:2 rows:6 cols:6] real[dihedral:2 colours:21 cyclic:1 cols:6] |C|=20000+ gain=12.70b outs=20000  progsD=60 progs=60 fitD=1.00 fitC=1.00 evals=5304 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: flipV
833dafe3  func[dihedral:3 colours:21 rows:5 cols:5] real[dihedral:1 colours:21] |C|=840 gain=8.71b outs=840  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=680 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: rot180 ; mirror4
84db8fc4  func[dihedral:3 colours:3 cyclic:2 rows:9 cols:9] real[dihedral:3 colours:1] |C|=64 gain=4.00b outs=64  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=3418 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: fillEnclosed 5 ; recolour 0 2
903d1b4a  func[dihedral:3 colours:36 cyclic:2 rows:15 cols:15] real[dihedral:3 colours:28 rows:2 cols:1] |C|=20001+ gain=12.29b outs=9354  progsD=2 progs=2 fitD=1.00 fitC=1.00 evals=5920 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: recolour 3 0 ; symmetrize4
94f9d214  func[dihedral:3 colours:3 cyclic:2 rows:4 cols:1] real[dihedral:2 colours:3 cyclic:2 rows:4 cols:1] |C|=9216 gain=11.17b outs=256  progsD=30 progs=30 fitD=1.00 fitC=1.00 evals=903 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitV nor 2
b1948b0a  func[dihedral:3 colours:3 cyclic:2 rows:5 cols:5] real[dihedral:3 cyclic:2 rows:5 cols:5] |C|=20000+ gain=12.70b outs=20000  progsD=44 progs=42 fitD=1.00 fitC=1.00 evals=1855 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: recolour 6 2
cd3c21df  func[dihedral:3 colours:36 cyclic:2 rows:11 cols:12] real[dihedral:1 colours:22 rows:9 cols:10] |C|=2688 gain=9.81b outs=56  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=3935 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: recolour 5 3 ; removeColour 3 ; cropLargest
d5d6de2d  func[dihedral:3 colours:1 cyclic:2 rows:24 cols:24] real[dihedral:3 rows:7 cols:8] |C|=40 gain=3.74b outs=20  progsD=3 progs=3 fitD=1.00 fitC=1.00 evals=4404 sym=0/2(ok 0) equiv=yes,yes det=yes,yes  SOLVED  :: fillEnclosed 3 ; recolour 2 0
d631b094  func[dihedral:2 colours:15 cyclic:2 rows:2 cols:3] real[dihedral:2 colours:15 cyclic:2 rows:2 cols:3] |C|=378 gain=6.56b outs=24  progsD=58 progs=44 fitD=1.00 fitC=1.00 evals=2244 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: countRow
f25fbde4  func[dihedral:3 colours:1 rows:4 cols:4] real[dihedral:3 colours:1 rows:1] |C|=48 gain=4.00b outs=28  progsD=2 progs=2 fitD=1.00 fitC=1.00 evals=959 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: scale 2 ; cropBBox
00576224  func[dihedral:2 colours:28 cyclic:1 rows:4 cols:4] real[n/a] |C|=2352 gain=10.20b outs=2352  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
794b24be  func[colours:1 cyclic:1 cols:1] real[n/a] |C|=114 gain=3.51b outs=36  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
992798f6  func[dihedral:3 colours:3 rows:2 cols:2] real[n/a] |C|=768 gain=7.58b outs=768  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
d037b0a7  func[dihedral:2 colours:15 cyclic:1 rows:2] real[n/a] |C|=1440 gain=8.91b outs=900  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 0) equiv=yes det=-  unfit  :: id
dc433765  func[dihedral:3 colours:1 rows:3 cols:3] real[n/a] |C|=792 gain=6.82b outs=464  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/2(ok 1) equiv=yes,yes det=-,-  unfit  :: id
017c7c7b  func[dihedral:3 colours:1 rows:5 cols:2] real[n/a] |C|=1296 gain=8.75b outs=540  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2c737e39  func[dihedral:3 colours:28 cyclic:2 cols:5] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
36fdfd69  func[dihedral:3 colours:15 cyclic:2 rows:16 cols:17] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4093f84a  func[dihedral:3 colours:15 cyclic:2 rows:12 cols:13] real[n/a] |C|=20001+ gain=12.70b outs=7637  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
642248e4  func[dihedral:3 colours:15 cyclic:2 rows:13 cols:15] real[n/a] |C|=20003+ gain=12.70b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
caa06a1f  func[dihedral:3 colours:28 cyclic:2 rows:10 cols:11] real[n/a] |C|=20000+ gain=12.70b outs=5154  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e66aafb8  func[dihedral:3 colours:36 cyclic:2 rows:22 cols:23] real[n/a] |C|=20016+ gain=11.97b outs=18031  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ecaa0ec1  func[dihedral:3 colours:6 cyclic:2 rows:7 cols:8] real[n/a] |C|=20002+ gain=12.29b outs=10707  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ecdecbb3  func[dihedral:3 colours:3 cyclic:2 cols:1] real[n/a] |C|=19464 gain=12.66b outs=16656  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0934a4d8  func[dihedral:3 colours:36 cyclic:2 rows:29 cols:29] real[n/a] |C|=20006+ gain=12.29b outs=13308  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
09629e4f  func[dihedral:3 colours:21 cyclic:2 rows:10 cols:10] real[n/a] |C|=20000+ gain=12.29b outs=7107  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7d1f7ee8  func[dihedral:3 colours:28 cyclic:2 rows:28 cols:27] real[n/a] |C|=20013+ gain=12.70b outs=14392  progsD=0 progs=0 fitD=0.33 fitC=0.22 evals=3231 sym=0/1(ok 0) equiv=NO det=-  unfit  :: recolourNonzero 1
845d6e51  func[dihedral:3 colours:28 cyclic:2 rows:17 cols:16] real[n/a] |C|=20002+ gain=12.70b outs=16888  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
```
