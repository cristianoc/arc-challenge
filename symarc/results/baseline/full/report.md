# SymArc run report

Selection: task list subsets/full.txt.

Workers: 12. Seed: 0. Loading + search wall time: 15.47 s (excludes report I/O and compilation).

## Predictive performance

One selected prediction per test input. **Task exact match** requires correct shape and every cell on every test output in a task; **test-grid exact match** scores individual test outputs. Counts and denominators are explicit.

| Metric | Public training tasks | Public evaluation tasks | All selected tasks |
|---|---:|---:|---:|
| Tasks | 400 | 400 | 800 |
| Test grids | 416 | 419 | 835 |
| **Task exact-match accuracy (primary)** | 57/400 (14.25%) | 23/400 (5.75%) | 80/800 (10.00%) |
| Test-grid exact-match accuracy | 62/416 (14.90%) | 27/419 (6.44%) | 89/835 (10.66%) |
| Prediction coverage (test grids) | 416/416 (100.00%) | 419/419 (100.00%) | 835/835 (100.00%) |
| Fits all within-task training examples | 56/400 (14.00%) | 21/400 (5.25%) | 77/800 (9.63%) |
| Fits training and solves every test | 53/400 (13.25%) | 21/400 (5.25%) | 74/800 (9.25%) |
| Task accuracy conditional on training fit | 53/56 (94.64%) | 21/21 (100.00%) | 74/77 (96.10%) |

Split labels describe task datasets, not a model trained across tasks. Each task is fitted independently. Test inputs participate in label-free constraints; test output labels are used only for scoring. Conditional accuracy applies only to the training-fitting subset.

Rates describe the selected tasks. A stratified quick set is not representative without reweighting. Public evaluation data used during development is not an untouched hidden test set.

## Symmetry diagnostics

| Metric | Public training tasks | Public evaluation tasks | All selected tasks |
|---|---:|---:|---:|
| Realizability rejected a generator (tasks) | 52/56 (92.86%) | 21/21 (100.00%) | 73/77 (94.81%) |
| Successful repair (tasks) | 1/56 (1.79%) | 0/21 (0.00%) | 1/77 (1.30%) |
| Capped final closure (tasks) | 270/400 (67.50%) | 314/400 (78.50%) | 584/800 (73.00%) |
| Median reported coverage gain (bits) | 12.70 | 12.70 | 12.70 |
| Test inputs reached by symmetry | 18/416 (4.33%) | 10/419 (2.39%) | 28/835 (3.35%) |
| Symmetry-answer accuracy on reached inputs | 9/18 (50.00%) | 5/10 (50.00%) | 14/28 (50.00%) |
| Program undefined at test input | 0 | 0 | 0 |
| Test-input equivariance: yes / no / unassessed | 397 / 19 / 0 | 401 / 18 / 0 | 798 / 37 / 0 |
| Survivor agreement: yes / no / unassessed | 50 / 5 / 361 | 18 / 1 / 400 | 68 / 6 / 761 |
| Tasks retaining dihedral: functional / realizable | 396 / 55 | 398 / 21 | 794 / 76 |
| Tasks retaining colours: functional / realizable | 398 / 50 | 399 / 21 | 797 / 71 |
| Tasks retaining cyclic: functional / realizable | 323 / 24 | 340 / 11 | 663 / 35 |
| Tasks retaining rows: functional / realizable | 339 / 39 | 368 / 19 | 707 / 58 |
| Tasks retaining cols: functional / realizable | 347 / 40 | 374 / 18 | 721 / 58 |

Rejection and repair denominators are tasks with a fitting program. Capped gains measure explored coverage; they are not exact closure entropies. Realizability uses sampling and may replace programs through repair. Program-count ratios are not automatically hypothesis-entropy reductions.

## Configuration

Enumeration depth: 2. Mutation length threshold: 3. Realizability filter: true.

Closure caps: functionality 1000, sampling 300, final 20000. Extra sample attempts: 64.

Hill climbing: 16 restarts × 200 steps; repair: 150 steps.

This is one run at one seed. Seed variation, cell accuracy, probability calibration, and a causal benefit from symmetry are not measured by this report. For harness runs, run.json records the exact command, revision, hashes, platform, and end-to-end subprocess timing.

## Task details

```text
00576224  func[dihedral:2 colours:28 cyclic:1 rows:4 cols:4] real[n/a] |C|=2352 gain=10.20b outs=2352  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
007bbfb7  func[dihedral:3 colours:10 rows:6 cols:6] real[n/a] |C|=160 gain=5.00b outs=160  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
009d5c81  func[dihedral:3 colours:3 cyclic:2 rows:13 cols:13] real[n/a] |C|=20023+ gain=11.97b outs=9646  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
00d62c1b  func[dihedral:3 colours:1 cyclic:2 rows:19 cols:19] real[dihedral:3] |C|=34 gain=2.77b outs=34  progsD=8 progs=8 fitD=1.00 fitC=1.00 evals=6657 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: fillEnclosed 4
00dbd492  func[dihedral:3 colours:1 cyclic:2 rows:19 cols:19] real[n/a] |C|=20002+ gain=12.29b outs=20002  progsD=0 progs=0 fitD=0.25 fitC=0.08 evals=3232 sym=0/1(ok 0) equiv=NO det=-  unfit  :: fillEnclosed 4
017c7c7b  func[dihedral:3 colours:1 rows:5 cols:2] real[n/a] |C|=1296 gain=8.75b outs=540  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
025d127b  func[dihedral:3 colours:10 rows:10 cols:6] real[n/a] |C|=20004+ gain=13.29b outs=15275  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
03560426  func[dihedral:3 colours:28 cyclic:2] real[n/a] |C|=20004+ gain=12.70b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
045e512c  func[dihedral:3 colours:28 cyclic:2 rows:1 cols:1] real[n/a] |C|=20001+ gain=12.70b outs=19998  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0520fde7  func[dihedral:3 colours:3 cols:4] real[dihedral:2 colours:1 cols:4] |C|=20 gain=2.74b outs=7  progsD=36 progs=33 fitD=1.00 fitC=1.00 evals=1038 sym=0/1(ok 0) equiv=yes det=NO  SOLVED  :: splitH and 2
05269061  func[dihedral:3 colours:15 cyclic:2] real[n/a] |C|=20000+ gain=12.70b outs=5884  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
05a7bcf2  func[dihedral:3 colours:6 cyclic:2 rows:17 cols:22] real[n/a] |C|=20021+ gain=12.70b outs=19981  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
05f2a901  func[dihedral:3 colours:3 cyclic:2 rows:4 cols:2] real[n/a] |C|=20000+ gain=12.70b outs=12902  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0607ce86  func[dihedral:3 colours:15 cyclic:2 rows:23 cols:21] real[n/a] |C|=20007+ gain=12.70b outs=6501  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0692e18c  func[dihedral:3 colours:10 rows:6 cols:6] real[n/a] |C|=35 gain=3.54b outs=35  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
06df4c85  func[dihedral:3 colours:21 cyclic:2 rows:25 cols:25] real[n/a] |C|=20005+ gain=12.70b outs=19925  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
070dd51e  func[dihedral:3 colours:36 cyclic:2 rows:28 cols:19] real[n/a] |C|=20001+ gain=13.29b outs=16062  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
08573cc6  func[dihedral:3 colours:21 cyclic:1 rows:1 cols:1] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
08ed6ac7  func[dihedral:3 colours:1 cyclic:2 rows:8 cols:8] real[n/a] |C|=20000+ gain=13.29b outs=15080  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0934a4d8  func[dihedral:3 colours:36 cyclic:2 rows:29 cols:29] real[n/a] |C|=20006+ gain=12.29b outs=13308  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
09629e4f  func[dihedral:3 colours:21 cyclic:2 rows:10 cols:10] real[n/a] |C|=20000+ gain=12.29b outs=7107  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0962bcdd  func[dihedral:3 colours:21 cyclic:2 rows:2 cols:3] real[n/a] |C|=20002+ gain=13.29b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
09c534e7  func[dihedral:3 colours:21 cyclic:2 rows:13 cols:18] real[n/a] |C|=20017+ gain=12.70b outs=20017  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0a1d4ef5  func[dihedral:3 colours:36 cyclic:2 rows:29 cols:29] real[n/a] |C|=20018+ gain=12.70b outs=10909  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0a2355a6  func[dihedral:3 colours:1 cyclic:2 rows:12 cols:16] real[n/a] |C|=20007+ gain=12.29b outs=20007  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0a938d79  func[dihedral:3 colours:15 cyclic:2 cols:2] real[n/a] |C|=20000+ gain=12.29b outs=2915  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0b148d64  func[dihedral:3 colours:15 cyclic:2 rows:18 cols:18] real[n/a] |C|=20011+ gain=12.70b outs=11766  progsD=0 progs=0 fitD=0.67 fitC=0.30 evals=3237 sym=0/1(ok 0) equiv=NO det=-  SOLVED  :: leftHalf ; cropLargest
0b17323b  func[dihedral:3 colours:1] real[n/a] |C|=16 gain=3.00b outs=16  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0bb8deee  func[dihedral:3 colours:28 cyclic:2 rows:12 cols:9] real[n/a] |C|=20000+ gain=12.70b outs=14458  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0becf7df  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20018+ gain=12.70b outs=20018  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0c786b71  func[dihedral:3 colours:36 cyclic:2 rows:3 cols:5] real[dihedral:1 colours:36 rows:3 cols:3] |C|=6048 gain=10.98b outs=6048  progsD=1 progs=1 fitD=1.00 fitC=0.77 evals=1017 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: rot180 ; mirror4
0c9aba6e  func[dihedral:3 colours:6 cyclic:2 rows:11 cols:3] real[dihedral:2 colours:3 cyclic:1 rows:7 cols:3] |C|=1152 gain=8.17b outs=192  progsD=52 progs=46 fitD=1.00 fitC=1.00 evals=3343 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitV nor 8
0ca9ddb6  func[dihedral:3 colours:6] real[n/a] |C|=336 gain=6.81b outs=336  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0d3d703e  func[dihedral:3 colours:4 cyclic:2 rows:2 cols:2] real[n/a] |C|=288 gain=6.17b outs=288  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0d87d2a6  func[dihedral:3 colours:3 cyclic:2 rows:22 cols:24] real[n/a] |C|=20013+ gain=12.70b outs=16062  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0dfd9992  func[dihedral:3 colours:36 cyclic:2 rows:20 cols:20] real[n/a] |C|=20060+ gain=12.71b outs=16602  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0e206a2e  func[dihedral:3 colours:21 cyclic:2 rows:18 cols:13] real[n/a] |C|=20001+ gain=12.70b outs=10791  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
0e671a1a  func[dihedral:3 colours:6 cyclic:2 rows:2 cols:3] real[n/a] |C|=20005+ gain=12.29b outs=20005  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 0) equiv=yes det=-  unfit  :: id
0f63c0b9  func[dihedral:3 colours:28 cyclic:2 rows:3 cols:8] real[n/a] |C|=20000+ gain=12.29b outs=8512  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
103eff5b  func[dihedral:3 colours:15 cyclic:2 rows:23 cols:25] real[n/a] |C|=20004+ gain=13.29b outs=18465  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
10fcaaa3  func[dihedral:3 colours:15 rows:6 cols:4] real[n/a] |C|=288 gain=6.17b outs=288  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
11852cab  func[dihedral:3 colours:15 cyclic:2 rows:8 cols:8] real[n/a] |C|=20004+ gain=12.70b outs=17411  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1190e5a7  func[dihedral:3 colours:15 cyclic:2 rows:26 cols:26] real[n/a] |C|=20003+ gain=12.70b outs=36  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
11e1fe23  func[dihedral:3 colours:21 cyclic:2 rows:3 cols:6] real[n/a] |C|=20001+ gain=13.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
12422b43  func[dihedral:3 colours:36 cyclic:2 rows:2 cols:4] real[n/a] |C|=20001+ gain=11.97b outs=19621  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
12997ef3  func[dihedral:3 colours:28 cyclic:2 rows:4 cols:6] real[n/a] |C|=20003+ gain=12.29b outs=8498  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
12eac192  func[dihedral:3 colours:10 cyclic:2 rows:8 cols:7] real[n/a] |C|=20004+ gain=12.29b outs=16890  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
136b0064  func[dihedral:3 colours:21 cyclic:2 rows:18 cols:9] real[n/a] |C|=20013+ gain=12.70b outs=12992  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
13713586  func[dihedral:3 colours:28 cyclic:2 rows:15 cols:17] real[n/a] |C|=20005+ gain=12.70b outs=16935  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
137eaa0f  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:8] real[n/a] |C|=20000+ gain=12.70b outs=18467  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
137f0df0  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=1000 gain=8.38b outs=1000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 0) equiv=yes det=-  unfit  :: id
140c817e  func[dihedral:3 colours:10 cyclic:2 rows:1 cols:1] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
14754a24  func[dihedral:3 colours:3 cyclic:2 rows:18 cols:18] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
150deff5  func[dihedral:3 colours:1 cyclic:2 rows:8 cols:10] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
15113be4  func[dihedral:3 colours:21 cyclic:2 rows:19 cols:21] real[n/a] |C|=20028+ gain=12.70b outs=20028  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
15663ba9  func[dihedral:3 colours:6 cyclic:2 rows:14 cols:15] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
15696249  func[dihedral:3 colours:15 rows:6 cols:6] real[n/a] |C|=2880 gain=9.49b outs=2880  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
16b78196  func[dihedral:3 colours:21 cyclic:2 rows:15 cols:24] real[n/a] |C|=20003+ gain=13.29b outs=7629  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
178fcbfb  func[dihedral:3 colours:6 rows:9 cols:8] real[n/a] |C|=20000+ gain=12.70b outs=6007  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
17b80ad2  func[dihedral:3 colours:36 cyclic:2 rows:15 cols:16] real[n/a] |C|=20031+ gain=12.29b outs=19192  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
17cae0c1  func[dihedral:1 colours:1 cyclic:2 cols:3] real[n/a] |C|=20001+ gain=12.29b outs=2014  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
18419cfa  func[dihedral:3 colours:3 cyclic:2 rows:11 cols:15] real[n/a] |C|=20002+ gain=12.70b outs=19189  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
184a9768  func[dihedral:3 colours:36 cyclic:2 rows:24 cols:25] real[n/a] |C|=20056+ gain=12.71b outs=12140  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
195ba7dc  func[dihedral:3 colours:3 cyclic:2 rows:2 cols:10] real[dihedral:2 colours:3 cyclic:1 rows:2 cols:7] |C|=5760 gain=10.49b outs=720  progsD=32 progs=29 fitD=1.00 fitC=1.00 evals=1798 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitH or 1
1990f7a8  func[dihedral:3 colours:1 cyclic:2 rows:16 cols:16] real[n/a] |C|=17984 gain=12.55b outs=2352  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
19bb5feb  func[dihedral:3 colours:28 cyclic:2 rows:14 cols:14] real[n/a] |C|=20002+ gain=12.70b outs=1456  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1a07d186  func[dihedral:3 colours:15 cyclic:2 rows:6 cols:15] real[n/a] |C|=20003+ gain=12.70b outs=4917  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1a2e2828  func[dihedral:3 colours:36 cyclic:2 rows:10 cols:12] real[n/a] |C|=20011+ gain=11.97b outs=9  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1a6449f1  func[dihedral:3 colours:36 cyclic:2 rows:26 cols:24] real[n/a] |C|=20004+ gain=12.70b outs=19297  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1acc24af  func[dihedral:3 colours:3 cyclic:2 rows:11 cols:11] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1b2d62fb  func[dihedral:3 colours:3 cyclic:2 rows:4 cols:6] real[dihedral:2 colours:1 cyclic:1 rows:4 cols:4] |C|=1440 gain=8.17b outs=190  progsD=34 progs=27 fitD=1.00 fitC=1.00 evals=1801 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitH nor 8
1b60fb0c  func[dihedral:3 colours:1 cyclic:2 rows:2 cols:2] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1bfc4729  func[dihedral:2 colours:21 rows:6 cols:7] real[n/a] |C|=20000+ gain=13.29b outs=672  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
1c02dbbe  func[dihedral:3 colours:28 cyclic:2 rows:12 cols:13] real[n/a] |C|=20004+ gain=12.70b outs=9287  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1c0d0a4b  func[dihedral:3 colours:1 cyclic:2 rows:12 cols:12] real[n/a] |C|=20011+ gain=12.70b outs=15660  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1c56ad9f  func[dihedral:3 colours:15 rows:1 cols:1] real[n/a] |C|=288 gain=6.17b outs=288  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1c786137  func[dihedral:3 colours:36 cyclic:2 rows:22 cols:20] real[n/a] |C|=20001+ gain=12.70b outs=14307  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1caeab9d  func[dihedral:3 colours:6 cyclic:2 rows:9 cols:9] real[n/a] |C|=20002+ gain=12.70b outs=9680  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1cf80156  func[dihedral:3 colours:10 cyclic:2 rows:7 cols:7] real[dihedral:3 colours:10 rows:7 cols:6] |C|=120 gain=5.32b outs=120  progsD=100 progs=58 fitD=1.00 fitC=1.00 evals=3803 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: cropBBox
1d0a4b61  func[dihedral:3 colours:21 cyclic:2 rows:24 cols:24] real[n/a] |C|=20017+ gain=12.70b outs=12329  progsD=0 progs=0 fitD=0.67 fitC=0.37 evals=3239 sym=0/1(ok 0) equiv=NO det=-  unfit  :: symmetrizeH ; recolour 0 1
1d398264  func[dihedral:3 colours:36 cyclic:2 cols:4] real[n/a] |C|=20011+ gain=12.70b outs=20011  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
1da012fc  func[dihedral:3 colours:28 cyclic:2 rows:18 cols:24] real[n/a] |C|=20007+ gain=13.29b outs=17851  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1e0a9b12  func[dihedral:3 colours:36 cyclic:2 rows:5 cols:4] real[dihedral:1 colours:36 cyclic:1 rows:1 cols:4] |C|=20001+ gain=12.70b outs=13617  progsD=45 progs=22 fitD=1.00 fitC=1.00 evals=3585 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: gravityDown
1e32b0e9  func[dihedral:3 colours:21 cyclic:2 rows:5 cols:4] real[n/a] |C|=20003+ gain=12.70b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1e81d6f9  func[dihedral:3 colours:36 cyclic:2 rows:14 cols:14] real[n/a] |C|=20023+ gain=12.70b outs=18540  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1e97544e  func[dihedral:3 colours:36 cyclic:2 rows:20 cols:22] real[n/a] |C|=20020+ gain=12.70b outs=20020  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1f0c79e5  func[dihedral:3 colours:21] real[n/a] |C|=1008 gain=7.98b outs=168  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1f642eb9  func[dihedral:3 colours:36 cyclic:2 rows:6 cols:7] real[n/a] |C|=20010+ gain=12.70b outs=20010  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1f85a75f  func[dihedral:3 colours:21 cyclic:2 rows:29 cols:29] real[dihedral:3 colours:21 rows:25 cols:25] |C|=7056 gain=11.78b outs=140  progsD=70 progs=11 fitD=1.00 fitC=0.55 evals=3916 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: cropLargest
1f876c06  func[dihedral:3 colours:28 cyclic:2 rows:3 cols:4] real[n/a] |C|=20001+ gain=12.70b outs=19219  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
1fad071e  func[dihedral:3 colours:3 cyclic:2 rows:7 cols:7] real[n/a] |C|=20001+ gain=12.70b outs=90  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2013d3e2  func[dihedral:1 colours:36 cyclic:2 rows:7 cols:7] real[n/a] |C|=20001+ gain=13.29b outs=19921  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2037f2c7  func[dihedral:3 colours:21 cyclic:2 rows:20 cols:19] real[n/a] |C|=20000+ gain=12.70b outs=405  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2072aba6  func[dihedral:1 colours:1 rows:4 cols:4] real[n/a] |C|=28 gain=3.22b outs=28  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
20818e16  func[dihedral:3 colours:28 cyclic:2 rows:8 cols:8] real[n/a] |C|=20000+ gain=12.70b outs=10022  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
20981f0e  func[dihedral:3 colours:3 cyclic:2 rows:13 cols:15] real[n/a] |C|=20007+ gain=12.70b outs=19356  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
212895b5  func[dihedral:3 colours:3 cyclic:2 rows:6 cols:13] real[n/a] |C|=20006+ gain=12.70b outs=20006  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
21f83797  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=2704 gain=10.40b outs=1352  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 0) equiv=yes det=-  unfit  :: id
2204b7a8  func[dihedral:3 colours:36 cyclic:2 rows:8 cols:8] real[n/a] |C|=20000+ gain=12.70b outs=12352  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
22168020  func[dihedral:3 colours:21 cyclic:2 rows:7 cols:6] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
22233c11  func[dihedral:3 colours:1] real[n/a] |C|=40 gain=3.74b outs=40  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2281f1f4  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
228f6490  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20002+ gain=12.70b outs=17800  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
22a4bbc2  func[dihedral:3 colours:3 cyclic:2 rows:17 cols:4] real[n/a] |C|=20002+ gain=12.29b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
22eb0ac0  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20006+ gain=12.70b outs=20006  progsD=0 progs=0 fitD=0.33 fitC=0.34 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
234bbc79  func[dihedral:3 colours:21 cyclic:2 rows:2 cols:6] real[n/a] |C|=20002+ gain=12.29b outs=9869  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
23581191  func[dihedral:3 colours:3 cyclic:2 rows:8 cols:8] real[n/a] |C|=15552 gain=12.92b outs=15552  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
239be575  func[dihedral:3 colours:3 cyclic:2 rows:7 cols:6] real[n/a] |C|=10656 gain=10.79b outs=4  progsD=0 progs=0 fitD=0.50 fitC=0.53 evals=3233 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: cropSmallest
23b5c85d  func[dihedral:3 colours:28 cyclic:2 rows:19 cols:19] real[n/a] |C|=20000+ gain=11.97b outs=48  progsD=0 progs=0 fitD=0.40 fitC=0.45 evals=3230 sym=0/1(ok 0) equiv=NO det=-  unfit  :: cropSmallest
25094a63  func[dihedral:3 colours:15 cyclic:2 rows:29 cols:29] real[n/a] |C|=20013+ gain=13.29b outs=20013  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
253bf280  func[dihedral:3 colours:1 rows:2 cols:6] real[n/a] |C|=20000+ gain=11.29b outs=20000  progsD=0 progs=0 fitD=0.25 fitC=0.22 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2546ccf6  func[dihedral:3 colours:21 cyclic:2 rows:25 cols:22] real[n/a] |C|=20003+ gain=13.29b outs=19021  progsD=0 progs=0 fitD=0.50 fitC=0.21 evals=3228 sym=0/1(ok 0) equiv=NO det=-  unfit  :: symmetrizeH
256b0a75  func[dihedral:3 colours:36 cyclic:2 rows:23 cols:25] real[n/a] |C|=20051+ gain=12.71b outs=17218  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
25d487eb  func[dihedral:3 colours:15 cyclic:2 rows:12 cols:11] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
25d8a9c8  func[dihedral:3 colours:28 cyclic:2 rows:2 cols:2] real[n/a] |C|=20002+ gain=12.29b outs=12  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
25ff71a9  func[dihedral:1 colours:3 cyclic:2 cols:2] real[dihedral:1 colours:3 cyclic:1 cols:2] |C|=42 gain=3.39b outs=42  progsD=6 progs=6 fitD=1.00 fitC=1.00 evals=243 sym=0/2(ok 0) equiv=yes,yes det=yes,yes  SOLVED  :: shiftDown
264363fd  func[dihedral:3 colours:28 cyclic:2 rows:14 cols:16] real[n/a] |C|=20032+ gain=12.71b outs=14824  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2685904e  func[dihedral:3 colours:36 cyclic:2 rows:4 cols:8] real[n/a] |C|=20013+ gain=11.70b outs=20013  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2697da3f  func[dihedral:3 colours:1 rows:6 cols:6] real[n/a] |C|=56 gain=3.81b outs=8  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
272f95fa  func[dihedral:1 colours:1 cyclic:1 cols:12] real[n/a] |C|=1368 gain=9.42b outs=1368  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2753e76c  func[dihedral:3 colours:15 cyclic:2 rows:13 cols:12] real[n/a] |C|=20000+ gain=12.70b outs=18349  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
27a28665  func[dihedral:3 colours:10 rows:2 cols:2] real[n/a] |C|=54 gain=2.95b outs=4  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=2/3(ok 2) equiv=yes,yes,yes det=-,-,-  unfit  :: id
27a77e38  func[dihedral:3 colours:36 rows:5 cols:5] real[n/a] |C|=20005+ gain=12.70b outs=20005  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
27f8ce4f  func[dihedral:3 colours:36 cyclic:2 rows:6 cols:6] real[n/a] |C|=20006+ gain=12.29b outs=20006  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
281123b4  func[dihedral:3 colours:15 cyclic:2 rows:3 cols:17] real[n/a] |C|=20005+ gain=11.70b outs=16243  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
28bf18c6  func[dihedral:3 colours:10 rows:2 cols:2] real[dihedral:2 colours:10 rows:2 cols:2] |C|=60 gain=4.32b outs=60  progsD=2 progs=2 fitD=1.00 fitC=1.00 evals=220 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: cropBBox ; hconcat
28e73c20  func[rows:3 cols:3] real[n/a] |C|=5 gain=0.00b outs=5  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
292dd178  func[dihedral:3 colours:15 cyclic:2 rows:8 cols:13] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
29623171  func[dihedral:3 colours:15 cyclic:2 rows:8 cols:9] real[n/a] |C|=20011+ gain=12.70b outs=4804  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
29700607  func[dihedral:3 colours:21 cyclic:2 rows:10 cols:14] real[n/a] |C|=20002+ gain=12.70b outs=13843  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
29c11459  func[dihedral:3 colours:28 cyclic:2 rows:1 cols:2] real[n/a] |C|=20002+ gain=13.29b outs=6148  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
29ec7d0e  func[dihedral:3 colours:36 cyclic:2 rows:17 cols:17] real[n/a] |C|=20037+ gain=12.29b outs=17509  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2a5f8217  func[dihedral:3 colours:28 cyclic:2 rows:12 cols:12] real[n/a] |C|=20000+ gain=12.70b outs=14998  progsD=0 progs=0 fitD=0.33 fitC=0.12 evals=3230 sym=0/1(ok 0) equiv=NO det=-  unfit  :: recolour 1 8
2b01abd0  func[dihedral:3 colours:28 cyclic:2 rows:4 cols:6] real[n/a] |C|=20000+ gain=12.70b outs=19654  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2bcee788  func[dihedral:3 colours:21 cyclic:2 rows:3 cols:3] real[n/a] |C|=20001+ gain=12.29b outs=6246  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2bee17df  func[dihedral:3 colours:3 cyclic:2 rows:13 cols:13] real[n/a] |C|=20006+ gain=12.70b outs=20006  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2c0b0aff  func[dihedral:3 colours:3 cyclic:2 rows:18 cols:17] real[n/a] |C|=20000+ gain=12.29b outs=19390  progsD=0 progs=0 fitD=0.50 fitC=0.13 evals=3234 sym=0/1(ok 0) equiv=NO det=-  unfit  :: bottomHalf ; cropSmallest
2c608aff  func[dihedral:3 colours:21 cyclic:1 rows:16 cols:19] real[n/a] |C|=20002+ gain=12.29b outs=17951  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2c737e39  func[dihedral:3 colours:28 cyclic:2 cols:5] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2dc579da  func[dihedral:3 colours:15 cyclic:2 rows:9 cols:8] real[n/a] |C|=20005+ gain=12.70b outs=952  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2dd70a9a  func[dihedral:3 colours:6 cyclic:2 rows:19 cols:19] real[n/a] |C|=20008+ gain=12.70b outs=20008  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2dee498d  func[dihedral:3 colours:15 cyclic:2 rows:4 cols:14] real[n/a] |C|=20000+ gain=12.70b outs=5910  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
2f0c5170  func[dihedral:3 colours:15 cyclic:2 rows:13 cols:15] real[n/a] |C|=20001+ gain=12.70b outs=6232  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
310f3251  func[dihedral:2 colours:21 rows:11 cols:11] real[n/a] |C|=224 gain=5.49b outs=224  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3194b014  func[dihedral:3 colours:36 cyclic:2 rows:19 cols:19] real[n/a] |C|=20010+ gain=12.70b outs=9  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
319f2597  func[dihedral:3 colours:36 cyclic:2 rows:19 cols:19] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
31aa019c  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20010+ gain=12.70b outs=1951  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
31adaf00  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20011+ gain=12.70b outs=20011  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
31d5ba1a  func[dihedral:3 colours:3 cyclic:2 rows:5 cols:4] real[dihedral:2 colours:1 cyclic:2 rows:3 cols:4] |C|=14400 gain=11.49b outs=1620  progsD=33 progs=29 fitD=1.00 fitC=1.00 evals=1656 sym=0/2(ok 0) equiv=yes,yes det=yes,yes  SOLVED  :: splitV xor 6
321b1fc6  func[dihedral:3 colours:21 cyclic:2 rows:2 cols:2] real[n/a] |C|=20000+ gain=13.29b outs=12358  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
32597951  func[dihedral:3 colours:3 cyclic:2 rows:16 cols:16] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
32e9702f  func[dihedral:1 colours:10 cyclic:1 rows:9 cols:6] real[dihedral:1 colours:10 cyclic:1 rows:9] |C|=1345 gain=8.81b outs=1345  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=513 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: shiftLeft ; recolour 0 5
332efdb3  func[dihedral:3 rows:2 cols:2] real[n/a] |C|=3 gain=0.00b outs=3  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3345333e  func[dihedral:3 colours:21 cyclic:2 rows:15 cols:15] real[n/a] |C|=20003+ gain=13.29b outs=9587  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3391f8c0  func[dihedral:3 colours:28 cyclic:2 rows:7 cols:15] real[n/a] |C|=20004+ gain=12.29b outs=19589  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
33b52de3  func[dihedral:3 colours:21 cyclic:2 rows:20 cols:22] real[n/a] |C|=20021+ gain=13.29b outs=16129  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3428a4f5  func[dihedral:3 colours:3 cyclic:2 rows:11 cols:4] real[dihedral:2 colours:1 cyclic:1 rows:7 cols:4] |C|=1920 gain=8.91b outs=960  progsD=35 progs=32 fitD=1.00 fitC=1.00 evals=2458 sym=0/2(ok 0) equiv=yes,yes det=yes,NO  SOLVED  :: splitV xor 3
3490cc26  func[dihedral:3 colours:3 cyclic:2 rows:22 cols:27] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
34b99a2b  func[dihedral:3 colours:6 cyclic:2 rows:3 cols:7] real[dihedral:2 colours:3 cyclic:1 rows:3 cols:5] |C|=5760 gain=10.49b outs=720  progsD=47 progs=45 fitD=1.00 fitC=1.00 evals=2658 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitH xor 2
351d6448  func[dihedral:3 colours:10 cyclic:2 rows:7 cols:5] real[n/a] |C|=20004+ gain=13.29b outs=1465  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
358ba94e  func[dihedral:3 colours:15 cyclic:2 rows:13 cols:15] real[n/a] |C|=20000+ gain=12.29b outs=3988  progsD=0 progs=0 fitD=0.75 fitC=0.39 evals=3232 sym=0/1(ok 0) equiv=NO det=-  SOLVED  :: cropSmallest
3618c87e  func[dihedral:3 colours:3 cyclic:2 rows:4 cols:4] real[n/a] |C|=10800 gain=11.81b outs=3600  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
3631a71a  func[dihedral:3 colours:36 cyclic:2 rows:29 cols:29] real[n/a] |C|=20000+ gain=12.29b outs=19419  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
363442ee  func[dihedral:3 colours:36 cyclic:2 rows:4 cols:6] real[n/a] |C|=20005+ gain=12.70b outs=15344  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
36d67576  func[dihedral:3 colours:10 cyclic:2 rows:4 cols:4] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
36fdfd69  func[dihedral:3 colours:15 cyclic:2 rows:16 cols:17] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
37d3e8b2  func[dihedral:3 colours:1 cyclic:2 rows:18 cols:17] real[n/a] |C|=20000+ gain=12.70b outs=19963  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3906de3d  func[dihedral:3 colours:3 cyclic:2 rows:5 cols:7] real[n/a] |C|=20002+ gain=12.70b outs=15931  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3979b1a8  func[dihedral:1 colours:21 cyclic:2 rows:6 cols:7] real[n/a] |C|=20000+ gain=13.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
39a8645d  func[dihedral:3 colours:21 cyclic:2 rows:11 cols:11] real[n/a] |C|=20001+ gain=12.70b outs=189  progsD=0 progs=0 fitD=0.33 fitC=0.30 evals=3231 sym=0/1(ok 0) equiv=NO det=-  SOLVED  :: cropLargest
39e1d7f9  func[dihedral:3 colours:15 cyclic:2 rows:28 cols:28] real[n/a] |C|=20023+ gain=12.70b outs=20023  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3a301edc  func[dihedral:3 colours:28 cyclic:2 rows:6 cols:7] real[n/a] |C|=20002+ gain=11.97b outs=20002  progsD=0 progs=0 fitD=0.20 fitC=0.13 evals=3232 sym=0/1(ok 0) equiv=NO det=-  unfit  :: box 1
3aa6fb7a  func[dihedral:3 colours:1 cyclic:2 rows:6 cols:6] real[n/a] |C|=20001+ gain=13.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3ac3eb23  func[dihedral:3 colours:21 cyclic:2 cols:2] real[n/a] |C|=5628 gain=11.46b outs=1876  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3af2c5a8  func[dihedral:3 colours:6 rows:3 cols:5] real[dihedral:1 colours:6 rows:2 cols:4] |C|=24 gain=3.00b outs=24  progsD=16 progs=8 fitD=1.00 fitC=1.00 evals=908 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: mirror4
3b4c2228  func[dihedral:3 colours:3 rows:5 cols:5] real[n/a] |C|=240 gain=5.58b outs=10  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
3bd67248  func[dihedral:1 colours:10] real[n/a] |C|=30 gain=3.32b outs=30  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3bdb4ada  func[dihedral:3 colours:15 cyclic:2] real[n/a] |C|=20000+ gain=13.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3befdf3e  func[dihedral:3 colours:28 cyclic:2 rows:3 cols:3] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3c9b0459  func[dihedral:3 colours:36 cyclic:1 cols:1] real[dihedral:3 colours:36] |C|=12672 gain=11.63b outs=12672  progsD=117 progs=77 fitD=1.00 fitC=1.00 evals=6998 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: rot180
3d31c5b3  func[dihedral:3 colours:10 cyclic:2 rows:11 cols:5] real[n/a] |C|=20001+ gain=11.70b outs=13046  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3de23699  func[dihedral:3 colours:15 cyclic:2 rows:13 cols:13] real[n/a] |C|=20001+ gain=12.29b outs=3732  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3e980e27  func[dihedral:3 colours:15 cyclic:2 rows:1] real[n/a] |C|=20004+ gain=12.29b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3ed85e70  func[dihedral:3 colours:28 cyclic:2 rows:12 cols:12] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3eda0437  func[dihedral:3 colours:3 cyclic:2 rows:3 cols:29] real[n/a] |C|=20002+ gain=12.29b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3ee1011a  func[dihedral:3 colours:28 cyclic:2 rows:16 cols:21] real[n/a] |C|=20001+ gain=12.70b outs=8122  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3f23242b  func[dihedral:2 colours:1 cyclic:1 rows:3 cols:3] real[n/a] |C|=104 gain=5.70b outs=104  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
3f7978a0  func[dihedral:3 colours:3 cyclic:2 rows:9 cols:6] real[n/a] |C|=20000+ gain=12.70b outs=7912  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
40853293  func[dihedral:3 colours:36 cyclic:2 rows:28 cols:19] real[n/a] |C|=20001+ gain=13.29b outs=16062  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4093f84a  func[dihedral:3 colours:15 cyclic:2 rows:12 cols:13] real[n/a] |C|=20001+ gain=12.70b outs=7637  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
40f6cd08  func[dihedral:3 colours:21 cyclic:2 rows:4 cols:4] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
414297c0  func[dihedral:3 colours:28 cyclic:2 rows:19 cols:19] real[n/a] |C|=20006+ gain=12.70b outs=17861  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
41e4d17e  func[dihedral:3 colours:3 cyclic:2 cols:1] real[n/a] |C|=20000+ gain=13.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
423a55dc  func[dihedral:1 colours:15 cyclic:1 rows:6 cols:3] real[n/a] |C|=20000+ gain=11.97b outs=19068  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4258a5f9  func[dihedral:3 colours:1 cyclic:2] real[dihedral:3] |C|=16 gain=3.00b outs=16  progsD=11 progs=10 fitD=1.00 fitC=1.00 evals=554 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: box 1
4290ef0e  func[dihedral:3 colours:28 cyclic:2 rows:12 cols:13] real[n/a] |C|=20011+ gain=12.70b outs=12022  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
42918530  func[dihedral:3 colours:36 cyclic:2 rows:16 cols:18] real[n/a] |C|=20021+ gain=12.29b outs=19862  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
42a15761  func[dihedral:3 colours:1 cyclic:2 rows:10 cols:18] real[n/a] |C|=20011+ gain=12.70b outs=20011  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
42a50994  func[dihedral:3 colours:15 cyclic:2 rows:16 cols:18] real[n/a] |C|=20006+ gain=12.29b outs=10501  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4347f46a  func[dihedral:3 colours:36 cyclic:2 rows:3 cols:4] real[n/a] |C|=20004+ gain=12.70b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4364c1c4  func[dihedral:3 colours:28 cyclic:2 rows:4] real[n/a] |C|=20000+ gain=12.70b outs=19968  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
444801d8  func[dihedral:3 colours:28 cyclic:2 rows:3 cols:7] real[n/a] |C|=20003+ gain=12.70b outs=17645  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
445eab21  func[dihedral:3 colours:28 cyclic:2 rows:9 cols:9] real[n/a] |C|=20001+ gain=12.70b outs=8  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
447fd412  func[dihedral:3 colours:3 cyclic:2 rows:4 cols:7] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
44d8ac46  func[dihedral:3 colours:1 cyclic:2 rows:11 cols:11] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.25 fitC=0.24 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
44f52bb0  func[dihedral:3 colours:1 cyclic:2 rows:2 cols:2] real[n/a] |C|=282 gain=5.55b outs=3  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
4522001f  func[dihedral:3 colours:3 rows:6 cols:6] real[n/a] |C|=24 gain=3.58b outs=12  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
456873bc  func[dihedral:3 colours:3 cyclic:2 rows:7 cols:8] real[n/a] |C|=20001+ gain=12.70b outs=19814  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
45737921  func[dihedral:3 colours:28 cyclic:2 rows:12 cols:11] real[n/a] |C|=20007+ gain=12.70b outs=20007  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
45bbe264  func[dihedral:3 colours:21 cyclic:2 rows:15 cols:15] real[n/a] |C|=20004+ gain=12.70b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4612dd53  func[dihedral:3 colours:1 cyclic:2 rows:12 cols:12] real[n/a] |C|=20011+ gain=12.70b outs=20011  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
46442a0e  func[dihedral:3 colours:11 rows:3 cols:3] real[n/a] |C|=380 gain=6.98b outs=100  progsD=0 progs=0 fitD=0.67 fitC=0.16 evals=3229 sym=0/1(ok 0) equiv=NO det=-  unfit  :: mirror4
469497ad  func[dihedral:3 colours:28 rows:16 cols:16] real[n/a] |C|=20003+ gain=12.70b outs=14742  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
46f33fce  func[dihedral:3 colours:21 cyclic:2 rows:17 cols:17] real[n/a] |C|=20011+ gain=12.70b outs=5832  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
477d2879  func[dihedral:3 colours:36 cyclic:2 rows:12 cols:12] real[n/a] |C|=20002+ gain=12.70b outs=13010  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
47996f11  func[dihedral:3 colours:36 cyclic:2 rows:29 cols:29] real[n/a] |C|=20028+ gain=12.29b outs=19339  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
47c1f68c  func[dihedral:3 colours:15 cyclic:2 rows:3 cols:3] real[n/a] |C|=20000+ gain=12.70b outs=1502  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
48131b3c  func[dihedral:3 colours:10 cyclic:2 rows:4 cols:4] real[n/a] |C|=830 gain=8.11b outs=830  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
484b58aa  func[dihedral:3 colours:36 cyclic:2 rows:28 cols:28] real[n/a] |C|=20037+ gain=12.71b outs=20009  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4852f2fa  func[dihedral:3 colours:3 cyclic:2 rows:2 cols:5] real[n/a] |C|=20003+ gain=11.97b outs=3618  progsD=0 progs=0 fitD=0.20 fitC=0.10 evals=3232 sym=0/2(ok 0) equiv=NO,NO det=-,-  unfit  :: cropLargest
48d8fb45  func[dihedral:3 colours:15 cyclic:2 rows:7 cols:7] real[n/a] |C|=20000+ gain=12.70b outs=648  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
48f8583b  func[dihedral:3 colours:36 rows:7 cols:7] real[n/a] |C|=20000+ gain=11.70b outs=16740  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4938f0c2  func[dihedral:3 colours:3 cyclic:2 cols:8] real[n/a] |C|=20001+ gain=12.70b outs=17881  progsD=0 progs=0 fitD=0.33 fitC=0.32 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
496994bd  func[dihedral:3 colours:10 rows:3 cols:2] real[dihedral:3 colours:10 rows:3 cols:2] |C|=160 gain=6.32b outs=80  progsD=32 progs=2 fitD=1.00 fitC=1.00 evals=220 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: symmetrizeH ; symmetrizeV
49d1d64f  func[dihedral:3 colours:15 rows:3 cols:3] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4aab4007  func[dihedral:3 colours:36 cyclic:2 rows:27 cols:27] real[n/a] |C|=20040+ gain=12.71b outs=19401  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4acc7107  func[dihedral:3 colours:36 cyclic:2 rows:4 cols:4] real[n/a] |C|=20002+ gain=12.29b outs=14179  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4b6b68e5  func[dihedral:3 colours:36 cyclic:2 rows:25 cols:23] real[n/a] |C|=20005+ gain=12.70b outs=16344  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4be741c5  func[dihedral:3 colours:28 cyclic:2 rows:11 cols:13] real[n/a] |C|=20002+ gain=12.70b outs=3356  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4c177718  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:12] real[n/a] |C|=20012+ gain=12.29b outs=3768  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
4c4377d9  func[dihedral:1 colours:15 cyclic:1 rows:4] real[dihedral:1 colours:15 cyclic:1 rows:3] |C|=840 gain=7.71b outs=840  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=195 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: flipV ; vconcatFlip
4c5c2cf0  func[dihedral:3 colours:21 cyclic:2 rows:1 cols:3] real[n/a] |C|=20001+ gain=12.70b outs=16932  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4cd1b7b2  func[dihedral:3 colours:10 cyclic:2 rows:3 cols:3] real[n/a] |C|=20003+ gain=12.70b outs=2070  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4e45f183  func[dihedral:3 colours:21 cyclic:2 rows:13 cols:13] real[n/a] |C|=20001+ gain=12.70b outs=9566  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4e469f39  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=4800 gain=10.64b outs=4800  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4f537728  func[dihedral:3 colours:10 cyclic:2 rows:19 cols:19] real[n/a] |C|=20012+ gain=13.29b outs=20012  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
4ff4c9da  func[dihedral:3 colours:6 cyclic:2 rows:26 cols:26] real[n/a] |C|=20018+ gain=12.70b outs=18855  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
505fff84  func[dihedral:3 colours:6 cyclic:2 rows:10 cols:11] real[n/a] |C|=20002+ gain=11.97b outs=6116  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
506d28a5  func[dihedral:3 colours:6 cyclic:2 rows:7 cols:3] real[dihedral:2 colours:3 cyclic:1 rows:5 cols:3] |C|=5760 gain=10.49b outs=260  progsD=49 progs=45 fitD=1.00 fitC=1.00 evals=2690 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitV or 3
50846271  func[dihedral:3 colours:3 cyclic:2 rows:19 cols:21] real[n/a] |C|=20010+ gain=12.29b outs=20010  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
508bd3b6  func[dihedral:3 colours:3 cyclic:2] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
50a16a69  func[dihedral:3 colours:36 cyclic:2 rows:3 cols:3] real[n/a] |C|=20004+ gain=12.70b outs=8186  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
50aad11f  func[dihedral:3 colours:28 cyclic:2 rows:11 cols:11] real[n/a] |C|=20000+ gain=12.70b outs=9808  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
50cb2852  func[dihedral:3 colours:6 cyclic:2] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
50f325b5  func[dihedral:3 colours:15 cyclic:2 rows:17 cols:17] real[n/a] |C|=20007+ gain=12.29b outs=20007  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5117e062  func[dihedral:3 colours:28 cyclic:2 rows:11 cols:10] real[n/a] |C|=20004+ gain=12.70b outs=380  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5168d44c  func[dihedral:3 colours:3 cyclic:2 rows:4 cols:10] real[n/a] |C|=20000+ gain=12.70b outs=15951  progsD=0 progs=0 fitD=0.33 fitC=0.17 evals=3231 sym=0/1(ok 0) equiv=NO det=-  unfit  :: flipV
516b51b7  func[dihedral:3 colours:1 cyclic:2 cols:2] real[n/a] |C|=5040 gain=10.71b outs=5040  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5207a7b5  func[dihedral:1 colours:1 cyclic:1 rows:4 cols:1] real[n/a] |C|=124 gain=5.37b outs=124  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5289ad53  func[dihedral:3 colours:15 cyclic:2 rows:12 cols:16] real[n/a] |C|=20001+ gain=12.29b outs=2015  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
52fd389e  func[dihedral:3 colours:21 cyclic:2 rows:7 cols:6] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
539a4f51  func[dihedral:3 colours:21 cyclic:2 rows:5 cols:5] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
53b68214  func[dihedral:1 colours:15 cyclic:1 rows:9 cols:3] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.33 fitC=0.19 evals=3233 sym=0/2(ok 0) equiv=NO,NO det=-,-  unfit  :: tile 2 1
543a7ed5  func[dihedral:3 colours:3 cyclic:2] real[n/a] |C|=20000+ gain=13.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
54d82841  func[dihedral:3 colours:10 cyclic:2 rows:1 cols:6] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
54d9e175  func[dihedral:3 colours:6 cyclic:2 rows:1 cols:4] real[n/a] |C|=20000+ gain=12.29b outs=4280  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
54db823b  func[dihedral:3 colours:3 cyclic:2 rows:14 cols:14] real[n/a] |C|=20001+ gain=12.29b outs=19625  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
55059096  func[dihedral:3 colours:1 cyclic:2 rows:4 cols:5] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
551d5bf1  func[dihedral:3 colours:1 cyclic:2 rows:22 cols:26] real[n/a] |C|=20003+ gain=13.29b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5521c0d9  func[dihedral:3 colours:6 cyclic:2 cols:1] real[n/a] |C|=20001+ gain=12.70b outs=18368  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
55783887  func[dihedral:3 colours:15 cyclic:2 rows:3 cols:1] real[n/a] |C|=20000+ gain=11.97b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5582e5ca  func[dihedral:3 colours:21 cyclic:2 rows:2 cols:2] real[dihedral:3 colours:21 cyclic:2 rows:2 cols:2] |C|=20003+ gain=12.70b outs=7  progsD=103 progs=53 fitD=1.00 fitC=1.00 evals=3872 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: majorityFill
5614dbcf  func[dihedral:3 colours:36 cyclic:2 rows:7 cols:7] real[n/a] |C|=20002+ gain=13.29b outs=9111  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
56dc2b01  func[dihedral:3 colours:3 cyclic:2 rows:1 cols:4] real[n/a] |C|=20002+ gain=12.70b outs=10374  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
56ff96f3  func[dihedral:3 colours:28 cyclic:2 rows:4 cols:6] real[n/a] |C|=20002+ gain=12.29b outs=12878  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
575b1a71  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5783df64  func[dihedral:3 colours:36 cyclic:2 rows:8 cols:8] real[n/a] |C|=20006+ gain=12.70b outs=16909  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
57aa92db  func[dihedral:3 colours:21 cyclic:2 rows:3 cols:13] real[n/a] |C|=20002+ gain=12.29b outs=19977  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5833af48  func[dihedral:3 colours:15 cyclic:2 rows:6 cols:12] real[n/a] |C|=20001+ gain=12.70b outs=5160  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
58743b76  func[dihedral:3 colours:28 cyclic:2 rows:13 cols:13] real[n/a] |C|=20000+ gain=13.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
58e15b12  func[dihedral:3 colours:3 cyclic:2] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
59341089  func[dihedral:2 colours:6 cols:9] real[n/a] |C|=336 gain=6.39b outs=168  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5a5a2103  func[dihedral:3 colours:36 cyclic:2 rows:13 cols:11] real[n/a] |C|=20006+ gain=13.29b outs=15389  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5ad4f10b  func[dihedral:3 colours:21 cyclic:2 rows:22 cols:25] real[n/a] |C|=20000+ gain=12.70b outs=744  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5af49b42  func[dihedral:3 colours:36 cyclic:2 rows:13 cols:9] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5b526a93  func[dihedral:3 colours:1 cyclic:2 rows:21 cols:29] real[n/a] |C|=20003+ gain=13.29b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5b692c0f  func[dihedral:3 colours:15 cyclic:2 rows:16 cols:18] real[n/a] |C|=20005+ gain=13.29b outs=18152  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5b6cbef5  func[dihedral:3 colours:10 rows:12 cols:12] real[n/a] |C|=200 gain=5.32b outs=200  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5bd6f4ac  func[dihedral:3 colours:36 cyclic:2 rows:8 cols:8] real[n/a] |C|=20023+ gain=12.29b outs=8161  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5c0a986e  func[dihedral:3 colours:3 cyclic:2] real[n/a] |C|=7200 gain=11.23b outs=7200  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 0) equiv=yes det=-  unfit  :: id
5c2c9af4  func[dihedral:3 colours:10 rows:5 cols:5] real[n/a] |C|=100 gain=5.06b outs=80  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5d2a5c43  func[dihedral:3 colours:3 cyclic:2 rows:4 cols:6] real[dihedral:2 colours:1 cyclic:1 rows:4 cols:5] |C|=14400 gain=11.49b outs=2520  progsD=34 progs=29 fitD=1.00 fitC=1.00 evals=1705 sym=0/2(ok 0) equiv=yes,yes det=NO,yes  SOLVED  :: splitH or 8
5daaa586  func[dihedral:3 colours:21 cyclic:2 rows:14 cols:12] real[n/a] |C|=20001+ gain=12.70b outs=13134  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
5ffb2104  func[dihedral:3 colours:21 cyclic:2 rows:5 cols:5] real[n/a] |C|=20012+ gain=12.70b outs=14944  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
604001fa  func[dihedral:3 colours:3 cyclic:2 rows:19 cols:15] real[n/a] |C|=20012+ gain=12.29b outs=10483  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
60a26a3e  func[dihedral:3 colours:1 cyclic:2 rows:13 cols:15] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
60b61512  func[dihedral:3 colours:1 cyclic:2 rows:8 cols:8] real[n/a] |C|=20001+ gain=13.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
60c09cac  func[dihedral:3 colours:21 rows:5 cols:5] real[dihedral:3 colours:21 rows:3 cols:3] |C|=672 gain=8.39b outs=672  progsD=43 progs=13 fitD=1.00 fitC=1.00 evals=1482 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: scale 2
6150a2bd  func[dihedral:3 colours:36 cyclic:2 rows:2 cols:2] real[dihedral:3 colours:36] |C|=16128 gain=12.98b outs=16128  progsD=62 progs=25 fitD=1.00 fitC=1.00 evals=3433 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: rot180
623ea044  func[dihedral:3 colours:10 rows:2 cols:2] real[n/a] |C|=80 gain=4.74b outs=80  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
626c0bcc  func[dihedral:2 colours:1 cyclic:2 rows:6 cols:4] real[n/a] |C|=20005+ gain=12.70b outs=20005  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
62ab2642  func[dihedral:3 colours:1 cyclic:2 rows:14 cols:11] real[n/a] |C|=20009+ gain=12.70b outs=20009  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
62b74c02  func[dihedral:3 colours:10 cols:4] real[n/a] |C|=7200 gain=11.23b outs=720  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
62c24649  func[dihedral:3 colours:6 rows:3 cols:3] real[dihedral:1 colours:6 rows:3 cols:3] |C|=144 gain=5.58b outs=144  progsD=19 progs=19 fitD=1.00 fitC=1.00 evals=838 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: mirror4
63613498  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20000+ gain=12.70b outs=15963  progsD=0 progs=0 fitD=0.67 fitC=0.38 evals=3236 sym=0/1(ok 0) equiv=NO det=-  unfit  :: recolour 9 5 ; recolour 6 5
639f5a19  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=16928 gain=13.05b outs=16928  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
642248e4  func[dihedral:3 colours:15 cyclic:2 rows:13 cols:15] real[n/a] |C|=20003+ gain=12.70b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
642d658d  func[dihedral:3 colours:36 cyclic:2 rows:26 cols:23] real[n/a] |C|=20002+ gain=12.70b outs=9  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6430c8c4  func[dihedral:3 colours:6 cyclic:2 rows:5 cols:1] real[dihedral:2 colours:3 cyclic:1 rows:5 cols:1] |C|=1152 gain=8.17b outs=144  progsD=50 progs=44 fitD=1.00 fitC=1.00 evals=2169 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitV nor 3
6455b5f5  func[dihedral:3 colours:1 cyclic:2 rows:10 cols:12] real[n/a] |C|=20003+ gain=12.29b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
64a7c07e  func[dihedral:3 colours:1 rows:3 cols:3] real[n/a] |C|=1000 gain=8.38b outs=370  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
662c240a  func[dihedral:3 colours:36 cyclic:2 rows:6] real[n/a] |C|=20000+ gain=12.29b outs=1889  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
66e6c45b  func[dihedral:3 colours:36 cyclic:2] real[n/a] |C|=20001+ gain=13.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
66f2d22f  func[dihedral:3 colours:3 cyclic:2 rows:3 cols:10] real[dihedral:2 colours:1 cyclic:2 rows:3 cols:7] |C|=5376 gain=10.39b outs=1092  progsD=31 progs=27 fitD=1.00 fitC=1.00 evals=1932 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitH nor 5
67385a82  func[dihedral:3 colours:1 rows:2 cols:3] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
673ef223  func[dihedral:3 colours:3 cyclic:2 rows:1 cols:2] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
67636eac  func[dihedral:3 colours:15 cyclic:1 rows:9 cols:11] real[n/a] |C|=20001+ gain=12.70b outs=9234  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6773b310  func[dihedral:3 colours:3 cyclic:2 rows:9 cols:8] real[n/a] |C|=20001+ gain=12.29b outs=162  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
67a3c6ac  func[dihedral:3 colours:10 cyclic:2 rows:6 cols:6] real[dihedral:2 colours:10 cyclic:1 rows:6] |C|=20003+ gain=12.70b outs=20003  progsD=48 progs=48 fitD=1.00 fitC=1.00 evals=3432 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: flipH
67a423a3  func[dihedral:3 colours:28 cyclic:2 rows:4 cols:4] real[n/a] |C|=12992 gain=12.08b outs=12992  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
67b4a34d  func[dihedral:3 colours:36 cyclic:2 rows:14 cols:14] real[n/a] |C|=20001+ gain=12.70b outs=7659  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
67c52801  func[dihedral:3 colours:28 cyclic:2 rows:8 cols:11] real[n/a] |C|=20014+ gain=12.29b outs=13543  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
67e8384a  func[dihedral:3 colours:30 cyclic:2 rows:3 cols:3] real[dihedral:1 colours:30 rows:3 cols:3] |C|=19152 gain=12.23b outs=19152  progsD=70 progs=68 fitD=1.00 fitC=1.00 evals=5922 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: mirror4
681b3aeb  func[dihedral:3 colours:28 cyclic:2 rows:7 cols:7] real[n/a] |C|=20002+ gain=12.70b outs=4032  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6855a6e4  func[dihedral:3 colours:3 cyclic:2 rows:6 cols:8] real[n/a] |C|=20000+ gain=12.70b outs=10582  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
68b16354  func[dihedral:3 colours:21 cyclic:2 rows:6 cols:6] real[dihedral:2 colours:21 cyclic:1 cols:6] |C|=20000+ gain=12.70b outs=20000  progsD=60 progs=60 fitD=1.00 fitC=1.00 evals=5304 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: flipV
68b67ca3  func[dihedral:3 colours:28 cyclic:1 rows:5 cols:5] real[n/a] |C|=20005+ gain=12.70b outs=13102  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
692cd3b6  func[dihedral:3 colours:3 cyclic:2 rows:6 cols:7] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
694f12f3  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=1600 gain=9.64b outs=1600  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
695367ec  func[dihedral:1 colours:10 rows:14 cols:14] real[n/a] |C|=15 gain=2.32b outs=15  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
696d4842  func[dihedral:3 colours:21 cyclic:2 rows:24 cols:20] real[n/a] |C|=20002+ gain=12.70b outs=19331  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
69889d6e  func[dihedral:3 colours:3] real[n/a] |C|=144 gain=5.17b outs=144  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6a11f6da  func[dihedral:3 colours:6 cyclic:2 rows:10 cols:2] real[n/a] |C|=20002+ gain=11.97b outs=19728  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6a1e5592  func[dihedral:3 colours:3 cyclic:2 rows:1 cols:7] real[n/a] |C|=20002+ gain=13.29b outs=6641  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6aa20dc0  func[dihedral:3 colours:21 cyclic:2 rows:20 cols:19] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6ad5bdfd  func[dihedral:3 colours:36 cyclic:2 rows:7 cols:9] real[n/a] |C|=20000+ gain=12.70b outs=15853  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6b9890af  func[dihedral:3 colours:15 cyclic:2 rows:14 cols:16] real[n/a] |C|=20000+ gain=12.70b outs=8352  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6c434453  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20001+ gain=13.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6cdd2623  func[dihedral:3 colours:36 cyclic:2 rows:15 cols:21] real[n/a] |C|=20017+ gain=12.70b outs=423  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6cf79266  func[dihedral:3 colours:10 cyclic:2 rows:19 cols:19] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6d0160f0  func[dihedral:3 colours:36 cyclic:2 rows:10 cols:10] real[n/a] |C|=20004+ gain=12.29b outs=10535  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6d0aefbc  func[dihedral:3 colours:6 cols:3] real[dihedral:1 colours:6 cols:3] |C|=168 gain=5.39b outs=168  progsD=36 progs=36 fitD=1.00 fitC=1.00 evals=1098 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: hconcatFlip
6d58a25d  func[dihedral:3 colours:36 cyclic:2 rows:18 cols:19] real[n/a] |C|=20016+ gain=12.70b outs=18208  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6d75e8bb  func[dihedral:3 colours:1 cyclic:2 rows:13 cols:10] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6df30ad6  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20004+ gain=11.97b outs=1625  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6e02f1e3  func[colours:6 rows:1] real[n/a] |C|=100 gain=4.32b outs=5  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6e19193c  func[dihedral:3 colours:6] real[n/a] |C|=64 gain=5.00b outs=64  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6e82a1ae  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6ea4a07e  func[dihedral:3 colours:4 cyclic:2 rows:2 cols:2] real[n/a] |C|=252 gain=5.39b outs=168  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
6ecd11f4  func[dihedral:3 colours:36 cyclic:2 rows:23 cols:21] real[n/a] |C|=20029+ gain=12.70b outs=11557  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6f473927  func[dihedral:3 colours:1 rows:8 cols:8] real[n/a] |C|=20001+ gain=12.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6f8cd79b  func[dihedral:3 rows:1] real[n/a] |C|=7 gain=0.81b outs=7  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
6fa7a44f  func[dihedral:3 colours:36 rows:4 cols:2] real[dihedral:1 colours:36 rows:3 cols:2] |C|=20002+ gain=12.29b outs=20002  progsD=66 progs=64 fitD=1.00 fitC=1.00 evals=6195 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: vconcatFlip
7039b2d7  func[dihedral:3 colours:15 cyclic:2 rows:26 cols:26] real[n/a] |C|=20000+ gain=12.70b outs=30  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
705a3229  func[dihedral:3 colours:21 rows:7 cols:10] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
712bf12e  func[dihedral:3 colours:3 cyclic:2 rows:12 cols:14] real[n/a] |C|=20010+ gain=12.70b outs=20010  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
72207abc  func[dihedral:3 colours:15 rows:1 cols:6] real[n/a] |C|=4320 gain=10.49b outs=4320  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
72322fa7  func[dihedral:3 colours:28 cyclic:2 rows:11 cols:11] real[n/a] |C|=20009+ gain=12.70b outs=20009  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
72a961c9  func[dihedral:3 colours:6 rows:3 cols:1] real[n/a] |C|=1104 gain=8.11b outs=1104  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
72ca375d  func[dihedral:3 colours:36 cyclic:2 rows:8 cols:9] real[dihedral:2 colours:22 rows:7 cols:5] |C|=5488 gain=10.84b outs=63  progsD=1 progs=1 fitD=1.00 fitC=0.89 evals=4195 sym=0/1(ok 0) equiv=yes det=-  fit-only  :: recolour 2 0 ; removeColour 8 ; cropLargest
73182012  func[dihedral:1 colours:28 cyclic:2 rows:8 cols:8] real[n/a] |C|=20000+ gain=12.70b outs=19819  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
73251a56  func[dihedral:3 colours:36 cyclic:2 rows:20 cols:20] real[n/a] |C|=20024+ gain=12.70b outs=15855  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
73c3b0d8  func[dihedral:3 colours:3 rows:4 cols:1] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
73ccf9c2  func[dihedral:3 colours:10 cyclic:2 rows:18 cols:17] real[n/a] |C|=20001+ gain=12.70b outs=12704  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7447852a  func[dihedral:1 colours:1 cyclic:1 cols:15] real[n/a] |C|=14688 gain=12.26b outs=14688  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7468f01a  func[dihedral:3 colours:21 cyclic:2 rows:11 cols:10] real[dihedral:2 colours:21 rows:9 cols:9] |C|=504 gain=7.39b outs=504  progsD=6 progs=6 fitD=1.00 fitC=1.00 evals=1146 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: flipH ; cropBBox
746b3537  func[dihedral:3 colours:21 cyclic:1 rows:2 cols:5] real[dihedral:3 colours:21 cyclic:1 cols:1] |C|=20000+ gain=11.97b outs=2352  progsD=2 progs=2 fitD=1.00 fitC=0.99 evals=897 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: dedupRows ; dedupCols
74dd1130  func[dihedral:3 colours:36 cyclic:2 rows:2 cols:2] real[dihedral:1 colours:36] |C|=4032 gain=9.98b outs=4032  progsD=107 progs=71 fitD=1.00 fitC=1.00 evals=7414 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: transpose
759f3fd3  func[dihedral:3 colours:1] real[n/a] |C|=32 gain=4.00b outs=32  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
75b8110e  func[dihedral:3 colours:10 cyclic:2 rows:7 cols:7] real[n/a] |C|=20001+ gain=11.97b outs=14175  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
760b3cac  func[dihedral:3 colours:3 cyclic:2] real[n/a] |C|=7776 gain=11.34b outs=7776  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
762cd429  func[dihedral:3 colours:21 cyclic:1 rows:1 cols:12] real[n/a] |C|=20003+ gain=12.70b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
770cc55f  func[dihedral:3 colours:15 cyclic:2 rows:6 cols:3] real[n/a] |C|=20001+ gain=12.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
776ffc46  func[dihedral:3 colours:10 cyclic:2 rows:19 cols:19] real[n/a] |C|=20010+ gain=12.29b outs=20010  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
77fdfe62  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20024+ gain=12.70b outs=7814  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
780d0b14  func[dihedral:3 colours:28 cyclic:2 rows:22 cols:27] real[n/a] |C|=20009+ gain=12.70b outs=8273  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
782b5218  func[dihedral:3 colours:15 cyclic:2 rows:9 cols:9] real[n/a] |C|=20005+ gain=12.70b outs=10588  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7837ac64  func[dihedral:3 colours:21 cyclic:2 rows:26 cols:26] real[n/a] |C|=20002+ gain=12.29b outs=3539  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
79369cc6  func[dihedral:3 colours:10 cyclic:2 rows:18 cols:16] real[n/a] |C|=20021+ gain=12.70b outs=20021  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
794b24be  func[colours:1 cyclic:1 cols:1] real[n/a] |C|=114 gain=3.51b outs=36  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
7953d61e  func[dihedral:3 colours:36 cyclic:2 rows:5 cols:5] real[n/a] |C|=20000+ gain=11.97b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
79fb03f4  func[dihedral:3 colours:6 cyclic:2 rows:18 cols:21] real[n/a] |C|=20002+ gain=11.70b outs=19976  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7b6016b9  func[dihedral:3 colours:10 cyclic:2 rows:16 cols:20] real[dihedral:3 colours:10 cyclic:1 rows:3 cols:2] |C|=20001+ gain=12.70b outs=20001  progsD=1 progs=1 fitD=1.00 fitC=0.99 evals=4856 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: fillEnclosed 2 ; recolour 0 3
7b7f7511  func[dihedral:3 colours:28 cyclic:2 rows:6 cols:6] real[n/a] |C|=20000+ gain=12.70b outs=15281  progsD=0 progs=0 fitD=0.67 fitC=0.65 evals=3233 sym=0/1(ok 0) equiv=NO det=-  unfit  :: leftHalf
7bb29440  func[dihedral:3 colours:6 cyclic:2 rows:18 cols:21] real[n/a] |C|=20000+ gain=11.97b outs=9135  progsD=0 progs=0 fitD=0.40 fitC=0.24 evals=3233 sym=0/1(ok 0) equiv=NO det=-  unfit  :: cropLargest
7c008303  func[dihedral:3 colours:36 cyclic:2 rows:5 cols:5] real[n/a] |C|=20007+ gain=12.70b outs=9604  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7c8af763  func[dihedral:3 colours:6 cyclic:2 rows:8 cols:8] real[n/a] |C|=20012+ gain=12.70b outs=20012  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7c9b52a0  func[dihedral:3 colours:28 cyclic:2 rows:12 cols:11] real[n/a] |C|=20001+ gain=12.70b outs=10969  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7d18a6fb  func[dihedral:3 colours:28 cyclic:2 rows:15 cols:13] real[n/a] |C|=20022+ gain=12.70b outs=13527  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7d1f7ee8  func[dihedral:3 colours:28 cyclic:2 rows:28 cols:27] real[n/a] |C|=20013+ gain=12.70b outs=14392  progsD=0 progs=0 fitD=0.33 fitC=0.22 evals=3231 sym=0/1(ok 0) equiv=NO det=-  unfit  :: recolourNonzero 1
7d419a02  func[dihedral:3 colours:3 cyclic:2 rows:20 cols:17] real[n/a] |C|=20011+ gain=12.70b outs=20011  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7ddcd7ec  func[dihedral:3 colours:10 rows:1 cols:1] real[n/a] |C|=240 gain=6.32b outs=200  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7df24a62  func[dihedral:3 colours:3 cyclic:2 rows:13 cols:17] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7e02026e  func[dihedral:3 colours:1 cyclic:2 rows:11 cols:11] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7e0986d6  func[dihedral:3 colours:21 cyclic:2 rows:12 cols:16] real[n/a] |C|=20024+ gain=13.29b outs=3964  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7ee1c6ea  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20010+ gain=12.70b outs=20010  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7f4411dc  func[dihedral:3 colours:10 cyclic:2 rows:16 cols:16] real[n/a] |C|=20001+ gain=12.70b outs=5208  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
7fe24cdd  func[dihedral:1 colours:21 cyclic:2 rows:3 cols:3] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
80af3007  func[dihedral:3 colours:1 rows:8 cols:10] real[n/a] |C|=48 gain=4.00b outs=18  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
810b9b61  func[dihedral:3 colours:1 cyclic:2 rows:14 cols:14] real[n/a] |C|=20004+ gain=12.70b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
817e6c09  func[dihedral:1 colours:1 cyclic:2 rows:4 cols:16] real[n/a] |C|=20000+ gain=11.97b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
81c0276b  func[dihedral:3 colours:21 cyclic:2 rows:16 cols:17] real[n/a] |C|=20002+ gain=12.70b outs=8003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
82819916  func[dihedral:3 colours:28 cyclic:2 rows:13 cols:9] real[n/a] |C|=20031+ gain=12.29b outs=18430  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
83302e8f  func[dihedral:3 colours:10 cyclic:2 rows:19 cols:26] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
833dafe3  func[dihedral:3 colours:21 rows:5 cols:5] real[dihedral:1 colours:21] |C|=840 gain=8.71b outs=840  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=680 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: rot180 ; mirror4
834ec97d  func[dihedral:1 colours:10 cyclic:1 rows:3 cols:3] real[n/a] |C|=130 gain=5.44b outs=130  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8403a5d5  func[dihedral:3 colours:6] real[n/a] |C|=72 gain=4.58b outs=72  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
845d6e51  func[dihedral:3 colours:28 cyclic:2 rows:17 cols:16] real[n/a] |C|=20002+ gain=12.70b outs=16888  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
846bdb03  func[dihedral:3 colours:21 cyclic:2 rows:5 cols:5] real[n/a] |C|=20002+ gain=12.29b outs=19411  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
84db8fc4  func[dihedral:3 colours:3 cyclic:2 rows:9 cols:9] real[dihedral:3 colours:1] |C|=64 gain=4.00b outs=64  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=3418 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: fillEnclosed 5 ; recolour 0 2
84f2aca1  func[dihedral:3 colours:10 cyclic:2 rows:15 cols:12] real[n/a] |C|=20021+ gain=12.29b outs=20021  progsD=0 progs=0 fitD=0.25 fitC=0.04 evals=3232 sym=0/1(ok 0) equiv=NO det=-  unfit  :: fillEnclosed 5
855e0971  func[dihedral:3 colours:21 cyclic:2 rows:16 cols:18] real[n/a] |C|=20001+ gain=12.29b outs=10587  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8597cfd7  func[dihedral:3 colours:6 cyclic:2 rows:10 cols:8] real[n/a] |C|=20000+ gain=12.29b outs=4  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
85b81ff1  func[dihedral:3 colours:6 cyclic:2 rows:12 cols:13] real[n/a] |C|=20002+ gain=12.29b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
85c4e7cd  func[dihedral:3 colours:36 cyclic:2 rows:13 cols:13] real[n/a] |C|=20010+ gain=12.29b outs=20010  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
85fa5666  func[dihedral:3 colours:10] real[n/a] |C|=3840 gain=9.91b outs=3360  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
868de0fa  func[dihedral:3 colours:1 cyclic:2 rows:19 cols:19] real[n/a] |C|=20004+ gain=11.97b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8719f442  func[dihedral:3 colours:1 rows:12 cols:12] real[n/a] |C|=14 gain=2.22b outs=14  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8731374e  func[dihedral:3 colours:36 cyclic:2 rows:26 cols:22] real[n/a] |C|=20007+ gain=12.70b outs=2263  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
88207623  func[dihedral:3 colours:28 cyclic:2 rows:13 cols:15] real[n/a] |C|=20007+ gain=13.29b outs=17705  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
88a10436  func[dihedral:3 colours:15 cyclic:2 cols:2] real[n/a] |C|=20003+ gain=12.70b outs=11282  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
88a62173  func[dihedral:3 colours:10 rows:4 cols:4] real[n/a] |C|=960 gain=8.32b outs=30  progsD=0 progs=0 fitD=0.67 fitC=0.29 evals=3233 sym=1/1(ok 0) equiv=NO det=-  unfit  :: cropLargest
890034e9  func[dihedral:3 colours:15 cyclic:2 rows:20 cols:20] real[n/a] |C|=20021+ gain=12.70b outs=20021  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
891232d6  func[dihedral:3 colours:3 cyclic:2 rows:2 cols:6] real[n/a] |C|=20001+ gain=12.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
896d5239  func[dihedral:3 colours:3 cyclic:2 rows:14 cols:17] real[n/a] |C|=20008+ gain=12.70b outs=20008  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8a004b2b  func[dihedral:3 colours:15 cyclic:2 rows:9 cols:7] real[n/a] |C|=20000+ gain=12.70b outs=17356  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8a371977  func[dihedral:3 colours:1 cyclic:2 rows:17 cols:22] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8b28cd80  func[colours:21 rows:6 cols:6] real[n/a] |C|=35 gain=2.81b outs=35  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
8ba14f53  func[dihedral:1 colours:28 cyclic:1 rows:1 cols:7] real[n/a] |C|=8320 gain=10.44b outs=3048  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8be77c9e  func[dihedral:3 colours:1 rows:3] real[dihedral:1 colours:1 rows:3] |C|=12 gain=2.00b outs=12  progsD=10 progs=6 fitD=1.00 fitC=1.00 evals=398 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: vconcatFlip
8cb8642d  func[dihedral:3 colours:21 cyclic:2 rows:6 cols:10] real[n/a] |C|=20000+ gain=12.70b outs=18485  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8d5021e8  func[dihedral:3 colours:10 rows:6 cols:1] real[n/a] |C|=80 gain=4.74b outs=40  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8d510a79  func[dihedral:3 colours:6 cyclic:2 rows:3 cols:8] real[n/a] |C|=20013+ gain=13.29b outs=16840  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8dae5dfc  func[dihedral:3 colours:36 cyclic:2 rows:19 cols:18] real[n/a] |C|=20022+ gain=12.29b outs=20022  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8e1813be  func[dihedral:3 colours:36 cyclic:2 rows:13 cols:16] real[n/a] |C|=20002+ gain=12.70b outs=7549  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8e2edd66  func[dihedral:3 colours:10 rows:6 cols:6] real[n/a] |C|=100 gain=5.06b outs=100  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8e5a5113  func[dihedral:3 colours:36 cols:4] real[n/a] |C|=20005+ gain=12.70b outs=19940  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8eb1be9a  func[dihedral:3 colours:6 cyclic:2 cols:4] real[n/a] |C|=20002+ gain=13.29b outs=12767  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8ee62060  func[dihedral:3 colours:10 rows:13 cols:13] real[n/a] |C|=20003+ gain=12.70b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8efcae92  func[dihedral:3 colours:3 cyclic:2 rows:16 cols:9] real[n/a] |C|=20002+ gain=12.70b outs=15142  progsD=0 progs=0 fitD=0.67 fitC=0.11 evals=3236 sym=0/1(ok 0) equiv=NO det=-  unfit  :: bottomHalf ; cropLargest
8f2ea7aa  func[dihedral:3 colours:10 cyclic:2] real[n/a] |C|=8100 gain=11.40b outs=7605  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
8fbca751  func[dihedral:3 colours:1 cyclic:2 rows:7 cols:11] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
90347967  func[dihedral:3 colours:28 cyclic:1 rows:2] real[n/a] |C|=20001+ gain=12.70b outs=19196  progsD=0 progs=0 fitD=0.67 fitC=0.56 evals=3229 sym=0/1(ok 0) equiv=NO det=-  unfit  :: rot180
903d1b4a  func[dihedral:3 colours:36 cyclic:2 rows:15 cols:15] real[dihedral:3 colours:28 rows:2 cols:1] |C|=20001+ gain=12.29b outs=9354  progsD=2 progs=2 fitD=1.00 fitC=1.00 evals=5920 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: recolour 3 0 ; symmetrize4
90c28cc7  func[dihedral:3 colours:28 cyclic:2 rows:18 cols:18] real[n/a] |C|=20008+ gain=12.70b outs=16348  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
90f3ed37  func[dihedral:3 colours:1 cyclic:2 cols:2] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9110e3c5  func[dihedral:3 colours:28 cyclic:2 rows:6 cols:6] real[n/a] |C|=20009+ gain=11.48b outs=76  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
913fb3ed  func[dihedral:3 colours:6 rows:3 cols:3] real[n/a] |C|=2352 gain=9.20b outs=2352  progsD=0 progs=0 fitD=0.25 fitC=0.06 evals=3231 sym=0/1(ok 0) equiv=NO det=-  unfit  :: box 4
91413438  func[dihedral:1 colours:15 rows:15 cols:17] real[n/a] |C|=252 gain=5.98b outs=252  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
91714a58  func[dihedral:3 colours:36 cyclic:2 rows:15 cols:15] real[n/a] |C|=20012+ gain=12.70b outs=700  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9172f3a0  func[dihedral:3 colours:21 rows:6 cols:6] real[dihedral:3 colours:21 rows:6 cols:6] |C|=2016 gain=9.98b outs=2016  progsD=45 progs=17 fitD=1.00 fitC=1.00 evals=1592 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: scale 3
917bccba  func[dihedral:3 colours:28 rows:10 cols:10] real[n/a] |C|=20007+ gain=12.70b outs=13268  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
928ad970  func[dihedral:3 colours:15 cyclic:2 rows:4 cols:7] real[n/a] |C|=20005+ gain=12.70b outs=20005  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
929ab4e9  func[dihedral:3 colours:36 cyclic:2 rows:23 cols:23] real[dihedral:3 colours:28 rows:1 cols:1] |C|=20006+ gain=12.29b outs=9244  progsD=2 progs=2 fitD=1.00 fitC=1.00 evals=8556 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: recolour 2 0 ; symmetrize4
92e50de0  func[dihedral:3 colours:15 cyclic:2 rows:4 cols:5] real[n/a] |C|=20002+ gain=12.70b outs=19200  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9356391f  func[dihedral:3 colours:28 cyclic:2 cols:1] real[n/a] |C|=20002+ gain=13.29b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
93b4f4b3  func[dihedral:3 colours:36 cyclic:2 rows:16 cols:11] real[n/a] |C|=20004+ gain=13.29b outs=17588  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
93b581b8  func[dihedral:3 colours:36 cyclic:2] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 0) equiv=yes det=-  unfit  :: id
93c31fbe  func[dihedral:3 colours:15 cyclic:2 rows:16 cols:24] real[n/a] |C|=20003+ gain=12.70b outs=15539  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
94133066  func[dihedral:3 colours:21 cyclic:2 rows:12 cols:21] real[n/a] |C|=20001+ gain=12.70b outs=19998  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
941d9a10  func[dihedral:1 colours:1 rows:4 cols:4] real[n/a] |C|=544 gain=7.50b outs=544  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
94414823  func[dihedral:3 colours:36 cyclic:2 rows:7 cols:7] real[n/a] |C|=20005+ gain=12.70b outs=20005  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
94be5b80  func[dihedral:3 colours:28 cyclic:2 rows:2 cols:9] real[n/a] |C|=20007+ gain=13.29b outs=17278  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
94f9d214  func[dihedral:3 colours:3 cyclic:2 rows:4 cols:1] real[dihedral:2 colours:3 cyclic:2 rows:4 cols:1] |C|=9216 gain=11.17b outs=256  progsD=30 progs=30 fitD=1.00 fitC=1.00 evals=903 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitV nor 2
952a094c  func[dihedral:3 colours:36 cyclic:2 rows:3 cols:2] real[n/a] |C|=20015+ gain=12.70b outs=16648  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9565186b  func[dihedral:3 colours:15 cyclic:2 rows:2 cols:2] real[n/a] |C|=20002+ gain=12.29b outs=414  progsD=0 progs=0 fitD=0.50 fitC=0.19 evals=3231 sym=0/1(ok 0) equiv=NO det=-  unfit  :: recolour 1 5 ; recolour 8 5
95990924  func[dihedral:1 colours:1 cyclic:1 rows:2] real[n/a] |C|=1040 gain=8.44b outs=1040  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
95a58926  func[dihedral:3 colours:15 cyclic:2 rows:21 cols:26] real[n/a] |C|=20000+ gain=12.70b outs=3183  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
963e52fc  func[dihedral:1 colours:10 cyclic:2 rows:4 cols:11] real[n/a] |C|=20000+ gain=12.70b outs=19665  progsD=0 progs=0 fitD=0.33 fitC=0.22 evals=3232 sym=0/1(ok 0) equiv=NO det=-  SOLVED  :: tile 1 2
963f59bc  func[dihedral:3 colours:15 cyclic:2 rows:1 cols:4] real[n/a] |C|=20001+ gain=12.29b outs=18983  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
96a8c0cd  func[dihedral:3 colours:6 cyclic:2 rows:1 cols:7] real[n/a] |C|=20001+ gain=12.29b outs=19110  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
97239e3d  func[dihedral:3 colours:28 cyclic:2 rows:13 cols:16] real[n/a] |C|=20029+ gain=12.70b outs=20019  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9772c176  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=19456 gain=13.25b outs=19456  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
97999447  func[dihedral:3 colours:15] real[n/a] |C|=1224 gain=8.67b outs=1224  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
97a05b5b  func[dihedral:3 colours:21 cyclic:2 rows:11 cols:10] real[n/a] |C|=20004+ gain=12.70b outs=17370  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
981571dc  func[dihedral:3 colours:36 cyclic:2 rows:29 cols:29] real[n/a] |C|=20041+ gain=12.29b outs=19255  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
98cf29f8  func[dihedral:3 colours:15 cyclic:2 rows:10 cols:15] real[n/a] |C|=20000+ gain=12.70b outs=13464  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
992798f6  func[dihedral:3 colours:3 rows:2 cols:2] real[n/a] |C|=768 gain=7.58b outs=768  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
99306f82  func[dihedral:3 colours:36 cyclic:2 rows:7 cols:7] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
995c5fa3  func[dihedral:1 colours:1 cyclic:1 rows:1 cols:13] real[n/a] |C|=20000+ gain=12.29b outs=6  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
99b1bc43  func[dihedral:3 colours:6 cyclic:2 rows:5 cols:1] real[dihedral:2 colours:3 cyclic:1 rows:5 cols:1] |C|=1152 gain=8.17b outs=132  progsD=50 progs=44 fitD=1.00 fitC=1.00 evals=2089 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitV xor 3
99fa7670  func[colours:21 cyclic:2 cols:2] real[n/a] |C|=10437 gain=11.35b outs=10437  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9a4bb226  func[dihedral:3 colours:36 cyclic:2 rows:12 cols:12] real[n/a] |C|=20032+ gain=12.71b outs=4200  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9aec4887  func[dihedral:3 colours:28 cyclic:2 rows:10 cols:10] real[n/a] |C|=20000+ gain=12.70b outs=19901  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9af7a82c  func[dihedral:2 colours:15 cyclic:1 rows:3] real[n/a] |C|=18720 gain=12.19b outs=17280  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9b2a60aa  func[dihedral:3 colours:10 cyclic:2 cols:1] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9b365c51  func[dihedral:3 colours:28 cyclic:2 rows:7 cols:25] real[n/a] |C|=20018+ gain=12.70b outs=9883  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9b4c17c4  func[dihedral:3 colours:6 cyclic:2 rows:7 cols:6] real[n/a] |C|=20001+ gain=12.29b outs=13100  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
9bebae7a  func[dihedral:3 colours:3 cyclic:2 rows:1 cols:2] real[n/a] |C|=20001+ gain=11.97b outs=5542  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9c1e755f  func[dihedral:3 colours:36 cyclic:2 rows:2 cols:4] real[n/a] |C|=20000+ gain=12.29b outs=19087  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9c56f360  func[dihedral:3 colours:3 cyclic:2 rows:8 cols:8] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9caba7c3  func[dihedral:3 colours:3 cyclic:2 rows:18 cols:18] real[n/a] |C|=20012+ gain=12.70b outs=20012  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9d9215db  func[dihedral:3 colours:15 cyclic:2] real[n/a] |C|=20001+ gain=12.70b outs=7866  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9ddd00f0  func[dihedral:3 colours:6 cyclic:2 rows:8 cols:8] real[dihedral:3 colours:6 rows:3 cols:3] |C|=32 gain=4.00b outs=8  progsD=27 progs=19 fitD=1.00 fitC=1.00 evals=1186 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: symmetrize4
9def23fe  func[dihedral:3 colours:15 cyclic:2 rows:18 cols:23] real[n/a] |C|=20005+ gain=12.70b outs=20005  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9dfd6313  func[dihedral:3 colours:28 cyclic:2 rows:5 cols:5] real[dihedral:1 colours:28] |C|=4144 gain=10.43b outs=4144  progsD=25 progs=25 fitD=1.00 fitC=1.00 evals=3638 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: transpose
9ecd008a  func[dihedral:3 colours:36 cyclic:2 rows:14 cols:14] real[n/a] |C|=20006+ gain=12.70b outs=9345  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9edfc990  func[dihedral:3 colours:36 cyclic:2 rows:15 cols:15] real[n/a] |C|=20024+ gain=12.70b outs=20024  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9f236235  func[dihedral:2 colours:15 cyclic:2 rows:19 cols:19] real[n/a] |C|=20000+ gain=12.70b outs=5998  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
9f27f097  func[dihedral:3 colours:15 cyclic:2] real[n/a] |C|=20001+ gain=12.70b outs=19991  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a04b2602  func[dihedral:3 colours:3 cyclic:2 rows:5 cols:15] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a096bf4d  func[dihedral:3 colours:36 cyclic:2 rows:24 cols:25] real[n/a] |C|=20010+ gain=12.70b outs=20010  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a1570a43  func[dihedral:3 colours:3 cyclic:2 rows:2 cols:2] real[n/a] |C|=20001+ gain=12.29b outs=16706  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a2fd1cf0  func[dihedral:3 colours:2 rows:5 cols:6] real[n/a] |C|=3456 gain=10.17b outs=3456  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a3325580  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:8] real[n/a] |C|=20003+ gain=11.70b outs=1223  progsD=0 progs=0 fitD=0.17 fitC=0.15 evals=3224 sym=0/1(ok 0) equiv=NO det=-  unfit  :: cropLargest
a3df8b1e  func[dihedral:3 colours:1] real[n/a] |C|=48 gain=4.00b outs=32  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a3f84088  func[dihedral:3 colours:1 cyclic:2 rows:11 cols:11] real[n/a] |C|=1644 gain=8.68b outs=1644  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a406ac07  func[dihedral:3 colours:28 cyclic:2 rows:9 cols:9] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a416b8f3  func[dihedral:3 colours:28 cyclic:2 rows:2 cols:4] real[dihedral:2 colours:28 cyclic:2 rows:2 cols:4] |C|=20001+ gain=12.70b outs=20001  progsD=72 progs=40 fitD=1.00 fitC=1.00 evals=3657 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: hconcat
a48eeaf7  func[dihedral:3 colours:3 cyclic:2] real[n/a] |C|=9600 gain=12.23b outs=9600  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a5313dff  func[dihedral:3 colours:1 cyclic:2 rows:11 cols:11] real[dihedral:3 colours:1] |C|=40 gain=3.74b outs=40  progsD=8 progs=6 fitD=1.00 fitC=1.00 evals=3902 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: fillEnclosed 1
a57f2f04  func[dihedral:3 colours:15 cyclic:2 rows:10 cols:6] real[n/a] |C|=20001+ gain=12.70b outs=19955  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a59b95c0  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20001+ gain=11.97b outs=20001  progsD=0 progs=0 fitD=0.40 fitC=0.43 evals=3230 sym=0/1(ok 0) equiv=yes det=-  unfit  :: tile 3 3
a5f85a15  func[dihedral:2 colours:10 cyclic:1 rows:11 cols:5] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a61ba2ce  func[dihedral:3 colours:21 cyclic:2 rows:9 cols:9] real[n/a] |C|=20002+ gain=13.29b outs=5471  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a61f2674  func[dihedral:3 colours:1 cyclic:2 rows:8 cols:8] real[n/a] |C|=20004+ gain=13.29b outs=2883  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a64e4611  func[dihedral:3 colours:10 cyclic:2 rows:25 cols:25] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a65b410d  func[dihedral:1 colours:1 cyclic:1 rows:1] real[n/a] |C|=88 gain=4.87b outs=88  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a680ac02  func[dihedral:3 colours:21 cyclic:2 rows:17 cols:13] real[n/a] |C|=20000+ gain=12.70b outs=1851  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a68b268e  func[dihedral:3 colours:15 cyclic:2 rows:8 cols:8] real[n/a] |C|=20001+ gain=11.70b outs=13745  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a699fb00  func[dihedral:3 colours:1 cyclic:2 rows:1] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a740d043  func[dihedral:3 colours:15 cyclic:2 rows:4 cols:4] real[dihedral:3 colours:10 rows:4 cols:4] |C|=400 gain=7.06b outs=280  progsD=2 progs=2 fitD=1.00 fitC=1.00 evals=1176 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: recolour 1 0 ; cropBBox
a78176bb  func[dihedral:3 colours:15 rows:1 cols:1] real[n/a] |C|=2400 gain=9.64b outs=288  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a79310a0  func[dihedral:2 colours:1 cyclic:1 cols:4] real[dihedral:1 cyclic:1 cols:4] |C|=23 gain=2.94b outs=23  progsD=4 progs=4 fitD=1.00 fitC=1.00 evals=364 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: shiftDown ; recolour 8 2
a85d4709  func[dihedral:1 colours:1 cyclic:1] real[n/a] |C|=40 gain=3.32b outs=17  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 0) equiv=yes det=-  unfit  :: id
a8610ef7  func[dihedral:3 colours:1 cyclic:2 rows:5 cols:5] real[n/a] |C|=20003+ gain=12.29b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a87f7484  func[dihedral:3 colours:36 cyclic:2 rows:13 cols:12] real[n/a] |C|=20003+ gain=12.29b outs=302  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a8c38be5  func[dihedral:3 colours:36 cyclic:2 rows:15 cols:14] real[n/a] |C|=20006+ gain=13.29b outs=19602  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a8d7556c  func[dihedral:3 colours:1 cyclic:2 rows:17 cols:17] real[n/a] |C|=20009+ gain=12.70b outs=20009  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a934301b  func[dihedral:3 colours:3 cyclic:2 rows:13 cols:14] real[n/a] |C|=20006+ gain=12.70b outs=7432  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
a9f96cdd  func[dihedral:1 colours:1] real[n/a] |C|=16 gain=2.00b outs=14  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
aa18de87  func[dihedral:3 colours:15 rows:3 cols:8] real[n/a] |C|=20001+ gain=12.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
aa300dc3  func[dihedral:3 colours:1 cyclic:2 rows:6 cols:6] real[n/a] |C|=20001+ gain=12.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
aa4ec2a5  func[dihedral:3 colours:3 cyclic:2] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
aab50785  func[dihedral:3 colours:36 cyclic:2 rows:14 cols:16] real[n/a] |C|=20019+ gain=11.97b outs=15671  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
aabf363d  func[dihedral:3 colours:15 cyclic:2 rows:6 cols:6] real[n/a] |C|=20000+ gain=13.29b outs=10928  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
aba27056  func[dihedral:3 colours:10 rows:1 cols:1] real[n/a] |C|=80 gain=4.74b outs=80  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ac0a08a4  func[dihedral:3 colours:36 rows:12 cols:12] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.33 fitC=0.33 evals=3231 sym=0/1(ok 0) equiv=yes det=-  unfit  :: scale 3
ac0c5833  func[dihedral:3 colours:3 cyclic:2 rows:1 cols:1] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ac2e8ecf  func[dihedral:3 colours:15 cyclic:2 rows:6 cols:12] real[n/a] |C|=20002+ gain=12.70b outs=13522  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ac3e2b04  func[dihedral:3 colours:3 cyclic:2 rows:3 cols:9] real[n/a] |C|=20005+ gain=12.29b outs=20005  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ac605cbb  func[dihedral:3 colours:10] real[n/a] |C|=1480 gain=7.95b outs=1460  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ad7e01d0  func[dihedral:3 colours:10 rows:13 cols:13] real[n/a] |C|=6120 gain=10.58b outs=6120  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ae3edfdc  func[dihedral:3 colours:10 cyclic:2 rows:2 cols:1] real[n/a] |C|=20000+ gain=12.70b outs=8653  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ae4f1146  func[dihedral:3 colours:3 cyclic:2 rows:8 cols:8] real[n/a] |C|=20002+ gain=12.29b outs=472  progsD=0 progs=0 fitD=0.75 fitC=0.19 evals=3240 sym=0/1(ok 0) equiv=NO det=-  SOLVED  :: rightHalf ; cropLargest
ae58858e  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:11] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
aedd82e4  func[dihedral:3 colours:1 rows:2 cols:1] real[n/a] |C|=2074 gain=9.02b outs=2074  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
aee291af  func[dihedral:3 colours:6 cyclic:2 rows:17 cols:17] real[n/a] |C|=20002+ gain=12.70b outs=5067  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
af22c60d  func[dihedral:3 colours:36 cyclic:2 rows:29 cols:29] real[n/a] |C|=20032+ gain=12.29b outs=19477  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
af24b4cc  func[dihedral:3 colours:36 cyclic:2 rows:8 cols:8] real[n/a] |C|=20005+ gain=12.70b outs=11033  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
af902bf9  func[dihedral:3 colours:1 rows:5 cols:5] real[n/a] |C|=4352 gain=10.50b outs=4352  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b0722778  func[dihedral:3 colours:36 cyclic:2 rows:10 cols:8] real[n/a] |C|=20007+ gain=13.29b outs=12493  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b0c4d837  func[dihedral:1 colours:3 rows:6 cols:8] real[n/a] |C|=360 gain=5.91b outs=60  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b0f4d537  func[dihedral:3 colours:15 cyclic:2 rows:11 cols:13] real[n/a] |C|=20001+ gain=12.29b outs=5437  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b15fca0b  func[dihedral:3 colours:3 cyclic:2 rows:2 cols:4] real[n/a] |C|=20000+ gain=11.97b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b190f7f5  func[dihedral:3 colours:15 rows:12 cols:11] real[n/a] |C|=20000+ gain=12.70b outs=5220  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b1948b0a  func[dihedral:3 colours:3 cyclic:2 rows:5 cols:5] real[dihedral:3 cyclic:2 rows:5 cols:5] |C|=20000+ gain=12.70b outs=20000  progsD=44 progs=42 fitD=1.00 fitC=1.00 evals=1855 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: recolour 6 2
b1fc8b8e  func[dihedral:3 colours:1 rows:1 cols:1] real[n/a] |C|=72 gain=3.85b outs=10  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
b20f7c8b  func[dihedral:3 colours:36 cyclic:2 rows:17 cols:21] real[n/a] |C|=20010+ gain=12.70b outs=14397  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b230c067  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20009+ gain=12.70b outs=20009  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b27ca6d3  func[dihedral:3 colours:1 cyclic:2 rows:1 cols:2] real[n/a] |C|=20000+ gain=13.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b2862040  func[dihedral:3 colours:3 cyclic:2 rows:15 cols:14] real[n/a] |C|=20002+ gain=12.29b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b457fec5  func[dihedral:3 colours:21 cyclic:2 rows:26 cols:22] real[n/a] |C|=20018+ gain=12.70b outs=18355  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b4a43f3b  func[dihedral:3 colours:15 cyclic:2 rows:7 cols:9] real[n/a] |C|=20004+ gain=12.29b outs=12711  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b527c5c6  func[dihedral:3 colours:3 cyclic:2] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b548a754  func[dihedral:3 colours:21 cyclic:2 rows:11 cols:12] real[n/a] |C|=20002+ gain=12.70b outs=8321  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b60334d2  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=2592 gain=10.34b outs=2592  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b6afb2da  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=2400 gain=10.23b outs=2400  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b7249182  func[dihedral:3 colours:28 cyclic:2 rows:6 cols:3] real[n/a] |C|=20001+ gain=12.70b outs=15695  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b775ac94  func[dihedral:3 colours:28 cyclic:2 rows:16 cols:18] real[n/a] |C|=20026+ gain=12.70b outs=20019  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b782dc8a  func[dihedral:3 colours:15 cyclic:2 rows:14 cols:23] real[n/a] |C|=20020+ gain=13.29b outs=20020  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b7999b51  func[dihedral:3 colours:28 cyclic:2 rows:14 cols:14] real[n/a] |C|=20001+ gain=12.70b outs=19612  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b7cb93ac  func[dihedral:3 colours:15 cyclic:2 rows:6 cols:9] real[n/a] |C|=20001+ gain=12.70b outs=13938  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b7f8a4d8  func[dihedral:3 colours:15 cyclic:2 rows:29 cols:29] real[n/a] |C|=20039+ gain=12.71b outs=20039  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b7fb29bc  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=5850 gain=10.93b outs=5850  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b8825c91  func[dihedral:3 colours:36 cyclic:2 rows:15 cols:15] real[dihedral:3 colours:28 rows:2] |C|=20002+ gain=12.29b outs=12680  progsD=4 progs=2 fitD=1.00 fitC=0.99 evals=6086 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: recolour 4 0 ; symmetrize4
b8cdaf2b  func[dihedral:3 colours:21 cyclic:2 rows:2 cols:3] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b91ae062  func[dihedral:3 colours:36 cyclic:1 rows:9 cols:9] real[n/a] |C|=20000+ gain=11.97b outs=20000  progsD=0 progs=0 fitD=0.40 fitC=0.39 evals=3231 sym=0/1(ok 0) equiv=NO det=-  unfit  :: scale 3
b942fd60  func[dihedral:3 colours:15 cyclic:1 rows:7 cols:8] real[n/a] |C|=20008+ gain=11.70b outs=19686  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b94a9452  func[dihedral:3 colours:21 rows:10 cols:9] real[n/a] |C|=1848 gain=9.27b outs=126  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b9630600  func[dihedral:3 colours:1 cyclic:2 rows:5 cols:7] real[n/a] |C|=20005+ gain=12.70b outs=20005  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
b9b7f026  func[dihedral:3 colours:36 cyclic:2 rows:15 cols:16] real[n/a] |C|=20012+ gain=12.70b outs=9  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ba26e723  func[dihedral:1 colours:1 cyclic:1 rows:2 cols:12] real[n/a] |C|=4320 gain=9.75b outs=4320  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ba97ae07  func[dihedral:3 colours:36 cyclic:2 rows:12 cols:12] real[n/a] |C|=20002+ gain=12.29b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ba9d41b8  func[dihedral:1 colours:28 cyclic:1 rows:2 cols:3] real[n/a] |C|=13136 gain=12.10b outs=13136  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
baf41dbf  func[dihedral:3 colours:3 cyclic:2 rows:2 cols:2] real[n/a] |C|=20000+ gain=12.70b outs=16189  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bb43febb  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=1600 gain=9.64b outs=1600  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bb52a14b  func[dihedral:3 colours:6 cyclic:2 rows:21 cols:21] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bbb1b8b6  func[dihedral:3 colours:21 cyclic:2 rows:3 cols:8] real[n/a] |C|=20007+ gain=11.48b outs=6465  progsD=0 progs=0 fitD=0.43 fitC=0.35 evals=3225 sym=0/2(ok 0) equiv=NO,NO det=-,-  unfit  :: leftHalf
bbc9ae5d  func[dihedral:1 colours:15 rows:4 cols:4] real[n/a] |C|=60 gain=3.58b outs=60  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bc1d5164  func[dihedral:3 colours:15 rows:2 cols:4] real[n/a] |C|=204 gain=5.35b outs=126  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bc4146bd  func[dihedral:3 colours:36 cyclic:2 rows:2 cols:17] real[n/a] |C|=20000+ gain=12.29b outs=19998  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bcb3040b  func[dihedral:3 colours:3 cyclic:2 rows:17 cols:17] real[n/a] |C|=20026+ gain=12.70b outs=20026  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bd14c3bf  func[dihedral:3 colours:3 cyclic:2 rows:18 cols:17] real[n/a] |C|=20011+ gain=12.70b outs=20011  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bd4472b8  func[dihedral:3 colours:11 rows:2] real[n/a] |C|=3200 gain=10.06b outs=3200  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bda2d7a6  func[dihedral:3 colours:36 cyclic:2 rows:7 cols:7] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
bdad9b1f  func[dihedral:3 colours:3 cyclic:2 rows:5 cols:5] real[n/a] |C|=20000+ gain=13.29b outs=216  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
be03b35f  func[dihedral:3 colours:3 rows:3 cols:3] real[n/a] |C|=144 gain=5.58b outs=24  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
be94b721  func[dihedral:3 colours:36 cyclic:2 rows:7 cols:11] real[dihedral:3 colours:36 rows:5 cols:8] |C|=20001+ gain=12.29b outs=382  progsD=121 progs=26 fitD=1.00 fitC=1.00 evals=5489 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: cropLargest
beb8660c  func[dihedral:3 colours:36 cyclic:2 rows:7 cols:3] real[n/a] |C|=20002+ gain=12.70b outs=18523  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bf32578f  func[dihedral:3 colours:10 cyclic:2] real[n/a] |C|=3440 gain=10.16b outs=1220  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bf699163  func[dihedral:3 colours:36 cyclic:2 rows:16 cols:16] real[n/a] |C|=20008+ gain=13.29b outs=356  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
bf89d739  func[dihedral:3 colours:1 cyclic:2 rows:1 cols:1] real[n/a] |C|=20001+ gain=12.29b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c074846d  func[dihedral:1 colours:3 cyclic:2 rows:2 cols:4] real[n/a] |C|=2208 gain=8.79b outs=2154  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
c0f76784  func[dihedral:3 colours:1 cyclic:2 rows:11 cols:11] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c1990cce  func[dihedral:1 colours:1 rows:1 cols:5] real[n/a] |C|=16 gain=2.42b outs=16  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c1d99e64  func[dihedral:3 colours:10 cyclic:2 rows:26 cols:24] real[n/a] |C|=20027+ gain=12.70b outs=20027  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c3202e5a  func[dihedral:3 colours:36 cyclic:2 rows:28 cols:28] real[n/a] |C|=20005+ gain=12.70b outs=1641  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c35c1b4c  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20000+ gain=12.70b outs=19644  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c3e719e8  func[dihedral:3 colours:36 cyclic:2 rows:6 cols:6] real[n/a] |C|=20008+ gain=12.70b outs=20008  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c3f564a4  func[dihedral:3 colours:36 cyclic:2 rows:15 cols:15] real[n/a] |C|=20013+ gain=12.70b outs=17705  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c444b776  func[dihedral:3 colours:36 cyclic:2 rows:1 cols:12] real[n/a] |C|=20006+ gain=13.29b outs=20006  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c48954c1  func[dihedral:3 colours:36 cyclic:1 rows:6 cols:6] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c59eb873  func[dihedral:3 colours:28 rows:5 cols:5] real[dihedral:3 colours:28 rows:4 cols:4] |C|=15456 gain=12.33b outs=15456  progsD=71 progs=2 fitD=1.00 fitC=1.00 evals=2214 sym=0/1(ok 0) equiv=yes det=yes  fit-only  :: scale 2 ; cropSmallest
c62e2108  func[dihedral:3 colours:15 cyclic:2 rows:2 cols:3] real[n/a] |C|=20000+ gain=12.70b outs=5831  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c64f1187  func[dihedral:3 colours:28 cyclic:2 rows:13 cols:14] real[n/a] |C|=20000+ gain=13.29b outs=11909  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c658a4bd  func[dihedral:3 colours:21 cyclic:2 rows:11 cols:13] real[n/a] |C|=20000+ gain=13.29b outs=9952  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c663677b  func[dihedral:3 colours:36 cyclic:2 rows:26 cols:26] real[n/a] |C|=20007+ gain=12.70b outs=16806  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c6e1b8da  func[dihedral:3 colours:28 cyclic:2 rows:18 cols:15] real[n/a] |C|=20003+ gain=12.70b outs=15093  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c7d4e6ad  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20000+ gain=13.29b outs=13194  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c87289bb  func[dihedral:3 colours:3 rows:2 cols:7] real[n/a] |C|=18816 gain=12.20b outs=18816  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c8b7cc0f  func[dihedral:3 colours:15 cyclic:2 rows:7 cols:8] real[n/a] |C|=20002+ gain=12.70b outs=448  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c8cbb738  func[dihedral:3 colours:28 cyclic:2 rows:14 cols:15] real[n/a] |C|=20000+ gain=12.70b outs=9608  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c8f0f002  func[dihedral:3 colours:6 cyclic:2 rows:2 cols:5] real[dihedral:3 colours:1 cyclic:2 rows:2 cols:5] |C|=20000+ gain=12.70b outs=20000  progsD=55 progs=53 fitD=1.00 fitC=1.00 evals=2438 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: recolour 7 5
c909285e  func[dihedral:3 colours:28 cyclic:2 rows:23 cols:24] real[n/a] |C|=20010+ gain=12.70b outs=15439  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c92b942c  func[colours:10 rows:11 cols:14] real[n/a] |C|=90 gain=4.49b outs=90  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c97c0139  func[dihedral:3 colours:1 cyclic:2 cols:1] real[n/a] |C|=7676 gain=11.91b outs=7676  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
c9e6f938  func[dihedral:3 colours:1 cols:3] real[dihedral:1 colours:1 cols:3] |C|=8 gain=1.42b outs=8  progsD=12 progs=8 fitD=1.00 fitC=1.00 evals=422 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: hconcatFlip
c9f8e694  func[dihedral:3 colours:28 cyclic:2 rows:11 cols:11] real[n/a] |C|=20008+ gain=13.29b outs=13063  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ca8de6ea  func[dihedral:2 colours:20 cyclic:2 rows:4 cols:4] real[n/a] |C|=20001+ gain=12.70b outs=9507  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ca8f78db  func[dihedral:3 colours:36 cyclic:2 rows:29 cols:29] real[n/a] |C|=20014+ gain=12.70b outs=19216  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
caa06a1f  func[dihedral:3 colours:28 cyclic:2 rows:10 cols:11] real[n/a] |C|=20000+ gain=12.70b outs=5154  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
cad67732  func[dihedral:3 colours:28 cyclic:1 rows:10 cols:10] real[n/a] |C|=20002+ gain=12.70b outs=19253  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
cb227835  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=3928 gain=10.35b outs=3928  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
cbded52d  func[dihedral:3 colours:28 cyclic:2 rows:4 cols:4] real[n/a] |C|=20001+ gain=12.70b outs=19827  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ccd554ac  func[dihedral:3 colours:15 cyclic:2 rows:12 cols:12] real[n/a] |C|=1332 gain=7.79b outs=1332  progsD=0 progs=0 fitD=0.33 fitC=0.36 evals=3231 sym=0/1(ok 0) equiv=NO det=-  unfit  :: tile 3 3
cce03e0d  func[dihedral:3 colours:3 rows:6 cols:6] real[n/a] |C|=120 gain=5.32b outs=120  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
cd3c21df  func[dihedral:3 colours:36 cyclic:2 rows:11 cols:12] real[dihedral:1 colours:22 rows:9 cols:10] |C|=2688 gain=9.81b outs=56  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=3935 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: recolour 5 3 ; removeColour 3 ; cropLargest
cdecee7f  func[dihedral:3 colours:36 cyclic:2 rows:7 cols:7] real[n/a] |C|=20005+ gain=12.70b outs=19628  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ce039d91  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20006+ gain=12.29b outs=20006  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ce22a75a  func[dihedral:3 colours:1 cyclic:2] real[dihedral:3] |C|=10 gain=2.32b outs=10  progsD=6 progs=6 fitD=1.00 fitC=1.00 evals=516 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: recolour 5 1 ; box 1
ce4f8723  func[dihedral:3 colours:6 cyclic:2 rows:6 cols:3] real[dihedral:2 colours:3 cyclic:1 rows:5 cols:3] |C|=1152 gain=8.17b outs=84  progsD=49 progs=46 fitD=1.00 fitC=1.00 evals=2510 sym=0/1(ok 0) equiv=yes det=NO  SOLVED  :: splitV or 3
ce602527  func[dihedral:3 colours:21 cyclic:2 rows:14 cols:13] real[n/a] |C|=20006+ gain=12.29b outs=2492  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ce8d95cc  func[dihedral:3 colours:36 rows:6 cols:7] real[dihedral:3 colours:36 rows:5 cols:4] |C|=20004+ gain=12.29b outs=14190  progsD=2 progs=2 fitD=1.00 fitC=1.00 evals=510 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: dedupRows ; dedupCols
ce9e57f2  func[dihedral:3 colours:1 cyclic:2 rows:2 cols:1] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
cf133acc  func[dihedral:3 colours:36 cyclic:2 rows:13 cols:14] real[n/a] |C|=20008+ gain=12.70b outs=18207  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
cf98881b  func[dihedral:3 colours:10 cyclic:2 rows:2 cols:11] real[n/a] |C|=20000+ gain=11.97b outs=14330  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
cfb2ce5a  func[dihedral:3 colours:36 cyclic:2 rows:3 cols:5] real[n/a] |C|=20018+ gain=12.70b outs=19733  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d017b73f  func[dihedral:3 colours:21 cyclic:2 rows:2 cols:5] real[n/a] |C|=20000+ gain=12.29b outs=17969  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d037b0a7  func[dihedral:2 colours:15 cyclic:1 rows:2] real[n/a] |C|=1440 gain=8.91b outs=900  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 0) equiv=yes det=-  unfit  :: id
d06dbe63  func[dihedral:1 colours:1 cyclic:1] real[n/a] |C|=104 gain=5.70b outs=104  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 0) equiv=yes det=-  unfit  :: id
d07ae81c  func[dihedral:3 colours:21 cyclic:2] real[n/a] |C|=20003+ gain=12.70b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d0f5fe59  func[dihedral:3 colours:1 cyclic:2 rows:12 cols:8] real[n/a] |C|=6400 gain=11.06b outs=32  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d10ecb37  func[dihedral:3 colours:21 cyclic:2 rows:11 cols:7] real[n/a] |C|=20000+ gain=12.70b outs=1776  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d13f3404  func[dihedral:3 colours:28 rows:3 cols:3] real[n/a] |C|=4032 gain=10.39b outs=3360  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d19f7514  func[dihedral:3 colours:3 cyclic:2 rows:8 cols:3] real[dihedral:2 colours:1 cyclic:2 rows:6 cols:3] |C|=4608 gain=10.17b outs=1008  progsD=30 progs=27 fitD=1.00 fitC=1.00 evals=1706 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitV or 4
d22278a0  func[dihedral:3 colours:15 rows:4 cols:4] real[n/a] |C|=780 gain=7.61b outs=780  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d23f8c26  func[dihedral:3 colours:36 cyclic:2 rows:6 cols:6] real[n/a] |C|=20005+ gain=12.70b outs=3903  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d282b262  func[dihedral:3 colours:36 cyclic:2 rows:11 cols:6] real[n/a] |C|=20013+ gain=12.70b outs=16978  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d2abd087  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d2acf2cb  func[dihedral:3 colours:3 cyclic:2 rows:9 cols:8] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d304284e  func[dihedral:3 colours:1] real[n/a] |C|=32 gain=4.00b outs=32  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d364b489  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=3200 gain=10.64b outs=3200  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d37a1ef5  func[dihedral:3 colours:3 cyclic:2 rows:11 cols:9] real[n/a] |C|=20000+ gain=12.70b outs=11234  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d406998b  func[dihedral:3 colours:1 rows:2 cols:12] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d43fd935  func[dihedral:3 colours:21 cyclic:2 rows:9 cols:9] real[n/a] |C|=20001+ gain=12.70b outs=19974  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d4469b4b  func[dihedral:3 colours:6 rows:3 cols:3] real[n/a] |C|=8064 gain=10.17b outs=9  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
d47aa2ff  func[dihedral:3 colours:21 cyclic:2 rows:9 cols:20] real[n/a] |C|=20000+ gain=12.70b outs=14820  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d492a647  func[dihedral:3 colours:10 cyclic:2 rows:14 cols:16] real[n/a] |C|=20013+ gain=13.29b outs=20013  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d4a91cb9  func[dihedral:3 colours:2 rows:6 cols:7] real[n/a] |C|=9344 gain=11.60b outs=9344  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d4b1c2b1  func[dihedral:3 colours:36 rows:10 cols:9] real[n/a] |C|=20000+ gain=11.48b outs=10791  progsD=0 progs=0 fitD=0.43 fitC=0.43 evals=3226 sym=0/1(ok 0) equiv=NO det=-  unfit  :: scale 2
d4c90558  func[dihedral:3 colours:36 cyclic:2 rows:22 cols:22] real[n/a] |C|=20000+ gain=12.70b outs=8934  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d4f3cd78  func[dihedral:3 colours:1 cyclic:2 rows:3 cols:7] real[n/a] |C|=20000+ gain=13.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 0) equiv=yes det=-  unfit  :: id
d511f180  func[dihedral:3 colours:36 cyclic:2 rows:4 cols:4] real[n/a] |C|=20007+ gain=12.70b outs=20007  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d56f2372  func[dihedral:3 colours:28 cyclic:2 rows:19 cols:17] real[n/a] |C|=20005+ gain=12.70b outs=1700  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d5c634a2  func[dihedral:1 colours:1 rows:14 cols:11] real[n/a] |C|=2736 gain=8.61b outs=384  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
d5d6de2d  func[dihedral:3 colours:1 cyclic:2 rows:24 cols:24] real[dihedral:3 rows:7 cols:8] |C|=40 gain=3.74b outs=20  progsD=3 progs=3 fitD=1.00 fitC=1.00 evals=4404 sym=0/2(ok 0) equiv=yes,yes det=yes,yes  SOLVED  :: fillEnclosed 3 ; recolour 2 0
d631b094  func[dihedral:2 colours:15 cyclic:2 rows:2 cols:3] real[dihedral:2 colours:15 cyclic:2 rows:2 cols:3] |C|=378 gain=6.56b outs=24  progsD=58 progs=44 fitD=1.00 fitC=1.00 evals=2244 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: countRow
d687bc17  func[dihedral:3 colours:28 cyclic:2 rows:11 cols:13] real[n/a] |C|=20001+ gain=12.70b outs=11467  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d6ad076f  func[dihedral:3 colours:28 cyclic:2 rows:2 cols:1] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d89b689b  func[dihedral:3 colours:36 cyclic:2 rows:6 cols:7] real[n/a] |C|=20004+ gain=12.70b outs=5242  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d8c310e9  func[dihedral:3 colours:15 cyclic:2 cols:1] real[n/a] |C|=20000+ gain=12.70b outs=18932  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d90796e8  func[dihedral:3 colours:6 rows:4 cols:6] real[n/a] |C|=20000+ gain=12.70b outs=14969  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d931c21c  func[dihedral:3 colours:1 cyclic:2 rows:10 cols:9] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.25 fitC=0.20 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d94c3b52  func[dihedral:3 colours:3 cyclic:2 rows:16 cols:24] real[n/a] |C|=20007+ gain=12.70b outs=20007  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d9f24cd1  func[dihedral:3 colours:3 cyclic:2] real[n/a] |C|=9600 gain=12.23b outs=9600  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
d9fac9be  func[dihedral:3 colours:15 cyclic:2 rows:11 cols:11] real[dihedral:1 colours:15 rows:11 cols:11] |C|=240 gain=5.91b outs=6  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=763 sym=0/1(ok 0) equiv=yes det=-  fit-only  :: shiftRight ; topHalf ; cropSmallest
da2b0fe3  func[dihedral:3 colours:10 cyclic:2 rows:7 cols:9] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
da515329  func[dihedral:2 colours:1 rows:4 cols:4] real[n/a] |C|=24 gain=3.00b outs=24  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
dae9d2b5  func[dihedral:3 colours:3 cyclic:2 rows:1 cols:3] real[dihedral:2 colours:1 cyclic:2 rows:1 cols:3] |C|=720 gain=7.17b outs=99  progsD=35 progs=29 fitD=1.00 fitC=1.00 evals=1137 sym=0/2(ok 0) equiv=yes,yes det=yes,yes  SOLVED  :: splitH or 6
db3e9e38  func[dihedral:3 colours:1 rows:1 cols:1] real[n/a] |C|=24 gain=3.58b outs=24  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
db93a21d  func[dihedral:3 colours:1 rows:10 cols:10] real[n/a] |C|=64 gain=4.00b outs=64  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
dbc1a6ce  func[dihedral:3 colours:1 cyclic:2 rows:8 cols:15] real[n/a] |C|=20004+ gain=12.29b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
dc0a314f  func[dihedral:3 colours:36 cyclic:2 rows:14 cols:14] real[n/a] |C|=20013+ gain=12.70b outs=11070  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
dc1df850  func[dihedral:3 colours:21 cyclic:2 rows:4 cols:4] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.33 fitC=0.17 evals=3225 sym=0/1(ok 0) equiv=NO det=-  unfit  :: box 1
dc2aa30b  func[dihedral:3 colours:3 cyclic:2 rows:10 cols:9] real[n/a] |C|=20009+ gain=12.70b outs=19991  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
dc2e9a9d  func[dihedral:3 colours:1 cyclic:2 rows:8 cols:10] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
dc433765  func[dihedral:3 colours:1 rows:3 cols:3] real[n/a] |C|=792 gain=6.82b outs=464  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/2(ok 1) equiv=yes,yes det=-,-  unfit  :: id
dd2401ed  func[dihedral:3 colours:6 cyclic:2 rows:4 cols:4] real[n/a] |C|=20004+ gain=12.29b outs=19240  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ddf7fa4f  func[dihedral:3 colours:36 cyclic:2 rows:9 cols:9] real[n/a] |C|=20000+ gain=12.70b outs=14100  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
de1cd16c  func[dihedral:3 colours:28 cyclic:2 rows:17 cols:18] real[n/a] |C|=20001+ gain=12.29b outs=8  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
de493100  func[dihedral:3 colours:36 cyclic:2 rows:29 cols:29] real[n/a] |C|=20010+ gain=12.29b outs=17916  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ded97339  func[dihedral:3 colours:1 cyclic:2] real[n/a] |C|=4000 gain=10.38b outs=4000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
df8cc377  func[dihedral:3 colours:28 cyclic:2 rows:16 cols:20] real[n/a] |C|=20008+ gain=12.70b outs=15176  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e0fb7511  func[dihedral:3 colours:1 cyclic:2 rows:12 cols:12] real[n/a] |C|=20004+ gain=12.70b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e133d23d  func[dihedral:3 colours:6 cyclic:2 rows:1 cols:4] real[dihedral:2 colours:3 cyclic:1 rows:1 cols:4] |C|=360 gain=6.17b outs=30  progsD=50 progs=44 fitD=1.00 fitC=1.00 evals=2029 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitH or 2
e179c5f4  func[dihedral:3 colours:1] real[n/a] |C|=48 gain=4.00b outs=28  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e1baa8a4  func[dihedral:3 colours:36 cyclic:2 rows:12 cols:12] real[dihedral:3 colours:36 rows:10 cols:11] |C|=20002+ gain=12.29b outs=17764  progsD=2 progs=2 fitD=1.00 fitC=1.00 evals=714 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: dedupRows ; dedupCols
e1d2900e  func[dihedral:3 colours:3 cyclic:2 rows:6 cols:7] real[n/a] |C|=20003+ gain=12.70b outs=8547  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e2092e0c  func[dihedral:3 colours:36 cyclic:2 rows:14 cols:14] real[n/a] |C|=20006+ gain=12.70b outs=20006  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e21a174a  func[dihedral:3 colours:28 cyclic:2 rows:7 cols:11] real[n/a] |C|=20004+ gain=12.70b outs=19912  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
e21d9049  func[dihedral:3 colours:15 cyclic:2 rows:5 cols:5] real[n/a] |C|=20000+ gain=13.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e26a3af2  func[dihedral:3 colours:36 cyclic:2 rows:14 cols:16] real[n/a] |C|=20021+ gain=12.70b outs=2934  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e345f17b  func[dihedral:3 colours:3 cyclic:2 rows:3 cols:7] real[dihedral:2 colours:1 cyclic:2 rows:3 cols:4] |C|=3072 gain=9.58b outs=304  progsD=31 progs=30 fitD=1.00 fitC=1.00 evals=1830 sym=0/2(ok 0) equiv=yes,yes det=yes,yes  SOLVED  :: splitH nor 4
e3497940  func[dihedral:3 colours:36 cyclic:2 rows:8 cols:5] real[dihedral:1 colours:36 cyclic:1 rows:8 cols:5] |C|=20000+ gain=12.70b outs=8221  progsD=1 progs=1 fitD=1.00 fitC=1.00 evals=567 sym=0/1(ok 0) equiv=yes det=-  SOLVED  :: symmetrizeH ; leftHalf
e4075551  func[dihedral:3 colours:28 cyclic:2 rows:13 cols:14] real[n/a] |C|=20007+ gain=12.70b outs=15135  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e40b9e2f  func[dihedral:3 colours:28 cyclic:2 rows:3 cols:2] real[n/a] |C|=20002+ gain=12.70b outs=12911  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e41c6fd3  func[dihedral:3 colours:21 cyclic:2 rows:16 cols:29] real[n/a] |C|=20031+ gain=12.70b outs=13472  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e48d4e1a  func[dihedral:3 colours:21 rows:1 cols:1] real[n/a] |C|=2016 gain=8.98b outs=168  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e5062a87  func[dihedral:3 colours:3 cyclic:2 rows:9 cols:9] real[n/a] |C|=20006+ gain=12.70b outs=20006  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e509e548  func[dihedral:3 colours:1 cyclic:2 rows:18 cols:20] real[n/a] |C|=20011+ gain=12.70b outs=20011  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e50d258f  func[dihedral:3 colours:6 cyclic:2 rows:8 cols:9] real[n/a] |C|=20002+ gain=12.70b outs=8605  progsD=0 progs=0 fitD=0.67 fitC=0.27 evals=3236 sym=0/1(ok 0) equiv=NO det=-  unfit  :: leftHalf ; cropSmallest
e57337a4  func[dihedral:3 colours:10 rows:12 cols:12] real[n/a] |C|=100 gain=5.06b outs=80  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e5790162  func[dihedral:3 colours:6 rows:3 cols:5] real[n/a] |C|=5952 gain=10.22b outs=3264  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e5c44e8f  func[dihedral:1 colours:3 cyclic:1] real[n/a] |C|=330 gain=6.78b outs=330  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e619ca6e  func[dihedral:3 colours:1] real[n/a] |C|=48 gain=4.00b outs=48  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e633a9e5  func[dihedral:3 colours:36 cyclic:2 rows:2 cols:2] real[n/a] |C|=20007+ gain=12.70b outs=20007  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e66aafb8  func[dihedral:3 colours:36 cyclic:2 rows:22 cols:23] real[n/a] |C|=20016+ gain=11.97b outs=18031  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e6721834  func[dihedral:3 colours:21 cyclic:2 rows:22 cols:12] real[n/a] |C|=20005+ gain=12.70b outs=13284  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e681b708  func[dihedral:3 colours:21 cyclic:2 rows:26 cols:25] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e69241bd  func[dihedral:3 colours:28 cyclic:2 rows:8 cols:8] real[n/a] |C|=20003+ gain=12.70b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e6de6e8f  func[dihedral:3 colours:1 rows:3 cols:7] real[n/a] |C|=768 gain=8.00b outs=128  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e73095fd  func[dihedral:3 colours:1 cyclic:2 rows:15 cols:18] real[n/a] |C|=20003+ gain=12.70b outs=20003  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e74e1818  func[dihedral:3 colours:28 cyclic:2 rows:11 cols:13] real[n/a] |C|=20003+ gain=12.70b outs=19132  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e760a62e  func[dihedral:3 colours:6 cyclic:2 rows:15 cols:22] real[n/a] |C|=20003+ gain=12.70b outs=11536  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e7639916  func[dihedral:3 colours:1 cyclic:2 rows:3 cols:5] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e76a88a6  func[dihedral:3 colours:21 cyclic:2] real[n/a] |C|=20000+ gain=13.29b outs=6987  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e78887d1  func[dihedral:3 colours:10 cyclic:2 rows:9 cols:7] real[n/a] |C|=20001+ gain=12.29b outs=18342  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e7a25a18  func[dihedral:3 colours:28 cyclic:1 rows:9 cols:6] real[n/a] |C|=20003+ gain=13.29b outs=7853  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e7b06bea  func[dihedral:3 colours:36 cyclic:1] real[n/a] |C|=20004+ gain=11.97b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e7dd8335  func[dihedral:3 colours:1 rows:5 cols:7] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e8593010  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e872b94a  func[dihedral:3 colours:1 cyclic:2 rows:11 cols:11] real[n/a] |C|=20000+ gain=12.29b outs=8  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e88171ec  func[dihedral:3 colours:15 cyclic:2 rows:20 cols:20] real[n/a] |C|=20008+ gain=12.70b outs=20008  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e8dc4411  func[dihedral:3 colours:21 cyclic:2 rows:3] real[n/a] |C|=20000+ gain=12.70b outs=19983  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e95e3d8e  func[dihedral:3 colours:28 cyclic:2 rows:21 cols:21] real[n/a] |C|=20011+ gain=12.70b outs=17321  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e9614598  func[dihedral:3 colours:1 cyclic:2 rows:2 cols:2] real[n/a] |C|=800 gain=8.64b outs=800  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
e98196ab  func[dihedral:3 colours:28 cyclic:2 rows:8 cols:10] real[n/a] |C|=20006+ gain=12.70b outs=12378  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e99362f0  func[dihedral:3 colours:15 cyclic:2 rows:10 cols:8] real[n/a] |C|=20000+ gain=11.70b outs=16084  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e9ac8c9e  func[dihedral:3 colours:36 cyclic:2 rows:10 cols:10] real[n/a] |C|=20000+ gain=12.70b outs=9853  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e9afcf9a  func[dihedral:1 colours:21 cyclic:1 rows:1] real[n/a] |C|=42 gain=4.39b outs=42  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=1/1(ok 1) equiv=yes det=-  unfit  :: id
e9b4f6fc  func[dihedral:3 colours:36 cyclic:2 rows:8 cols:8] real[n/a] |C|=20006+ gain=12.29b outs=7154  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e9bb6954  func[dihedral:3 colours:36 cyclic:2 rows:17 cols:15] real[n/a] |C|=20019+ gain=12.29b outs=18431  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
e9c9d9a1  func[dihedral:3 colours:1 rows:8 cols:6] real[n/a] |C|=20000+ gain=12.70b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ea32f347  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20006+ gain=12.29b outs=20006  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ea786f4a  func[dihedral:3 colours:10 cyclic:2 rows:4 cols:4] real[n/a] |C|=415 gain=7.11b outs=415  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ea959feb  func[dihedral:3 colours:36 cyclic:2 rows:21 cols:24] real[n/a] |C|=20004+ gain=12.70b outs=19526  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ea9794b1  func[dihedral:3 colours:10 cyclic:2 rows:9 cols:9] real[n/a] |C|=20002+ gain=11.70b outs=17095  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
eb281b96  func[dihedral:3 colours:6 rows:10 cols:14] real[n/a] |C|=20000+ gain=13.29b outs=14333  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
eb5a1d5d  func[dihedral:3 colours:21 cyclic:1 rows:19 cols:21] real[dihedral:3 colours:21 rows:17 cols:18] |C|=8736 gain=11.51b outs=1092  progsD=2 progs=2 fitD=1.00 fitC=1.00 evals=560 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: dedupRows ; dedupCols
ec883f72  func[dihedral:3 colours:28 cyclic:2 cols:1] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ecaa0ec1  func[dihedral:3 colours:6 cyclic:2 rows:7 cols:8] real[n/a] |C|=20002+ gain=12.29b outs=10707  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ecdecbb3  func[dihedral:3 colours:3 cyclic:2 cols:1] real[n/a] |C|=19464 gain=12.66b outs=16656  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ed36ccf7  func[dihedral:2 colours:10] real[colours:10] |C|=20 gain=2.32b outs=20  progsD=25 progs=15 fitD=1.00 fitC=1.00 evals=694 sym=0/1(ok 0) equiv=yes det=NO  SOLVED  :: rot270
ed74f2f2  func[dihedral:3 colours:1 rows:2 cols:6] real[n/a] |C|=96 gain=4.00b outs=52  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ed98d772  func[dihedral:1 colours:15 rows:4 cols:3] real[n/a] |C|=120 gain=4.58b outs=120  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ef135b50  func[dihedral:3 colours:1 cyclic:2 rows:9 cols:9] real[n/a] |C|=20004+ gain=12.70b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ef26cbf6  func[dihedral:3 colours:28 cyclic:2 rows:10 cols:10] real[n/a] |C|=20023+ gain=13.29b outs=16520  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f0afb749  func[dihedral:3 colours:10 rows:7 cols:7] real[n/a] |C|=380 gain=6.98b outs=220  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f0df5ff0  func[dihedral:3 colours:36 cyclic:2 rows:14 cols:14] real[n/a] |C|=20019+ gain=12.70b outs=20019  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f15e1fac  func[dihedral:3 colours:3 rows:6 cols:6] real[n/a] |C|=3648 gain=10.25b outs=1344  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f1cefba8  func[dihedral:3 colours:15 cyclic:2 rows:18 cols:17] real[n/a] |C|=20014+ gain=12.70b outs=14264  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f21745ec  func[dihedral:3 colours:36 cyclic:2 rows:8 cols:6] real[n/a] |C|=20000+ gain=12.70b outs=13678  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f25fbde4  func[dihedral:3 colours:1 rows:4 cols:4] real[dihedral:3 colours:1 rows:1] |C|=48 gain=4.00b outs=28  progsD=2 progs=2 fitD=1.00 fitC=1.00 evals=959 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: scale 2 ; cropBBox
f25ffba3  func[dihedral:3 colours:28 cyclic:2] real[dihedral:2 colours:28 cyclic:1] |C|=20001+ gain=13.29b outs=12348  progsD=56 progs=23 fitD=1.00 fitC=1.00 evals=2205 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: symmetrizeV
f2829549  func[dihedral:3 colours:6 cyclic:2 rows:2 cols:4] real[dihedral:2 colours:3 cyclic:1 rows:2 cols:3] |C|=1296 gain=8.02b outs=132  progsD=50 progs=47 fitD=1.00 fitC=1.00 evals=2264 sym=0/1(ok 0) equiv=yes det=NO  SOLVED  :: splitH nor 3
f35d900a  func[dihedral:3 colours:15 cyclic:2 cols:1] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f3b10344  func[dihedral:3 colours:21 cyclic:2 rows:1 cols:3] real[n/a] |C|=20001+ gain=12.70b outs=20001  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f3cdc58f  func[dihedral:3 colours:10 cyclic:2 rows:9 cols:8] real[n/a] |C|=20008+ gain=12.70b outs=6204  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f3e62deb  func[dihedral:2 colours:6 rows:2 cols:2] real[n/a] |C|=544 gain=6.50b outs=184  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=2/2(ok 0) equiv=yes,yes det=-,-  unfit  :: id
f4081712  func[dihedral:3 colours:36 cyclic:2 rows:22 cols:22] real[n/a] |C|=20007+ gain=11.97b outs=8154  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f45f5ca7  func[dihedral:3 colours:10 rows:2 cols:2] real[n/a] |C|=19200 gain=12.64b outs=19200  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f5aa3634  func[dihedral:3 colours:36 cyclic:2 rows:14 cols:13] real[n/a] |C|=20000+ gain=12.70b outs=4370  progsD=0 progs=0 fitD=0.67 fitC=0.49 evals=3224 sym=0/1(ok 0) equiv=NO det=-  SOLVED  :: cropLargest
f5b8619d  func[dihedral:2 colours:10 cyclic:2 rows:6 cols:6] real[n/a] |C|=830 gain=8.11b outs=830  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f5c89df1  func[dihedral:3 colours:6 cyclic:2] real[n/a] |C|=20000+ gain=12.70b outs=3924  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f76d97a5  func[dihedral:3 colours:15 cyclic:2 rows:4 cols:4] real[n/a] |C|=20001+ gain=12.70b outs=8540  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f823c43c  func[dihedral:3 colours:21 cyclic:2 rows:15 cols:18] real[n/a] |C|=20004+ gain=13.29b outs=11150  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f83cb3f6  func[dihedral:3 colours:15 cyclic:2 rows:13 cols:11] real[n/a] |C|=20003+ gain=12.70b outs=6530  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f8a8fe49  func[dihedral:3 colours:3 cyclic:2 rows:4 cols:6] real[n/a] |C|=20004+ gain=12.70b outs=11953  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f8b3ba0a  func[dihedral:3 colours:21 cyclic:2 rows:13 cols:15] real[n/a] |C|=20005+ gain=12.29b outs=420  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f8be4b64  func[dihedral:3 colours:28 cyclic:2 rows:29 cols:27] real[n/a] |C|=20000+ gain=12.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f8c80d96  func[dihedral:3 colours:10 cyclic:2] real[n/a] |C|=6000 gain=10.97b outs=6000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f8ff0b80  func[dihedral:3 colours:28 cyclic:2 rows:9 cols:9] real[n/a] |C|=20001+ gain=12.70b outs=672  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f9012d9b  func[dihedral:3 colours:15 rows:5 cols:5] real[n/a] |C|=480 gain=7.32b outs=126  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f9a67cb5  func[dihedral:3 colours:3 cyclic:2 rows:1 cols:1] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
f9d67f8b  func[dihedral:3 colours:36 cyclic:2 rows:29 cols:29] real[n/a] |C|=20024+ gain=12.29b outs=19818  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
fafd9572  func[dihedral:3 colours:15 cyclic:2 rows:13 cols:17] real[n/a] |C|=20010+ gain=13.29b outs=16442  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
fafffa47  func[dihedral:3 colours:3 cyclic:2 rows:3] real[dihedral:2 colours:1 cyclic:2 rows:3] |C|=720 gain=7.17b outs=72  progsD=34 progs=27 fitD=1.00 fitC=1.00 evals=1052 sym=0/1(ok 0) equiv=yes det=yes  SOLVED  :: splitV nor 2
fb791726  func[dihedral:1 colours:10 rows:12 cols:12] real[n/a] |C|=990 gain=8.37b outs=990  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
fc754716  func[dihedral:3 colours:15 cyclic:2 rows:2] real[n/a] |C|=804 gain=7.65b outs=804  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
fcb5c309  func[dihedral:3 colours:15 cyclic:2 rows:15 cols:15] real[n/a] |C|=20000+ gain=12.70b outs=7192  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
fcc82909  func[dihedral:3 colours:28 cyclic:2 rows:5 cols:8] real[n/a] |C|=20004+ gain=12.70b outs=20004  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
fd096ab6  func[dihedral:3 colours:36 cyclic:2 rows:13 cols:12] real[n/a] |C|=20000+ gain=13.29b outs=20000  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
fd4b2b02  func[dihedral:3 colours:1 rows:1 cols:1] real[n/a] |C|=64 gain=4.42b outs=64  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
fe9372f3  func[dihedral:3 colours:1 rows:15 cols:2] real[n/a] |C|=32 gain=4.00b outs=32  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
fea12743  func[dihedral:3 colours:1 cyclic:2 rows:11 cols:9] real[n/a] |C|=20002+ gain=12.70b outs=20002  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
feca6190  func[dihedral:1 colours:28 rows:11 cols:11] real[n/a] |C|=2960 gain=9.21b outs=2216  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ff28f65a  func[dihedral:1 colours:1 rows:5 cols:4] real[n/a] |C|=52 gain=2.70b outs=11  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/3(ok 0) equiv=yes,yes,yes det=-,-,-  unfit  :: id
ff72ca3e  func[dihedral:3 colours:3 rows:20 cols:10] real[n/a] |C|=7296 gain=10.83b outs=7296  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
ff805c23  func[dihedral:3 colours:28 cyclic:2 rows:22 cols:22] real[n/a] |C|=20002+ gain=12.70b outs=2465  progsD=0 progs=0 fitD=0.00 fitC=0.00 evals=3217 sym=0/1(ok 0) equiv=yes det=-  unfit  :: id
```
