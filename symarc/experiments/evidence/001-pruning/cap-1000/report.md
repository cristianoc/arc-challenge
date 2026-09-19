# 001-pruning: program-entropy policy comparison

Data: `../data/training`. Tasks: 400. Workers: 12. Seeds: [0, 1, 2, 3, 4].

## Before symmetry

Fixed depth-2 syntax class; no repair or search fallback. **55/400 tasks have a fitting program**; 345 are excluded from conditional metrics and unsolved in all-task accuracy.

| Initial measurement | Value |
|---|---:|
| Correct selected programs | 53/400 (13.25%) |
| Tasks with at least one correct program | 54 |
| Already determined among eligible | 32/55 |
| Undetermined among eligible | 23 |
| Mean program entropy, eligible tasks | 4.21 bits |
| Mean answer entropy, eligible tasks | 0.68 bits |

## Predictive results

Single chosen program, exact match on every test output in a task. Means and ranges vary the seed on the **same tasks**; seed repetitions are not independent task observations.

| Policy | Mean correct tasks (min–max) | All-task accuracy | Eligible-task accuracy | Test-grid accuracy (mean) |
|---|---:|---:|---:|---:|
| no-symmetry | 53.00 (53–53) / 400 | 13.25% | 96.36% | 13.70% |
| fixed-order | 53.00 (53–53) / 400 | 13.25% | 96.36% | 13.70% |
| random-order | 51.00 (50–52) / 400 | 12.75% | 92.73% | 13.22% |
| marginal-pruning | 51.00 (51–51) / 400 | 12.75% | 92.73% | 13.22% |

## Entropy and failure diagnostics

Answer entropy includes undefinedness as an outcome; determination requires unanimous **defined** answers at every test input. Entropies average over eligible tasks (within each task, average over its test inputs). Lost-correct counts tasks where every initially available correct program was removed.

| Policy | Mean pruning bits | Mean final program entropy | Mean final answer entropy | Newly determined tasks | Unanimous wrong tasks | Lost all correct programs |
|---|---:|---:|---:|---:|---:|---:|
| no-symmetry | 0.00 | 4.21 | 0.68 | 0.00 | 1.00 | 0.00 |
| fixed-order | 0.41 | 3.79 | 0.02 | 18.00 | 2.00 | 1.00 |
| random-order | 0.47 | 3.74 | 0.02 | 17.60 | 4.00 | 3.00 |
| marginal-pruning | 0.57 | 3.63 | 0.02 | 18.00 | 4.00 | 3.00 |

## Paired task changes

Wins/losses compare identical tasks and seeds; values are mean counts per seed.

| Policy | Wins vs no symmetry | Losses vs no symmetry | Wins vs fixed | Losses vs fixed |
|---|---:|---:|---:|---:|
| no-symmetry | 0.00 | 0.00 | 1.00 | 1.00 |
| fixed-order | 1.00 | 1.00 | 0.00 | 0.00 |
| random-order | 0.00 | 2.00 | 0.00 | 2.00 |
| marginal-pruning | 1.00 | 3.00 | 0.00 | 2.00 |

## Work and approximation

Shared preparation (enumeration, functionality prefilter, cached program answers): 2.35 s. Each policy/seed batch uses 12 workers and is run separately. Charged work covers program-grid checks, not closure construction; actual work is not assumed equal merely because the ceilings match.

| Policy | Mean selection wall seconds | Mean charged checks | Mean attempted trials | Mean capped completed trials | Mean budget-exhausted tasks |
|---|---:|---:|---:|---:|---:|
| no-symmetry | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| fixed-order | 0.14 | 975645.40 | 247.00 | 68.00 | 0.00 |
| random-order | 1.02 | 1094267.80 | 282.60 | 78.40 | 0.00 |
| marginal-pruning | 1.08 | 830637.00 | 1205.00 | 0.00 | 0.00 |

Closure cap: 1000. Extra sample attempts: 64. Per-task/per-policy charge ceiling: 500000. Full closures are checked exactly when completed; otherwise constraints are sampled. Surviving sets are nested, but they are **observed admissibility sets**, not proven full-closure version spaces. Selection also checks test-input commutation. Fixed/random can accept zero-pruning generators; marginal pruning stops when none strictly prunes.

This is development evidence from public training tasks only. The comparison disables stable-solver repair/fallback and stops on answer determination; it does not compare the complete stable solver or establish performance on public evaluation tasks.

## Per-seed task accuracy

| Seed | Policy | Correct tasks |
|---:|---|---:|
| 0 | no-symmetry | 53/400 |
| 0 | fixed-order | 53/400 |
| 0 | random-order | 50/400 |
| 0 | marginal-pruning | 51/400 |
| 1 | no-symmetry | 53/400 |
| 1 | fixed-order | 53/400 |
| 1 | random-order | 51/400 |
| 1 | marginal-pruning | 51/400 |
| 2 | no-symmetry | 53/400 |
| 2 | fixed-order | 53/400 |
| 2 | random-order | 51/400 |
| 2 | marginal-pruning | 51/400 |
| 3 | no-symmetry | 53/400 |
| 3 | fixed-order | 53/400 |
| 3 | random-order | 51/400 |
| 3 | marginal-pruning | 51/400 |
| 4 | no-symmetry | 53/400 |
| 4 | fixed-order | 53/400 |
| 4 | random-order | 52/400 |
| 4 | marginal-pruning | 51/400 |

## Outcomes for initially undetermined eligible tasks

Tasks whose answers already agree cannot change under the common stopping rule. `correct` scores the chosen shortest survivor; `lost` means all initially correct programs were removed.

| Task | Seed | Policy | Programs before → after | Pruning bits | Answer bits before → after | Correct | Determined | Lost | Budget hit | Chosen program |
|---|---:|---|---:|---:|---:|---|---|---|---|---|
| 0520fde7 | 0 | no-symmetry | 36 → 36 | 0.00 | 0.18 → 0.18 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 0 | fixed-order | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 0 | random-order | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 0 | marginal-pruning | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 1 | no-symmetry | 36 → 36 | 0.00 | 0.18 → 0.18 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 1 | fixed-order | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 1 | random-order | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 1 | marginal-pruning | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 2 | no-symmetry | 36 → 36 | 0.00 | 0.18 → 0.18 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 2 | fixed-order | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 2 | random-order | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 2 | marginal-pruning | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 3 | no-symmetry | 36 → 36 | 0.00 | 0.18 → 0.18 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 3 | fixed-order | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 3 | random-order | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 3 | marginal-pruning | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 4 | no-symmetry | 36 → 36 | 0.00 | 0.18 → 0.18 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 4 | fixed-order | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 4 | random-order | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 0520fde7 | 4 | marginal-pruning | 36 → 33 | 0.13 | 0.18 → 0.20 | true | false | false | false | `splitH and 2` |
| 1cf80156 | 0 | no-symmetry | 100 → 100 | 0.00 | 1.87 → 1.87 | true | false | false | false | `cropBBox` |
| 1cf80156 | 0 | fixed-order | 100 → 58 | 0.79 | 1.87 → 0.00 | true | true | false | false | `cropBBox` |
| 1cf80156 | 0 | random-order | 100 → 6 | 4.06 | 1.87 → 0.00 | false | true | true | false | `cropBBox ; dedupCols` |
| 1cf80156 | 0 | marginal-pruning | 100 → 6 | 4.06 | 1.87 → 0.00 | false | true | true | false | `cropBBox ; dedupCols` |
| 1cf80156 | 1 | no-symmetry | 100 → 100 | 0.00 | 1.87 → 1.87 | true | false | false | false | `cropBBox` |
| 1cf80156 | 1 | fixed-order | 100 → 58 | 0.79 | 1.87 → 0.00 | true | true | false | false | `cropBBox` |
| 1cf80156 | 1 | random-order | 100 → 6 | 4.06 | 1.87 → 0.00 | false | true | true | false | `cropBBox ; dedupCols` |
| 1cf80156 | 1 | marginal-pruning | 100 → 6 | 4.06 | 1.87 → 0.00 | false | true | true | false | `cropBBox ; dedupCols` |
| 1cf80156 | 2 | no-symmetry | 100 → 100 | 0.00 | 1.87 → 1.87 | true | false | false | false | `cropBBox` |
| 1cf80156 | 2 | fixed-order | 100 → 58 | 0.79 | 1.87 → 0.00 | true | true | false | false | `cropBBox` |
| 1cf80156 | 2 | random-order | 100 → 6 | 4.06 | 1.87 → 0.00 | false | true | true | false | `cropBBox ; dedupCols` |
| 1cf80156 | 2 | marginal-pruning | 100 → 6 | 4.06 | 1.87 → 0.00 | false | true | true | false | `cropBBox ; dedupCols` |
| 1cf80156 | 3 | no-symmetry | 100 → 100 | 0.00 | 1.87 → 1.87 | true | false | false | false | `cropBBox` |
| 1cf80156 | 3 | fixed-order | 100 → 58 | 0.79 | 1.87 → 0.00 | true | true | false | false | `cropBBox` |
| 1cf80156 | 3 | random-order | 100 → 6 | 4.06 | 1.87 → 0.00 | false | true | true | false | `cropBBox ; dedupCols` |
| 1cf80156 | 3 | marginal-pruning | 100 → 6 | 4.06 | 1.87 → 0.00 | false | true | true | false | `cropBBox ; dedupCols` |
| 1cf80156 | 4 | no-symmetry | 100 → 100 | 0.00 | 1.87 → 1.87 | true | false | false | false | `cropBBox` |
| 1cf80156 | 4 | fixed-order | 100 → 58 | 0.79 | 1.87 → 0.00 | true | true | false | false | `cropBBox` |
| 1cf80156 | 4 | random-order | 100 → 58 | 0.79 | 1.87 → 0.00 | true | true | false | false | `cropBBox` |
| 1cf80156 | 4 | marginal-pruning | 100 → 6 | 4.06 | 1.87 → 0.00 | false | true | true | false | `cropBBox ; dedupCols` |
| 1e0a9b12 | 0 | no-symmetry | 45 → 45 | 0.00 | 2.38 → 2.38 | true | false | false | false | `gravityDown` |
| 1e0a9b12 | 0 | fixed-order | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 0 | random-order | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 0 | marginal-pruning | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 1 | no-symmetry | 45 → 45 | 0.00 | 2.38 → 2.38 | true | false | false | false | `gravityDown` |
| 1e0a9b12 | 1 | fixed-order | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 1 | random-order | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 1 | marginal-pruning | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 2 | no-symmetry | 45 → 45 | 0.00 | 2.38 → 2.38 | true | false | false | false | `gravityDown` |
| 1e0a9b12 | 2 | fixed-order | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 2 | random-order | 45 → 24 | 0.91 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 2 | marginal-pruning | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 3 | no-symmetry | 45 → 45 | 0.00 | 2.38 → 2.38 | true | false | false | false | `gravityDown` |
| 1e0a9b12 | 3 | fixed-order | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 3 | random-order | 45 → 24 | 0.91 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 3 | marginal-pruning | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 4 | no-symmetry | 45 → 45 | 0.00 | 2.38 → 2.38 | true | false | false | false | `gravityDown` |
| 1e0a9b12 | 4 | fixed-order | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 4 | random-order | 45 → 24 | 0.91 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1e0a9b12 | 4 | marginal-pruning | 45 → 25 | 0.85 | 2.38 → 0.00 | true | true | false | false | `gravityDown` |
| 1f85a75f | 0 | no-symmetry | 70 → 70 | 0.00 | 1.45 → 1.45 | true | false | false | false | `cropLargest` |
| 1f85a75f | 0 | fixed-order | 70 → 25 | 1.49 | 1.45 → 0.00 | true | true | false | false | `cropLargest` |
| 1f85a75f | 0 | random-order | 70 → 2 | 5.13 | 1.45 → 0.00 | false | true | true | false | `cropLargest ; recolour 2 0` |
| 1f85a75f | 0 | marginal-pruning | 70 → 2 | 5.13 | 1.45 → 0.00 | false | true | true | false | `cropLargest ; recolour 2 0` |
| 1f85a75f | 1 | no-symmetry | 70 → 70 | 0.00 | 1.45 → 1.45 | true | false | false | false | `cropLargest` |
| 1f85a75f | 1 | fixed-order | 70 → 25 | 1.49 | 1.45 → 0.00 | true | true | false | false | `cropLargest` |
| 1f85a75f | 1 | random-order | 70 → 11 | 2.67 | 1.45 → 0.00 | true | true | false | false | `cropLargest` |
| 1f85a75f | 1 | marginal-pruning | 70 → 2 | 5.13 | 1.45 → 0.00 | false | true | true | false | `cropLargest ; recolour 2 0` |
| 1f85a75f | 2 | no-symmetry | 70 → 70 | 0.00 | 1.45 → 1.45 | true | false | false | false | `cropLargest` |
| 1f85a75f | 2 | fixed-order | 70 → 25 | 1.49 | 1.45 → 0.00 | true | true | false | false | `cropLargest` |
| 1f85a75f | 2 | random-order | 70 → 11 | 2.67 | 1.45 → 0.00 | true | true | false | false | `cropLargest` |
| 1f85a75f | 2 | marginal-pruning | 70 → 2 | 5.13 | 1.45 → 0.00 | false | true | true | false | `cropLargest ; recolour 2 0` |
| 1f85a75f | 3 | no-symmetry | 70 → 70 | 0.00 | 1.45 → 1.45 | true | false | false | false | `cropLargest` |
| 1f85a75f | 3 | fixed-order | 70 → 25 | 1.49 | 1.45 → 0.00 | true | true | false | false | `cropLargest` |
| 1f85a75f | 3 | random-order | 70 → 11 | 2.67 | 1.45 → 0.00 | true | true | false | false | `cropLargest` |
| 1f85a75f | 3 | marginal-pruning | 70 → 2 | 5.13 | 1.45 → 0.00 | false | true | true | false | `cropLargest ; recolour 2 0` |
| 1f85a75f | 4 | no-symmetry | 70 → 70 | 0.00 | 1.45 → 1.45 | true | false | false | false | `cropLargest` |
| 1f85a75f | 4 | fixed-order | 70 → 25 | 1.49 | 1.45 → 0.00 | true | true | false | false | `cropLargest` |
| 1f85a75f | 4 | random-order | 70 → 25 | 1.49 | 1.45 → 0.00 | true | true | false | false | `cropLargest` |
| 1f85a75f | 4 | marginal-pruning | 70 → 2 | 5.13 | 1.45 → 0.00 | false | true | true | false | `cropLargest ; recolour 2 0` |
| 3428a4f5 | 0 | no-symmetry | 35 → 35 | 0.00 | 0.09 → 0.09 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 0 | fixed-order | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 0 | random-order | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 0 | marginal-pruning | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 1 | no-symmetry | 35 → 35 | 0.00 | 0.09 → 0.09 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 1 | fixed-order | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 1 | random-order | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 1 | marginal-pruning | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 2 | no-symmetry | 35 → 35 | 0.00 | 0.09 → 0.09 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 2 | fixed-order | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 2 | random-order | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 2 | marginal-pruning | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 3 | no-symmetry | 35 → 35 | 0.00 | 0.09 → 0.09 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 3 | fixed-order | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 3 | random-order | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 3 | marginal-pruning | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 4 | no-symmetry | 35 → 35 | 0.00 | 0.09 → 0.09 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 4 | fixed-order | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 4 | random-order | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3428a4f5 | 4 | marginal-pruning | 35 → 32 | 0.13 | 0.09 → 0.10 | true | false | false | false | `splitV xor 3` |
| 3af2c5a8 | 0 | no-symmetry | 16 → 16 | 0.00 | 1.75 → 1.75 | true | false | false | false | `mirror4` |
| 3af2c5a8 | 0 | fixed-order | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 0 | random-order | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 0 | marginal-pruning | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 1 | no-symmetry | 16 → 16 | 0.00 | 1.75 → 1.75 | true | false | false | false | `mirror4` |
| 3af2c5a8 | 1 | fixed-order | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 1 | random-order | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 1 | marginal-pruning | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 2 | no-symmetry | 16 → 16 | 0.00 | 1.75 → 1.75 | true | false | false | false | `mirror4` |
| 3af2c5a8 | 2 | fixed-order | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 2 | random-order | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 2 | marginal-pruning | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 3 | no-symmetry | 16 → 16 | 0.00 | 1.75 → 1.75 | true | false | false | false | `mirror4` |
| 3af2c5a8 | 3 | fixed-order | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 3 | random-order | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 3 | marginal-pruning | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 4 | no-symmetry | 16 → 16 | 0.00 | 1.75 → 1.75 | true | false | false | false | `mirror4` |
| 3af2c5a8 | 4 | fixed-order | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 4 | random-order | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3af2c5a8 | 4 | marginal-pruning | 16 → 8 | 1.00 | 1.75 → 0.00 | true | true | false | false | `mirror4` |
| 3c9b0459 | 0 | no-symmetry | 117 → 117 | 0.00 | 2.34 → 2.34 | true | false | false | false | `rot180` |
| 3c9b0459 | 0 | fixed-order | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 0 | random-order | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 0 | marginal-pruning | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 1 | no-symmetry | 117 → 117 | 0.00 | 2.34 → 2.34 | true | false | false | false | `rot180` |
| 3c9b0459 | 1 | fixed-order | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 1 | random-order | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 1 | marginal-pruning | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 2 | no-symmetry | 117 → 117 | 0.00 | 2.34 → 2.34 | true | false | false | false | `rot180` |
| 3c9b0459 | 2 | fixed-order | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 2 | random-order | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 2 | marginal-pruning | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 3 | no-symmetry | 117 → 117 | 0.00 | 2.34 → 2.34 | true | false | false | false | `rot180` |
| 3c9b0459 | 3 | fixed-order | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 3 | random-order | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 3 | marginal-pruning | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 4 | no-symmetry | 117 → 117 | 0.00 | 2.34 → 2.34 | true | false | false | false | `rot180` |
| 3c9b0459 | 4 | fixed-order | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 4 | random-order | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 3c9b0459 | 4 | marginal-pruning | 117 → 77 | 0.60 | 2.34 → 0.00 | true | true | false | false | `rot180` |
| 496994bd | 0 | no-symmetry | 32 → 32 | 0.00 | 1.50 → 1.50 | true | false | false | false | `symmetrizeV` |
| 496994bd | 0 | fixed-order | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 496994bd | 0 | random-order | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 496994bd | 0 | marginal-pruning | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 496994bd | 1 | no-symmetry | 32 → 32 | 0.00 | 1.50 → 1.50 | true | false | false | false | `symmetrizeV` |
| 496994bd | 1 | fixed-order | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 496994bd | 1 | random-order | 32 → 22 | 0.54 | 1.50 → 0.00 | true | true | false | false | `symmetrizeV` |
| 496994bd | 1 | marginal-pruning | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 496994bd | 2 | no-symmetry | 32 → 32 | 0.00 | 1.50 → 1.50 | true | false | false | false | `symmetrizeV` |
| 496994bd | 2 | fixed-order | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 496994bd | 2 | random-order | 32 → 21 | 0.61 | 1.50 → 0.00 | true | true | false | false | `symmetrizeV` |
| 496994bd | 2 | marginal-pruning | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 496994bd | 3 | no-symmetry | 32 → 32 | 0.00 | 1.50 → 1.50 | true | false | false | false | `symmetrizeV` |
| 496994bd | 3 | fixed-order | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 496994bd | 3 | random-order | 32 → 22 | 0.54 | 1.50 → 0.00 | true | true | false | false | `symmetrizeV` |
| 496994bd | 3 | marginal-pruning | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 496994bd | 4 | no-symmetry | 32 → 32 | 0.00 | 1.50 → 1.50 | true | false | false | false | `symmetrizeV` |
| 496994bd | 4 | fixed-order | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 496994bd | 4 | random-order | 32 → 22 | 0.54 | 1.50 → 0.00 | true | true | false | false | `symmetrizeV` |
| 496994bd | 4 | marginal-pruning | 32 → 2 | 4.00 | 1.50 → 0.00 | true | true | false | false | `symmetrizeH ; symmetrizeV` |
| 5582e5ca | 0 | no-symmetry | 103 → 103 | 0.00 | 0.76 → 0.76 | true | false | false | false | `majorityFill` |
| 5582e5ca | 0 | fixed-order | 103 → 83 | 0.31 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 0 | random-order | 103 → 68 | 0.60 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 0 | marginal-pruning | 103 → 78 | 0.40 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 1 | no-symmetry | 103 → 103 | 0.00 | 0.76 → 0.76 | true | false | false | false | `majorityFill` |
| 5582e5ca | 1 | fixed-order | 103 → 83 | 0.31 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 1 | random-order | 103 → 55 | 0.91 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 1 | marginal-pruning | 103 → 78 | 0.40 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 2 | no-symmetry | 103 → 103 | 0.00 | 0.76 → 0.76 | true | false | false | false | `majorityFill` |
| 5582e5ca | 2 | fixed-order | 103 → 83 | 0.31 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 2 | random-order | 103 → 75 | 0.46 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 2 | marginal-pruning | 103 → 78 | 0.40 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 3 | no-symmetry | 103 → 103 | 0.00 | 0.76 → 0.76 | true | false | false | false | `majorityFill` |
| 5582e5ca | 3 | fixed-order | 103 → 83 | 0.31 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 3 | random-order | 103 → 78 | 0.40 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 3 | marginal-pruning | 103 → 78 | 0.40 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 4 | no-symmetry | 103 → 103 | 0.00 | 0.76 → 0.76 | true | false | false | false | `majorityFill` |
| 5582e5ca | 4 | fixed-order | 103 → 83 | 0.31 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 4 | random-order | 103 → 68 | 0.60 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 5582e5ca | 4 | marginal-pruning | 103 → 78 | 0.40 | 0.76 → 0.00 | true | true | false | false | `majorityFill` |
| 6150a2bd | 0 | no-symmetry | 62 → 62 | 0.00 | 3.27 → 3.27 | true | false | false | false | `rot180` |
| 6150a2bd | 0 | fixed-order | 62 → 25 | 1.31 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 0 | random-order | 62 → 26 | 1.25 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 0 | marginal-pruning | 62 → 26 | 1.25 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 1 | no-symmetry | 62 → 62 | 0.00 | 3.27 → 3.27 | true | false | false | false | `rot180` |
| 6150a2bd | 1 | fixed-order | 62 → 25 | 1.31 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 1 | random-order | 62 → 25 | 1.31 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 1 | marginal-pruning | 62 → 26 | 1.25 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 2 | no-symmetry | 62 → 62 | 0.00 | 3.27 → 3.27 | true | false | false | false | `rot180` |
| 6150a2bd | 2 | fixed-order | 62 → 25 | 1.31 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 2 | random-order | 62 → 26 | 1.25 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 2 | marginal-pruning | 62 → 26 | 1.25 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 3 | no-symmetry | 62 → 62 | 0.00 | 3.27 → 3.27 | true | false | false | false | `rot180` |
| 6150a2bd | 3 | fixed-order | 62 → 25 | 1.31 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 3 | random-order | 62 → 26 | 1.25 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 3 | marginal-pruning | 62 → 26 | 1.25 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 4 | no-symmetry | 62 → 62 | 0.00 | 3.27 → 3.27 | true | false | false | false | `rot180` |
| 6150a2bd | 4 | fixed-order | 62 → 25 | 1.31 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 4 | random-order | 62 → 25 | 1.31 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 6150a2bd | 4 | marginal-pruning | 62 → 26 | 1.25 | 3.27 → 0.00 | true | true | false | false | `rot180` |
| 67e8384a | 0 | no-symmetry | 70 → 70 | 0.00 | 0.11 → 0.11 | true | false | false | false | `mirror4` |
| 67e8384a | 0 | fixed-order | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 0 | random-order | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 0 | marginal-pruning | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 1 | no-symmetry | 70 → 70 | 0.00 | 0.11 → 0.11 | true | false | false | false | `mirror4` |
| 67e8384a | 1 | fixed-order | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 1 | random-order | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 1 | marginal-pruning | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 2 | no-symmetry | 70 → 70 | 0.00 | 0.11 → 0.11 | true | false | false | false | `mirror4` |
| 67e8384a | 2 | fixed-order | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 2 | random-order | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 2 | marginal-pruning | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 3 | no-symmetry | 70 → 70 | 0.00 | 0.11 → 0.11 | true | false | false | false | `mirror4` |
| 67e8384a | 3 | fixed-order | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 3 | random-order | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 3 | marginal-pruning | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 4 | no-symmetry | 70 → 70 | 0.00 | 0.11 → 0.11 | true | false | false | false | `mirror4` |
| 67e8384a | 4 | fixed-order | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 4 | random-order | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 67e8384a | 4 | marginal-pruning | 70 → 68 | 0.04 | 0.11 → 0.00 | true | true | false | false | `mirror4` |
| 74dd1130 | 0 | no-symmetry | 107 → 107 | 0.00 | 2.25 → 2.25 | true | false | false | false | `transpose` |
| 74dd1130 | 0 | fixed-order | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 0 | random-order | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 0 | marginal-pruning | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 1 | no-symmetry | 107 → 107 | 0.00 | 2.25 → 2.25 | true | false | false | false | `transpose` |
| 74dd1130 | 1 | fixed-order | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 1 | random-order | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 1 | marginal-pruning | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 2 | no-symmetry | 107 → 107 | 0.00 | 2.25 → 2.25 | true | false | false | false | `transpose` |
| 74dd1130 | 2 | fixed-order | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 2 | random-order | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 2 | marginal-pruning | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 3 | no-symmetry | 107 → 107 | 0.00 | 2.25 → 2.25 | true | false | false | false | `transpose` |
| 74dd1130 | 3 | fixed-order | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 3 | random-order | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 3 | marginal-pruning | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 4 | no-symmetry | 107 → 107 | 0.00 | 2.25 → 2.25 | true | false | false | false | `transpose` |
| 74dd1130 | 4 | fixed-order | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 4 | random-order | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 74dd1130 | 4 | marginal-pruning | 107 → 71 | 0.59 | 2.25 → 0.00 | true | true | false | false | `transpose` |
| 9172f3a0 | 0 | no-symmetry | 45 → 45 | 0.00 | 3.15 → 3.15 | true | false | false | false | `scale 3` |
| 9172f3a0 | 0 | fixed-order | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 0 | random-order | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 0 | marginal-pruning | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 1 | no-symmetry | 45 → 45 | 0.00 | 3.15 → 3.15 | true | false | false | false | `scale 3` |
| 9172f3a0 | 1 | fixed-order | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 1 | random-order | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 1 | marginal-pruning | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 2 | no-symmetry | 45 → 45 | 0.00 | 3.15 → 3.15 | true | false | false | false | `scale 3` |
| 9172f3a0 | 2 | fixed-order | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 2 | random-order | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 2 | marginal-pruning | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 3 | no-symmetry | 45 → 45 | 0.00 | 3.15 → 3.15 | true | false | false | false | `scale 3` |
| 9172f3a0 | 3 | fixed-order | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 3 | random-order | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 3 | marginal-pruning | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 4 | no-symmetry | 45 → 45 | 0.00 | 3.15 → 3.15 | true | false | false | false | `scale 3` |
| 9172f3a0 | 4 | fixed-order | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 4 | random-order | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 9172f3a0 | 4 | marginal-pruning | 45 → 17 | 1.40 | 3.15 → 0.00 | true | true | false | false | `scale 3` |
| 99b1bc43 | 0 | no-symmetry | 50 → 50 | 0.00 | 0.56 → 0.56 | true | false | false | false | `splitV xor 3` |
| 99b1bc43 | 0 | fixed-order | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 0 | random-order | 50 → 46 | 0.12 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 0 | marginal-pruning | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 1 | no-symmetry | 50 → 50 | 0.00 | 0.56 → 0.56 | true | false | false | false | `splitV xor 3` |
| 99b1bc43 | 1 | fixed-order | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 1 | random-order | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 1 | marginal-pruning | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 2 | no-symmetry | 50 → 50 | 0.00 | 0.56 → 0.56 | true | false | false | false | `splitV xor 3` |
| 99b1bc43 | 2 | fixed-order | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 2 | random-order | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 2 | marginal-pruning | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 3 | no-symmetry | 50 → 50 | 0.00 | 0.56 → 0.56 | true | false | false | false | `splitV xor 3` |
| 99b1bc43 | 3 | fixed-order | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 3 | random-order | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 3 | marginal-pruning | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 4 | no-symmetry | 50 → 50 | 0.00 | 0.56 → 0.56 | true | false | false | false | `splitV xor 3` |
| 99b1bc43 | 4 | fixed-order | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 4 | random-order | 50 → 46 | 0.12 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| 99b1bc43 | 4 | marginal-pruning | 50 → 44 | 0.18 | 0.56 → 0.00 | true | true | false | false | `splitV xor 3` |
| a416b8f3 | 0 | no-symmetry | 72 → 72 | 0.00 | 2.21 → 2.21 | true | false | false | false | `hconcat` |
| a416b8f3 | 0 | fixed-order | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 0 | random-order | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 0 | marginal-pruning | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 1 | no-symmetry | 72 → 72 | 0.00 | 2.21 → 2.21 | true | false | false | false | `hconcat` |
| a416b8f3 | 1 | fixed-order | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 1 | random-order | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 1 | marginal-pruning | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 2 | no-symmetry | 72 → 72 | 0.00 | 2.21 → 2.21 | true | false | false | false | `hconcat` |
| a416b8f3 | 2 | fixed-order | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 2 | random-order | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 2 | marginal-pruning | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 3 | no-symmetry | 72 → 72 | 0.00 | 2.21 → 2.21 | true | false | false | false | `hconcat` |
| a416b8f3 | 3 | fixed-order | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 3 | random-order | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 3 | marginal-pruning | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 4 | no-symmetry | 72 → 72 | 0.00 | 2.21 → 2.21 | true | false | false | false | `hconcat` |
| a416b8f3 | 4 | fixed-order | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 4 | random-order | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| a416b8f3 | 4 | marginal-pruning | 72 → 40 | 0.85 | 2.21 → 0.00 | true | true | false | false | `hconcat` |
| b8825c91 | 0 | no-symmetry | 4 → 4 | 0.00 | 1.00 → 1.00 | false | false | false | false | `recolour 4 0 ; symmetrizeH` |
| b8825c91 | 0 | fixed-order | 4 → 2 | 1.00 | 1.00 → 0.00 | true | true | false | false | `recolour 4 0 ; symmetrize4` |
| b8825c91 | 0 | random-order | 4 → 2 | 1.00 | 1.00 → 0.00 | false | true | true | false | `recolour 4 0 ; symmetrizeH` |
| b8825c91 | 0 | marginal-pruning | 4 → 2 | 1.00 | 1.00 → 0.00 | true | true | false | false | `recolour 4 0 ; symmetrize4` |
| b8825c91 | 1 | no-symmetry | 4 → 4 | 0.00 | 1.00 → 1.00 | false | false | false | false | `recolour 4 0 ; symmetrizeH` |
| b8825c91 | 1 | fixed-order | 4 → 2 | 1.00 | 1.00 → 0.00 | true | true | false | false | `recolour 4 0 ; symmetrize4` |
| b8825c91 | 1 | random-order | 4 → 2 | 1.00 | 1.00 → 0.00 | false | true | true | false | `recolour 4 0 ; symmetrizeH` |
| b8825c91 | 1 | marginal-pruning | 4 → 2 | 1.00 | 1.00 → 0.00 | true | true | false | false | `recolour 4 0 ; symmetrize4` |
| b8825c91 | 2 | no-symmetry | 4 → 4 | 0.00 | 1.00 → 1.00 | false | false | false | false | `recolour 4 0 ; symmetrizeH` |
| b8825c91 | 2 | fixed-order | 4 → 2 | 1.00 | 1.00 → 0.00 | true | true | false | false | `recolour 4 0 ; symmetrize4` |
| b8825c91 | 2 | random-order | 4 → 2 | 1.00 | 1.00 → 0.00 | false | true | true | false | `recolour 4 0 ; symmetrizeH` |
| b8825c91 | 2 | marginal-pruning | 4 → 2 | 1.00 | 1.00 → 0.00 | true | true | false | false | `recolour 4 0 ; symmetrize4` |
| b8825c91 | 3 | no-symmetry | 4 → 4 | 0.00 | 1.00 → 1.00 | false | false | false | false | `recolour 4 0 ; symmetrizeH` |
| b8825c91 | 3 | fixed-order | 4 → 2 | 1.00 | 1.00 → 0.00 | true | true | false | false | `recolour 4 0 ; symmetrize4` |
| b8825c91 | 3 | random-order | 4 → 2 | 1.00 | 1.00 → 0.00 | false | true | true | false | `recolour 4 0 ; symmetrizeH` |
| b8825c91 | 3 | marginal-pruning | 4 → 2 | 1.00 | 1.00 → 0.00 | true | true | false | false | `recolour 4 0 ; symmetrize4` |
| b8825c91 | 4 | no-symmetry | 4 → 4 | 0.00 | 1.00 → 1.00 | false | false | false | false | `recolour 4 0 ; symmetrizeH` |
| b8825c91 | 4 | fixed-order | 4 → 2 | 1.00 | 1.00 → 0.00 | true | true | false | false | `recolour 4 0 ; symmetrize4` |
| b8825c91 | 4 | random-order | 4 → 2 | 1.00 | 1.00 → 0.00 | false | true | true | false | `recolour 4 0 ; symmetrizeH` |
| b8825c91 | 4 | marginal-pruning | 4 → 2 | 1.00 | 1.00 → 0.00 | true | true | false | false | `recolour 4 0 ; symmetrize4` |
| be94b721 | 0 | no-symmetry | 121 → 121 | 0.00 | 1.15 → 1.15 | true | false | false | false | `cropLargest` |
| be94b721 | 0 | fixed-order | 121 → 98 | 0.30 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 0 | random-order | 121 → 30 | 2.01 | 1.15 → 0.21 | true | false | false | false | `cropLargest` |
| be94b721 | 0 | marginal-pruning | 121 → 28 | 2.11 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 1 | no-symmetry | 121 → 121 | 0.00 | 1.15 → 1.15 | true | false | false | false | `cropLargest` |
| be94b721 | 1 | fixed-order | 121 → 98 | 0.30 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 1 | random-order | 121 → 26 | 2.22 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 1 | marginal-pruning | 121 → 28 | 2.11 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 2 | no-symmetry | 121 → 121 | 0.00 | 1.15 → 1.15 | true | false | false | false | `cropLargest` |
| be94b721 | 2 | fixed-order | 121 → 98 | 0.30 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 2 | random-order | 121 → 30 | 2.01 | 1.15 → 0.21 | true | false | false | false | `cropLargest` |
| be94b721 | 2 | marginal-pruning | 121 → 28 | 2.11 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 3 | no-symmetry | 121 → 121 | 0.00 | 1.15 → 1.15 | true | false | false | false | `cropLargest` |
| be94b721 | 3 | fixed-order | 121 → 98 | 0.30 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 3 | random-order | 121 → 64 | 0.92 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 3 | marginal-pruning | 121 → 28 | 2.11 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 4 | no-symmetry | 121 → 121 | 0.00 | 1.15 → 1.15 | true | false | false | false | `cropLargest` |
| be94b721 | 4 | fixed-order | 121 → 98 | 0.30 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 4 | random-order | 121 → 27 | 2.16 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| be94b721 | 4 | marginal-pruning | 121 → 28 | 2.11 | 1.15 → 0.00 | true | true | false | false | `cropLargest` |
| c59eb873 | 0 | no-symmetry | 71 → 71 | 0.00 | 4.11 → 4.11 | true | false | false | false | `scale 2` |
| c59eb873 | 0 | fixed-order | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 0 | random-order | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 0 | marginal-pruning | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 1 | no-symmetry | 71 → 71 | 0.00 | 4.11 → 4.11 | true | false | false | false | `scale 2` |
| c59eb873 | 1 | fixed-order | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 1 | random-order | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 1 | marginal-pruning | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 2 | no-symmetry | 71 → 71 | 0.00 | 4.11 → 4.11 | true | false | false | false | `scale 2` |
| c59eb873 | 2 | fixed-order | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 2 | random-order | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 2 | marginal-pruning | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 3 | no-symmetry | 71 → 71 | 0.00 | 4.11 → 4.11 | true | false | false | false | `scale 2` |
| c59eb873 | 3 | fixed-order | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 3 | random-order | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 3 | marginal-pruning | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 4 | no-symmetry | 71 → 71 | 0.00 | 4.11 → 4.11 | true | false | false | false | `scale 2` |
| c59eb873 | 4 | fixed-order | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 4 | random-order | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| c59eb873 | 4 | marginal-pruning | 71 → 2 | 5.15 | 4.11 → 0.00 | false | true | true | false | `scale 2 ; cropSmallest` |
| ce4f8723 | 0 | no-symmetry | 49 → 49 | 0.00 | 0.29 → 0.29 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 0 | fixed-order | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 0 | random-order | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 0 | marginal-pruning | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 1 | no-symmetry | 49 → 49 | 0.00 | 0.29 → 0.29 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 1 | fixed-order | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 1 | random-order | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 1 | marginal-pruning | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 2 | no-symmetry | 49 → 49 | 0.00 | 0.29 → 0.29 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 2 | fixed-order | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 2 | random-order | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 2 | marginal-pruning | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 3 | no-symmetry | 49 → 49 | 0.00 | 0.29 → 0.29 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 3 | fixed-order | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 3 | random-order | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 3 | marginal-pruning | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 4 | no-symmetry | 49 → 49 | 0.00 | 0.29 → 0.29 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 4 | fixed-order | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 4 | random-order | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| ce4f8723 | 4 | marginal-pruning | 49 → 46 | 0.09 | 0.29 → 0.15 | true | false | false | false | `splitV or 3` |
| d631b094 | 0 | no-symmetry | 58 → 58 | 0.00 | 1.39 → 1.39 | true | false | false | false | `countRow` |
| d631b094 | 0 | fixed-order | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 0 | random-order | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 0 | marginal-pruning | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 1 | no-symmetry | 58 → 58 | 0.00 | 1.39 → 1.39 | true | false | false | false | `countRow` |
| d631b094 | 1 | fixed-order | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 1 | random-order | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 1 | marginal-pruning | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 2 | no-symmetry | 58 → 58 | 0.00 | 1.39 → 1.39 | true | false | false | false | `countRow` |
| d631b094 | 2 | fixed-order | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 2 | random-order | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 2 | marginal-pruning | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 3 | no-symmetry | 58 → 58 | 0.00 | 1.39 → 1.39 | true | false | false | false | `countRow` |
| d631b094 | 3 | fixed-order | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 3 | random-order | 58 → 44 | 0.40 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 3 | marginal-pruning | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 4 | no-symmetry | 58 → 58 | 0.00 | 1.39 → 1.39 | true | false | false | false | `countRow` |
| d631b094 | 4 | fixed-order | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 4 | random-order | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| d631b094 | 4 | marginal-pruning | 58 → 45 | 0.37 | 1.39 → 0.00 | true | true | false | false | `countRow` |
| ed36ccf7 | 0 | no-symmetry | 25 → 25 | 0.00 | 2.08 → 2.08 | true | false | false | false | `rot270` |
| ed36ccf7 | 0 | fixed-order | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 0 | random-order | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 0 | marginal-pruning | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 1 | no-symmetry | 25 → 25 | 0.00 | 2.08 → 2.08 | true | false | false | false | `rot270` |
| ed36ccf7 | 1 | fixed-order | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 1 | random-order | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 1 | marginal-pruning | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 2 | no-symmetry | 25 → 25 | 0.00 | 2.08 → 2.08 | true | false | false | false | `rot270` |
| ed36ccf7 | 2 | fixed-order | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 2 | random-order | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 2 | marginal-pruning | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 3 | no-symmetry | 25 → 25 | 0.00 | 2.08 → 2.08 | true | false | false | false | `rot270` |
| ed36ccf7 | 3 | fixed-order | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 3 | random-order | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 3 | marginal-pruning | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 4 | no-symmetry | 25 → 25 | 0.00 | 2.08 → 2.08 | true | false | false | false | `rot270` |
| ed36ccf7 | 4 | fixed-order | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 4 | random-order | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| ed36ccf7 | 4 | marginal-pruning | 25 → 15 | 0.74 | 2.08 → 0.57 | true | false | false | false | `rot270` |
| f25ffba3 | 0 | no-symmetry | 56 → 56 | 0.00 | 3.13 → 3.13 | true | false | false | false | `symmetrizeV` |
| f25ffba3 | 0 | fixed-order | 56 → 23 | 1.28 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 0 | random-order | 56 → 24 | 1.22 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 0 | marginal-pruning | 56 → 24 | 1.22 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 1 | no-symmetry | 56 → 56 | 0.00 | 3.13 → 3.13 | true | false | false | false | `symmetrizeV` |
| f25ffba3 | 1 | fixed-order | 56 → 23 | 1.28 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 1 | random-order | 56 → 24 | 1.22 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 1 | marginal-pruning | 56 → 24 | 1.22 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 2 | no-symmetry | 56 → 56 | 0.00 | 3.13 → 3.13 | true | false | false | false | `symmetrizeV` |
| f25ffba3 | 2 | fixed-order | 56 → 23 | 1.28 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 2 | random-order | 56 → 24 | 1.22 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 2 | marginal-pruning | 56 → 24 | 1.22 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 3 | no-symmetry | 56 → 56 | 0.00 | 3.13 → 3.13 | true | false | false | false | `symmetrizeV` |
| f25ffba3 | 3 | fixed-order | 56 → 23 | 1.28 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 3 | random-order | 56 → 24 | 1.22 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 3 | marginal-pruning | 56 → 24 | 1.22 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 4 | no-symmetry | 56 → 56 | 0.00 | 3.13 → 3.13 | true | false | false | false | `symmetrizeV` |
| f25ffba3 | 4 | fixed-order | 56 → 23 | 1.28 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 4 | random-order | 56 → 23 | 1.28 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f25ffba3 | 4 | marginal-pruning | 56 → 24 | 1.22 | 3.13 → 0.00 | true | true | false | false | `symmetrizeV` |
| f2829549 | 0 | no-symmetry | 50 → 50 | 0.00 | 0.14 → 0.14 | true | false | false | false | `splitH nor 3` |
| f2829549 | 0 | fixed-order | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 0 | random-order | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 0 | marginal-pruning | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 1 | no-symmetry | 50 → 50 | 0.00 | 0.14 → 0.14 | true | false | false | false | `splitH nor 3` |
| f2829549 | 1 | fixed-order | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 1 | random-order | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 1 | marginal-pruning | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 2 | no-symmetry | 50 → 50 | 0.00 | 0.14 → 0.14 | true | false | false | false | `splitH nor 3` |
| f2829549 | 2 | fixed-order | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 2 | random-order | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 2 | marginal-pruning | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 3 | no-symmetry | 50 → 50 | 0.00 | 0.14 → 0.14 | true | false | false | false | `splitH nor 3` |
| f2829549 | 3 | fixed-order | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 3 | random-order | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 3 | marginal-pruning | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 4 | no-symmetry | 50 → 50 | 0.00 | 0.14 → 0.14 | true | false | false | false | `splitH nor 3` |
| f2829549 | 4 | fixed-order | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 4 | random-order | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
| f2829549 | 4 | marginal-pruning | 50 → 47 | 0.09 | 0.14 → 0.15 | true | false | false | false | `splitH nor 3` |
