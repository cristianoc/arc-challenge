# ARC1 has improved through complementary predictions

## Answer

The unchanged experiment 018 learner had not previously been measured separately
on ARC1. Experiment 019 now runs it on all 400 training and 400 evaluation tasks
and reruns the stable Rust solver. A label-blind combination increases exact
solutions from **57 to 78 training tasks** and **23 to 27 evaluation tasks**, with
no previously correct task lost. These are test-query scores, not reconstruction
of demonstrations. Neither solver's search or representation was modified.

| Frozen policy | ARC1 training | ARC1 evaluation | Correct query grids, training / evaluation |
|---|---:|---:|---:|
| Stable solver | 57/400 | 23/400 | 62/416; 27/419 |
| Experiment 018 union alone | 27/400 | 5/400 | 29/416; 5/419 |
| Relational complete override | 78/400 | 27/400 | 83/416; 31/419 |
| Relational override only after all outer checks pass | 63/400 | 25/400 | 68/416; 29/419 |
| Relational fallback only when stable does not fit demonstrations | **78/400** | **27/400** | **83/416; 31/419** |
| Correct-either oracle, not a policy | 78/400 | 27/400 | Not used for selection |

The standalone learner is not a replacement for the stable solver. Of its 27
correct training answers only six overlap the stable solver; of its five
correct evaluation answers only one overlaps. Complementarity produces **25
additional solved tasks overall**, taking the total from 80/800 to 105/800.

The no-fit fallback rule is operational and uses no query answers:

```text
if the stable solver fits every demonstration:
    use the stable prediction
else if experiment 018 determines every query grid completely:
    use its prediction
else:
    use the stable prediction
```

Selection is once per task, not cherry-picking correct query grids. Complete
means every required grid exists and contains no abstention. The broader
complete-override policy happens to realize the same correct-either ceiling.
This equality is measured here, not guaranteed on future tasks.

018 alone has one complete wrong training answer (`6e82a1ae`) and none on ARC1
evaluation; other failures are incomplete. The combination still returns the
stable solver's guesses elsewhere. Its remaining 322 training and 373 evaluation
errors are not erased by describing only its gains. Outer gating is not needed
to obtain the measured increase and discards many complementary successes.

## What was fixed before scores

Baseline `83df79f78f059fe18b356d51a4e0354367966f72`; protocol registered at
`8e6f89f7a05287373deea827d3cd537851c61bf4`. Execution revision
`a5b3cccf79b0ad9c2d88550769030b3fcc08b438`, GitHub Actions run 35427286878.
The three combination policies above were declared before either arm was scored.

The Rust adapter imports the existing library and calls `search::run_task` with
unchanged defaults, seed 0. It reads projected task files whose query outputs
are fixed 1x1 zero placeholders and exports no placeholder accuracy. All
predictions therefore precede reading the real query answers. The existing
solver already excludes test outputs from its candidate construction and search.
The Python arm is byte-identical to 018. Both prediction files were written and
hashed before the separate scorer read the answer file. Twelve workers per arm;
arms ran serially. Runtime is provenance, not a Python-versus-Rust speed claim.

All 400+400 tasks are included for comparison with the historical baseline.
Excluding 018's three inspected fixtures gives stable **56/398 training and
23/399 evaluation**, versus fallback **76/398 and 27/399**. The fixtures are
`00d62c1b`, `29c11459` in ARC1 training and `e88171ec` in ARC1 evaluation. Thus
24 of the 25 gains remain after these exclusions.

## Every complementary gain

Training: `0d3d703e`, `2281f1f4`, `25d8a9c8`, `29c11459`, `3618c87e`,
`3bd67248`, `4347f46a`, `67385a82`, `6f8cd79b`, `7f4411dc`, `810b9b61`,
`91714a58`, `aedd82e4`, `bb43febb`, `c0f76784`, `ce9e57f2`, `d037b0a7`,
`d511f180`, `e8593010`, `ea32f347`, `ea786f4a`.

Evaluation: `7e02026e`, `84f2aca1`, `aa18de87`, `e0fb7511`.

The matched 018 exact arm alone scores 21/400 and 4/400; order-only scores 24/400
and 4/400; guarded-exact scores 24/400 and 5/400; guarded-order/union scores
27/400 and 5/400. These are not comparisons with the object fallback from 003;
that separate candidate was not included in this combination.

## Overlap: a practical gain, not new independent validation

ARC1 and ARC2 are not independent corpora. Of ARC1's 400 training tasks, 391
have identical labelled examples in the pinned ARC2 data, allowing example
reordering. Of its 400 evaluation tasks, 382 share IDs and 381 share labelled
content. All **25 newly solved tasks occur in ARC2 training with the same
labelled content** and were therefore already part of the prior research.

All 772 shared-content tasks reorder demonstrations. The audit aligns these
orders and checks that every selected model, query prediction and outer-fold
result matches 018's retained ARC2 run. This confirms reproducibility, not an
independent generalisation test. The result establishes that the research has
already added useful capability relative to the stable solver on reused public
ARC1 tasks. It does not establish hidden ARC benchmark performance.

## Checks and reproduction

The stable build passes all 11 Rust tests and the CLI/regression suite. All 15
018 synthetic tests pass. Independently recomputing the frozen selection rules,
whole-grid equality and individual-grid scores verifies **8,000 policy/task
decisions**. Every stable task's fit, task score and grid score agrees with the
retained 011 ARC1 baseline, across all 800 tasks.

A complete independent local Python run, in eight serial 100-task batches with
12 workers each, reproduces the Actions relational predictions byte-for-byte.
The independent overlap audit additionally verifies 772 permuted ARC2 task
replays. The audit does not independently reimplement 018's entire learner.

Relational prediction SHA-256:
`e47927459066e61bc19b1b1d3e6d2678d62a26aebf4da7f5bc9cc359c8f1ed18`.

The complete evidence archive contains projected problems, separate answers,
stable predictions, all 018 candidates/outer folds, scores, hashes, source,
compiled-package lockfile and regression logs. Actions artifact
`10579147646` retains the original run (expires 18 December 2026); the companion
archive also retains the independent audits and local replay evidence.

The stable core and accepted mathematics are unchanged. Retain this as an
integration candidate and an executable composition of existing learners, not
another parameter-tuned solver. Research continuation is separately registered
as 020: operation sharing without merging guards.
