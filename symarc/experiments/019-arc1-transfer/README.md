# 019 — Has the relational learner already improved ARC1?

Status: active at registration; outcomes will be recorded in the evidence report.
Baseline: `83df79f78f059fe18b356d51a4e0354367966f72` (PR #32).

Run the unchanged 018 learner on all 400 training and 400 evaluation tasks in
this repository's `data/`, and rerun the complete stable Rust solver. Existing
stable reference: 57/400 training and 23/400 evaluation. Do not compare the
new learner only in isolation: measure complementary correct tasks and actual
label-blind combined policies. No core, representation, operation, cost,
feature bound or selection change is permitted in this comparison.

## Frozen policies, before scoring

For every task retain the complete stable solver prediction S and 018 union
prediction R. A relational prediction is complete only if every supplied query
grid is present, rectangular and contains no abstention. P means that 018's
outer check has at least two folds and predicts every held-out grid exactly.
The stable solver's training-fit flag is computed only from demonstrations.

- Stable alone; each of 018's four arms and its union alone.
- Complete override: choose R when complete, otherwise S.
- Strict override: choose R when complete and P, otherwise S.
- No-fit fallback: choose R when complete and S does not fit all demonstrations;
  otherwise S.
- Correct-either oracle: retrospective upper bound, NOT a deployed policy.

Choose once per task, not by test answer or per-query correctness. No tuning
based on scores. Report correct tasks and grids, complete errors, paired gains
and losses, all changed task IDs, union/intersection of correct sets and
regressions. All 400+400 tasks are the requested historical comparison; also
report excluding the three previously inspected development fixtures from 018.
All these public datasets were used previously. ARC1/ARC2 overlap must be
measured by content and discussed; dataset names do not imply independent tests.

## Execution and isolation

Use the pinned repository revision for data and source. Project query inputs
and store their answers separately. The Rust adapter imports `symarc` and calls
`search::run_task` with defaults; it does not copy the solver. It receives input
files with query outputs replaced by a fixed 1x1 zero placeholder, which cannot
enter candidate construction or search. It exports predictions and the training
fit, never the meaningless placeholder score. The existing regression suite and
the historical per-task baseline are independent controls.

018 reads only the projected problems; retain its frozen source/dependency
hashes, full candidates, outer folds and predictions. Persist and hash both
prediction files before scoring the separate answers. Twelve workers per arm,
serial arms, deterministic ordering and seed 0 for the stable solver. Runtime
is provenance only, not a matched speed claim across Python and Rust.

Independent audit: recompute exact answers and policy selections from frozen
predictions without the scorer; verify 018 predictions on content-identical
ARC2 tasks against retained experiment 018 outputs; compare every stable
per-task score with the existing baseline report; rerun synthetic 018 and stable
core tests. Report failures rather than silently updating historical fixtures.

## Interpretation and research continuation

A gain in correct-either coverage is not a usable solver gain unless a declared
label-blind policy realizes it. A policy gain on reused public ARC1 data is a
measured integration candidate, not proof of generalisation or blind benchmark
competitiveness. Preserve the stable core while investigating the separate
question from 018: sharing evidence for operations across distinct guards.
