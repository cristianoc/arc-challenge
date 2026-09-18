# 001-pruning: does aggressive hypothesis pruning improve predictions?

Ledger: [001-pruning](../RESULTS.md#001-pruning--does-aggressive-hypothesis-pruning-improve-predictions).

## Hypothesis and decision

Choosing the symmetry with largest feasible marginal reduction in a fixed
program class improves single-prediction task exact-match accuracy, compared
with fixed and random generator order. Reduced entropy without improved
correctness is not sufficient for integration.

A candidate for further validation must improve mean training-task accuracy
against fixed order and no symmetry, compare favorably with random order,
and not merely produce more unanimous wrong answers. Repeat promising effects
with the prespecified larger closure cap before considering integration.
This is an exploratory development experiment, not a significance claim.

## Fixed protocol (recorded before scientific runs)

- Data: all 400 public **training** tasks. No public evaluation tasks are used.
- Program class: all syntax programs of length ≤2 from the stable task-specific
  DSL pool that fit the task's training examples. Enumerate once and freeze it.
  No repair, hill-climbing fallback, length prior, or symmetry-only prediction
  fallback. Tasks with no fitting program are explicitly excluded from
  conditional metrics and count as unsolved in all-400 metrics.
- Common generator candidates: the stable functionality-greedy survivors,
  computed once with cap 1000 and the stable candidate order. We are testing
  realizability selection within that common prefilter, not all possible
  generator sets. This is a controlled comparator, not the full stable solver.
- Uniform prior over syntax programs. Select the shortest surviving program,
  with original enumeration order breaking ties. Undefined prediction is an
  explicit answer category in entropy; determination requires one **defined**
  answer at every test input.
- Policies: no symmetry; fixed order; seeded random order; maximal marginal
  pruning among nonempty survivors. Fixed/random may accept zero-pruning
  generators; maximal pruning stops when no candidate prunes. All stop once
  answers are determined. Thus this compares whole selection/stopping policies,
  not order alone. Zero-pruning generators can have useful joint effects.
- Checking: closure cap 300. Use the full closure when completed, otherwise
  stable one-step + 64 random-walk sample attempts. Check commutation for each
  added generator at every test input. Reject witnessed functionality conflicts.
  Sort generator sets canonically for traversal and derive sampling seeds from
  the set and run seed: evaluating alternatives does not consume a global RNG
  or change another alternative's sample.
- Each trial intersects the current survivors with the new constraints, so
  survivors are nested. Accepted sampled constraints remain enforced through
  that intersection. Report entropy of this observed finite survivor set—not
  entropy under a claimed exact full closure. Report capped trial frequency.
- Shared per-task/per-policy budget: 500,000 charged program-grid evaluations.
  A closure-sample evaluation costs one; a core commutation check is charged
  two (a conservative bound). Closure building is outside this counter and
  included in measured selection time. Same budget ceiling does not mean
  equal actual work; record budget exhaustion, charges, and wall time.
  On exhaustion, retain the last accepted state; do not select from an
  incompletely ranked marginal-pruning round.
- Seeds: 0,1,2,3,4; separate RNG for random order. All arms use the same
  deterministic set-conditioned samples for a given seed.
- Workers: 12. Preparation is shared; each policy/seed batch runs serially
  relative to the others. Record preparation and selection timing separately.
- Primary: task exact-match accuracy, both all 400 and eligible tasks;
  paired wins/losses against fixed order and no symmetry. Secondary: program
  entropy/pruning bits, mean per-task answer entropy (mean over that task's
  test inputs), determination, unanimous wrong answers, and loss of every
  correct program when one was initially available. Labels score only after
  selection. Distinguish per-seed observations from independent tasks.
- Budget for this first round: one complete five-seed run at cap 300; one
  cap-1000 confirmation if effects or sampled-constraint sensitivity warrant
  it. Investigate correctness failures before making scientific conclusions.
  No public evaluation run or automatic integration in this experiment.

## Checks

Test entropy with undefined answers, nonempty/nested filtering, deterministic
set-conditioned samples, budget exhaustion, toy closure counts, and independence
of selection from held-out labels. Verify core source files remain unchanged.

## Commands

From `symarc/`:

```sh
cargo test --release --manifest-path experiments/001-pruning/Cargo.toml
experiments/001-pruning/run.sh
# Prespecified sensitivity/confirmation, when warranted:
experiments/001-pruning/run.sh --cap 1000
```

The runner records a direct Markdown report and source/data/configuration hashes
under `out/experiments/001-pruning/`. Decisive findings move to the ledger before
any pruning of the experiment directory.
