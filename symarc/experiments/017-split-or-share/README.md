# 017 — Split a context, or share a better operation?

Status: active; primary protocol registered before execution. Baseline PR #32
`71c9efdfb112c35ac61030c758fdbdb3144a3436`. No stable-core or accepted-math changes.

## Question

016 supplies exact feature-necessity certificates relative to constants plus
CopyCentre. Its sensitivity audit showed that richer shared actions can remove
a particular conflict without making the whole task fit. Test the two repair
directions together, with a matched finite constructor extension and exact
multi-example action intersections. This is a supplied-grammar study, not
primitive-free abstraction discovery.

## Fixed constructor extension and controls

Reuse 016's fifteen input features and its constants 0..9 plus CopyCentre.
Introduce one directional walk constructor: from the pointed cell, walk north,
south, west or east while colour equals the start colour. The walk includes the
start cell and stops at the canvas edge or the first different-coloured cell.
Its count gives four new input features. Reading the first cell after that walk
gives four new output operations, undefined if the walk exits the canvas.
Also include four one-step directional copies, the control studied in 016.
All directions, all colours and all tasks use identical definitions. No
rectangle primitive, task-specific constants, default-copy or tie-breaking
based on query answers.

Four matched arms, all with at most three selected features:

| Arm | Features | Actions |
|---|---|---|
| base | 016's 15 | constants + centre copy (11) |
| split | 15 + four directional run lengths | same 11 |
| share | same 15 | 11 + four adjacent + four beyond-run copies (19) |
| joint | all 19 | all 19 |

This finite vocabulary is constructed before labels are read. Conflict-guided
search learns feature subsets and intersections learn context-to-action sets;
no claim is made that the walk constructor itself is learned.

## Correct conflict certificates for richer operations

For a labelled point u, let A(u) be its set of allowed actions. A feature class
is consistent exactly when the intersection of A(u) over all its observations
is nonempty. Pairwise nonempty intersections do NOT imply this in general.
Use deletion-minimal inconsistent sets K, which can have more than two points.
A successful refinement must select a feature nonconstant on K. Branch on all
such features and retain all inclusion-minimal sufficient subsets of size <=3;
no beam, timeout or entropy pruning. Full-vocabulary conflicts are separately
recorded from feature-bound exhaustion.

For each retained feature, remove it and retain a conflicting set proving its
necessity relative to the remaining features and that arm's action language.
For old full-vocabulary conflicts, record separately whether the new terms
separate their points, the new actions explain the entire witness with one
operation, and the extended learner actually fits the whole task. Resolving
one certificate never counts as solving a task.

## Selection, isolation and scoring

Within each arm use 016's rule: whole-demonstration exact reconstruction first,
then sum of per-grid correct-known fractions, then feature count, supplied
constructor costs and syntax order. Refit the action table for each held-out
input group. Outer validation rebuilds the entire feature search and internal
selection without the outer labels. Duplicate demonstration inputs are held
out together.

In addition evaluate a union selector over all four arms, preserving the base
arm's candidates even when richer action languages make their feature sets
nonminimal. After validation score, ties use feature count, constructor costs,
action-library size, arm order base/split/share/joint and feature syntax. Retain
an outer-validation-gated version as a coverage/error diagnostic. No policy
changes after test scoring. Internal validation is model selection, not an
unbiased estimate of the final selected model.

Unknown feature keys abstain. For a known key, a cell is determined only when
EVERY surviving action is defined there and all evaluate to the same colour.
Do not silently discard an action that becomes undefined on a query.
More actions may increase training expressivity but also increase query
ambiguity. Minimal feature sets need not be nested across action languages.

Predict and hash all queries before a separate command reads query answers.
Use all 1,000 training and 120 evaluation tasks at official ARC-AGI-2
`f3283f727488ad98fe575ea6a5ac981e4a188e49`; the two known development fixtures
`00d62c1b` and `e88171ec` are excluded from primary counts. Eligibility is the
same as 016, determined from demonstration shapes and distinct inputs alone.
The projected problem and answer files are reused from 016 with verified hashes.
These previously used public data are not an untouched holdout. Twelve workers,
deterministic ordering, no random seed; wall time is provenance only.

Report per arm and split: training fits, complete/correct/wrong query tasks,
outer passes and gated coverage, candidate oracle, unknown-key and action
ambiguity/undefinedness, and paired changes against base. Score partial
predictions separately, never as solved tasks. Report construction gains and
selection gains separately. No minimum-fit stop is needed: 016 already provides
a nontrivial control cohort. Negative or coverage-limited results are retained.

## Pre-run checks and post-run audit

Before corpus execution: synthetic triple-intersection counterexample, conflict
minimality, general guided-search equality with exhaustive small subsets,
constant/copy equivalence with 016, run-length and operation semantics, unseen
keys, undefined surviving actions, duplicate handling and outer-label isolation.
Freeze code hashes after these controls and before the corpus run.

After prediction: verify base-arm pools, selections and queries against 016;
independently check action intersections, conflict/necessity certificates,
complete task scores and a deterministic sample of full feature-subset pools.
Repeat prediction with 12 workers and compare byte-for-byte. Store predictions,
fold decisions, all conflicts and provenance. Any code correction affecting
scientific outputs must be recorded explicitly and the affected run repeated.

## Interpretation

A useful result must distinguish three effects: new terms separate observed
cases; new operations allow different cases to share a rule; either change
improves prediction after the full selection process. An increased fit count
alone establishes only expressivity. Novelty is not claimed for conflict-driven
search or program unification. Relevant precedents include STUN (Alur et al.,
2015), https://www.microsoft.com/en-us/research/publication/synthesis-through-unification/,
and conflict-driven synthesis (Feng et al., 2017),
https://arxiv.org/abs/1711.08029.
