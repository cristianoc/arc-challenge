# Split a context, or share a better operation?

## Result in one paragraph

Experiment 017 compares two ways of resolving a training contradiction: add
features that distinguish the situations, or allow an input-dependent operation
they can share. A frozen combined selector improves from 22 to 25 exact query
solutions on the 998 nondevelopment tasks in ARC2's public training split, with
complete wrong answers increasing from four to five. No evaluation task is
solved. A concrete task needs both extensions. The mathematical correction is
also real: richer operations produce minimal inconsistent sets of three, four
and five observations, so 016's two-cell certificate scheme is not complete for
the extended action language. A training-only majority probe repairs a four-cell
witness on `7e0986d6`, but not that whole task.

## Frozen comparison and scope

Baseline: PR #32 at `71c9efdfb112c35ac61030c758fdbdb3144a3436`. The primary
protocol was committed at `f37093f7`; source hashes, runnable code and an active
ledger entry were published at `2fe31f5a` before corpus execution. Sixteen
synthetic controls passed before that run. Temporary source/evidence
writeback workflows remove themselves; they do not run the scientific corpus
experiment or change the stable solver.

The data are all 1,000 training and 120 evaluation tasks at official ARC-AGI-2
`f3283f727488ad98fe575ea6a5ac981e4a188e49`. Two previously inspected development
fixtures, `00d62c1b` and `e88171ec`, are excluded from primary counts. There are
678 eligible training and 81 eligible evaluation tasks: demonstration dimensions
must match and there must be at least two distinct demonstration inputs.
Primary corpus denominators remain 998 and 120. These previously used public
data are not an untouched holdout.

The new constructor walks north/south/west/east while the colour equals the
starting cell's colour. Its length supplies a feature; reading the first cell
beyond the run supplies an output action. Exiting the canvas makes that action
undefined. Four adjacent-cell copies are included as well. These constructors
and directions are supplied priors, not automatically invented primitives.

| Arm | Input features | Output actions |
|---|---|---|
| Base | The 15 features from 016 | Constants 0–9 and CopyCentre |
| Split | Those 15 plus four directional run lengths | Same 11 actions |
| Share | Original 15 | 11 plus four adjacent and four beyond-run copies |
| Joint | All 19 | All 19 |

All arms enumerate inclusion-minimal sufficient feature sets of size at most
three. No beam, timeout, majority default or guessed unknown output is used.
All consistent actions are retained. A query cell is determined only if all
surviving actions are defined and evaluate to the same colour. An undefined
survivor is not silently removed.

Selection follows 016: whole withheld-demonstration accuracy first, then the
sum of per-grid correct-known fractions, then supplied feature costs and syntax.
An outer check rebuilds the search and inner selection without the outer
labels. Duplicate inputs are withheld together. A union selector keeps all four
arms, preserving the original candidates even when richer actions make their
feature sets nonminimal. After validation, its fixed tie order prefers fewer
features, lower feature costs, smaller action libraries, and syntax.

Query predictions were persisted and hashed before a separate scoring process
read query answers. Twelve workers were used, with deterministic ordering and
no random seed. A second 12-worker prediction run is byte-identical. The original
19.147 seconds is provenance only, not a speed comparison with earlier systems.

## Main outcome

The following are **test-query scores for tasks in the public training split**,
not demonstration reconstruction scores.

| Policy | Tasks fitted | Complete correct answers | Complete wrong answers | Correct answers after strict outer gate |
|---|---:|---:|---:|---:|
| Base | 72 | 22 | 4 | 10 |
| Split | 150 | 24 | 5 | 12 |
| Share | 73 | 21 | 2 | 9 |
| Joint | 156 | 24 | 3 | 10 |
| Union selector | 156 | **25** | **5** | **12** |

The union gains `25d8a9c8`, `29c11459`, and `3618c87e`, losing no previously
correct task. It nevertheless has five complete wrong answers: `3aa6fb7a`,
`5ad8a7c0`, `6c434453`, `6e82a1ae`, and `b7256dcd`. It fixes the old error on
`3618c87e` and introduces two new complete errors. The outer gate removes all
five wrong answers but also discards 13 of the 25 correct answers. Fourteen
tasks pass that gate; two have incomplete queries. This is a coverage/error
tradeoff, not a general correctness guarantee or an unqualified improvement.

Evaluation fits are 4/base, 5/split, 4/share, 5/joint and 5/union; all predictions
remain incomplete and all policies solve 0/120. The five fitting models have
2,824 unknown-key cells and 35 action-disagreement cells, versus 523 determined
cells. The determined cells happen to be correct, but this is not a solved-task
result or a calibration claim.

These are comparisons between the experimental learners. The stable Rust
solver was not changed, rerun, or combined with their scores.

## A task where both changes matter: 29c11459

The two demonstrations contain these nonzero rows (other rows are zero):

```text
3 0 0 0 0 0 0 0 0 0 7   ->   3 3 3 3 3 5 7 7 7 7 7
1 0 0 0 0 0 0 0 0 0 2   ->   1 1 1 1 1 5 2 2 2 2 2
```

The selected joint key consists of the left and right same-colour run lengths.
For some observed length pairs the common action is CopyBeyondRunWest; for
others it is CopyBeyondRunEast; the central pair selects constant 5. The final
query contains two rows with new endpoint colours, and both are predicted
exactly:

```text
4 0 0 0 0 0 0 0 0 0 8   ->   4 4 4 4 4 5 8 8 8 8 8
6 0 0 0 0 0 0 0 0 0 9   ->   6 6 6 6 6 5 9 9 9 9 9
```

Neither base nor share fits. Split fits by distinguishing additional training
positions, but its available constants do not transfer to the new endpoint
colours. Joint supplies both position distinctions and input-dependent colour
sources. This is a concrete interaction between the two forms of repair, not
merely a larger fitting-program count.

There is an important limitation. The learner has an exact table of observed
length pairs; it has NOT inferred a general inequality such as “copy the nearer
endpoint”, and no arbitrary-width extrapolation is established. Its fixed-key
internal validation predicts 0/2 entire withheld demonstrations, though its
partial score selects it. The test is correct despite failing the strict gate.
No new demonstration/query labels were supplied by this analysis.

The two other gains are also inspectable. `25d8a9c8` selects left/right run
lengths and predicts all four held-out demonstrations, while the old selected
model predicts none. `3618c87e` selects colour plus northward run length; the
split arm predicts all three held-out demonstrations versus one for base.
Their gains should not be described as automatically discovered line primitives:
the directional-run constructor was supplied.

## Why a larger operation library can lose a correct answer

On `d5d6de2d`, both base and share select colour plus component boundary
reachability. The base table returns constant 0 for the key `(2,1)`. In the
richer library, north/west/east beyond-run copies also fit the demonstrations.
At query 1, cell `(0,0)`, north and west copies are undefined, while constant 0
and east copy return 0. The conservative predictor abstains.

Thus share and joint lose a previously complete correct solution without losing
training fit. The union can retain the base candidate and restores that answer.
This is not a bug to repair by discarding inconvenient undefined hypotheses:
those hypotheses genuinely survived the demonstrations. More expressivity and
more determined predictions are different objectives. Moreover, inclusion-minimal
feature pools need not be nested across action languages.

## General conflict sets replace the special pair theorem

For a labelled pointed input `(u,y)`, define its admissible actions by

$$
A(u,y)=\{\theta\in\Theta:E(\theta,u)=y\}.
$$

A feature class C can share a rule exactly when

$$
\bigcap_{(u,y)\in C} A(u,y)\ne\varnothing.
$$

With constants and centre copy, an empty intersection has a two-point witness.
Not so with general actions: the sets `{a,b}`, `{a,c}`, `{b,c}` intersect in every
pair but not jointly. The implementation finds deletion-minimal inconsistent
sets K. Any successful refinement must select a feature nonconstant on K:

$$
F\cap\{f:|\{f(u):(u,y)\in K\}|>1\}\ne\varnothing.
$$

This clause is necessary; finding one separator is not sufficient to solve the
whole class. Repeated exact checking and branching establish sufficiency.
Alternatively, a new operation can remove that conflict by satisfying every
observation in K. This run compares fixed libraries; it does not dynamically
invent arbitrary operations.

The issue occurs in real data, not only a synthetic control. The audit verifies
303 three-point, 14 four-point, and 32 five-point minimal certificates, in
addition to 9,262 pairs. Counts include different arms and model-necessity
certificates and are not independent tasks. Every point is checked against the
original demonstration, and removing any point makes that certificate's action
intersection nonempty.

### Four points on 7e0986d6

These four points in demonstration 0 have identical values for all 19 features:

| Point | North | South | West | East | Required output |
|---|---:|---:|---:|---:|---:|
| (2,10) | 0 | 2 | 2 | 2 | 2 |
| (5,5) | 0 | 0 | 2 | 0 | 0 |
| (8,12) | 0 | 2 | 0 | 0 | 0 |
| (11,12) | 2 | 2 | 2 | 0 | 2 |

All centres have colour 1. Omit the first point and north-copy fits the other
three; omit the second and west-copy fits; omit the third and south-copy fits;
omit the fourth and east-copy fits. No available operation fits all four.
A two- or three-point-only audit would miss this exact obstruction.

The labels suggest another shared operation: take a strict majority among the
in-bounds orthogonal neighbours. A separate training-only protocol was committed
at `bb8bf2e2` after the primary run and before this sensitivity check. The new
operation was manually proposed from training evidence, not invented by the
primary learner. It ignores the centre, rejects ties/no majority, and is undefined
with no neighbours. No alternate radius or tie rule was tried.

That operation explains all four points. But `7e0986d6` still has another
inconsistent class: its full-feature conflicting-class count decreases from
three to one, not to zero. Two other training tasks, `5751f35e` and `caa06a1f`,
become fully training-compatible. Full-feature fits rise from 374 to 376 among
the 678 eligible training tasks; evaluation stays 32/81. These are unrestricted
full-feature consistency counts, not <=3-feature fits. No new query prediction
or score is produced by this audit. Fixing a certificate is not fitting a task,
and fitting a task is not solving its query.

The operation's access to neighbour colours is explicit. This does not make
input information unnecessary; it moves that dependence from rule selection
into execution. The full prediction remains `E(h(a(u)),u)`.

## Where the bottleneck moved

Under joint, training failures split into 304 full-vocabulary conflicts and
218 tasks requiring more than three features; 156 have a bounded fit. Base had
602 full-vocabulary conflicts, four feature-bound failures and 72 fits. In
evaluation, joint has 49 vocabulary conflicts, 27 bound failures and five fits.
The new terms distinguish many previously collapsed cases, but frequently by
producing more specific keys that require more features or have little query
coverage. These are exact classifications under the finite grammar, not
predictions of what a larger synthesizer would do.

Among the joint-selected training models, 19,696 query cells have unknown keys,
964 have disagreeing actions, 235 have undefined surviving actions, and 9,919 are
determined. This is a different cohort from base, so raw count differences are
not a controlled per-task estimate of harm from adding features. The joint
predictor has 596 wrong defined cells; union has 665. Cell counts are correlated
and many cells are unchanged; they are not independent reliability observations.

The union candidate oracle solves 26 training tasks; the selector solves 25.
Only `6e82a1ae` remains recoverable by selecting among the already retained
candidate predictions. Another scalar selector score has little headroom here.

A more concrete next direction is to learn **relations among numeric features**
and the associated conditional actions instead of memorising exact tuples.
For example, a comparison between west/east distances might replace several
individual table entries. Its applicability must be expressed too: an endpoint
must exist, and a blank row is not the same domain as a row bounded by markers.
That is a proposed next experiment, not a rule learned by 017. Both its predicate
construction and output-operation selection must be rebuilt inside the outer
holdout. More feature-subset search remains a separate expressivity question.

## Position relative to prior work

Conflict-driven synthesis and unifying partial solutions are established
approaches. STUN (Alur, Cerny and Radhakrishna, CAV 2015) combines programs correct
on different parts of an input space using domain-specific unification operators.
Feng, Martins, Bastani and Dillig's 2017 conflict-driven synthesis learns lemmas
that prune related spurious programs. This experiment claims neither novelty
for those ideas nor a prior-free definition of generalisation. Its contribution
to this project is an executed split/share comparison, qualified certificates,
and reusable task-level evidence identifying the next bottleneck.

Sources: https://www.microsoft.com/en-us/research/publication/synthesis-through-unification/
and https://arxiv.org/abs/1711.08029.

## Checks, evidence and disposition

Sixteen pre-corpus tests include 150 exact guided/exhaustive subset comparisons,
729 small constant/copy consistency comparisons, a three-way conflict, directional
semantics, undefined actions, duplicate handling and outer-label isolation.

The post-run audit independently evaluates directional features using array
slices and the older features using union-find; action checking uses explicit
sets rather than learner bitsets. It verifies 7,539 feature-necessity certificates,
2,072 full-vocabulary certificates, 30,032 exhaustive subset checks on 12
hash-selected fitted tasks (32 nontrivial complete pools), 3,059 candidate query
grids, 9,251 internal and 11,760 outer fold predictions, and all 5,600 policy/task
scores. The full base arm matches 016 on 1,120 eligibility checks, all 761
eligible pools, 378 candidate models and 2,352 outer folds. The auditor replays
outer predictions but does not independently reimplement outer selection;
source review and synthetic label isolation cover that boundary.

The supplementary majority operation also passes all 81 ternary four-neighbour
configurations and three boundary cases. It was not added to the frozen primary
learner. The primary predictions reproduce byte-for-byte with 12 workers:

`17cbb1bbca38c9ff3743a633bfe995498cd614291c042f13ba7bfbadb919fc05`.

Retain this experiment as a research candidate and its exact audit as a reusable
control. No stable-core integration, theorem about arbitrary ARC tasks, blind
holdout claim, or general confidence guarantee is made. Compact results and case
certificates belong in the branch evidence directory; the accompanying archive
contains complete predictions, inputs, separately scored answers, sources,
protocols, hashes and a reproduction script.

Publication note: a mistyped compressed upload was rejected by its checksum before any evidence was written. The corrected transport was verified against the local payload hash; scientific sources and results were unchanged.
