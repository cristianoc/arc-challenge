# Conflict-guided refinement: useful distinctions, conditional certificates

Experiment 016 continues 015 by constructing feature combinations from labelled
training conflicts. The frozen method improves from 11 to 22 exact solutions on
the nondevelopment public training tasks, at the cost of four complete wrong
answers. It produces no complete evaluation answer. The stronger result is a
small, inspectable learning mechanism: every selected feature has an exact
certificate explaining why the remaining features are insufficient for the
supplied output operations.

## What was frozen

Baseline: PR #32 at `d9e4ff652bcc07e3d0066175b99c67a0ae088e83`.
The [primary protocol](../../016-conflict-refinement/README.md) and source hashes
were published at `40c4e78` before corpus computation. The active entry was
registered in the local working ledger before the run, then consolidated in
the branch ledger with the findings.

Data: all 1,000 training and 120 evaluation tasks at ARC-AGI-2 `f3283f7`.
`00d62c1b` and `e88171ec` are development fixtures excluded from primary counts.
Training-only eligibility requires at least two distinct demonstration inputs
and matching demonstration input/output dimensions. This leaves 678 eligible
training and 81 evaluation tasks. Reported primary corpus denominators are 998
and 120, respectively. Public data have been used previously; this is not an
untouched holdout.

The grammar supplies 15 terms. Three are cell colour, canvas-border membership,
and same-colour four-neighbour degree. The other twelve combine three scopes
with four aggregators. Scopes are the point plus its same-colour neighbours,
its same-colour connected component, and all same-colour cells anywhere.
Aggregators are count, existence of a canvas-border point, minimum degree and
maximum degree. Thus boundary reachability is a composition of supplied
constructors, not discovered recursion. The method learns which terms to
combine; it does not invent primitives outside this grammar.

The output language is 015's constants plus CopyCentre. Unknown keys and actions
that disagree on their resulting colour abstain. The complete prediction still
has the form `E(h(a(u)),u)`: sharing an action need not mean equal output colours.

## An exact role for training contradictions

For a labelled pointed input `(u,y)`, let `A(u,y)` contain the output actions
that produce y at u. A feature key is sufficient when the intersection of those
action sets is nonempty in every class of identical keys.

For constants plus CopyCentre, every incompatible class has a two-cell witness:
one cell must change, while another requires a different output. Copy fails the
changed cell; no constant satisfies both. Conversely, without such a pair,
either all outputs agree or all occurrences are unchanged.

For an incompatible pair u,v, let

$$
S_{u,v}=\{f:f(u)\ne f(v)\}.
$$

A sufficient feature set F must satisfy

$$
F\cap S_{u,v}\ne\varnothing
$$

for every incompatible pair. With this particular action language, these
constraints are also sufficient. An empty separator certifies failure of the
entire supplied feature vocabulary, not just the current search bound.

The learner starts with no features, finds a conflict, and branches on all
terms that separate it. It retains all inclusion-minimal sufficient feature
sets of size at most three. There is no beam or node timeout. For each retained
feature f it records a conflicting pair indistinguishable under `F - {f}` but
distinguished by f. Necessity is relative to this F, vocabulary and action
language; it is not a claim that f is intrinsically required by the task.

This is a finite constraint-based feature learner, not a novelty claim about
abstraction refinement or relational feature learning. One relevant primary
account of relational feature construction is Dutta and Srinivasan,
[Consensus-Based Modelling using Distributed Feature Construction](https://arxiv.org/abs/1409.3446)
(2014).

## Selection is evaluated as part of learning

The validation selector ranks fitting feature tuples by their predictions of
withheld demonstrations: complete exact grids first, then per-grid correct-known-
cell fractions. Ties use feature count, supplied constructor costs and syntax
order. This is internal model selection, not an unbiased evaluation of the
selected model.

The outer check withholds an entire distinct demonstration input and all its
duplicates, reruns the feature search and internal selector on the remaining
data, then predicts the outer grid. Thus its label is not used to construct or
select the representation being evaluated. Input-derived features may be cached;
held-out labels may not. Final query predictions are saved and hashed before a
separate scoring command reads query answers.

## Primary results

| Policy | Training tasks fitted | Complete correct training-task answers | Complete wrong answers | Evaluation tasks fitted | Complete evaluation answers |
|---|---:|---:|---:|---:|---:|
| One feature, validation selection | 25 | 11 | 0 | 0 | 0 |
| Up to three features, no reach constructor | 37 | 15 | 4 | 1 | 0 |
| Up to three features, cost selection | 72 | 16 | 2 | 4 | 0 |
| Up to three features, validation selection | 72 | **22** | 4 | 4 | 0 |
| Consensus of validation-tied candidates | 72 | 22 | 4 | 4 | 0 |
| Coordinate-key control | 306 | 0 | 0 | 35 | 0 |

Identity solves no eligible task. These are new learner variants, not scores
for or comparisons against the stable Rust solver.

Relative to one feature, refinement gains twelve correct tasks and loses one:
`6e82a1ae`. Relative to the no-reach control it gains seven and loses none.
Relative to cost selection it gains seven and loses one. These are descriptive
paired results on previously used public data, not population-effect or
calibration claims.

The four complete errors are `3618c87e`, `3aa6fb7a`, `6e82a1ae`, `b7256dcd`.
The finite refined candidate oracle solves 24 training tasks; the selector misses
`3618c87e` and `6e82a1ae`. There are only two remaining correct complete answers
available through selection within this candidate pool.

The strict outer-validation gate leaves ten complete answers, all correct.
It eliminates all four complete wrong answers but also discards twelve of the
22 correct answers. Twelve tasks pass the outer check; two still have incomplete
queries. This is a measured coverage/error tradeoff, not a general correctness
guarantee or an unqualified improvement.

## The enclosure example is now an executed construction result

On development task `00d62c1b`, no single term fits. Refinement constructs eight
minimal sufficient feature combinations. The selected key is

$$
a(u)=\bigl(\operatorname{Colour}(u),
\exists v.\operatorname{SameColourStep}^{*}(u,v)\land\operatorname{Border}(v)\bigr).
$$

The colour-plus-component-size alternative fits but predicts 0/5 whole withheld
demonstrations. Colour plus boundary reachability predicts 5/5. The full nested
outer learner also predicts every withheld grid correctly and its final query
answer is exact. Cost selection chooses the size alternative and does not
complete the query correctly.

The composed boundary predicate is supplied by the frozen grammar. What the
learner constructs is its combination with colour and the associated action
table, chosen from training without task-specific exemptions. This is not
primitive-free discovery, and the development task is excluded from the scores.

Two nondevelopment tasks, `84db8fc4` and `a5313dff`, select the same feature pair,
solve their queries, and pass every outer check. Other reach-dependent gains
are `7b6016b9`, `810b9b61`, `c0f76784`, `d5d6de2d`, `ea32f347`. The two-task result
demonstrates reuse under the supplied prior, not statistical independence of
ARC tasks or recovery of an author's unique intended program.

The previous rectangle fixture `e88171ec` remains outside this vocabulary. In
its second demonstration, points `(5,5)` and `(6,4)` have identical values for
all fifteen terms but must output 8 and 0. That is a training-only failure
certificate even without a three-feature search limit. A component can contain
points with different roles in a contained rectangle; these aggregate features
do not retain that distinction.

## Most remaining failures are not search depth

Among the 678 eligible nondevelopment training tasks, 602 have an exact
full-vocabulary conflict, four are consistent with all features but require more
than three, and 72 have a bounded fit. Evaluation gives 77 full-vocabulary
conflicts and four fits. The four evaluation fits still have incomplete query
predictions: 2,105 unknown-key cells and 35 action-ambiguous cells, versus 513
determined cells. Defined predictions are not thereby correct.

On training tasks, the validation-selected models have 8,641 unknown-key cells,
169 action-ambiguous cells and 5,907 determined cells. There are 303 wrong defined
cells across 24 tasks; 232 errors occur among 1,645 predicted changed cells.
These are correlated descriptive cell counts, not independent samples.

This distinguishes three bottlenecks: a language that cannot fit the training
labels, a bound that misses available combinations, and a fitted rule whose
query applicability/interpretation remains unresolved. More feature-subset
search cannot fix the first.

## A necessary qualification: perhaps the shared operation is missing

After the primary run, a separate [protocol](../../016-conflict-refinement/ACTION_AUDIT.md)
was registered at `8bc397a` for a training-only sensitivity audit. It adds copying
north, south, west or east to the action language, leaving the fifteen features
unchanged. Out-of-bounds copies are undefined. No new query predictions or scores
are produced.

Of the primary full-vocabulary pair conflicts, 60 training and ten evaluation
pairs now admit a shared operation. Yet the number of entire training-compatible
tasks is unchanged: 76 training and four evaluation tasks using all fifteen
features. The 76 include the four cases beyond the three-feature limit. Other
occurrences still contradict the extended action language in every newly
pair-resolved task. Resolving one witness is not solving a task.

The point is conceptual as well as empirical: **a conflict can be removed by
retaining more information, or by allowing a more appropriate shared operation**.
A feature-necessity certificate must name its action language. Moreover, the
special two-cell empty-intersection property need not survive an action-language
extension; the audit intersects all occurrences, rather than assuming pairwise
consistency is sufficient.

## Assessment and next target

This is a useful positive construction result within a declared prior. Feature
combinations add predictive coverage, the reach ablation isolates useful scope,
and the procedure explains each selected distinction using labelled examples.
The results do not establish a prior-free theory, a robust general-purpose ARC
solver, or automatically invented primitives. Retain this as a research
candidate; no stable-core integration is claimed.

The next target should not be another score for the existing candidates: the
candidate oracle offers only two more complete answers. Instead, use the
full-vocabulary certificates to direct a bounded search for **new separating
terms or new shared operations**, explicitly recording which branch removes
which contradiction. This is different from manually supplying a full repaired
program after inspecting the query answer. Any such extension must be rebuilt
inside the outer held-out evaluation and its bias made explicit.

## Evidence and reproduction

Thirteen pre-corpus synthetic controls passed, including exhaustive small action
checks and 120 deterministic tables comparing guided search against exhaustive
subset enumeration. The independent post-run audit uses union-find for features,
explicit action sets and exhaustive bounded subsets. It verifies 50,643 subset
checks, 203 complete candidate pools, 1,233 necessity certificates, 2,080
full-vocabulary certificates across the three families, 7,840 policy/task
complete-and-correct decisions, and 11,760 outer fold predictions. Counts over
families/models are not independent task counts. The audit reconstructs outer
predictions from retained labels but does not independently reimplement the
outer selector; source inspection and synthetic subset isolation cover that
boundary.

A second twelve-worker run produces byte-identical predictions. Original
prediction time is 3.507 seconds, provenance only; no runtime comparison with
older solvers is claimed. No GitHub Actions run was used for 016.

Prediction SHA-256:
`c1d47fa2687bd147f884cdf63c33de41c16980d764b519e52c43abc64e818955`.

[Compact results](results.json) retain decisive counts, source/data revisions,
paired changes and hashes. The accompanying investigation archive retains
projected inputs, answer file, full predictions, per-model necessity
certificates, fold records, scores, audits, dependency sources and manifests.
The stable solver and accepted mathematics are unchanged.
