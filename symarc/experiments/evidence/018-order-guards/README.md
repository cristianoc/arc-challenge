# Relations transfer; applicability can still be confounded

## Outcome

Experiment 018 adds order comparisons and explicit operation-validity predicates
to the frozen 017 learner. On the 997 nondevelopment tasks in ARC2's public
training split, the combined selector improves from **23 to 30 correct complete
query answers**, with complete wrong answers decreasing from three to two.
There are seven gains and no losses. All policies still solve **0/120** public
evaluation tasks. These are comparisons among experimental learners, not scores
for the stable Rust solver or an untouched-holdout claim.

The positive mechanism is concrete: a learned comparison transfers to unseen bar
heights. The negative mechanism is equally concrete: on a separate controlled
family, the correct conditional model solves all 28 declared query grids, but
validation selects an accidental guard that solves only 14. We preserved that
failure rather than changing the selector or adding another predicate after
seeing the scores.

## Frozen setup and evidence boundaries

Baseline: `aa0afd5a97a58e02461ca57012fe2c4d7bf264d7`. Protocol and exact runner/test
hashes were committed at `8edf6342`, then the active ledger entry at `f50726ed`,
before scientific execution. Runnable files were published subsequently with
those same hashes. Official ARC-AGI-2 data revision:
`f3283f727488ad98fe575ea6a5ac981e4a188e49`.

All 1,000 training and 120 evaluation tasks were processed. `00d62c1b`,
`e88171ec`, and the motivating endpoint task `29c11459` are development fixtures
excluded from primary scores. This leaves 677 eligible training and 81 eligible
evaluation tasks: at least two distinct demonstration inputs and matching input/
output shapes. The primary corpus denominators are 997 and 120. The data have
been used in earlier research and are not independent holdouts.

Every arm has the same 19 output operations from 017: constants, centre copy,
four adjacent copies and four first-different-colour copies beyond directional
same-colour runs. The maximum selected feature count is three. Every consistent
action survives; unknown keys, action disagreements and undefined survivors
abstain. There is no default copying, new majority operator or query-label tuning.

The factorial is:

| Arm | Input features |
|---|---|
| Exact | The 19 features from 017's joint arm |
| Order | Exact plus all six pairwise directional-run trichotomies (<,=,>) |
| Guarded exact | Exact plus four beyond-run-action definedness predicates |
| Guarded order | All 29 features |
| Union | Select among retained candidates from all four arms |

These comparisons and guards are supplied constructors, not newly invented
primitives. Colours are never numerically ordered. The guard treats a returned
colour 0 as defined; only the -1 sentinel denotes absence. Feature selection uses
exact conflict-driven enumeration, not a beam. It ranks internal withheld-grid
accuracy, then per-grid correct-known-cell fractions, feature count, supplied
constructor costs, arm and syntax order. Counting a compound comparison as one
selected term is an explicit prior, even though its cost includes both operands.

An outer holdout reruns both feature construction and internal model selection
without the outer labels. Duplicate inputs are withheld together. All query
predictions were persisted before a separate scoring process read query answers.

## Primary scores

These are **test-query scores for public training tasks**, not demonstration
reconstruction rates.

| Policy | Tasks fitted | Complete correct queries | Complete wrong queries | Correct after strict outer gate |
|---|---:|---:|---:|---:|
| Exact | 155 | 23 | 3 | 10 |
| Order | 156 | 26 | 3 | 11 |
| Guarded exact | 163 | 27 | 2 | 11 |
| Guarded order | 163 | 30 | 2 | 12 |
| Union | 163 | 30 | 2 | 12 |

The exact arm is 017's **joint arm**, not its union selector. Its apparent change
from the earlier 24 correct answers to 23 is solely the additional exclusion of
`29c11459`. The entire underlying candidate pools, predictions and outer folds
match 017, before any development filtering.

Order-only gains: `3bd67248`, `ce9e57f2`, `ea786f4a`.
Guard-only gains: `2281f1f4`, `25ff71a9`, `aa18de87`, `d037b0a7`.
The combined selector retains all seven and loses none of the exact arm's correct
answers. Remaining complete errors: `6e82a1ae` and `b7256dcd`. The former complete
error `5ad8a7c0` becomes incomplete, not correct.

The strict outer gate retains twelve correct complete answers and no complete
errors. It eliminates both complete errors but discards eighteen of the thirty
correct answers. Fourteen tasks pass the outer check; two have incomplete
queries. This is a measured error/coverage tradeoff, not a correctness guarantee.

All evaluation arms fit five tasks but produce no complete answer. The union's
finite candidate oracle is 32 training solutions and zero evaluation solutions.
Only two complete training answers remain recoverable by selection within this
pool. These are descriptive results on reused public data, not population-effect
estimates or claims of benchmark competitiveness.

## Positive case: a relation survives a height change

On `ce9e57f2`, the selected exact key is colour plus northward and southward run
lengths. The order arm instead selects

```math
a(u)=(\operatorname{Colour}(u),\operatorname{sign}(d_N(u)-d_S(u))).
```

Both fit the demonstrations. The comparison model predicts all three internally
withheld demonstrations, passes every outer check and predicts the taller query
exactly. Demonstration grid heights are 8, 7 and 9; the query height is 11.
On the supplied palette, colour-2 bar cells whose upward same-colour run is longer
than their downward run become colour 8; the others remain unchanged. This
colours the lower half of each bar, leaving an odd bar's middle unchanged.

There is no numeric threshold chosen from the query and no added task-specific
rule. The relation can reuse an action for previously unseen distance pairs.
The complete learner is still an exact table on **relational signatures**. It is
not arbitrary arithmetic synthesis, and observed success does not establish a
rule for previously unseen colours or every possible grid.

A second case, `2281f1f4`, selects the pair of definedness predicates for the north
and east beyond-run operations. Its table colours the intersection positions
identified by these input conditions and copies other observed contexts. It
passes all three inner demonstrations, every outer fold and the query. The
input conditions were selected from the common vocabulary, not hand-selected
symmetries or a task-specific marker detector.

## Why comparison is abstraction rather than more input information

The comparison code is a function of the old distances. If all old features are
equal at two points, adding their comparisons cannot distinguish them. Thus
order-only features cannot remove a full-vocabulary conflict. The audit checks
this on every eligible task.

Indeed the training classifications are:

| Arm | Full-vocabulary conflict | More than three selected features needed | Bounded fit |
|---|---:|---:|---:|
| Exact | 304 | 218 | 155 |
| Order | 304 | 217 | 156 |
| Guarded exact | 246 | 268 | 163 |
| Guarded order | 246 | 268 | 163 |

The useful effect of comparison is to **forget magnitude while retaining order**.
For any common strictly increasing reparameterisation phi, comparison of phi(a)
and phi(b) agrees with comparison of a and b. This is a property of numeric
features, not a claim that every such reparameterisation is a realizable ARC
grid transformation or a valid task symmetry.

The guards are different: they expose whether an existing decoder operation is
defined, information absent from the old feature vector. They can remove some
full-vocabulary conflicts. Both changes still rely on explicit representation
and output-operation priors.

## Controlled failure: relation learned, wrong applicability selected

Before execution, we declared two artificial teachers: copy the nearer endpoint
or copy the farther endpoint of a zero row bounded by two different colours;
ties output 5. Everything else copies. Each teacher has two demonstrations,
widths 7 and 11, with endpoint colours (1,2) and (3,4), in five-row grids.

Each is tested on widths 5,9,13,17,21,25,29: once with colours (6,8) and height 5,
once with colours (7,9) and height 7. These are 28 **teacher-labelled constructed
queries**, not new ARC annotations, independent tasks, or evidence of author
intent. Both the family and all test values were fixed before execution.

| Selected method, across both teachers | Correct grids | Complete wrong grids | Incomplete grids |
|---|---:|---:|---:|
| Exact | 0 | 0 | 28 |
| Guarded exact | 0 | 0 | 28 |
| Order / guarded order / union | 14 | 2 | 12 |
| Available endpoint-validity model, not selected | 28 | 0 | 0 |

The selected model uses `compare(W,E)` and `near.any_border`, where the latter
asks whether the cell or an immediate same-colour neighbour touches the canvas
border. That condition accidentally separates the active row from background
in the five-row demonstrations. It is not the same as being horizontally
bounded by endpoint colours.

The model transfers to every tested width at height 5 but fails at height 7.
At width 5, height 7, it confidently colours a blank background cell 5. At the
other new-height grids it also encounters undefined surviving operations and
abstains. Both teachers exhibit the same pattern. The strict outer test rejects
all these selected controlled models; this is not a failure of a passed strict
gate, but of the partial-validation ranking.

The correct alternative is already in the candidate pool:

```math
(\operatorname{compare}(d_W,d_E),
\operatorname{Defined}(\operatorname{CopyBeyondRun}_W),
\operatorname{Defined}(\operatorname{CopyBeyondRun}_E)).
```

Its frozen predictions solve all 28 constructed grids. The known ARC endpoint
fixture `29c11459`, however, still selects an exact distance pair in every arm.
We therefore cannot claim that 018 prospectively inferred the intended general
endpoint rule from that ARC task's demonstrations.

### The wrong model wins validation for an exact, inspectable reason

For each teacher, neither candidate predicts a complete internally withheld
demonstration. The sum of their per-grid correct-known-cell fractions is:

| Candidate | Validation fraction sum |
|---|---:|
| Border-neighbourhood guard, selected | 134/77 |
| Endpoint-validity guards, available | 634/385 |

The difference is

```math
\frac{134}{77}-\frac{634}{385}
=\frac{36}{385}=\frac{2}{35}+\frac{2}{55}.
```

It consists exactly of the two endpoint cells in each withheld grid. With only
one remaining demonstration, the correct guard's separate left/right endpoint
classes retain both a literal colour and CopyCentre; on the new colour these
actions disagree, causing abstention. The accidental guard pools differently
coloured endpoints with other unchanged contexts. That eliminates the constants
and identifies CopyCentre, gaining those four validation predictions.

Thus the wrong guard is not merely a cheaper tie-break winner. Its partial
validation score is strictly higher. Pooling examples provides useful evidence
for an operation while making an unjustified claim about the contexts to which
that operation applies. Both demonstrations share the same confounding height.
These findings use the saved candidates and scores; no policy was changed.

## Interpretation and next research target

018 establishes useful relative comparisons and explicit applicability selection
within a supplied grammar. It does not discover primitive-free concepts or
remove the need for priors. A compact context can improve action identification
and still extrapolate its scope incorrectly. Conversely, retaining finer
contexts can create literal/copy ambiguity and unnecessary abstention.

The next issue is therefore not just adding comparison operators. We need to
investigate **which observations ought to share evidence for an operation,
without automatically sharing the same applicability condition**. A conditional
program can share a copy operation across different branches while retaining
different guards. Exact lookup tables conflate those two choices. Any comparison
with such a conditional learner must keep the primitive/action vocabulary fixed
and rebuild its guard and operation selection inside outer validation.

Conditional synthesis and separating predicates have established precedents:
[STUN](https://www.microsoft.com/en-us/research/publication/synthesis-through-unification/)
(Alur, Cerny, Radhakrishna, CAV 2015) and
[divide-and-conquer synthesis](https://www.microsoft.com/en-us/research/publication/scaling-enumerative-program-synthesis-via-divide-and-conquer/)
(Alur, Radhakrishna, Udupa, TACAS 2017). The latter explicitly separates partial
expressions from predicates and combines them using decision trees. No novelty
is claimed for that general architecture. Our concrete deliverable is the
executed comparison and the controlled failure that a next learner must address.

## Checks, runtime and disposition

Fifteen pre-corpus controls pass, including 150 guided/exhaustive subset checks,
zero-valued endpoints, undefined survivors and outer-label isolation.
The independent audit uses array slices for directions, union-find for the older
component features, and explicit operation sets instead of learner bitsets.
It checks 1,120 baseline eligibility decisions, all 761 exact-arm pools,
1,129 candidate models and 2,352 exact-arm outer folds against 017.
It additionally verifies 110,355 exhaustive subset checks on twelve hash-selected
tasks, 17,937 feature-necessity certificates, 1,290 full-vocabulary certificates,
19,598 internal and 11,760 outer predictions, and all 5,600 policy/task scores.
The outer selector is not independently reimplemented on the corpus; source
inspection and synthetic isolation tests cover that boundary. Replaying a
selected model is not equivalent to independently validating its selection.

Monolithic invocations exceeded the tool execution window. The identical,
task-independent learner was completed in fourteen serial batches of eighty
tasks, using twelve workers per batch. No task was dropped, search bound changed
or score inspected to choose a batch. All inputs were reassembled and checked
against the complete projected corpus before scoring. The repeated batched run
produces byte-identical complete predictions. Sum of successful first-run batch
times: 65.018s; repeat: 65.691s. These are provenance, not speed comparisons.

Prediction SHA-256:
`97ad00b4c93a2fc0a71003581480cf6ec6db09499aa9c0934197c119025fdad6`.

Retain as a research candidate and reproducible conditional-transfer/selection
control; no stable-solver integration or accepted-mathematics change. The report,
compact evidence and all source files are in PR #32. The accompanying archive
retains complete predictions, separately stored answers, candidate tables/folds,
audits, baseline evidence, all source dependencies and a reproduction script.
The one-shot ledger workflows only publish registration/results and remove
themselves; scientific computation is local.
