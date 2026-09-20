# Totality is a constraint, not a truth signal

## Result

Experiment 024 implements the distinction proposed after 023: require a concrete
candidate program to be defined on every supplied query input, instead of merely
preferring a model family that already has an unambiguous answer. It filters
whole program families before projecting their possible actions, including the
coupling imposed by a shared default.

On ARC1's 400 public training tasks, base relational inference increases from
28 to 30 exact solutions, while complete wrong answers increase from one to two.
Its stable-solver composition increases from 79 to 80 solutions. Default-based
inference gains no solutions and increases from four to nine complete errors;
its composition remains 81/400. All compositions stay 27/400 on evaluation.
The default totality variant also adds a complete relational evaluation error.
These are test-query scores on reused public data, not fresh holdout evidence.
No stable code, old wrapper, primitive, feature bound or ranking cost changed.

The most useful outcome is an exact account of what the domain constraint can
and cannot establish, with both a clean completion and a wrong completion.

## Registered scope and frozen assumptions

Baseline: `f2a926f30783de3f2ef985d582c748e57c603907`. The
[protocol](../../024-query-totality/README.md) was committed at `81d09eb5`,
and the active ledger entry was confirmed before execution. The source/test
hashes and 16 successful synthetic controls were committed at `2e0738ce`
before either scientific prediction run. Source files were published afterwards
with those hashes, not claimed to have been in GitHub before the run.

Reuse all 800 projected ARC1 tasks and all 24 controlled specifications from 023.
The controlled data are six previously inspected teachers, crossed with four
palette/position demonstration conditions. Each problem has the same two labelled
demonstrations and the same 36 crossed plus 14 overlapping legacy query inputs.
All query outputs stay in separate files until both modes' predictions are
written and hashed. No new ARC2 run or artificial ARC ground-truth label is added.

The 29 features, 19 operations, <=3-feature minimal-subset search, internal
predictive scores and tie-breaks are unchanged. The primary computation reuses
023's frozen full-data and outer candidate pools; it reconstructs each operation
table from its own labelled fitting subset. Cached outer scores are stripped.
The new conditioner sees the outer held-out INPUTS, not their outputs. It never
changes internal validation evidence, and never reoptimizes a prior after seeing
query definedness. This isolates the new constraint from a change of the learner.
A separate fresh replay reruns search and selection as described below.

## The precise logical step

For a fitted representation a, let P_D be its concrete program family. A base
program chooses one operation from V_z for each observed context z, and uses that
same operation at every occurrence of z. It is undefined on unobserved contexts.

Impose the finite-domain requirement

$$
P_{D,Q}=\{p\in P_D:Q\subseteq\operatorname{dom}(p)\}.
$$

This is not the requirement that every original candidate already be total.
It also is not permission to pick a different operation at each query occurrence.
Let

$$
T_z(Q)=\bigcap_{u\in Q:\,a(u)=z}\{\theta:E(\theta,u)\text{ is defined}\}.
$$

With independent per-key choices, P_{D,Q} is nonempty precisely when every query
key is observed and V_z intersect T_z(Q) is nonempty at every queried key.
Conditional action projections are then V_z intersect T_z(Q). A nonempty family
can still disagree on output colours: those cells remain abstentions.
If the entire family is empty, it supplies no predictions, rather than falsely
using vacuous agreement or combining incompatible partial programs.

The closed-table assumption matters. A missing key is outside these declared
programs' domains; this is NOT a proof that no extension of the table, or no more
general symbolic program, can explain it. This experiment does not construct
such extensions. Nor does query totality establish the model's global domain.

If a correct query-total program belongs to P_D, the constraint cannot eliminate
it. But the intended program may not belong to the selected family, or even to
the entire retained pool. A uniquely determined answer after filtering need not
be correct. This is conditional hypothesis elimination, not a confidence theorem.

## Shared defaults: condition before projecting

The default mode uses 021's original TRAINING-optimal defaults D_min. For a fixed
default d, the choices at key z are {d} if d belongs to V_z, otherwise V_z.
A default branch survives only if all of its query-key choices satisfy totality.
Only afterwards are surviving branches projected into per-key action sets.
A more expensive default is not silently adopted when every old optimum fails.

Why this order matters is visible in a two-key example. Suppose the only
programs are (a,a) and (b,b). Their per-key projections both contain {a,b}.
The first query requires a to be defined, and the second requires b. Independent
filtering leaves (a,b), apparently feasible, but that program never existed.
Whole-program conditioning correctly returns an empty family.

The synthetic suite checks this false-feasibility case exhaustively. In the real
ARC1 tables, exact coupling produces four stricter nonempty projections than
naive filtering, on outer models of `25ff71a9` and `d037b0a7`. No actual false-
feasibility case occurs in these retained ARC1 or controlled pools. Those four
projection differences are not claimed as four newly solved tasks.

## Four frozen selection policies

1. Original training-ranked model, without conditioning.
2. The unchanged 023 tie-complete reference: among best-evidence models, prefer
   a fully determined query answer before applying the old cost/syntax tie-break.
3. Total-ranked: discard empty query-total families, select the originally
   best-ranked surviving family, and report its exact conditional prediction.
   Nonempty but disagreeing families are allowed and can give partial answers.
4. Total-tie-complete: among the best-evidenced surviving total families, apply
   the previous completeness tie-break. This adds that heuristic to the domain
   constraint; it is not part of the logical conditioning operation itself.

If no total family survives, the last two policies abstain. They do not preserve
partial predictions from a family that cannot produce one program for all Q.
The two totality policies happen to give identical final query predictions in
this run. Their outer-fold decisions differ in some cases, so they are not the
same algorithm. Query-set dependence is intrinsic to the stated domain condition.

## ARC1 results

Each entry below is an exact test-query score for all 400 tasks in the named
public split, not a training-reconstruction score.

| Mode / policy | Training correct | Training complete wrong | Evaluation correct | Evaluation complete wrong | Hybrid train / evaluation |
|---|---:|---:|---:|---:|---:|
| Base / original | 27 | 1 | 5 | 0 | 78 / 27 |
| Base / 023 tie-complete | 28 | 1 | 5 | 0 | 79 / 27 |
| Base / total-ranked | **30** | **2** | 5 | 0 | **80 / 27** |
| Base / total-tie-complete | 30 | 2 | 5 | 0 | 80 / 27 |
| Defaults / original | 29 | 4 | 5 | 0 | 80 / 27 |
| Defaults / 023 tie-complete | 30 | 4 | 5 | 0 | 81 / 27 |
| Defaults / total-ranked | 30 | **9** | 5 | **1** | 81 / 27 |
| Defaults / total-tie-complete | 30 | 9 | 5 | 1 | 81 / 27 |

The hybrid is exactly the frozen 019 no-fit rule with saved stable predictions:
retain stable whenever its program fits all demonstrations; otherwise substitute
a complete relational answer, else retain stable. Low error counts of a mostly
abstaining relational learner are not low error counts of that guessing hybrid.

Base gains `a699fb00` and `d5d6de2d` relative to 023 tie-complete, and adds the
complete error `f76d97a5`. It loses no previously solved relational task.
Only `a699fb00` improves the hybrid: the stable program fits the demonstrations
of `d5d6de2d`, so the unchanged composition deliberately keeps its wrong guess.
This is why two new standalone solutions produce only one new hybrid solution.

The new default-mode complete training errors are `6c434453`, `a79310a0`,
`d4f3cd78`, `d5d6de2d` and `f76d97a5`; the new evaluation error is `27a77e38`.
There are no new default-mode exact task solutions. One extra individual query
is correct on `d5d6de2d`, but the task is still wrong as a whole. All paired raw
and hybrid task/grid outcomes are retained, including these failures.

Excluding the three prior fixtures leaves base hybrid 77/398 to 78/398 training,
default hybrid 79/398 unchanged, and 27/399 evaluation unchanged. The gains are
not those fixtures, but remain part of the reused public corpus.

The strict outer gate under total-ranked retains 15 base training answers and
three evaluation answers, all correct; total-tie-complete retains 16 and three.
Defaults retain 14 and three, all correct. These are measured rejection/coverage
tradeoffs, not a totality-implies-correctness result. The new unfiltered policies
are not integrated into any previous wrapper or the stable core.

## A clean completion without a new representation

On `a699fb00`, base inference keeps precisely the old selected key:
(colour, west-endpoint-defined, east-endpoint-defined). Three query cells become
determined, completing the correct task. At query position (0,0), the context's
surviving training operations were:

```text
constant 0; copy centre; copy north
```

The north copy is undefined at that top-edge point. Constant 0 and centre copy
both remain defined and return 0. The domain constraint removes north copy;
no observed output colour or invented feature was needed for this elimination.
The intended output there is 0. This is an exact completion of a previously
ambiguous model, not a new feature-discovery result.

On `d5d6de2d`, base inference keeps (colour, component-boundary-reachability).
Eight cells become determined. For context (2,1), constant 0 and beyond-run east
copy survive; beyond-run north/west copies cannot execute at query corners.
The same program choices must work across both queries. Both query grids become
correct, subject to the hybrid limitation described above.

## A wrong completion explains the limitation

On `f76d97a5`, the previously preferred model has three unknown query keys.
It is rejected under the closed-table semantics. The newly selected model has
weaker internal evidence: zero complete demonstrations and fraction zero,
versus zero complete and 9/25 for the old model. Its features are local minimum
degree, local maximum degree and maximum degree among same-colour peers.

At query cell (0,3), whose input colour is 5, its context allows:

```text
constant 6; copy north; copy east; copy beyond north run; copy beyond east run
```

North operations are undefined there. East operations return the intended colour
3 at THIS point, but are undefined at another query occurrence of the SAME key,
position (2,4). One shared operation must cover both occurrences, so those east
operations are also removed. Only constant 6 remains. The predictor becomes
unanimous and wrong: the official output at (0,3) is 3.

This is not repaired by ignoring undefinedness at the inconvenient occurrence:
that would pick different operations inside a context declared to share one.
It exposes a wrong identification or an insufficient operation/program family.
The conditioning itself is exact relative to that family.

A separately labelled post-score oracle diagnostic intersects each family's
operation constraints with the actual complete query outputs, respecting shared
defaults. In base mode, some concrete correct program exists in the retained pool
for 34 training tasks, but a unanimously correct conditioned family exists for
31; selection solves 30. For `f76d97a5`, NO retained family contains a concrete
correct query program. For the older error `6e82a1ae`, a correct family does exist.
Defaults have 34/34 such training opportunities and solve 30. Both modes have
five/five evaluation opportunities. These are finite-pool ceilings, not deployed
policies or claims about the full ARC program space. Query labels enter this
post-score diagnostic only and never change predictions.

## Controlled outcomes: definedness does not supply missing contrasts

The entries are correct / complete-wrong / incomplete crossed query grids,
out of 216 dependent queries from six already-inspected teachers.

| Palette varies | Row varies | 023 tie-complete, base | Total-ranked, base | 023 tie-complete, defaults | Total-ranked, defaults |
|---|---|---|---|---|---|
| No | No | 12 / 76 / 128 | **12 / 204 / 0** | 12 / 76 / 128 | **12 / 204 / 0** |
| Yes | No | 36 / 84 / 96 | 156 / 60 / 0 | 96 / 64 / 56 | 156 / 60 / 0 |
| No | Yes | 0 / 0 / 216 | 0 / 0 / 216 | 72 / 0 / 144 | 72 / 0 / 144 |
| Yes | Yes | 216 / 0 / 0 | 216 / 0 / 0 | 216 / 0 / 0 | 216 / 0 / 0 |

The two totality policies have the same final outcomes. The fully crossed,
discriminating demonstrations retain their 216/216 success. Without either
contrast, totality turns all 128 abstentions into wrong complete answers.
Colour-only designs improve but retain 60 wrong answers. This is not a general
recommendation to answer whenever a total explanation can be found. The primary
and overlapping legacy banks are reported separately in the machine-readable
results; their counts are not pooled as independent observations.

## What to do next

Retain the exact conditioning implementation as a domain-analysis instrument,
not a new confidence heuristic. It can remove impossible concrete programs
without inventing labels, but cannot validate a scope identification that made
the intended program unavailable. The useful completion and the failure above
are both consequences of precisely the same rule.

A more productive next construction experiment should use domain failures to
identify where a context needs refinement or a partial program needs a guarded
extension, instead of merely replacing it by the next total candidate. In the
counterexample, east-copy is appropriate at one occurrence but cannot cover the
whole class. A proposed guard must explain that split; its new branch must not
receive a made-up label. Such extensions change the program family and require
a separate frozen test, including literal-output controls and nested evaluation.
The oracle's distinction between missing correct programs and poor selection
should remain explicit before adjusting another score.

Transductive use of query inputs and constraint-based synthesis have established
precedents. Lee et al., *Program Synthesis via Test-Time Transduction* (2025),
uses LLM-predicted labels to eliminate hypotheses; this experiment does not.
STUN (Alur, Cerny and Radhakrishna, CAV 2015) unifies partial solutions under a
specification. Neither paper makes the correctness of this new policy follow
from a willingness to answer. No novelty is claimed for conditioning a finite
hypothesis set on an additional domain constraint.

Sources: https://arxiv.org/abs/2509.17393 and
https://www.microsoft.com/en-us/research/publication/synthesis-through-unification/.

## Verification, reproducibility and disposition

Sixteen new synthetic tests pass, including 1,200 comparisons against explicit
concrete-program enumeration, no reoptimization of defaults, shared-key domain
conflicts, empty-family handling, literal zero, undefined actions and outer-label
isolation. All 42 inherited 018/021/023 tests also pass.

The independent explicit-set audit reconstructs raw action sets and default
branches from the specified training indices, implements a separate domain
conditioner and rank/filter selector, and replays all selected outputs. It checks
59,944 tables, 2,344,478 context projections, 32,647 default branches, 133,504
unchanged candidate records, 1,088,864 candidate-grid predictions, 20,488 policy
choices, 13,896 outer-grid scores, 6,592 raw policy/task scores and 6,400 hybrid
scores. These counts repeat related models/folds, not independent samples.
Feature/action extraction is reused; the auditor does not independently invent
or exhaustively resynthesize the underlying feature vocabulary.

Both full 800-task ARC1 cached computations repeat byte-for-byte. A fresh
end-to-end replay, without cached candidates, matches all 24 controlled
specifications in both modes, and 12 deterministic hash-selected fitting ARC1
tasks in each mode. It rechecks the original search as well as conditioning;
it is not an independent implementation of that search.

Two attempted monolithic controlled invocations were interrupted; the tool
reported a transport timeout and no completed evidence files were produced.
Memory pressure was observed, and the remaining orphaned workers were terminated. The same frozen code was then completed in six fixed four-task
batches per mode, still using twelve workers, and reassembled with exact task-set
and source-hash checks. No task, feature bound or score-dependent policy changed.
All four complete prediction files were frozen before either query answer file
was used for scoring. Runtime is provenance only, especially because the primary
run reuses earlier candidate pools.

Prediction SHA-256:
- ARC1/base: `bb2f4bd3688c62d32cca16bb81386bfcdd108d389225d50496c26cc598637ca2`
- ARC1/default: `29b4b81317680d9c2b09de3461fd47790fca88a52e2b0f5fdf2be7b964f4ec40`
- Controlled/base: `f77ffd6d5460f4197da6deb295fdaed8306202f958489e2e6bd84a0d79ef968d`
- Controlled/default: `fc8c084dffcb29d34f4f2ffafa0298a24e43d24b9023011078f8f83f2a3f964a`

Close without integration. Retain the exact finite-family conditioner, synthetic
program-enumeration tests, audit and case certificates as reusable controls for
future guarded-extension work. Full predictions, source dependencies, projected
inputs, separate answers and checked reproduction instructions are in the
companion archive. The stable solver, old wrappers and accepted mathematics are
unchanged.
