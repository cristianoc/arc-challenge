# 027 — Construct a spatial set expression and reuse it generatively

Baseline: `37a312e7725f4ac0a62850e7b4ff0f8ee891585c`. Status is in the ledger.
No change to stable solver or accepted mathematics. This follows 026's exact
count/span collision for frame recognition, rather than changing another selector.

## Supplied language and construction target

An input is an already segmented nonempty finite set S of integer coordinate
pairs. The carrier B(S) is the Cartesian product of the inclusive coordinate
ranges from each minimum to maximum. This range/axis/segmentation prior is
explicit; constructing it is NOT an outcome. Within B(S), compare each point's
row or column with the corresponding minimum or maximum over S. These four
comparisons are generic expanded expressions, not frame/interior atoms.

Construct point predicates from those four comparisons, true, false, not, and,
and or, through seven Boolean syntax nodes (a comparison counts as one Boolean
node but its entire coordinate/projection/extremum AST is charged in expanded
cost). Exact semantic equivalence is computed on ALL sixteen Boolean valuations
of the comparisons, not agreement only on the demonstrations. Retain a minimum
expanded-cost representative per truth table with a deterministic syntax tie.
This quotient is sound for the declared point-predicate language. Do not use
shape labels or query inputs when building it.

A candidate shape observation is equality of S with the constructed set
`{p in B(S) : phi(p,S)}`. Compare both binary output-label assignments, including
constants, and retain every minimum expanded-cost semantic program fitting the
source labels. This is a deliberately restricted generative-template grammar,
not arbitrary first-order geometry or primitive-free abstraction discovery.
Keep exact counts/spans as a negative observation-language baseline: the old
collision must remain inseparable for every function of those three scalars.

## Frozen source and test data

Source: all 511 nonempty subsets of the 3x3 coordinate universe, labelled by the
same independent row/edge frame teacher as the known 026 control. This is dense
supervision, not sparse ARC. A registered sparse diagnostic uses only the exact
known collision: masks 255 (negative) and 495 (positive). It is a development
ablation, not a new randomly held-out task. Do not change the grammar or ranking
if its extrapolation is wrong or underdetermined.

Recognition queries: all 65,535 nonempty 4x4 subsets, reporting the 511 exact
source overlaps separately; and larger shapes in 5x7,7x5,8x8,11x13 boxes. For each
larger box include its complete perimeter, each single perimeter-cell deletion,
each single interior-cell addition, and 32 seeded mixed deletion/addition cases.
Use three translations (0,0),(-9,4),(12,-6), include transposes, and deduplicate
actual point sets. Report positives/negatives, not just majority-class accuracy.
Source examples select the library before query data are accessed.

## A genuinely different computation using the frozen body

Fit two set-valued target tasks from two supplied demonstrations each. One repairs
a damaged/noisy outline; the other returns its interior as a NEW set of cells.
The fixed target operation grammar is identity, constructed set, carrier-minus-
constructed-set, union, intersection, each directional difference, and symmetric
difference of the input and constructed set. No operation names the target shape.
The shape predicate is frozen before target examples enter fitting.

For each target, demonstrate a 5x7 outline with one noncorner edge cell removed,
and a translated 7x5 outline with one edge cell removed and one interior cell
added. Teacher outputs are independently generated complete edges or strict
interior. Test on the larger recognition bank. All inputs preserve the intended
extrema; no claim of recovering lost bounds is permitted. Full spatial outputs
are compared, not output relabellings or only counts.

Compare library reuse with fresh enumeration of the SAME source-fitting predicate
bodies and target operations using the SAME source and target evidence. Charge
expanded bodies, not a free macro. The expected reusable-work advantage is cache
amortization; no speedup claim against an optimized caching solver is allowed.
Retain all minimum target programs and their disagreements, never choose by test
correctness. Report target grammar checks and source enumeration separately.

## Semantic coverage and negative control

Record which of the sixteen point roles are actually constrained by positive
source examples. If they determine the entire truth table, explain why a huge
query count is then exhaustive checking of one finite semantic table rather than
thousands of independent generalisation successes. Reproduce candidate sets and
costs under duplicated aliases, different constructor iteration order, and
expanded versus named expressions. Macro invariance must not hide the body cost.

Use two 5x5 sets: the full box minus its centre and the full box minus (1,1).
They share count/spans and have the same sets of occupied/unoccupied extrema
roles. A thickness-two outline teacher labels them differently. Every observation
of the frozen template form must identify them. Retain that exact language-level
obstruction; do not add distance predicates after seeing it.

## Execution, audits and stopping

Register before execution; save source/code hashes and learned library before
query predictions, and hash predictions before the separate answer scorer.
Standard-library deterministic semantic/work-count study, not timing benchmark.
Use a single process as in 026; no comparison of CPU time to earlier 12-worker
ARC runs. Fixed seed 2701, no score-dependent parameter changes. No new ARC score.

Synthetic controls precede scientific execution. An independent auditor must
reconstruct the bounded grammar by a different algorithm, verify all retained
truth tables and minimum programs, independently generate teacher geometry,
verify the old and new collision certificates, and recompute all final scores.
Use neither scientific evaluator nor fit routine in the auditor. Repeat outputs
byte-for-byte. Report separately where low-level primitives remain supplied and
where sparse labels fail. No new theoretical or empirical generalisation claim
is justified merely by successful execution of this construction.
