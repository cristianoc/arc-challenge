# 013 — Can constant lifting identify repairs from training alone?

Registered before execution. Baseline `c9413a8`; 120 external programs/data pinned
by 006/inputs.json. These public tasks have been used in previous audits; this
is not an untouched holdout. The repair grammar and enumeration procedure are
newly frozen here, without inspecting further individual answers. No SymArc
DSL or stable solver changes.

Question: how often does a uniform, bounded constant-to-input-feature repair
space contain an improvement, and can training consistency identify its answer?
This is a narrow automated step beyond the manual 012 studies, not an
implementation of path/rank/global-scope repair or abstraction discovery.

## Frozen language

Preserve each original Python program. Replace exactly one integer literal in
a comparison operand or a `range` argument with a read-only input scalar.
Eligible literals: 2 through 30 inclusive; include nested expressions, but
exclude literals under nested comprehensions/lambdas/functions relative to the
operand when enumerating. Deduplicate positions, sort by source position, take
first 16 sites per program. One replacement at a time, no other edits.

Twelve supplied features: height, width, minimum/maximum dimension, height-1,
width-1, number of distinct colours, number of non-background colours,
minimum/maximum non-background colour population, number of monochromatic
4-connected non-background components, maximum such component area.
Background = most frequent colour, smallest numeric ID on ties. Empty
non-background feature collections have value 0. Features are computed once
from each original input, not from intermediate mutable grids. Colour IDs and
integer magnitudes share Python's integer type; no inferred semantic typing.

At most 16 × 12 = 192 mutants per program plus the original. Identity-equivalent
bindings on all training inputs are retained and marked; their fit is inherited
rather than evidence distinguishing the feature from the constant.

## Procedure and scoring

Read training inputs/outputs and test inputs; fit each candidate on every
training pair. Collect predictions on all test inputs only after fitting.
Freeze candidate list, fit decisions, predictions and answer agreement before
scoring against intended test outputs. Preserve errors, CPU timeouts and
undefined predictions as failures, not agreements.

No ranking prior: predict only when **all** training-fitting candidates are
defined on every query and agree on the complete tuple of query outputs.
The original is included if it fits. No majority voting, hand priority or
shortest-program tie-break. Primary metric: oracle coverage of repairs for training-fitting but test-wrong
originals, together with the number of competing test answers. Also report
changed, correct, unanimously identified answers, fitting coverage, disagreement,
undefined candidates, original accuracy and oracle repair coverage (diagnostic,
not a chosen solution). A mutation that fixes a training failure is distinguished
from changing a training-perfect original.

Original CPU budget 5s for fitting, 5s for query prediction. Each mutant gets
0.25s for fitting and separately 0.25s for queries. Twelve workers across tasks;
serial benchmark execution. Report timing sensitivity where timeouts occur.
The original has a larger budget as reference; no equal-compute speed claim.

Logical limitation known before running: when the original fits training and
is defined on queries, retaining it makes a different unanimous answer
impossible. The experiment measures repair-space coverage and ambiguity, not
this elementary fact. Changed unanimity can only repair a training failure.
Training consistency alone cannot select a conflicting repair; an additional
selection principle would be required, and is deliberately not fabricated here.

Controls before corpus run: a literal-width program repaired by width binding;
an ambiguous constant/width pair; an undefined query candidate that must block
unanimity. Check source/data hashes. No stochastic seed.

## Interpretation and disposition

Replacing a literal by a feature expression is constrained hole filling. The
constant and feature variants have a common schematic program with one hole,
but no anti-unification algorithm is implemented and no theorem says the feature
variant is semantically more general. All feature extractors are supplied.
No MDL/Kolmogorov score is measured. A negative outcome closes this grammar;
retain protocol/evidence and a runnable Git revision. Even a positive outcome
requires broader validation before integration.

## Timeout sensitivity (registered after main run, before retries)

The main run had 13 fitting-phase CPU timeouts and no query-phase timeouts.
Rerun exactly those candidates with 5 CPU seconds per phase, unchanged feature
and site grammar, same 12 workers. Keep main results intact. This targeted check
is to qualify the negative coverage result; it does not broaden the grammar or
choose candidates based on whether their test answers were correct.
