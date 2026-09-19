# 026 — Construct a predicate, rather than select a supplied shape feature

Baseline: `92a73f4944d48afe3ff3b518e0a5d03d93cdc3ed`; accepted mathematical
integration is separate from this experiment. This is a controlled construction
study, not an ARC score or a claim of primitive-free concept discovery.

## Question and common language

Can operation-compatibility constraints construct a useful symbolic abstraction
from a lower-level expression language, and can that abstraction be reused
without assigning it an artificial one-token description cost?

Inputs are nonempty finite SETS of integer coordinate pairs. No segmentation is
learned in this study. Supplied scalar observations are count, inclusive span of
row coordinates, inclusive span of column coordinates, zero and one. Spans use
minimum and maximum; axes, counting and ordered coordinates are explicit priors.
There is no rectangle, square, enclosure, border-shape or learnt-predicate atom.
Integer expressions use +, - and multiplication. Boolean expressions use equality,
less-than, negation and conjunction. Enumerate the finite typed grammar by full
expanded AST size through predicate size 5 (integer subexpressions accordingly).
Only exact commutative canonicalization and duplicate-syntax removal are allowed;
no identification based solely on agreeing on training inputs is performed.

A complete program is `if predicate then constant y1 else constant y0`.
Its cost is the predicate AST size plus three nodes (if and two constants).
Both methods use the SAME finite expression set, expanded costs, output palette
observed in the task, and fixed ordering. No named macro receives a cost discount.

- Direct reference explicitly checks every pair of branch constants for each
  candidate predicate, stopping a candidate at its first inconsistent observation.
- Constructive learner synthesizes predicates and solves each class's nonempty
  operation intersection, using the accepted `coherent_iff_fitting` criterion.
  It records the incompatible labelled observations that reject earlier predicates.
  It retains all minimum-cost fitting programs, not arbitrary semantic aliases.

Compare the exact sets of minimal programs, not merely final accuracy. Count
predicate evaluations and label-consistency checks separately; do not claim that
avoiding branch-constant enumeration is a new synthesis algorithm. No selector,
query-totality policy, primitive extractor or existing Rust solver is changed.

## Data fixed before construction

Source task: every nonempty subset of a 3x3 coordinate universe, ordered by mask
(511 examples). Output 1 when the set is a solid axis-aligned rectangle, otherwise
0. The teacher checks consecutive nonempty rows with an identical consecutive
column set; it must not use the learner's count/span formula.

Primary withheld input bank: every nonempty subset of a 4x4 universe (65,535).
Secondary bank: positive rectangles and deterministic seeded random/deleted-cell
shapes in larger bounding universes (5x7, 7x5, 8x8, 11x13), translated including
negative coordinates and with axes swapped. Fix seed 2601, 64 random masks per
universe, plus full rectangles and each single-cell deletion. Deduplicate sets.
Prepare inputs and answers separately; freeze all learned programs and query
predictions before scoring. Query inputs do not enter expression construction,
canonicalization, ranking or source-program choice.

These exhaustive/constructed examples are NOT sparse ARC demonstrations and NOT
independent ARC tasks. Their purpose is a checkable first construction, transfer
mechanism and counterexample, not a headline benchmark gain.

Negative expressivity control: using the same 3x3 input sets, label exact hollow
rectangular frames by an independent boundary-coordinate interpreter. Before
search, check for differently labelled sets with identical supplied scalar values.
Retain such a pair as a vocabulary-insufficiency certificate; do not add another
primitive to repair the result in this run.

## Reuse and robustness

Extract the source predicate as a callable symbolic definition with its body
retained. Reuse it in four new branch-label tasks with output pairs (2,7), (8,4),
(6,3), (9,1). Each supplies one positive and one negative example; the source
construction data remain the provenance of the reusable definition. Freeze that
definition before these tasks. Compare fitting only its branch constants against
cold direct re-enumeration constrained by BOTH source and target evidence.
This is a multi-task reuse check, not an unfair comparison with a baseline denied
source observations. Report construction plus transfer work and the break-even
point; no per-call AST-cost reduction or runtime guarantee is asserted.

Run front-end alias/renaming controls: aliases are expanded before canonicalization,
including duplicated primitive spellings and a named expression body. The set of
canonical candidates, full expanded costs and predictions must not change. This
is invariance to presentation aliases, not arbitrary changes of semantic primitives.

Independent audit reconstructs the direct candidate set, verifies every surviving
formula on the full withheld bank with a separate scalar interpreter, checks
rejection witnesses and full costs, and checks all source/target label isolation.
The source task deliberately tests a relation expressible in this small grammar;
that researcher choice is a bias and not evidence that arbitrary abstractions
emerge automatically. Report no useful abstraction if none fits; no tuning after
scores. No new ARC1/ARC2 score or stable-solver policy integration.

## Scope and provenance

The stable theory concerns operation-relative coherence, not the optimality or
novelty of this learner. Relational expression synthesis and library reuse have
established precedents (Alur et al., TACAS 2017; Ellis et al., DreamCoder, 2021).
A successful result must include the actually constructed body, matched reference,
independent geometry test, full cost, reuse accounting and a genuine failure
certificate—not just the assertion that a supplied rectangle feature is useful.
