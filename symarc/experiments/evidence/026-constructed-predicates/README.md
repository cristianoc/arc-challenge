# Core mathematics integrated; a first constructed shape abstraction

## What changed

The accepted Lean core now includes operation-relative abstraction and concrete
program-family conditioning. Separately, experiment 026 constructs a symbolic
predicate from a small arithmetic language instead of selecting a supplied
rectangle feature. The stable Rust solver and its experimental combinations
are unchanged. This study does not report another ARC1 or ARC2 score.

## Accepted mathematical integration

`SymArc/Learning.lean` is imported by `SymArc.lean`, so the normal `lake build`
checks it. `LEARNING.md` is the accepted account and is linked from `MATH.md`
and the project README. The existing closure/equivariance development remains.

Fifteen theorems cover the following stable material:

1. A representation can share an operation on each class exactly when every
   class has a nonempty intersection of compatible operations. This is equivalent
   to existence of a fitting selector, not a generalisation theorem.
2. Explicit refinement preserves representability. A labelled conflict cannot
   be placed in one class by a fitting representation. Two coherent quotients
   need not have a coherent common coarsening; a three-point example proves
   that there need not be a greatest coherent quotient.
3. Predictions require a nonempty family as well as agreement. Filtering preserves
   an already determined answer only with surviving programs. Correctness of a
   conditioned prediction requires membership and totality of the target program.
4. Independent per-key query conditioning has an exact intersection characterization
   and an exact projection theorem. Coupled alternatives must be conditioned as
   whole branches before projecting. A checked two-key counterexample shows how
   projecting first can manufacture a program absent from the original family.
5. If a primary operation fits all data, every constant fallback fits too. At an
   input where the primary is undefined these alternatives can return any output.
   A domain guard supplies no label for its unobserved branch.

These facts have no entropy, default-cost or model-ranking assumption. The general
existence results use standard classical choice where needed; the finite prototype
uses enumeration, not executable code extracted from those proofs. Total selectors
and additional finite-table domain restrictions are explicitly distinguished.

GitHub Actions run **35452954813** successfully built both Lean modules with the
pinned **Lean 4.34.0**, printed all 15 axiom dependencies (only ordinary foundational
axioms; no `sorryAx` or custom axiom), and passed the 11 stable Rust tests and CLI
regressions. Its PR-merge source revision is `316e59c05e4625d358715528e85fa4385c4884bb`,
with head `c06e6ea23d06c8750163e292341cbe15687eb537`. This checks the mathematics,
not the Python synthesis implementation. A persistent CI workflow retains the checks.

## Experiment 026: what is supplied, and what is constructed

The input is an already-given nonempty finite set S of integer coordinate pairs.
The supplied observations are cardinality, row span, column span, zero and one.
The supplied expression constructors are addition, subtraction, multiplication,
equality, less-than, negation and conjunction. Axes, ordering, extrema and counting
are explicit priors. Object segmentation is not learned.

There is no rectangle or frame predicate in the grammar. It exhaustively constructs
all typed Boolean expressions through AST size five: **945 canonical predicates**.
Only exact commutativity and expanded syntax aliases are normalized. Expressions
that merely agree on examples are not merged or assigned a common meaning.

For each candidate predicate, the construction procedure asks whether observations
with the same Boolean key admit a common output operation. Here those operations
are constants from the observed label palette. Incompatible classes reject the
predicate with a two-observation witness. Compatible classes construct the branch
assignments directly. This is the finite instance of the integrated coherence theorem.

A direct reference enumerates the same predicates and all branch assignments in
`if predicate then constant else constant`. It has the same label palette, fixed
ordering and cost. The complete program costs predicate size plus three nodes.
Both retain **all minimum-cost fitting programs**.

## The actual result

Construction data contain every nonempty subset of a 3x3 coordinate universe:
511 labelled sets. An independent row/interval teacher labels solid rectangles.
It checks consecutive nonempty rows having the same consecutive column set; it
does not use the arithmetic expression being synthesized.

The chosen definition is:

```math
\operatorname{Rect}(S)\ :=\ |S|
  =\operatorname{span}(\pi_r S)\,\operatorname{span}(\pi_c S).
```

The body has **five AST nodes**. Its complete classification program has **eight**.
The other minimum program tests strict inequality and exchanges its output labels.
The two programs define the same classification on legal inputs, because a finite
set cannot have more elements than its bounding rectangle. They were both retained.

This is actual expression construction, not selection of a previously named shape
feature. It is also a deliberately small, target-compatible grammar, not primitive-
free discovery or evidence that arbitrary abstractions can already be constructed.
The source supplies 511 labels, not a sparse two- or three-example ARC specification.

There is an ordinary mathematical explanation of why the learned definition works:
S is contained in its finite bounding rectangle B, and |B| is the product of the
coordinate spans. Equality of these finite cardinalities implies S=B. Conversely,
a solid rectangle has that product cardinality. This argument is separate from
the integrated Lean proofs; this particular finite-geometry characterization is
not formalized in the current core.

## Transfer checks and the overlap correction

The primary query bank is all 65,535 nonempty subsets of a 4x4 universe. Both
minimum programs classify every case correctly: **100 rectangles and 65,435
nonrectangles**. Reporting only overall accuracy would hide this imbalance.

The protocol called this bank withheld. It includes **511 exact source sets**,
because a 3x3 subset also belongs to the 4x4 universe. The audit reports that
explicitly rather than presenting all 65,535 as unseen. There are **65,024 genuinely
new sets**, including 64 positive cases; all are correct. In 63,728 query cases
the whole count/span tuple is also absent from source data. This is a case count,
not that many distinct scalar tuples.

A secondary, fixed-seed bank contains **3,085** larger, translated and axis-swapped
sets: 21 rectangles and 3,064 nonrectangles. All are correct and none is an exact
source set. Both programs and the selected definition were frozen before these
query inputs were used; answers were read only after predictions were saved.

These are exhaustive/constructed geometry checks, not independent ARC tasks or
a new benchmark score. The source task was intentionally chosen to be expressible
in the declared language. No classifier or primitive was retuned after scoring.

## What the direct comparison says—and does not say

| Source-construction work | Direct reference | Class-compatibility construction |
|---|---:|---:|
| Predicates considered | 945 | 945 |
| Predicate/observation evaluations | 482,895 | 482,895 |
| Label-consistency checks | 12,230 | 5,919 |
| Branch assignments enumerated or constructed | 3,780 | 2 |
| Minimum fitting programs | 2 | 2 |

The methods have exactly the same minimal program set, full cost and predictions.
The constructive procedure avoids enumerating many incompatible output assignments.
**It does not yet reduce predicate enumeration.** These are specified work units,
not a wall-time speedup claim. They do not establish a novel synthesis algorithm.

After freezing the predicate, four tasks attach different literal output pairs to
its two classes. Each supplies one positive and one negative example. Reusing the
predicate correctly fits those heads and preserves all query classifications.
The cold reference has the SAME source evidence: it re-enumerates the shared
predicate constrained by source data and then fits the new output assignments.

Counting initial construction plus four adaptations gives **5,935 label checks**
for construction/reuse versus **61,198** for cold direct resynthesis. Predicate
observations are 482,911 versus 2,414,491. The declaration body retains cost five;
its name is not treated as a one-node shortcut. Reuse amortizes construction; a
direct solver caching equivalent work could benefit too. This is relabelling the
same partition, not discovering that it is relevant to four unrelated ARC tasks.

Duplicating primitive spellings and introducing nested aliases for the constructed
body leaves all 945 canonical candidates, expanded costs, work counts and learned
programs unchanged. This is invariance to presentation aliases by design, not to
arbitrary changes in semantic primitives or cost models.

## A real failure certificate, before attempting more search

For hollow rectangular frames, the same observations lose essential information.
The procedure finds this pair among the labelled 3x3 sets:

```text
Not a frame       A frame
###               ###
###               #.#
##.               ###
```

Both have eight cells, row span three and column span three. Their frame labels
differ. No expression of these scalar observations can distinguish them, even
without the size-five limit. More arithmetic search is therefore not a repair.
The certificate identifies missing spatial information rather than encouraging
another cost or confidence adjustment. No frame primitive was added after this
failure. The two source indices are 254 and 494 (masks 255 and 495).

This is the next useful boundary: construction works for a property determined
by the observations, while a closely related shape property provably requires a
richer observation language. A future extension should construct spatial tests
from generic point/coordinate relations, rather than install a named frame feature.
That extension is not claimed to have been implemented here.

## Verification, provenance and disposition

Protocol `8ca83c78` and source hashes `4b70b9e1` preceded execution. The active
ledger entry was confirmed before preparing the data. The sources were published
unchanged afterwards; the frozen hashes, not an assertion of earlier publication,
identify the implementation actually run. One serial construction/work-count study
was registered, not a runtime or worker-scaling comparison.

Twelve synthetic tests include 120 direct/constructive finite comparisons, type
checks, aliases and full cost, translation, and observation-collision controls.
The independent auditor imports neither scientific module. It reconstructs the
945-predicate grammar, its minimum programs and direct work counts, validates
943 rejection witnesses, the frame collision and all recorded output predictions.
It replays 480,340 outputs across the source, minimum alternatives and transfer
heads; those replays are checks, not independent scientific samples.

An additional input-isolation check changes all query inputs and transfer labels
and reproduces the source library byte-for-byte. The repeated full experiment
reproduces the library, construction, predictions, scores and audit exactly.

SHA-256:
- Library: `7ea82c7a4157ab85e0440f0319ef7f48339b8e4ab1704bf84d76fa0deb1c3195`
- Predictions: `e3b3a78942378b6de83a10b89ad9d296e1706e2327c42d1574619d954bd5c372`
- Scores: `c781d53d64e4f5c17a1b8bd34d922638193ad1627c02c11f6227ac60b1b4b4e3`
- Independent audit: `41abb70c03e3cc25b2b01119ecb5f7695ecf90a9051c483c929d2edb9526f140`

Retain the small construction harness and its positive/negative cases as a research
candidate. Mathematical results are integrated; this prototype is not a new stable
solver policy. No earlier score or wrapper is changed. The scientific next target
is automatic construction of richer observations and nontrivial downstream reuse,
not further tuning of this deliberately simple rectangle classifier.

## Primary precedents

Predicate/partial-expression separation is established in Alur, Radhakrishna and
Udupa, *Scaling Enumerative Program Synthesis via Divide and Conquer* (TACAS 2017).
Library learning and reuse are established in Ellis et al., *DreamCoder* (2021).
This study claims an executed bounded construction and exact controls, not novelty
for their general architectures.

- https://www.microsoft.com/en-us/research/publication/scaling-enumerative-program-synthesis-via-divide-and-conquer/
- https://arxiv.org/abs/2006.08381
