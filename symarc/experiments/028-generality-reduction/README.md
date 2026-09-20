# 028 — What does reducing generality tell us?

Baseline: `ed4b76bd202202976dfeee82dccd1cd6d32d7bc7`. This tests the user's
suggestion to study the opposite of generalisation. No stable solver, accepted
Lean theorem, grammar constructor or ARC score is changed.

## Distinguish three different reductions

A classifier specialises another when its positive extension is a subset.
Shrinking a version space is not the same: it can select a broader classifier.
For the nonempty set V of all fitting semantic classifiers define

    L(V) = intersection of their positive extensions
    U(V) = union of their positive extensions.

Additional consistent labels grow L and shrink U. Record separately positive
possibilities removed (U_old minus U_new), negative possibilities removed
(L_new minus L_old), and ambiguity |U minus L|. Empty V is an inconsistency,
NOT complete certainty. These elementary guarantees require the true classifier
to remain in the declared family. No scalar is a prior-free correctness score.

## Fixed finite semantics and comparison

Reuse 027's Boolean point grammar through size 7, its four extrema comparisons,
set-template equality, and both output orientations/constant maps. Retain the
full fitting pool rather than only shortest programs. Expanded costs remain
unchanged and serve only a reported shortest-program comparator.

Represent a set by required-occupied and required-empty point-role masks. If a
role contains both an occupied and an empty point, no template can match.
Otherwise collapse each coordinate's interior to a single representative. The
exact scope domain consists of normalized nonempty subsets fitting in 3x3 plus
one nonuniform-role case (4x4 box minus an interior point). A proof of this finite
quotient and exhaustive checks on all 4x4 sets accompany the experiment. Collapse
classifiers with identical behaviour on this whole domain, not merely their
training traces. Counts weight each semantic input class equally; they are NOT
probabilities under a natural ARC distribution.

Use the same fixed two seed inputs (3x3 masks 255 and 495) for every teacher.
Simulated teachers: perimeter, filled box, box minus lower-right corner, and
four corners; also the label complement of each. The latter controls stop a
preference for few positive answers masquerading as progress. These are eight
constructed rules in a supplied geometry, not independent ARC tasks.

Labels for additional evidence may be requested only from the remaining 3x3
source catalogue. There is an explicit teacher oracle in this controlled study;
ordinary ARC supplies no such interface. Compare four acquisition policies,
with at most EIGHT additional labels after the same two seeds:

- first: first informative source input in increasing original mask order;
- halving: minimise the largest remaining semantic-hypothesis count;
- negative_reduction: maximise removal from U conditional on a negative answer;
- scope: minimise worst-case remaining |U minus L| over both possible answers.

Only informative queries are considered. Ties use original source mask order,
not a target name, label, cost adjustment or future test score. Record every
prefix, including 0/1/2/4/8 acquired labels; stop querying only if no informative
source input remains. Simulations use identical budgets and expose only requested
labels to query selection. Scope and halving should be label-complement invariant;
the negative-only policy deliberately need not be. No new policy after scoring.

Primary outcomes: exact identified semantic classifiers, remaining ambiguity,
number of labels, and guaranteed reductions. Also score must/may envelopes and
shortest fitting comparator on all 4x4 nonempty sets and 027's larger bank.
Report positive/negative errors separately, source overlaps and per-teacher
outcomes. Large grid counts are dependent finite checks, not sample sizes for
statistical confidence. Freeze acquisition traces before scoring those banks.

## Fixed mechanism and failure controls

For the original frame seed pair, compare adding (i) a positive singleton and
(ii) a negative solid 3x3 box. In a supplementary unbounded-template diagnostic,
consider all 65,536 point truth tables and both label orientations. Report BOTH
point-table counts and distinct classifier counts: they need not agree. Compare
hypothesis elimination with semantic scope reduction, without pretending these
two label choices were discovered blindly.

Report whether changes in the shortest fitting classifier are true set-inclusion
specialisations, generalisations or incomparable replacements. For an already
refuted classifier, certify rejected consequences rather than assigning a score
to syntax. A thickness-two outline/nonoutline pair from 027 is a misspecified-
language control: contradictory labels on identical observable profiles must
produce an explicit empty-space result, not a spurious confident answer.

## Execution and verification

Deterministic standard-library Python, one process, no runtime benchmark. Reuse
027's checked expression enumerator but independently reconstruct the geometry
and the finite-domain signatures in the auditor. Recompute all acquisition
choices by explicit sets, all recorded eliminations, envelope monotonicity,
label-swap symmetry, shortest-program scores and the full-domain mechanism.
Test duplicate-hypothesis and duplicate-observation invariance, false-negative
tradeoffs, empty families, and finite-state versus concrete grid evaluation.
Retain source hashes, code, traces, independent audits and a runnable package.
Protocol and ledger registration precede scientific execution; implementation
hashes are frozen before predictions. No score-based retuning or ARC result.

## Prior art and scientific boundary

Generality as extension inclusion, general/specific boundaries and candidate
elimination are established: Mitchell, *Version Spaces: A Candidate Elimination
Approach to Rule Learning*, IJCAI 1977 (https://www.ijcai.org/Proceedings/77-1/Papers/048.pdf).
This experiment does not rename version spaces as a new generalisation theory.
It asks whether measuring consequences ruled out, rather than hypotheses removed
or a selected program's size, exposes a useful decision in our existing fixture.
All conclusions stay relative to this finite language and its declared evidence
oracle. The ratio of progress to label cost is measured, not presumed.
