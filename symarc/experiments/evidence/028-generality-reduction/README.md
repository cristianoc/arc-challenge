# Learning by reducing generality

## Finding

The reverse direction is useful when it means **ruling out unsupported output
possibilities**, rather than simply accepting fewer positive inputs or eliminating
more candidate programs. Experiment 028 compares those interpretations using the
unchanged spatial-template language from 027.

There are two concrete results. First, two labelled examples can each halve the
number of fitting classifiers while having dramatically different consequences:
one leaves 65,399 false-positive possibilities on the 4x4 bank, while the other
leaves 24. Second, a label-request policy measuring worst-case reduction of the
remaining disputed predictions identifies all eight constructed teacher rules
within five extra labels. A policy balancing hypothesis counts identifies four
by that point, although its cheapest selected classifiers already predict the
bank correctly for all eight. The improvement is **identification and an explicit
evidence certificate**, not a new ARC benchmark score.

These are four designed geometric rules and their label complements, not eight
independent discovery problems. Labels come from an explicit synthetic teacher
oracle. No new feature, geometric primitive, solver policy or Lean result is
integrated.

## Three notions that must not be conflated

For a classifier h, write A_h = {x : h(x)=1}. A classifier h' is a specialisation
of h when A_h' is a subset of A_h. Fewer positive predictions are not necessarily
better: this can lose genuine positives and depends on which class is called 1.

For a fixed hypothesis family H, labelled observations D define V_D, the members
of H agreeing with D. Adding consistent evidence shrinks V_D, but can select a
more general classifier. Hypothesis elimination is not classifier specialisation.

For nonempty V_D, define instead the two semantic envelopes:

\[
L_D=\bigcap_{h\in V_D}A_h,\qquad
U_D=\bigcup_{h\in V_D}A_h.
\]

L_D contains inputs necessarily accepted; U_D contains those still possibly
accepted. For additional consistent evidence e,

\[
L_D\subseteq L_{D+e}\subseteq U_{D+e}\subseteq U_D.
\]

The actual loss of positive generality is U_D minus U_(D+e). The dual change,
L_(D+e) minus L_D, removes the possibility of a negative answer. Together these
reduce the disputed region W_D = U_D minus L_D. Equivalently, track
R_D(x)={h(x):h in V_D} and remove output possibilities, not program spellings.
An empty V_D is an inconsistency, never maximal confidence.

If the target belongs to H and the observed labels are correct, it stays between
these envelopes. This is a conditional logical fact, not evidence that H contains
an unknown ARC rule. Outside the declared family, both envelopes can be wrong.

This framework is established version-space/candidate-elimination reasoning,
not a new definition of generalisation. Mitchell's IJCAI 1977 paper explicitly
orders rules by their positive extensions and separates deductive elimination
from choosing a best hypothesis. Our contribution here is the measured distinction
between hypothesis counts and the consequences eliminated in the existing fixture.

## The finite semantic domain

The underlying templates remain unchanged: Boolean formulas through size seven
in the four tests for a point lying at the minimum/maximum input row/column,
lifted to a set inside the supplied bounding rectangle. The classifier compares
that set with the input, with both output orientations and constant maps allowed.
No frame predicate is added. The same expanded syntax costs are retained only
for a reported shortest-fitting-program comparator.

A set imposes occupied and empty requirements on the sixteen point roles. If a
role contains both kinds of point, no role-uniform template can match. Otherwise,
collapse all interior coordinates of each axis to one representative. A matching
set then has exactly the same requirements as a normalized set of size at most
3x3. Consequently **400 normalized sets plus one nonuniform-role case** form an
exact 401-class semantic domain for this entire classifier template, at arbitrary
finite grid sizes. The nonuniform case is realized by a 4x4 box missing one
interior point.

This is stronger than deduplication on the observed labels. Two classifiers with
the same extension on these 401 classes are extensionally equal throughout the
declared geometric domain. The 366 bounded point predicates and four label maps
reduce to **260 distinct classifier functions**. Raw syntax or repeated aliases
cannot become extra votes.

The policy below gives each of these 401 input classes equal weight. This is an
explicit measure of semantic distinctions, not a natural probability distribution
on ARC grids. It is not invariant to a substantive change of observation language.

## A decisive controlled contrast

Start with the same two examples used in 027: the 3x3 square missing its bottom-
right corner, labelled negative; and the eight-cell hollow outline, positive.
For this diagnostic, remove the syntax-size bound and allow all 65,536 Boolean
point truth tables and both label orientations.

After the two labels there are 256 fitting point-table/orientation descriptions,
representing 36 distinct classifier functions. Compare adding either a positive
singleton or a negative solid 3x3 square. Both halve BOTH counts:

| Additional evidence | Point-table programs | Distinct classifiers | Disputed semantic classes | Still possibly positive 4x4 inputs | False-positive possibilities |
|---|---:|---:|---:|---:|---:|
| None | 256 | 36 | 393 | 65,499 | 65,399 |
| A positive singleton | 128 | 18 | 392 | 65,499 | 65,399 |
| A negative solid square | 128 | 18 | 7 | 124 | 24 |

The singleton certifies 16 additional concrete singleton placements as positive
on the 4x4 bank; it removes no positive possibility. The negative square excludes
65,375 previously possible positive classifications. Both updates have the same
hypothesis-count reduction, even after classifier-level alias elimination.

The explanation is structural. One orientation means "accept a matching
geometric template"; the other means "accept everything except a matching
template". Two distinct negative shapes with the same bounds rule out the latter:
one generated set cannot equal both. A positive singleton instead leaves the
broad inverse family alive while constraining a degenerate point role.

The 18 remaining classifiers after the solid-square label still disagree on
seven degenerate classes. Thus the third example does not identify the full
unbounded template. It does identify predictions on all 2,317 larger examples
in 027's bank. Large query counts do not make this an independent statistical
sample; they instantiate the same few semantic constraints.

## Choosing evidence by the consequences it removes

The primary experiment uses the original bounded language, retaining all fitting
semantic classifiers rather than only minimum-cost ones. All teachers start with
the same two input sets. Four acquisition policies may request up to eight more
labels from the remaining 3x3 catalogue:

* First informative input in fixed mask order.
* Hypothesis halving: minimise the largest surviving classifier count.
* Negative-only reduction: maximise the shrinkage of U if the answer is negative.
* Two-sided scope reduction: minimise the worst-case remaining size of U minus L.

Only informative inputs are considered. The last policy is

\[
q^*=\arg\min_q\max_{y:\,V_{D+(q,y)}\ne\varnothing}
             |U_{D+(q,y)}\setminus L_{D+(q,y)}|.
\]

It reads no prospective label. All ties use the original input order. The two
initial labels are additional to every budget below. Requested labels are the
only source evidence supplied to the chooser; test banks are scored after traces
are persisted and hashed.

For the bounded frame fixture, four classifiers survive the two seeds. Halving
asks about the singleton: the two possible outcomes leave 2/2 classifiers, but
up to 388 disputed semantic classes. Scope reduction asks about the broken row
`#.#`: the outcomes leave 1/3 classifiers, but at most five disputed classes.
The teacher answers negative and the frame function is uniquely identified
within the bounded language. This is a three-labelled-object result under that
language, not the unbounded six-example certificate from 027.

| Acquisition policy | Rules uniquely identified within 5 extra labels | Within 8 extra labels | Cheapest selected classifier already correct on the full 4x4 bank after 5 |
|---|---:|---:|---:|
| First informative | 4/8 | 4/8 | 6/8 |
| Hypothesis halving | 4/8 | 8/8 | 8/8 |
| Negative-only reduction | 5/8 | 6/8 | 7/8 |
| Two-sided scope reduction | **8/8** | **8/8** | **8/8** |

All rules identified exactly are also correct on the larger bank. The table
must not be read as eight new ARC solutions. It distinguishes correct answers
selected by a cost preference from correct answers forced by the remaining
finite family. By five labels, halving and scope have the same selected accuracy;
scope has removed the remaining alternative explanations.

The extra-label counts for scope versus halving are: perimeter 1 versus 2,
filled box 5 versus 7, missing-corner set 3 versus 2, and four corners 5 versus 6.
Their complemented labels give the same paths. Scope is not uniformly superior:
it loses on the missing-corner rule. Across this deliberately selected set it
uses 28 requested labels in total versus 34 for halving to identify all targets.
These sums are descriptive, not an estimate of a general sample-complexity gain.

Negative-only reduction can be fast when its assumed direction is appropriate,
but fails badly on the complement of the filled-box teacher: 118 candidate
functions still survive its eight requests. Calling fewer inputs positive is
not a label-independent objective. The symmetric envelope criterion avoids that
particular dependence; twelve label-complement control pairs are checked.

## Limits and implications

The cheapest fitting program does not change only by specialisation. Under the
scope policy, its 28 changes comprise 18 identical choices, three genuine
specialisations, three generalisations and four incomparable replacements.
The sound monotone object is the pair of envelopes, not one irreversible chain
of successively narrower chosen programs. No previously acquired label is ignored.

The thickness-two outline and nonoutline from 027 have the same nonuniform role
profile and different supplied labels. Their version space becomes empty. The
procedure reports an observation-language conflict rather than converting that
collapse into certainty.

The useful next interface is now precise: return an input on which competing
rules disagree, and the possible reductions caused by each answer. That can
support active experimentation or dataset design. It cannot fabricate the answer,
and the eight controlled teachers provide an oracle ordinary ARC does not have.
Without additional labels, any change in committed predictions is a prior or a
policy decision, not evidence created by the reduction metric.

No abstract principle here requires the labels to be Boolean: R_D(x) can retain
several output alternatives. But multi-output actions, partiality and the cost of
computing semantic consequences need a separate implementation and evaluation.
We have not turned this controlled label-query result into an ARC solver gain.

## Verification and disposition

The protocol was published at `ef49ae6c`; the active ledger was checked before
execution; source hashes were registered at `e72ab8ce`. Twelve synthetic controls
passed first, including 300 independent envelope comparisons. Scientific source
files were frozen locally and are published unchanged; no earlier remote source
publication is implied.

An independent auditor imports neither scientific module. It re-enumerates
90,690 raw syntax trees, verifies all 260 classifier extensions and costs,
checks 133 acquisition decisions, 1,855 elimination witnesses, 165 trace states,
12 complement pairs and the three unbounded-template contrasts. It independently
interprets all 67,852 geometric test inputs and rechecks 330 bank/state scores.
No inferred envelope prediction is wrong in this realizable/noiseless study;
that follows conditionally from target containment, not a learned calibration.

This is a deterministic one-process semantic study, not a runtime benchmark.
Both prediction and score hashes match a repeated run. The package contains
frozen source dependencies, traces, all inputs, scores, independent audits and
an executable reproduction command. The original 027 catalogue overlap is 511
primary inputs and zero secondary inputs; actual selected fitting sets are the
two seeds and the explicitly listed requests, not all 511 catalogue labels.

Retain the consequence-reduction analyser and its controls as a bounded research
candidate. Stable Rust code, existing ARC wrappers and accepted Lean mathematics
remain unchanged.

Reference: Tom M. Mitchell, *Version Spaces: A Candidate Elimination Approach
to Rule Learning*, IJCAI 1977, pp. 305–310.
https://www.ijcai.org/Proceedings/77-1/Papers/048.pdf
