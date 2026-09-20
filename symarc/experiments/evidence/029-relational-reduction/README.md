# Reduction can establish a relation without identifying either answer

## Result

Experiment 029 extends 028's consequence reduction from separate possible labels
to pairs of labels produced by the SAME candidate program. It finds concrete
relations which are lost by a per-input summary. But the richer pair objective
chooses exactly the same questions as the previous marginal-scope objective.
There is no new evidence-acquisition improvement from that extension here.

A broader comparison also qualifies 028's positive result. Previously, scope
reduction identified eight chosen geometric rules within five extra labels,
while hypothesis halving identified four. Here every one of the 260 functions
in the same bounded semantic language is treated as a target. After eight extra
labels, halving identifies 188 targets and scope/pair reduction identifies 172.
After five and six labels, scope is ahead. Neither is uniformly better.

This is an exhaustive CLOSED-WORLD exercise with a supplied noiseless oracle,
not 260 independent natural tasks, a probability model for ARC, or a new solver
score. No primitive, cost, hypothesis language, stable solver or accepted Lean
module changed. The most useful outcome is the exact relation between shrinking
candidate families and constructing a coarser, universally respected input
quotient, together with actual unlabelled-input witnesses.

## Frozen design

Baseline `2f534a92f728bde2446b368cc03f68a8ad14cd34`; protocol `f8cd4a95`.
An active ledger entry and the actual source/test hashes were committed by the
registration workflow before scientific execution. Twelve synthetic tests,
including 300 independent finite comparisons, passed first. Source files were
published with their frozen hashes after the local run; earlier source
publication is not claimed.

Reuse 028's 401 exact semantic input classes, 260 extensionally distinct
classifiers, 511 nonempty 3x3 source shapes, and seed inputs 255 and 495. Shape
labels come from the chosen target function, only at requested inputs. The
selector receives the current version space, not the target's identity. Each
of three policies may request at most eight additional labels after the two
seeds. All target functions are included, with no post-score exclusions.

The geometry, axes, extrema, bounding carrier and finite template language are
inherited assumptions. Syntax aliases are not additional hypotheses. The 401
classes are exact for this language, not a sample of independent ARC grids.
The source catalogue covers 400 classes and separates all 260 classifier
functions. No run stops early with unresolved observationally indistinguishable
targets; the unresolved cases are censored by the fixed eight-request budget.

## What the three policies measure

For a nonempty family V of total classifiers, write

    R_V(x) = {h(x): h in V}
    R_V(x,x') = {(h(x),h(x')): h in V}.

The pair must use one common h. Multiplying the two marginal sets can invent
joint outcomes produced by no candidate. Define

    W1(V) = sum_x (|R_V(x)| - 1)
    W2(V) = sum_{x<x'} (|R_V(x,x')| - 1).

For Boolean labels W1 is exactly 028's disputed-region size. The policies
minimize, over informative source questions, the worst-case remaining value of
|V| (halving), W1 (marginal) or W2 (pair). They use increasing source-mask order
for ties. There are no target names, future labels, syntax-cost preferences or
majority votes in their selection. All output pairs get equal weight; this is
an explicit choice of measure, not a representation-independent scalar.

If V shrinks, both R sets shrink. Empty V is an inconsistency, not complete
knowledge. The program-level premise matters: these consequences are justified
only relative to the surviving family and the assumption that it contains the
target. They do not validate an incorrectly supplied grammar.

## The broader evidence-selection result

| Extra requested labels, after the two seeds | Hypothesis halving | Marginal scope reduction | Pair scope reduction |
|---|---:|---:|---:|
| 2 | 8/260 identified | 4/260 | 4/260 |
| 5 | 20/260 | 26/260 | 26/260 |
| 6 | 64/260 | 72/260 | 72/260 |
| 7 | 136/260 | 122/260 | 122/260 |
| 8 | 188/260 | 172/260 | 172/260 |

At budget eight, scope identifies 16 targets that halving does not, while
halving identifies 32 that scope does not. The capped sums of label requests
are 1,834 for halving and 1,838 for each scope policy. These are censored sums,
not the numbers required to identify all targets.

The marginal and pair policies have identical complete question/answer paths
for every target. Their numerical scores differ; their minimizing choices do
not. This does NOT prove policy equivalence on arbitrary hypothesis classes.
More precise representation of consequences need not improve a greedy query
objective, and W2 changes the weighting as well as keeping correlations.

The original eight teacher comparisons reproduce exactly: perimeter requires
one extra label under scope versus two under halving; filled box five versus
seven; missing-corner three versus two; corners five versus six. Label complements
have the same paths. The previous finding remains true for those chosen rules,
but was not evidence of a general advantage across the entire supplied language.
The expanded target distribution is still artificial: uniform weighting of all
semantic classifiers is not established as a natural task distribution either.

## An actual relation learned before either answer is known

The deterministic first retained witness concerns two unlabelled sets:

    A: ##       B: ###
       #.          ##.

The existing source labels in this trace are masks 255->0, 495->0, 7->1,
73->0 and 9->0. The next requested label is a negative singleton (mask 1->0).
All sets use the original row-major 3x3 source encoding; actual coordinates
and the whole acquisition history are retained in the case certificate.

Before that new label, the surviving classifiers permit these outcomes for
(A,B): (0,0), (1,0), (1,1). Afterwards they permit only (0,0) and (1,1).
Each separate label still has possibilities {0,1}. Nevertheless,

    every surviving classifier gives A and B the SAME answer.

No label was requested for A or B. The new observation eliminates a relational
possibility, not either individual answer. The classifier family shrinks from
25 to six; its output-equality quotient on the 401 semantic inputs shrinks from
33 classes to seven. This observation creates 28 new forbidden output pairs
whose coordinates both remain individually ambiguous.

Across the recorded decision trees, 254 distinct acquisition edges contain
such relational discoveries. The 2,952 witness occurrences in per-target traces
repeat shared edges and are not independent empirical observations. A witness
is selected by fixed source-mask/version order, not by a preferred teacher or
subsequent accuracy. The auditor verifies every recorded first witness and its
lexicographic ordering independently.

## A canonical abstraction obtained by reduction

There is a useful mathematical answer to the original quotient question. Define

    x ~_V x'  iff  for every h in V, h(x)=h(x').

This is an equivalence relation. Every surviving h factors through its quotient.
Moreover, it is the GREATEST equivalence respected by every h in V: any other
relation whose classes all these h treat uniformly is contained in ~_V.

For nonempty V' subset V,

    ~_V is contained in ~_V'.

Thus eliminating hypotheses can make MORE input identifications justified.
Reduction on the program side constructs a coarser abstraction on the input
side. On a finite domain the implementation groups inputs by their surviving
output-column vectors; removing rows can make two columns identical. It needs
no guessed labels for those inputs. This is a computable finite quotient, not
yet a compact geometric formula implementing the equivalence at arbitrary inputs.

This does not contradict the accepted no-greatest-coherent-quotient result. That
result asks whether SOME operation selector can fit a proposed quotient, and
alternative quotients can require incompatible choices. Here EVERY surviving
classifier must respect each identification. The different quantifier provides
a canonical conservative object. It may be much finer than the target's own
kernel while alternatives survive, and its interpretation still depends on H.

A missing-pair relation can express equality, complementarity or implication.
All are consequences of the same surviving programs. Only equality is used for
the quotient above; arbitrary pairwise constraints are not themselves an
equivalence relation. The audit verifies 664 distinct strict quotient-coarsening
edges. Some of those simply merge now-fixed outputs; the separate ambiguous-pair
witnesses are needed to demonstrate nontrivial relational progress.

## Limits of pairwise consequence summaries

Pair summaries are still an abstraction. The three-bit even-parity family
{000,011,101,110} has exactly the same singleton and pair projections as all eight
three-bit vectors. Its genuine ternary constraint is invisible to W1 and W2.
The synthetic suite checks this and a three-valued-label correlation example.
The primary acquisition engine remains binary; this is not a multiclass ARC run.

More generally, retain R_V(Q)={h restricted to Q: h in V} for an input tuple Q.
Higher arity can expose dependencies hidden at lower arity. At the complete
semantic domain, R_V(X) is just the extensional version space itself. This is a
hierarchy of consequence abstractions, not a new universal definition of learning.

The Cartesian approximation gap is also not a monotone progress measure. If
w=W1(V) and n=|X|, the number of spurious allowed pair outcomes is

    G2(V) = (n-1) w + w(w-1)/2 - W2(V).

Both the true relation and its marginal product change as labels arrive. G2 can
increase or decrease. The recorded relational discoveries instead count newly
forbidden pairs while both coordinates stay individually ambiguous.

## Implication for the research

Retain the relational analyzer and canonical finite quotient as diagnostics.
Do not promote pair-weighted acquisition: it changes no choices here. Do not
repeat the earlier eight-teacher comparison as a general superiority result.

The next substantive construction interface is to turn justified input
identifications into a compact reusable relation, while keeping their dependence
on the candidate family explicit. A grouping of prediction columns already
provides an exact finite abstraction; a useful geometric definition needs a
separate synthesis/compression result. Asking for another label requires an
oracle or actual external observation. No reduction score creates that evidence.

Candidate elimination, generalized binary search and relational abstract
interpretation are established. The independently checked contribution of this
step is the concrete consequence/quotient mechanism and the enlarged comparison,
not novelty of those frameworks. Primary context: Mussmann and Liang,
Generalized Binary Search for Split-Neighborly Problems (AISTATS 2018),
https://arxiv.org/abs/1802.09751; Patrick Cousot, Abstract Interpretation in a
Nutshell, https://www.di.ens.fr/~cousot/AI/IntroAbsInt.html.

## Verification and reproduction

The frozen predictor uses row-wise bitsets of pair outputs. The independent
auditor imports neither learner and instead intersects hypothesis columns,
weighted by their multiplicities. It checks 104,260 geometry/classifier values,
2,838 independently measured candidate subfamilies, all 996 recorded policy
states, 6,290 trace states, 5,510 acquired labels and all relational witnesses.
All 16 historical scope/halving traces match 028; 390 complementary-target
policy pairs are verified. These are finite checks, not statistical sample sizes.
The underlying grammar and its completeness proof are inherited, not re-created
as a new scientific discovery.

Twelve synthetic tests pass. A repeat prediction/summary run and the complete
reproduction command reproduce all four expected semantic hashes, including
028's source-family trace and the independent audit. No runtime speedup is
claimed. Full traces, source inputs, candidate functions, consequences, audit,
source hashes and executable commands are retained in the companion archive.

    python3 symarc/experiments/029-relational-reduction/reproduce.py --out /tmp/symarc029

The stable Rust solver, previous ARC configurations and accepted Lean mathematics
remain unchanged. Retain this as a consequence/quotient diagnostic, not a new
acquisition winner or an integrated solver improvement.
