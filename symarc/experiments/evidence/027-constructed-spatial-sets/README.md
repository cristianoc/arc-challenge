# From a shape classifier to a reusable spatial set

## Result and boundaries

Experiment 027 constructs a point predicate and lifts it to a set of positions.
The learned expression recognizes rectangular outlines. After freezing it, two
small target learners reuse it to complete damaged/noisy outlines and generate
their interiors. This is more than changing the output labels of one classifier.

Dense-source recognition is correct on all 65,535 nonempty 4x4 sets and 2,317
larger/translated sets. Each set-valued target is correct on 2,317 queries, of
which 2,316 are not target demonstrations. The fixed two-example source control,
however, selects a much cheaper wrong explanation: 136/65,535 correct. A separate
post-score certificate finds six examples that uniquely identify the desired
semantic program within the entire declared template class, without a size prior.

These are constructed teaching problems, not ARC tasks or new benchmark gains.
Axes, coordinate extrema, a Cartesian bounding carrier, segmentation, Boolean
connectives and set operations remain supplied. The observation language itself
was chosen to investigate the known frame/count-span collision from 026. There
is no frame atom, but there is a strong frame-friendly geometric prior.

## What is constructed

For a nonempty finite set S of integer points, the supplied carrier B(S) is the
Cartesian product of the inclusive ranges between its coordinate extrema. The
learner considers point predicates built from four comparisons:

```
row(p) = min(rows(S))       row(p) = max(rows(S))
col(p) = min(cols(S))       col(p) = max(cols(S))
```

It combines them with true, false, not, and, and or through seven Boolean syntax
nodes. The compact comparison spellings expand to equality, coordinate, variable,
projection and extremum syntax before costing. The selected predicate costs 27
expanded nodes, its generated set 44, and its complete binary classifier 49.
The bounding carrier and repeated syntax are charged; naming the library body
never makes it a free primitive.

For each candidate phi, the observation is

```
S = {p in B(S) : phi(p,S)}.
```

Both output orientations and constant label maps are considered. Source labels
therefore do not secretly prescribe which truth value of this observation means
"frame". Minimum semantic programs all survive; aliases are not extra votes.

The result constructed from all 511 nonempty 3x3 sets is:

```math
\phi(p,S) =
(r(p)=r_{\min})\lor(r(p)=r_{\max})\lor
(c(p)=c_{\min})\lor(c(p)=c_{\max}).
```

Its generated set F(S) is the rectangular perimeter of the supplied carrier.
The classifier returns 1 precisely when S=F(S). This separates 026's exact
count/span collision without adding an `isFrame` observation.

## Exact semantic enumeration, not training-trace merging

Each point predicate has a complete truth table on the sixteen valuations of
the four comparisons. All sixteen are realizable in carriers of size at most
3x3, including one-row, one-column and singleton cases. Predicate equivalence
is therefore checked on the full semantic input space, not merely on labelled
source examples. Boolean operators respect this equivalence.

The constructive enumerator keeps a cheapest expanded representative per truth
table and Boolean-size layer. Its independent reference enumerates all 90,690
ordered syntax trees within the bound, including redundancies. Both recover the
same 366 semantic candidates and their exact minimum costs/representatives.
Reversing constructor iteration leaves the candidates and counts unchanged;
nested aliases expand to the same bodies and costs.

Direct if/constant/constant enumeration and class-coherence construction consider
these same 366 predicates and recover exactly the same minimum program:

| Work | Direct | Constructive |
|---|---:|---:|
| Candidate/profile checks | 187,026 | 187,026 |
| Label checks | 7,119 | 4,757 |
| Branch assignments tested/constructed | 1,464 | 1 |

This is an exact finite search and avoids redundant branch assignment checks.
Semantic quotienting is also available to the reference. No novel general
search algorithm or runtime speedup is claimed.

## Recognition and source overlap

The source labels come from an explicit row/edge teacher, not the synthesized
Boolean formula. The primary bank enumerates all nonempty 4x4 subsets. The
secondary bank contains full outlines, each one-cell edge deletion, each one-cell
interior addition, and seeded mixed corruptions in 5x7, 7x5, 8x8 and 11x13 carriers.
Three translations and transpositions are included; actual sets are deduplicated.

| Bank | Dense-source correct | Positive cases correct | Negative cases correct |
|---|---:|---:|---:|
| All nonempty 4x4 sets | 65,535/65,535 | 100/100 | 65,435/65,435 |
| Larger/translated/corrupted | 2,317/2,317 | 21/21 | 2,296/2,296 |

Of the primary cases, 511 repeat dense source inputs, leaving 65,024/65,024
correct genuinely new sets. There is no dense-source overlap in the larger bank.
These counts are highly dependent finite checks and class-imbalanced, not numbers
of independent tasks. In fact the source positives already constrain every one
of the sixteen point-role truth values. Once the semantic table is fixed, the
large banks mainly check that its interpreter applies it correctly at new sizes.
This is useful size/translation generality under a finite representation, not a
statistical generalisation estimate.

## Reuse generates new spatial outputs

The source library is saved before target demonstrations or query inputs are
read. Each of two targets then has two examples: a 5x7 outline missing one edge
cell, and a translated 7x5 outline with both a missing edge cell and an extra
interior cell. Targets are complete perimeter and strict interior, respectively.

The target grammar contains only generic set operations: input, generated set,
carrier complement, union, intersection, the two differences and symmetric
difference. In both tasks a single operation survives the target demonstrations:

```text
Repair:    F(S)
Interior:  B(S) minus F(S)
```

Both are exact on all 2,317 target inputs. Each target has one exact demonstration
input repeated in the query bank, so the new-input score is 2,316/2,316. Repair
changes 2,296 of the supplied sets; interior extraction changes all 2,317. The
outputs include cells absent from the input. All corruption preserves the
original extrema; recovering bounds that were entirely erased is not tested.
The set generator deliberately applies also when the damaged input no longer
satisfies the source classifier.

This is generative reuse of a learned body, not just output relabelling. Its
transfer is nevertheless within the same explicitly structured family. The
learner is not discovering which objects to segment or how to locate a hidden
frame in an unrestricted scene.

A matched cold procedure receives the same source and target evidence, rebuilds
the source-compatible bodies, and tests the same target operations. Its results
match exactly. Reuse needs ten and nine target-example checks; cold execution
also repeats 7,119 source label checks for each target. This is amortization of
stored work, not an advantage over a solver that already caches the same body.
The generator and interior program retain expanded costs 44 and 60.

## The two-example source control fails dramatically

The fixed sparse control is exactly 026's known pair: an eight-cell 3x3 frame
and an eight-cell 3x3 square missing its lower-right corner. Both have count/span
observations (8,3,3), but different labels.

The cheapest fitting classifier does not recognize frames. It constructs the
carrier with its lower-right corner removed and returns 0 when the input equals
that set, 1 otherwise. Its point predicate is

```math
\neg(r(p)=r_{\max}\land c(p)=c_{\max}).
```

That costs 14 expanded predicate nodes, versus the frame predicate's 27. Both
source labels fit, but on 4x4 sets it is correct only 136 times: all 100 positives
and just 36 of 65,435 negatives. It is complete and wrong on the other 65,399.
The larger bank score is 21/2,317, recognizing all positives but no negatives.
No cost or grammar was adjusted to remove this outcome.

The actual sparse-source overlap in the primary bank is two sets, yielding
134/65,533 correct new sparse-control cases. The original frozen scorer's
`source_overlap` and `unseen_correct` fields always compare against the common
511-set bank, even for the sparse row. Those fields do NOT describe its fitting
set. The independent auditor explicitly supplies actual per-mode overlaps;
classification counts are unaffected. This naming limitation is retained rather
than silently claiming a changed prediction implementation matched its old hash.

## Six examples provide a stronger, precisely scoped identification result

After scoring, a separate deterministic diagnostic chooses positive source
examples covering as many still-unseen point roles as possible, then two distinct
negative sets having the same carrier. It selects six labelled sets:

```
positive:  ###     ###     #     #
           #.#             #
           ###             #

negative:  ..#            #.#
           #..            #..
```

The four positives cover all sixteen roles: ordinary edge/corner/interior roles,
one-row roles, one-column roles and the singleton role. They determine the
membership value of the generated set at every possible role.

The two negatives are different sets with identical bounds. A single generated
set depending only on these bounds cannot equal both. Consequently the inverse
label orientation ("match means negative") is impossible. Constant classifiers
are ruled out by the mixed labels. The resulting semantic program is unique.

The auditor verifies this by examining ALL 65,536 Boolean truth tables and all
four label maps, not just the size-seven syntax class. This is an exact teaching
certificate relative to the declared template semantics; no minimum-length prior
is needed for identification. It is not a proof that six examples are minimal,
a randomly sampled few-shot result, or a new preregistered performance score.
Selecting the certificate uses the already-labelled source catalogue and was
introduced as a post-score explanation. The two-example failure is not repaired
by pretending those four additional labels were originally available.

## A remaining observation-language obstruction

Consider two 5x5 sets: all cells except the centre, and all cells except (1,1).
The first is a thickness-two outline; the second is not. Both have 24 cells,
identical bounds, and identical occupied/unoccupied extrema-role profiles.
Both contain occupied and unoccupied points in the undistinguished interior role.

No Boolean formula over these four point comparisons can distinguish the two
through the template-equality observation, even without a syntax bound. The
independent audit checks all 65,536 truth tables. Thus the new construction
resolves the old scalar collision but still cannot represent finer within-
interior geometry. Additional search depth cannot fix that information loss.

The next useful direction is to construct or choose finer coordinate relations
and the candidate carrier, rather than insert a thickness-two-frame atom.
Their cost and evidence must remain explicit. The present result demonstrates
construction and reuse within a strong supplied geometric scaffold, not the
emergence of that scaffold itself.

## Verification and repository disposition

Protocol `c3078d3a` and the active ledger preceded scientific execution. Source
hashes were registered at `df0ab32c` after thirteen synthetic tests passed; files
were published unchanged afterwards. Preparation, source construction, target
prediction and scoring are separate commands. The library hash is frozen before
target examples and query inputs are read; predictions are hashed before query
answers are opened.

The independent auditor imports neither scientific module. It reconstructs the
raw grammar, checks all 366 minima and 365 rejection witnesses, independently
interprets 67,852 geometry cases, replays 135,704 recognizer predictions and
4,634 set-valued predictions, and verifies both negative certificates. These
are exact finite implementation checks, not independent scientific samples.
The reproducibility command reruns all tests and stages, then matches seven
scientific output hashes exactly. Wall times are not compared.

Retain as a bounded construction candidate and regression instrument. The stable
solver, ARC scores and accepted Lean mathematics are unchanged. The package is
standalone standard-library Python and does not extend the chain of dependencies
on earlier experimental learners.

```sh
E=symarc/experiments/027-constructed-spatial-sets
python3 "$E/reproduce.py" --out /tmp/symarc027 \
  --reference symarc/experiments/evidence/027-constructed-spatial-sets/scientific-hashes.json
```
