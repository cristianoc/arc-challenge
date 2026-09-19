# 018 — Ordering relations and applicability guards

Question: can selected comparisons among directional run lengths transfer where
017's exact numeric keys do not, and is explicit endpoint validity needed?

Baseline: PR #32 at `aa0afd5a97a58e02461ca57012fe2c4d7bf264d7`.
Official ARC-AGI-2 data: `f3283f727488ad98fe575ea6a5ac981e4a188e49`, all 1000
training and 120 evaluation tasks. Previously used public data, not an untouched
holdout. Known `00d62c1b`, `e88171ec`, and the motivating `29c11459` are excluded
from primary counts (997/120 corpus denominators; eligibility from training).

## Frozen factorial comparison

All arms keep the same 19 operations from 017 (constants, centre/adjacent/beyond-
run copy). The exact arm is precisely 017's joint arm, not its union selector.
Unknown keys, conflicting action results and undefined surviving actions abstain.
No copying fallback, majority operation, output-size changes or query-label tuning.

- Exact: 19 original features.
- Order: those 19 plus all six pairwise directional-run comparisons (<,=,>).
- Guarded exact: 19 features plus four beyond-run-action definedness booleans.
- Guarded order: all 29 features.
- Union: retain candidates from all four arms and select by the frozen rule.

Comparisons operate on counts, never numerical colour order. Definedness treats
colour zero as a valid value; the -1 sentinel is undefined. These constructors,
costs and the four directions are supplied. At most three selected features;
exact conflict-driven enumeration of every inclusion-minimal sufficient tuple.
Comparisons have constructor cost 9, validity predicates 5; inherited costs are
unchanged. Rank by complete held-out-demo predictions, per-demo correct-known
fraction, number of terms, constructor cost, fixed arm order and term order.
Number of selected terms precedes their cost, an explicit bias favouring a
compound predicate over several raw features. No measured MDL claim.

Internal validation refits action tables with each distinct demonstration input
(and all its duplicates) withheld. Outer validation reruns both feature search
and internal selection without the outer labels. Only input-derived values are
cached. Query predictions are saved and hashed before separate scoring. Twelve
workers, deterministic, no random seed, beam, node cap or corpus-score pilot.

Primary units are tasks and complete query grids, not independent cells. Report
fit/complete/correct/wrong/abstaining tasks by split, paired changes, candidate
oracle, outer-gate retention/losses, and exact reasons for abstention. Preserve
all model/fold predictions and action-relative necessity certificates. Verify
the exact arm against 017's retained joint arm on every task after accounting
for the extra development exclusion.

## Constructed width-transfer control

Separate from ARC, freeze two explicit teachers: copy the nearer endpoint, or
copy the farther endpoint; ties output 5. All other cells copy. A 5-row grid has
one middle row with differently coloured endpoints and zeros between them.
Train each teacher at widths 7 and 11, endpoint pairs (1,2) and (3,4). Test at
widths 5,9,13,17,21,25,29, each with palette (6,8), height 5, and palette (7,9),
height 7. These 28 supplied teacher-labelled queries are not new ARC labels,
independent task samples or evidence of an author's intended semantics. Persist
predictions before scoring. No teacher/width/tie-rule tuning after execution.

The known ARC endpoint example is a development mechanism check only. No new
synthetic outputs are declared its ground truth. Width extrapolation is claimed
only for the explicitly defined controlled teachers and actually tested widths.

## Mathematical scope

For point u, raw distances d(u), and selected relational code c(d(u)), fit an
action map h with prediction E(h(c(d(u))),u). Equality of relation values shares
an operation, not necessarily a resulting colour. Validity predicates can bound
where operations are plausible but do not magically justify extrapolation.

Comparisons are functions of old numeric features: if the entire old feature
vector is equal at two points, their comparisons are equal too. Therefore
order-only additions cannot remove a full-vocabulary training conflict. They
can instead compress two or more coordinates into one categorical feature,
change the <=3-feature search space, and merge unseen numerical values with
observed relational cases. The four validity predicates expose information
not previously present in the feature vector, although used by the decoder.

A quotient valid for rule sharing still needs a supplied prior. The finite
relational code can overgeneralise. A better whole-demonstration score is evidence
about this learner, not a universal correctness or novelty claim. Relation/
conditional synthesis has precedents, including Alur, Cerny and Radhakrishna,
Synthesis Through Unification (CAV 2015), and divide-and-conquer synthesis.

## Pre-corpus checks and stop rule

Fifteen synthetic controls pass: directional comparisons/validity, zero-valued
endpoints, blank rows, order under increasing remapping, a fixed conditional key
on a novel width, abstention, undefined survivors, three-way conflicts, 150
exhaustive subset comparisons, exact-arm equivalence, duplicate handling,
outer-label isolation and training-only eligibility. The fixed-key width test
checks expressivity, not learned selection; the controlled run measures selection.

Source SHA-256 before scientific runs:
- run.py: `4df4f38f789dbfbb7ce9bf43fedc0947301ea519d2edacd7622186b352efba34`
- test_run.py: `4964b0794938ded8caa7925245c7fa23154ca1315c788b770cbcf499e78a1ccc`

No arbitrary five-task gate: a matched baseline exists and the factorial has
clear representational predictions even if accuracy gains are zero. Record
negative results without modifying the grammar. Any post-run diagnostic or
proposed repair must be labelled separately and must not alter primary scores.
No stable-core or accepted-mathematics changes.

```sh
E=symarc/experiments/018-order-guards
python3 "$E/test_run.py"
python3 "$E/run.py" predict --problems input/problems.json --out out/018 --workers 12
python3 "$E/run.py" score --predictions out/018/predictions.json --answers input/answers.json --out out/018
python3 "$E/run.py" controlled --out out/018
```
