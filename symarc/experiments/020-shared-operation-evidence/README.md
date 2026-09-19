# 020 — Share operation evidence without merging guard contexts

Status: active at registration. Baseline learner is unchanged experiment 018
at `83df79f78f059fe18b356d51a4e0354367966f72`. This is a bounded mechanism study
on 018's already-inspected controlled teachers, not a new blind ARC benchmark.

## Question and prior

018's endpoint-validity guard retained literal/copy ambiguity at endpoints,
while an accidentally pooled border guard identified copying and won partial
validation. Can operations share evidence without forcing context keys to merge?

Keep all 29 features, 19 operations, three-feature search bound and selector
ranking unchanged. For each fitted table with nonempty action sets V_z, choose
ALL minimum-cardinality palettes B of operations such that B intersects V_z for
every observed context z. At each context retain exactly those actions appearing
in some optimal B and in V_z. Do not select an arbitrary optimal palette or
arbitrarily prefer copy over a constant. Unknown context keys still abstain;
undefined surviving operations still cause abstention. A palette preference is
an explicit additional inductive bias, not a logical consequence of the labels.
No primitive, guard, teacher, test grid, cost or tie-break may be changed after
scoring. Fitting remains equivalent to nonempty V_z; only prediction ambiguity
and consequently internal validation/selection can change.

Cross-context sharing happens because the same operation can hit several V_z.
Each guard remains distinct. A supplied constant can share too; copying is not
given a privileged cost. Deduplicate equal constraints but retain all optimal
palettes. Empty data keeps an empty table, not a default operation.

## Frozen comparison

Use the same nearer and farther endpoint teachers as 018: two demonstrations
at widths 7/11, height 5, palettes (1,2)/(3,4); fourteen queries per teacher at
widths 5/9/13/17/21/25/29 with palette-height pairs (6,8)-5 and (7,9)-7.
Teacher labels are constructed, not ARC annotations. Compare ordinary 018 and
the palette-sharing learner. Report complete/correct/wrong/incomplete grids,
selected feature keys, internal scores, all outer folds, optimal palettes, and
the available fixed endpoint-validity candidate [24,27,28] separately from
learned selection. Persist predictions before consulting teacher query labels.

Rerun both feature construction and internal selection inside each outer fold.
No full ARC corpus extension is authorized by this protocol: first determine
whether the targeted mechanism improves operation identification and whether
that actually resolves guard selection. An unchanged or worse selected result
must be retained rather than repaired by further tuning.

## Checks and stopping

Before the teacher run, compare the palette algorithm against exhaustive subset
enumeration on finite synthetic action-set families. Check multiple optimal
covers, distinct unchanged colours selecting copy, one-colour ambiguity,
mandatory constants, unchanged context identities, unknown-key abstention,
undefined actions, and isolation of labels outside an explicit training subset.

This is finite minimum hitting set applied to the already-built table, not a
new synthesis architecture or prior-free abstraction principle. Independent
checks must verify each returned palette hits every observed class and no
smaller palette does. Report the remaining distinction between identifying an
operation, identifying its domain, and obtaining a correct query prediction.
