# Supplementary training-only audit: a common operation for a four-point core

Registered AFTER the frozen 017 prediction/score run, before this audit runs.
This is a post-primary, training-only sensitivity analysis. It does not change
or rescore any primary prediction and is not an additional blind solution.

The exact joint-arm full-vocabulary conflict for `7e0986d6` consists of four
points in demonstration 0: (2,10), (5,5), (8,12), (11,12), all zero-based.
They have identical values for all 19 supplied features and require outputs
2,0,0,2. Removing any one point permits a shared adjacent-copy direction;
no one constant, centre copy, adjacent copy or beyond-run copy explains all
four. Each required label appears at three of that point's four orthogonal
neighbours. This suggests a new action, not another fixed direction feature.

Supply one operation: StrictNeighbourMajority. Read the existing in-bounds
north/south/west/east neighbours and return a colour iff it appears strictly
more than half the number of those neighbours. No neighbours, or no strict
majority, makes the operation undefined. Exclude the centre. Do not try other
radii, connectivity, tie rules or task-specific exemptions in this audit.
The operation is manually proposed from the training witness, not invented by
the primary learner. Its input access is explicit: it reads neighbour colours
that are not necessarily recoverable from the feature key.

Keep the 19 features fixed and compare intersections of the original 19 actions
with the 20-action extension, over ALL demonstration occurrences in each exact
full-vocabulary class. Check whether the four-point witness is explained,
whether the entire `7e0986d6` demonstration specification becomes compatible,
and how many nondevelopment eligible training/evaluation tasks become fully
training-compatible. This is a full-feature expressivity check, not a <=3-feature
search or a test-accuracy comparison. Evaluation-task demonstrations are allowed;
query inputs and query answers are not read by the audit.

Also record whether every label in each newly compatible class has an explicit
shared-action witness, and identify any remaining contradictory class in
`7e0986d6`. Fixing four selected cells does not count as fitting a whole task.
If the extension supplies no whole-task gains, report that without changing it.
