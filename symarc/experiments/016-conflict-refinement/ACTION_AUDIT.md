# Supplementary audit: is a distinction necessary, or is an operation missing?

Registered after the frozen primary 016 run and before this audit's computation.
It must not change primary feature selection, predictions, policies or scores.
No query labels or query inputs are needed by the audit.

Question: how often does a full-vocabulary training conflict disappear when the
shared operation language admits copying an adjacent cell, not only the centre?

Use the same 15 feature values and full-vocabulary partition. Compare the
original constants plus CopyCentre with constants plus CopyCentre, CopyNorth,
CopySouth, CopyWest and CopyEast. Adjacent-copy actions are undefined beyond the
canvas and cannot satisfy a labelled occurrence there. The five offsets are
supplied, not learned or selected per task.

For each eligible nondevelopment task, intersect action sets over every class
of identical full feature vectors. Report the number of training-compatible
tasks under each language and each split, preserving the primary development
exclusions. For each primary full-vocabulary two-cell certificate, count whether
its endpoints now admit a shared action; separately check whether the WHOLE
class/task becomes compatible. A resolved pair is not a solved task. Do not
report any new test accuracy or add these operators to the primary learner.

Important boundary: constants-plus-copy-centre has a two-cell empty-intersection
certificate property. The extended action language need not have this property;
empty intersections must be computed over the entire class. Pairwise agreement
is insufficient for arbitrary action sets. Thus the original hitting-set search
is not silently reused as a complete learner for the extended language.

Mathematically, enlarging the action language can make a previously necessary
feature unnecessary. All necessity claims in 016 are relative to the supplied
feature and action languages, not intrinsic facts about the task.

Retain training-only action intersections/counts and at least one source-grid
witness where an added shared copy action removes a previously certified pair
conflict. The audit is a sensitivity diagnostic, not a selector improvement or
an automatically discovered operator language.
