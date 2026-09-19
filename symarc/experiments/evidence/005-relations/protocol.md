# 005-relations: Relational object expressivity and informative evidence

Baseline `df33dbf`; stable core unchanged, hash
`e76a7031af456aeba4348a32b08e32017f44641eb17256e1b01e09a5985a7f5c`.
Reuse the 003 object representation, extraction and selectors through its library.
No copied solver, no changes to accepted mathematics or production policy.

## One bounded relational language

`segment → select unique source → select unique reference → relate → render`.
Use the eight existing segmentations and existing selectors (including palette
colours); each must select exactly one object, and source/reference must differ.
Nine relations, in this order: recolour the source using the reference's dominant
colour; move above/below/left/right of reference; copy above/below/left/right.
Placement uses adjacent bounding boxes with top/left alignment on the other
axis. Original canvas size/background is preserved; movement clears the source,
copying preserves it. Any out-of-bounds cell or overwrite of remaining foreground
makes the transformation undefined. Transfer colour uses the existing renderer.
No multiple targets, nesting, arbitrary offsets, new object definitions or
multistep relational programs. Source/reference selectors follow 003 order.

## Real-task coverage

400 public training tasks; grid depth 2 plus 003 object AST are the control.
Enumerate all fitting relational ASTs over a vocabulary constructed from training
pairs and test INPUTS only. No hill climbing, symmetry or entropy policy.
Compare first-fitting selection, program-pool oracle, fit and prediction coverage,
and union with the control (grid first, 003 objects second, relations last).
Oracle means a single program correct on every test output, scored afterwards.
Report gains, counterexamples, exact program witnesses and counts, not only syntax
pool growth. Save relation fits/predictions for follow-up analysis.

Audit core depth 3 on up to 16 newly fitting tasks, selected by training fit and
FNV-1a ID order before test correctness. This distinguishes a representation gain
from a shallower search advantage; it is not a full-corpus depth-3 benchmark.
Retain the relational language as a candidate only if it adds at least one
oracle-correct task beyond the control and the audited depth-3 grid class.
Otherwise close the language extension, without retrofitting it to observed tasks.

## Controlled identification study (mechanism check, not an ARC benchmark)

32 scene seeds × TWO equally represented ground-truth rules = 64 cases:

- relational: recolour the largest object to the smallest object's colour;
- grid: recolour the initial source colour to a fixed initial reference colour.

The object rule uses the actual relational interpreter; the grid rule uses the
core recolour primitive. Teacher outputs are constructed separately by replacing
source-coloured cells. Hypothesis space is explicitly these two fixed rules,
with equal prior, not the complete enumerated DSL. Background zero, uniquely
largest source and uniquely smallest marker. Some seeds include an intermediate
sized distractor. Vary colours, locations and source sizes; both rules must be
defined on all queries, to remove the undefinedness confound from 004.

The initial example makes the two rules agree. Each case offers four candidate
example INPUTS: three preserve marker colour, one changes it. Geometry changes
in every candidate. The informative candidate occupies seed mod 4, balanced over
positions. The test input has a third marker colour; both rules give different
answers there. Only the selected example's teacher output is revealed to the
learner; the latent rule and test answer are used for scoring only.

Compare no additional evidence, one guaranteed uninformative example, one known
informative example (oracle query control), the first menu example, uniform random
query choice (exact expectation over four queries), and maximum expected
information gain. Query IG is entropy of the model-predicted answer distribution;
with deterministic hypotheses it equals expected reduction in hypothesis entropy.
It is computed without example answers or rule labels. Condition the equal prior
on the revealed answer; choose the first surviving hypothesis, grid before object,
for deterministic output. Also score posterior mass on the true rule and Brier
loss. Report possible improvement and query choice, not only accuracy.

By construction: uninformative evidence cannot identify the rule, informative
evidence can, and this two-model setting is deliberately favourable to IG. This
is a check of identifiability and mechanism, not evidence of automatic abstraction
discovery or of performance on unseen ARC tasks. Expected outcomes can be derived
before running; generator diversity does not turn them into independent research
hypotheses. No query oracle is assumed available in ordinary ARC solving.

## Checks, cost and commands

Test relative placement/collision/bounds, moving versus copying, unique-selection
requirements, colour transfer and observed training fit. On every generated case,
check independent teacher/interpreter agreement, initial observational equivalence,
valid predictions, and test disagreement. Check query choice is label-blind, exact
random-query averaging, entropy 1→1 for uninformative versus 1→0 for informative
examples, and balanced truth labels (so fixed guessing cannot win).

12 workers; serial harness; 24 hash-selected real tasks in pilot, then full 400
if <120s/<4GiB. Controlled cases run in both. 600s process limit; no task-specific
language changes after first scientific run. All decisions and reports are emitted
directly; source/data/binary/artifact hashes retained. No public evaluation data.

```
cargo test --release --manifest-path experiments/005-relations/Cargo.toml
python3 experiments/005-relations/run.py pilot
python3 experiments/005-relations/run.py full
```
