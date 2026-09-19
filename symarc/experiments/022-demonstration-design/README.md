# 022 — Separate operation evidence from scope evidence by varying demonstrations

Baseline: PR #32 at `3f75fee8854870f7c438077cc55fd7ec2b129309`.
Question: does independent variation of colour and active-row position improve
the unchanged learner, when the number of demonstration grids AND labelled
output cells are held fixed? No new features, operations, priors or tie-breaks.

021's learned default removes endpoint action ambiguity but leaves a tie in
scope selection. Rather than changing that tie-break, vary the evidence. This
is a controlled experiment-design study on already inspected artificial task
families. It is not an automatic data-augmentation rule for ARC and does not
supply additional ground-truth labels to an ordinary ARC solver.

## Fixed factorial and units

Use all six teachers from 021: nearer/farther endpoint copying, nearer/farther
literal interior colours, and nearer/farther copying with literal background.
Reuse their implementation, extending it only by moving the entire active row
(and corresponding output row) to another strictly interior row. The other
rows retain each teacher's declared operation. Teacher choice and position
semantics are supplied by the researcher, not discovered by the learner.

Every training specification contains EXACTLY two grids: width 7 and width 11,
both height 5. Each has two coloured endpoints in one active row. Thus every
condition supplies 90 labelled output cells, the same number of input cells,
and four nonzero input cells. The first grid always has palette (1,2), row 2.
The second grid independently varies two binary factors:

| Factor | Unvaried | Varied |
|---|---|---|
| Palette | (1,2) again | (3,4) |
| Active row, zero-based | 2 again | 1 |

All four combinations are run. Palette-varied/row-unvaried reproduces 021's
original demonstration design. A row change preserves input/output colour
multisets and cell count; it is not a larger-example-budget intervention.

Learners are unchanged 018 (base) and 021 (default), including all 29 features,
19 operations, at most three selected terms, all surviving actions, internal
selection and nested whole-demonstration reconstruction. Query predictions use
the union policy. The other policies/candidate pools remain in raw evidence.
The model sees no teacher names: task IDs are opaque hashes and metadata are
not passed to the learner. Both algorithms rerun feature/operation selection
inside each outer fold. No new primitive or default-on-unknown behaviour.

## Queries fixed before execution

Primary crossed bank: widths 5,13,21; heights 5,7; palettes (6,8),(7,9);
active rows 1, floor(height/2), height-2. Fully cross these factors: 36 grids
per teacher, 216 per demonstration design and learner. Widths are unseen in
all training specifications. Palette is NOT coupled to height, unlike 018's
original query bank. All grids stay inside the existing ARC size bound.

Secondary legacy bank: the same fourteen centred-row queries per teacher as
021 (widths 5,9,13,17,21,25,29 and the paired palette/heights (6,8)-5, (7,9)-7).
Use its original-design arm as a regression check against 021's documented
42/84 correct, 18 complete wrong, 24 incomplete outcomes for both learners.
The two banks overlap; NEVER add their counts as independent observations.

The study has six fixed teachers, not 216 independent tasks. Report complete,
correct, wrong and abstaining grids; all-queries-correct teacher counts;
selected feature keys and action choices; internal and outer validation;
available-candidate ceilings; paired changes stratified by teacher, height,
position and palette. A fixed endpoint-validity candidate is a supplied
expressivity control, not automatically the intended answer to every task.
No confidence intervals or general ARC accuracy claims.

## Execution, checks and stop rule

Separate prepare, predict and score processes. Prepare stores projected
problems, query answers, and design metadata separately. Prediction receives
only problems. Write and hash both learners' predictions before either answer
file is used to score. Use 12 processes per learner; learners run serially,
with no seed, beam, time cap or policy tuning. Batch only if required by the
tool window, preserving every job and the exact search. Runtime is provenance.

Eight pre-run tests passed. They compare translated teachers with an independent
coordinate-level interpreter, check exact 90-cell budgets, the full query
factorial, equality to original centred teachers, colour-multiset preservation,
invalid domains, fitting-subset label isolation and unknown-key abstention.

Frozen pre-run SHA-256:
- run.py: `19bb19929c3d1a5c090db49644e95b53e8345357452dff1aa66febdf44906892`
- test_run.py: `322eb3bdb5f3a74a05d008a07de1f480d6989baf23645512ad4c5590f1720513`

Independently verify all teacher labels and score every frozen selected query.
Replay selected full-data models by explicitly intersecting operation sets and
applying the default prior, without reusing the learner's table-fitting code.
Check a repeated prediction run for byte equality. The audit does not claim to
independently reimplement primitive extraction or the complete outer selector.

Record every factorial cell, including adverse outcomes. A wrong guard becoming
inconsistent is a different outcome from the algorithm selecting a right one;
passing fixed-model validation is different from the entire outer learner
passing. Do not change row positions, query banks or costs after seeing scores.
No broad ARC run or integration follows automatically from this controlled
result; keep the stable core and accepted mathematics unchanged.

## Mathematical question

For a declared class H and labelled data D, the version space is
V(D)={h in H : h fits D}. Evidence can remove a hypothesis only when its labelled
prediction is contradicted; changes to a preference can instead select among
unchanged alternatives. This experiment varies D at equal size, not the prior.
A measured benefit remains relative to these teachers, feature/action grammar
and learner. It does not show prior-free generalisation or learn the teacher's
intervention semantics.

Training-set design and learner-dependent teaching have established precedents;
see Liu and Zhu, The Teaching Dimension of Linear Learners (2015),
https://arxiv.org/abs/1512.02181. No teaching dimension is computed here.

```sh
E=symarc/experiments/022-demonstration-design
python3 "$E/test_run.py"
python3 "$E/run.py" prepare --out /tmp/arc022/input
python3 "$E/run.py" predict --mode base --problems /tmp/arc022/input/problems.json --out /tmp/arc022/base --workers 12
python3 "$E/run.py" predict --mode default --problems /tmp/arc022/input/problems.json --out /tmp/arc022/default --workers 12
# Only after both prediction files exist:
for mode in base default; do
  python3 "$E/run.py" score --predictions /tmp/arc022/$mode/predictions.json --answers /tmp/arc022/input/answers.json --metadata /tmp/arc022/input/metadata.json --out /tmp/arc022/$mode
done
```
