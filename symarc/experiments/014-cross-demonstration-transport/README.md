# 014 — Can an abstraction predict a whole withheld demonstration?

## Reflection and falsifiable question

013-minimal-sufficient-distinction did not implement its claimed experiment.
Its support booleans were constants, and calling `012.case` scored test answers
before discarding those scores. More fundamentally, comparing two programs
already selected to fit all demonstrations cannot find a direct labelled
contrast between them. That is a consequence of their selection, not a new
negative empirical discovery. It does not show an absence of all evidence.

Retain the useful distinction from 012: an unlabelled collision proves a
representation boundary relative to a proposed repair, not the repair's truth.
Now change the unit of analysis and the prediction protocol.

**Question:** with a fixed, transparent family of cell-context abstractions,
can relationships learned from some demonstration grids predict a different
whole demonstration? Does that evidence filter erroneous predictions on the
supplied test inputs better than training compatibility alone?

This is a bounded study of cross-demonstration transport, not a general
abstraction-discovery algorithm, MDL experiment, or definition of intelligence.

## Fixed hypothesis family

Domain: pointed grids `(x,r,c)`, target: the output colour at `(r,c)`. Apply only
to tasks with at least two distinct demonstration inputs and matching input /
output dimensions in every demonstration. Select eligibility from training
only; a test shape change is a failure, never a post-hoc exclusion.

For radii `0,1,2,3`, `a_k(x,r,c)` is the raw-colour square neighbourhood of side
`2k+1`, with outside-grid sentinel 10. The abstraction erases absolute position
and the rest of the scene. The colour names, square neighbourhood, output
alignment and radius bound are **explicit supplied priors**.

Fit an exact table from context to output colour. A conflicting table rejects
that radius. Select the smallest consistent radius. Predict only seen keys;
unseen keys abstain (-1). No smoothing, synthetic labels, majority vote,
default copying, learned weights, or existing task-specific repaired programs.

Controls use the same table procedure: add `(height,width,row,column)` to each
key (`positioned`), or retain the entire input and cell coordinates
(`memorise`). Identity is an additional explicit baseline, not a fallback.
The controls are not equally expressive language families; report their
coverage as well as their correctness, not just an accuracy ranking.

## Cross-demonstration evidence without reusing held-out answers

Withhold each distinct demonstration input and all its duplicates together.
Relearn both the radius choice and the table on the remaining demonstrations.
Predict every cell of the held-out output before comparing it with its label.
**Do not choose the radius using all demonstrations before cross-validation.**
The gate passes only when all held-out grids are predicted completely and
exactly. It is a prospective prediction check for a fixed algorithm, not an
independence or causal-identification claim about the supplied examples.

Afterwards, refit on all demonstrations and freeze query predictions.
Compare raw predictions with the same predictions gated by cross-validation.
The gate can only abstain: it cannot increase the number of solved tasks.
Its question is selective reliability (errors removed versus correct answers
lost), not a new source of solution coverage.

## Important mathematical boundary

For a representation `a`, constancy of a predictor on the fibres of `a` is
equivalent to factorisation through the image of `a`. This elementary fact is
not a novelty claim. A training collision with different output labels refutes
that representation's sufficiency *on observed data*.

There is a further useful check. Suppose `a = q o b`, and both exact tables are
consistent with the same training data. Wherever the finer table `b` has an
observed key, the coarser table `a` has that key's image and predicts the same
label. Proof: take the training observation witnessing the finer key; it also
witnesses the coarser key, whose label is unique by consistency.

Consequently, nested neighbourhoods cannot disagree where both lookup
predictions are defined; larger contexts only lose lookup coverage once a
smaller one is consistent. Do not describe this study as choosing among
competing complete answers using a kernel score. Likewise, for a fixed
consistent table, leave-one-grid-out success is just cross-grid key coverage.
Relearning the radius inside every fold tests more than that repetition count.

Neither repeated transports nor cross-validation logically determine unseen
outputs without assumptions. No amount of this evidence proves arbitrary
future contexts safe to identify. This is a measured prior, not prior removal.

## Frozen protocol (before the corpus run)

Baseline: PR #32 research state `16c4193183a8357ea85ec58fe42e2e22b4274988`.
Data: all 1,000 training and 120 evaluation tasks from official ARC-AGI-2
`f3283f727488ad98fe575ea6a5ac981e4a188e49`. Preserve split labels. These are
public datasets already used in this research, not an untouched holdout.

Prepare projected problems and a separate answer file. The prediction command
reads only projected problems; its output is written and hashed before the
separate scoring command reads test answers. This gives algorithmic label
isolation, not a claim that the researcher has never seen these public tasks.

Use 12 workers, deterministic ordering, no random seed, no task-specific
exceptions and no tuning after scores. Synthetic unit tests precede the run.
No pilot score is used to alter the protocol. Wall time is provenance, not a
comparison with earlier Rust runtimes.

Primary units are **tasks** and whole demonstration / query grids, not cells
assumed independent. Report eligible tasks, training fits, cross-validation
passes, complete predictions, exact solutions, complete wrong answers, and
abstentions by split and policy. Also report known-cell errors and changed-cell
coverage / correctness so that background copying does not dominate.

Retain per-task radius, each fold's selected radius and scores, frozen query
predictions, file hashes and run manifest. Check source isolation, duplicate
handling, real abstention, conflict-driven refinement, and a positive transport
control. Independently recompute task scores from frozen predictions.

Interpretation criterion: useful positive evidence requires nonzero correct
whole-grid transport beyond identity, and an interpretable error/coverage
tradeoff for the gate. A gate that merely removes all predictions is vacuous.
Any confident failures become retrospective counterexamples to this *specific*
locality assumption, not evidence of defective ARC tasks.

## Reproduction

```sh
E=symarc/experiments/014-cross-demonstration-transport
python3 "$E/test_run.py"
python3 "$E/run.py" prepare --data /path/to/ARC-AGI-2/data --out /tmp/arc014/input
python3 "$E/run.py" predict --problems /tmp/arc014/input/problems.json --out /tmp/arc014/run --workers 12
python3 "$E/run.py" score --predictions /tmp/arc014/run/predictions.json --answers /tmp/arc014/input/answers.json --out /tmp/arc014/run
```

The stable solver and accepted mathematics are unchanged. Findings belong in
`../evidence/014-cross-demonstration-transport/` and the experiment ledger.

Pre-corpus source freeze (SHA-256):
- `run.py`: `572c8870bd00abf97a49fa4280c994ae9bca56cc14391408ec7fd94f166aa3f4`
- `test_run.py`: `d375184adaafd11741b68e63910359191b686d450e690158ed0ef04890163863`
Nine synthetic controls passed before any corpus prediction or score.
