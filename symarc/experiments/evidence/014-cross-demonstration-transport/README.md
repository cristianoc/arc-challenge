# From compatible abstractions to predictive transports

## What changed after reflection

The useful findings from 001–012 remain: aggressive pruning can discard correct
programs; representation coverage and selection are different bottlenecks;
hand-selected symmetry contracts can expose accidental commitments without
being justified by the examples; and an unlabelled collision exposes what a
statistic cannot represent relative to a proposed alternative.

The previous 013-minimal-sufficient-distinction step overstated its result.
Two programs chosen to fit every demonstration cannot be separated by checking
those demonstrations again. The proposed runner also hard-coded its negative
support flags and invoked 012's test scorer. It was not executed as a valid
training-only experiment. Its [review](../013-minimal-sufficient-distinction/README.md)
is recorded separately; it supplies no new empirical evidence against generalisation.

The factorisation observation is elementary, not a new theory of learning.
A predictor factors through the image of `a` precisely when it is constant on
`a`'s fibres. On unseen inputs, the decision to identify those fibres still
requires assumptions. The necessity of inductive bias is not a new consequence
of our three ARC examples; see [Mitchell (1980)](https://www.cs.cmu.edu/~tom/pubs/NeedForBias_1980.pdf).

The actionable question is narrower: **does a fixed abstraction-learning
procedure predict observations it did not use to choose its rule?**

## Experiment 014: predict complete withheld demonstrations

The [protocol](../../014-cross-demonstration-transport/README.md) and exact
source hashes were committed at `f0998ec` before the corpus run. The predictor
uses a deliberately small, transparent family rather than task-specific repairs.
For each output cell, its input feature is a raw-colour neighbourhood of radius
0, 1, 2 or 3. It chooses the smallest radius admitting a consistent context-to-
colour table. Unknown contexts abstain; there is no default copying.

Each distinct demonstration input is withheld in turn, together with any
duplicates. Both the radius and table are relearned from the remaining grids.
This matters: choosing the radius on all demonstrations before validating would
let the held-out output influence model selection. A strict gate requires
complete, exact reconstruction of every withheld demonstration.

The final predictor is refitted on all demonstrations. Query predictions are
written and hashed before a separate process scores query answers. Controls
retain explicit coordinates and grid dimensions, or the entire input grid.
Identity is reported separately. The study uses all 1,000 training and 120
public evaluation tasks at ARC-AGI-2 `f3283f7`, with 12 workers and no tuning.
The public data were used in earlier research; this is not an untouched holdout.

Eligibility is decided from training: at least two distinct inputs, and matching
input/output dimensions. Test shape changes are failures, not exclusions.

## Primary results

| Corpus split | Eligible tasks | Local tables fit all demonstrations | Pass every withheld-demonstration prediction | Complete test predictions | Exact test solutions |
|---|---:|---:|---:|---:|---:|
| Training, 1,000 tasks | 680 | 395 | 5 | 6 | 6 |
| Evaluation, 120 tasks | 81 | 36 | 0 | 0 | 0 |

The strict gate retains **4 of the 6 correct training-task answers**. It removes
no complete wrong answer, because the raw predictor made none. The fifth
cross-validation-passing task (`95990924`) has an incomplete query prediction.
This experiment therefore **does not establish improved whole-task reliability**.
At the cell level the gate removes errors, but does so by discarding almost all
predictions; that is not a substitute for the declared task-level comparison.

Coordinate-retaining tables fit 588 training and 70 evaluation tasks, but give
no complete test answer. Whole-input memorisation fits all 761 eligible tasks
and gives no complete test answer. Identity solves none of the eligible tasks.
These are useful coverage controls, not equally expressive competing solvers.

Raw local successes: `0d3d703e`, `3618c87e`, `6f8cd79b`, `b1948b0a`, `c8f0f002`,
`d511f180`. The gate loses the first two. These scores are not an improvement
claim over the Rust solver, which was neither changed nor run here.

The gap between **431 fitting tasks and 6 complete answers** is mainly a lookup-
coverage problem: many required contexts have never been seen. Raw colour names
also prevent a lookup table from treating “copy the input colour” as one shared
relation across previously unseen colours. This restricted family cannot answer
the broad question of whether ARC supplies evidence for structural abstraction.

## A more informative outcome: supported local rules can still be wrong

The raw local predictor makes some defined cell predictions on otherwise
incomplete grids. Across the two splits, **359 such cells are wrong on 62 tasks**.
Each is an exact collision: a context that had a unique training label receives
a different official query label. The audit records one concrete witness per
failing task, including training coordinates and the first larger radius that
distinguishes the two inputs.

A supplementary, post-score diagnostic counted distinct demonstration inputs
supporting each predicted context. It did not change any policy or prediction.

| Split | Supporting demonstration inputs | Known query-cell predictions | Wrong cells | Tasks containing errors |
|---|---|---:|---:|---:|
| Training | One | 6,132 | 212 | 49 |
| Training | Two or more | 7,235 | 63 | 10 |
| Evaluation | One | 802 | 84 | 8 |
| Evaluation | Two or more | 341 | 0 | 0 |

Query contexts repeated across distinct demonstrations occur on **171 training
and 9 evaluation tasks**. So literal local reuse exists where earlier whole-grid
orbit matching found none. But **10 tasks still violate a context-to-label rule
supported in at least two demonstration grids**. Those are local-rule failures,
not failures of the much stricter whole-demonstration gate, which those tasks
did not pass.

Repeated-context predictions have fewer errors descriptively, but cells are
correlated and the two support groups differ in content and difficulty. Of the
7,576 multi-demonstration-supported predictions, only 623 concern cells that
actually change; 60 of those are wrong. We claim neither calibration nor causal
benefit from the raw cell percentages. In evaluation, only four of the 341
multi-supported predictions concern changed cells.

## A concrete rectangle example: e88171ec

The uniform audit surfaced `e88171ec`. In each of its three demonstrations a
3x3 all-zero neighbourhood occurs with centre output colour 8. At test cell
`(16,11)` (zero-based), the identical neighbourhood must instead remain 0:

```text
input context in all four cases     demonstration label     test label
          0 0 0
          0 0 0                            8                    0
          0 0 0
```

The representative training points are `(4,6)`, `(5,5)` and `(11,8)`.
Consequently, **no function of that 3x3 context alone can fit all four labelled
cases**, regardless of the downstream program or search budget.

After inspecting the failure, we supplied and checked a structural explanation:
find the unique maximum-area all-zero rectangle, fill its strict interior with
colour 8, and copy everything else. The fill colour is inferred from training;
the rectangle concept and rule were supplied by the researcher after seeing the
test. Unique maximum rectangles have areas 16, 30 and 24 in training, and 30 in
the test. This prototype matches **3/3 demonstrations and 1/1 test** exactly.
It is a retrospective mechanism check, not a seventh blind solution.

In the test, the marked cell lies on the bottom edge of the chosen rectangle.
Extra zeros protrude beneath it, so its immediate neighbourhood looks like an
interior point. The distinction needed by this explanation is **position within
the complete rectangle**, not just local emptiness. The label is surprising
under the local rule, but the rectangle hypothesis explains it without alleging
a defective task.

The checked prototype and [rectangle coordinates](rectangle-case.json) are
retained. No tie rule is inferred, and nothing here proves this explanation
correct on every future grid.

## Consequences for the research direction

Do not integrate the strict gate: it sacrificed two correct complete answers
without an opportunity to reject a complete wrong one. Retain the collision
audit as a diagnostic, because it identifies precise lost distinctions and
produces reusable, labelled witnesses without hand-selecting task invariants.

A kernel alone is not a general explanation of generalisation. It must be
attached to a decision, a domain of applicability, and a specified output
interpretation. Whole-grid equivariance, for example, transforms outputs rather
than necessarily making them equal. Pointed cells provide one well-defined
setting for equality of output labels; they do not cover every ARC operation.

The next useful target is **domain-qualified structural transfer**: a procedure
must state which relation it proposes, what makes it applicable, and which
withheld observations it predicts. For this case, the contrast is between local
emptiness and membership in the strict interior of a recovered rectangle. The
examples do not uniquely force the latter. Choosing it prospectively still
requires a representation/learning prior whose cost and provenance are explicit.

Before another broad language sweep, establish multiple applicable development
cases and actual selection headroom. Freeze the candidate structural relations
and the full learning procedure before inspecting new answers. Include raw
colour versus colour-relative output interpretation as a separate factor, not
an invisible default-copy advantage. Measure coverage, errors, abstention and
unsupported assumptions separately. More agreeable training examples or a
larger quotient is not, by itself, the research objective.

## Evidence and checks

Nine synthetic controls passed. An independent audit recomputed all **7,840
policy/task complete-and-correct decisions**, checked context labels directly
from source grids, and verified the frozen prediction hash. A second 12-worker
local prediction run was byte-identical. Original prediction time was 3.071s;
this is provenance only, not a comparison with earlier runtimes.

A separate [GitHub Actions reproduction](https://github.com/cristianoc/arc-challenge/actions/runs/35418719391)
at `45c968c` succeeded. Its predictions, scores, summary, audit and rectangle
results are byte-identical to the original local run. The two temporary
read-only workflows were removed after completion.

Primary prediction SHA-256:
`3d328547540d6d97eabc228bc5999b554b2ebdacae826f0638e183a9511cce22`.

Compact [outcomes and provenance](results.json) and all 62 [collision certificates](collisions.json)
are retained here. The Actions artifact `symarc-014-full-evidence` (ID
10576219463, expiration 18 December 2026) retains full predictions,
fold-specific selections/scores, source/data hashes, the complete audit and
reproducible code. It can also be regenerated with the retained commands.
The stable core and accepted mathematics were not changed. Code is retained
as a reusable falsification instrument and case catalogue, not as an integrated
solver heuristic.
