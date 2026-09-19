# Equal-sized demonstrations can supply very different evidence

## Resume point and new result

The interrupted step had completed experiment 021. Its default/exception
variant raises the measured ARC1 composition from 78/400 to 80/400 training
tasks; evaluation stays 27/400. Its standalone relational component also makes
three additional complete wrong answers. Those recorded results are not a new
ARC run in this experiment and the existing 019 combination remains unchanged.
See [021's report](../021-default-exceptions/README.md).

Experiment 022 instead keeps both learners fixed and changes the demonstrations.
With exactly two grids and 90 labelled output cells per teacher, varying both
endpoint colours and the active row raises complete correct predictions from
36/216 to 144/216 on a fully crossed constructed query bank. No complete wrong
answers remain in that condition, but 72 grids are still incomplete. Varying
only the row makes performance worse: all 216 queries become incomplete.

These are six already-inspected artificial teacher rules, NOT 216 independent
ARC tasks. The effect is a controlled, teacher-relative demonstration-design
result. It does not measure an ARC score increase, discover new primitives,
automatically select demonstrations, or infer a valid augmentation law for ARC.

## What was fixed before execution

Baseline `3f75fee8854870f7c438077cc55fd7ec2b129309`. The
[protocol](../../022-demonstration-design/README.md) and exact source hashes were
published at `2c264eb5`, the runner at `ea14fb92`, and tests at `32febd71` before
scientific prediction. Workflow 35433346449 completed the active ledger
registration before the run. The workflow only edited the ledger and removed
itself; computation is local.

The learners are unchanged 018 and unchanged 021, with the same 29 features,
19 output operations, three-feature bound, exact search, internal ranking and
nested whole-demonstration reconstruction. All consistent operations and tied
defaults are retained. Unknown contexts and undefined operations still abstain.
No prior, operation cost, feature cost or tie-break was retuned after scoring.

Teacher rules are the same six from 021: copy the nearer/farther endpoint;
use literal interior colours instead; or use literal colour 6 for background
while filling from endpoints. The only extension moves their active row within
the canvas, with the corresponding teacher output. This translation semantics
is supplied by the researcher. It is independently verified, not learned.

Each specification has a 5x7 and a 5x11 grid: 90 labelled output cells in all
conditions. Each input has just two nonzero endpoint cells. The first example
always uses colours (1,2) in row 2. The second independently varies:

| Dimension of variation | Absent | Present |
|---|---|---|
| Palette | Colours (1,2) again | Colours (3,4) |
| Active-row position, zero-based | Row 2 again | Row 1 |

Moving the row preserves both input and output colour multisets. Thus neither
more grids, more pixels, nor more changed-output pixels explains its effect.
Palette diversity changes observed colour identities, not the pixel budget.

The primary query bank crosses widths 5/13/21, heights 5/7, palettes (6,8)/(7,9),
and active rows 1/middle/height-2. That gives 36 grids per teacher. Widths are
unseen in every training specification. A separate 14-grid legacy bank per
teacher reproduces the original setup; the banks overlap and are not added as
independent observations. Teacher identities and design metadata are not passed
to the learner; opaque IDs identify jobs.

Projected problems and query answers live in separate files. Both learners'
predictions were persisted and hashed before either scoring process read the
query answers. Each learner used 12 worker processes, with learners run serially.

## All four factorial outcomes

Both the base and default/exception learner have the following grid outcomes:

| Palette varies | Row varies | Correct / 216 | Complete wrong | Incomplete | Teachers solving all 36 queries |
|---|---|---:|---:|---:|---:|
| No | No | 12 | 76 | 128 | 0/6 |
| Yes | No — original design | 36 | 84 | 96 | 0/6 |
| No | Yes | 0 | 0 | 216 | 0/6 |
| Yes | Yes | **144** | **0** | **72** | **4/6** |

The successful four are the two endpoint-copy teachers and the two literal-fill
teachers. Each goes from 6/36 under the original design to 36/36 with both
variations. The two literal-background teachers go from 6/36 to 0/36, through
abstention rather than complete wrong answers. Aggregates must not conceal those
losses: the intervention is not a uniform improvement across all six teachers.

On the secondary legacy bank, the original condition reproduces 021's 42/84
correct, 18 complete wrong and 24 incomplete grids in both modes. With both
variations it gives 56/84 correct and 28 incomplete. This regression prevents a
change of the queried row distribution from masquerading as a change to the
original result. The primary bank is harder for the old guard because it also
includes off-centre active rows; its original-design score is 36/216, not 42/84.

## A labelled certificate explains why row variation matters

The old copy teacher selected

```text
(near.any_border, compare(W,E))
```

After the row intervention, that key is no longer consistent. For the near-copy
teacher, consider these two actual demonstration points:

| Demonstration | Point, zero-based | Old key | Required label | Operations that can produce it |
|---|---|---|---:|---|
| First, active row 2 | (2,2) | (false, W<E) | 1 | constant 1; copy beyond west run |
| Second, active row 1 | (2,4) | (false, W<E) | 0 | constant 0; centre or adjacent copy |

The action sets are disjoint. The first point belongs to the row being filled;
the second is now background. No single operation in the supplied library can
serve both under the old key. This is an actual training contradiction, not a
synthetic unlabelled counterexample and not a changed ranking preference.

The audit retains deletion-minimal conflict certificates for all six teachers'
old guards. For literal-background variants, the contradiction can already occur
inside the shifted demonstration. These certificates are relative to the
existing action library, as established in 017.

Row variation alone does not suffice: with the palette fixed, the selected key
becomes `(colour, compare(W,E), west-endpoint-defined)`. It fits by remembering
colour-specific contexts, but new endpoint colours have unobserved keys. Every
query is consequently incomplete. Geometric variation removes one shortcut
while leaving a different one available.

## What combined variation actually selects

For the four successful teachers the selected key is

```text
(canvas-border membership, compare(W,E), west-endpoint-defined)
```

It is NOT exactly the hand-supplied pair of endpoint-validity guards. It is an
alternative consistent description whose predictions match every tested width,
height, palette and row in this controlled family. The endpoint markers remain
at the left and right canvas edges in all these teachers; no conclusion follows
about markers placed inside a larger canvas or arbitrary multi-object scenes.

The copy variants use input-dependent operations and the literal-fill variants
use the literal colours their teachers require. Thus the gain cannot be explained
as a universal preference for copying. In this run, the base and default priors
have the same selected grid outcomes, although internal scores and surviving
operations can differ.

There remain many consistent candidate models. Even the four successful selected
models are not proofs that all possible fitting programs have the same outputs.
Other retained literal-fill candidates make some complete wrong query-grid
predictions. The claim is about the frozen learner's selected predictions, not
unique identification of a target function on all possible inputs.

## The two unresolved teachers isolate a selection tie

With both variations, literal-background teachers select

```text
(colour, compare(W,E), west-endpoint-defined)
```

All non-endpoint query cells are correctly determined. Exactly two cells per
query—the newly coloured endpoints—have unknown keys. Across 36 grids that is
72 unknown cells per teacher, with zero wrong determined cells.

The available key

```text
(compare(W,E), west-endpoint-defined, east-endpoint-defined)
```

predicts all 36 primary and all 14 legacy queries correctly for each teacher.
But both keys have the same internal predictive score, 634/385, and three terms.
The former has supplied cost 15, the latter cost 19. The unchanged tie-break
therefore selects the colour-specific key and abstains at the endpoints.

This is neither inadequate output-operation expressivity nor absence of a correct
candidate. The fixed scoring/cost preference still fails to select it. We did
not increase the cost of colour or start selecting for query completeness after
observing the result. Such a query-input-based applicability policy would be a
separate experiment and would have to count new wrong complete answers too.

## Why the strict outer gate rejects even the successes

No teacher/design combination passes every fully relearned outer fold in either
learner. This includes the four conditions whose selected full-data models solve
all primary queries. Every outer fit sees only one demonstration; it reconstructs
cheaper, unsuccessful exact-feature explanations instead of the two-example
relational model. A fixed representation can transfer when refitted in a fold
while the entire one-example learning procedure chooses another representation.

This is not evidence of a broken validation implementation. The outer check
asks how this procedure learns from one example, whereas the final learner uses
two. Two examples can supply complementary contrasts; removing either removes
some of the evidence used to construct/select the final representation.

For a finite hypothesis class, this distinction is elementary: if one example
eliminates alternative h1 and another eliminates h2, both together can identify
h while neither singleton does. We do not prove that these actual ARC-like
specifications uniquely identify their teachers, compute a teaching dimension,
or establish that their chosen rules will generalise universally. The measured
point is narrower: demanding leave-one-example-out success can discard useful
full-data predictions, particularly at such small demonstration counts.

## Interpretation and next step

The previous operation priors changed preferences while leaving the examples
unchanged. This experiment changes the examples at fixed size and leaves the
learning machinery unchanged. In this family, independent variation in what is
incidental is much more informative than either kind of variation alone.

The intervention semantics remain supplied by a known teacher. Ordinary ARC
provides no oracle for labelling shifted grids. This is therefore not a usable
ARC augmentation scheme by itself, nor evidence that an ARC task is defective
whenever a learned guard fails. Dataset design, learner behaviour, and author
intent must remain separate questions.

The concrete next opportunity is to detect which contrasts are missing or which
applicability distinctions remain unresolved, without pretending a preferred
operation supplies that evidence. Input-only checks of whether candidate rules
apply to a query could address the remaining colour-key tie, but they can also
favour confidently wrong rules. That must be evaluated as a new decision policy,
not silently built into the reported 022 scores. The measured 019 composition
and experimental 021 gains are independent of this controlled result.

Training-set design relative to a learner is an established machine-teaching
problem; see Liu and Zhu, [The Teaching Dimension of Linear Learners](https://arxiv.org/abs/1512.02181).
No novelty is claimed for that general principle or for factorial experiments.
Our deliverable is the executed equal-budget comparison, its explicit conflicts,
and the remaining selection failure in the same unchanged learner.

## Verification and reproduction

Eight pre-run design tests pass, as do all 12 existing 021 controls and all 15
018 controls. The independent coordinate-level teacher interpreter checks 1,248
training/query grids. Actual budgets are checked for all 24 specifications.

A set-based audit independently reconstructs selected action tables and defaults,
without calling learner fit/compress/apply. It verifies 2,400 selected query
predictions and scores, 96 fixed-model internal folds, 96 outer-fold predictions,
and 48 full-data ranking choices from the retained candidate scores. It also
checks the six old-guard rejection certificates and the available endpoint key.
Primitive extraction is reused; neither feature search nor complete outer
selection is independently reimplemented. Replaying a chosen outer model is not
an independent verification of how that model was chosen.

A second 12-worker run is byte-identical for both learners. Primary wall times
were 16.622s/base and 16.606s/default; repetitions were 15.650s and 15.785s.
These are provenance, not a speed comparison with ARC corpus runs.

Prediction SHA-256:
- Base: `75a251c9c28f75985d3862581f0ca8889beb7be09ef539203c91a6e82f8e4931`
- Default: `86e9d198f5e8f6b9ac1bed42b5fa14b4681f152ff6bf71d76c8f3c14d41019e2`

[Compact results](results.json) retain every factorial cell and teacher outcome;
[audit certificates](audit.json) retain the key contradictions and cost ties.
The investigation archive contains frozen sources/dependencies, projected inputs,
separate answers, compressed full predictions and reproduction instructions.

Close as a controlled design result, without solver integration or new ARC
scoring. Retain the small driver and certificates as regression fixtures for
future scope-learning or evidence-selection procedures. The stable core and
accepted mathematics are unchanged.
