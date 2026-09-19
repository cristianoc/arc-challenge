# Query applicability breaks some ties, but does not establish correctness

## Outcome

Experiment 023 changes only model selection. Among candidates tied on the
existing demonstration-prediction score, it prefers a candidate that can
completely answer the supplied query inputs. No query output is used.

This resolves 022's remaining controlled tie: with both palette and row variation,
all six constructed teachers now solve all 36 crossed queries, **216/216 instead
of 144/216**, in both the base and default/exception learners. On ARC1 it adds
one task, `e9afcf9a`: the existing base composition goes **78 to 79 training tasks**,
and the experimental default composition **80 to 81**. Evaluation stays **27/400**.
These are reused public/development data, not independent benchmark validation.

A broader policy that prefers any complete candidate, even over better validated
ones, also creates complete wrong answers. The experiment does not support
replacing predictive evidence with willingness to answer. The stable solver and
all previous compositions remain unchanged; this is a separate research wrapper.

## What was frozen

Baseline `784f282d9b528afba86b003c264a8e185bea61f4`. The primary
[protocol](../../023-query-applicability/README.md) and exact runner/test hashes
were committed at `0f721f6a`; active registration at `a52a5c77` completed before
scientific execution. The matching source files were published with the findings,
not claimed to have been in the remote branch before the run.

Reuse 018's 29 input features, 19 operations, exact search for minimal sufficient
feature sets of size at most three, and internal validation. Run both its base
operation inference and 021's learned observed-context defaults. The new driver
instantiates those existing classes directly; it changes neither their feature
search nor operation tables. Unknown keys, action disagreement and undefined
surviving operations remain abstentions.

All projected ARC1 400 training + 400 evaluation problems from 019 are retained.
The stable predictions are the same saved predictions used in 019/021. No-fit
composition keeps stable whenever it fits all demonstrations; otherwise it
substitutes a complete relational answer, falling back to stable if unavailable.
No new Rust or ARC2 run is made here. Public ARC1/ARC2 overlap was documented in
019; this experiment does not turn those reused tasks into a fresh holdout.

For controlled comparison, reuse all 24 specifications from 022: six known
teachers by four palette/row designs. Their two demonstrations still contain
exactly 90 output labels. Every learner sees the same 36 crossed and 14 legacy
query inputs, with opaque problem IDs and no teacher/design metadata. The
query banks overlap and are scored separately. They are not independent ARC tasks.

Both modes' predictions are persisted and hashed before scoring reads either
answer file. Twelve worker processes, serial modes, deterministic contiguous
batches. Two grouped invocations exceeded the tool window; their unfinished
batches were rerun unchanged. All 800 ARC1 tasks and all 24 controlled problems
were checked on reassembly; no search bound, task or score-dependent policy changed.

## Four selection policies

Let R be the retained models, ordered by the original rank: complete internal
held-out demonstrations, rational correct-known-cell fraction, feature count,
constructor cost, arm, then syntax. Let B be those tied for best on the first
two evidence terms. Let C_Q be models whose current conservative predictions
are complete for every input in the supplied query set Q.

- **Baseline:** the original first-ranked model.
- **Tie complete:** first-ranked in B intersect C_Q; if empty, keep baseline.
- **Complete first:** first-ranked in C_Q, regardless of internal evidence;
  if empty, keep baseline.
- **Tie consensus:** if B intersect C_Q is nonempty, return its common joint
  output only when all those models agree; otherwise abstain on the entire task.
  If that set is empty, keep baseline, including any partial predictions.

There is no majority vote or benefit from duplicating equivalent syntax.
Completeness means a valid colour at every cell, not agreement with a target.
Moreover, a table is itself conservative over surviving operations. Its
incompleteness may reflect uncertainty among concrete programs, not absence of
any correct or defined concrete program. This policy is model selection, not a
proof that incomplete model families can safely be discarded.

For each outer fold, the feature search and internal ranking are recomputed on
the remaining labels. The new selection sees only the excluded demonstration
INPUTS; its pool and decisions are recorded before those outputs are scored.
Duplicate inputs are withheld together. Original query inputs do not influence
these outer choices. The independent audit checks every outer pool's selection.

### Preservation is structural, not evidence of general safety

If baseline is already query-complete, it belongs to C_Q and B, and is first in
the original order. Both non-consensus policies therefore retain it. Any fully
correct baseline task is preserved automatically for a fixed candidate pool.
That includes preservation of a previously complete WRONG answer. A count of
zero raw solved-task losses is consequently not an independent safety result.
New complete answers may be wrong. Consensus can reject earlier complete answers.

The no-fit composition has no analogous automatic preservation guarantee: a new
relational prediction might replace a stable guess that happened to be correct
without fitting the demonstrations. Its paired gains/losses are measured separately.

## ARC1 results

The following are test-query scores for the 400 public training tasks, not
fits to their demonstrations. All rows are experimental learners or wrappers.

| Operation inference | Selection | Relational correct | Relational complete wrong | Stable + relational correct |
|---|---|---:|---:|---:|
| Base | Baseline | 27 | 1 | 78 |
| Base | Tie complete | **28** | **1** | **79** |
| Base | Complete first | 28 | 2 | 79 |
| Base | Tie consensus | 28 | 1 | 79 |
| Default/exception | Baseline | 29 | 4 | 80 |
| Default/exception | Tie complete | **30** | **4** | **81** |
| Default/exception | Complete first | 31 | 7 | 82 |
| Default/exception | Tie consensus | 29 | 4 | 80 |

All evaluation rows retain five correct relational answers and **27/400 combined
solutions**. Complete-first/default creates one new complete relational evaluation
error, `27a77e38`; other variants have none. The combined solver still returns
stable guesses on most unsolved tasks, so low relational error counts must not
be mistaken for low hybrid error rates.

Tie-complete changes one ARC1 task in each mode: `e9afcf9a`. Combined correct
training query grids go 83 to 84 for base and 85 to 86 for defaults; evaluation
remains 31/419. Excluding the three previously inspected fixtures gives base
**76/398 to 77/398**, default **78/398 to 79/398**, and **27/399 evaluation** unchanged.
The gain is not one of those fixtures, but remains part of the reused public corpus.

Complete-first adds the extra default-mode success `9565186b`, at the cost of
three additional complete wrong training answers (`a79310a0`, `d4f3cd78`,
`d5d6de2d`) and the evaluation error above. Base gains no extra correct task
from prioritising completeness and adds the complete error `3aa6fb7a`.
Tie-consensus/default rejects the already-correct development fixture `29c11459`
because 79 equally evidenced complete candidates give five distinct joint answers.
It still retains the old complete errors; agreement in a restricted pool is
not truth.

The strict outer gate behind tie-complete retains 13 correct base training answers
or 12 default training answers, and three evaluation answers, with no complete
errors. The newly solved `e9afcf9a` does NOT pass this gate. These are separate
coverage/error tradeoffs, not correctness certificates for final predictions.

## A concrete new success: e9afcf9a

The two demonstrations have this pattern, with colour pairs (3,9) and (4,8):

```text
3 3 3 3 3 3      3 9 3 9 3 9
9 9 9 9 9 9  ->  9 3 9 3 9 3
```

The original model selects (colour, westward same-colour run length). The query
has new colours (6,2), so all twelve keys are unknown and every cell abstains.
The query-complete alternative selects (westward run length, whether the
northward beyond-run operation is defined). It encodes horizontal position and
a role distinguishing the two rows, without retaining literal colour identities.
It gives the exact answer:

```text
6 6 6 6 6 6      6 2 6 2 6 2
2 2 2 2 2 2  ->  2 6 2 6 2 6
```

Both models have internal evidence (0 complete demonstrations, fraction 0).
The old one wins only on cost, 5 versus 9. Eight complete fitting models remain
at that same evidence level and agree on this query. The policy selects one;
no query label supplied that choice. Full inputs, targets, model keys and scores
are retained in [case certificates](case-certificates.json).

This is reuse of existing input-dependent operations under a different context,
not discovery of a parity operator. All observed and queried grids are 2x6;
the model still tabulates exact horizontal run lengths. Arbitrary-width or
arbitrary-shape extrapolation is not established. Its zero validation score is
also a reminder that this tie has no positive full-demonstration predictive
support; correctness is measured on this query, not certified by the selection.

## Controlled result: the remaining tie is resolved

Under 022's combined palette-and-row variation, baseline solved 144/216 crossed
grids and left 72 incomplete. All three query-aware policies solve **216/216**,
with zero complete wrong or incomplete grids, in BOTH operation modes. On the
separate legacy bank the comparison is **56/84 to 84/84**. All six teachers
now pass every supplied query, including both literal-background variants.

The two formerly unresolved teachers select

```text
(near.max_degree, compare(W,E), west-endpoint-defined)
```

rather than the old colour-specific key. Its validation fraction is still
634/385, three terms, and cost 19 rather than 15. Eleven query-complete models
tied on evidence agree on every supplied output, so even tie-consensus succeeds.
The chosen key is not exactly the hand-proposed two-endpoint-validity key.
No claim is made that it is the unique or universally correct scope description.

The other demonstration designs remain informative controls. The table gives
correct / complete-wrong / incomplete crossed query grids, out of 216:

| Palette varies | Row varies | Baseline, both modes | Tie complete, base | Tie complete, defaults |
|---|---|---|---|---|
| No | No | 12 / 76 / 128 | 12 / 76 / 128 | 12 / 76 / 128 |
| Yes | No | 36 / 84 / 96 | 36 / 84 / 96 | 96 / 64 / 56 |
| No | Yes | 0 / 0 / 216 | 0 / 0 / 216 | 72 / 0 / 144 |
| Yes | Yes | 144 / 0 / 72 | **216 / 0 / 0** | **216 / 0 / 0** |

Completeness-first on colour-only designs gives 156 correct and 60 complete wrong
queries in each mode. Input applicability is therefore not a replacement for
the missing geometric contrast. All factorial cells and policies, including
consensus losses, are retained in [results](results.json). Teacher grid counts
are dependent and the six rules were already inspected; these are not 216 new
independent ARC successes or a new augmentation law.

Every fully relearned outer check still fails in the combined-variation condition,
including the models now solving all queries. Those folds reconstruct from one
demonstration and lose complementary evidence, as in 022. The new policy did not
make those one-example learners reliable.

## Query-set dependence is real

The complete learner now has the form L(D,Q), not merely h_D applied to each
query. A model can be complete for one input but incomplete for a second;
requiring one common model for both may change the first answer.

The predeclared diagnostic runs selection independently for each query, without
adding labels or changing candidate fits. On ARC1, completeness-first changes one
grid's prediction on `d5d6de2d` when queries are separated, in both modes.
Default tie-consensus also changes one grid there. Tie-complete has no ARC1
batch differences in this run. In the controlled data it has differences on
four specifications / 32 grids in base and six / 54 in defaults. All occur
outside the combined-variation condition: its 216/216 result is unchanged when
queries are selected independently. None of these diagnostic predictions is
added to the primary scores.

## What was learned, and the next precise distinction

Input applicability can resolve a tie that a training-only cost preference gets
wrong. It does not by itself provide an output label, establish a transformation
law, or justify a new observation. The small ARC1 gain and controlled success
are useful, but broad completeness-first ranking incurs additional errors.
Do not tune away those negative outcomes or silently replace the stable solver.

The next more principled question separates two quantifiers. A conservative
model answers only if ALL its surviving operations are defined and agree. A
concrete intended program, however, need only be one consistent operation choice
that is defined on the required query domain. Instead of preferring an already
unambiguous model, one could condition concrete operation hypotheses on required
query definedness, retaining disagreement among all remaining defined choices.
For an observed context z this would use

```math
V_z^Q = V_z \cap \bigcap_{u\in Q:\,a(u)=z}
                \{\theta:E(\theta,u)\text{ is defined}\}.
```

That is a proposed next experiment, NOT an operation change in 023. It requires
an explicit candidate-program interpretation and query-totality requirement;
it cannot invent actions for unseen contexts or claim that remaining agreement
is correct outside the hypothesis class. It would distinguish a logical domain
constraint from the present heuristic preference for model-level determination.

Test-input-aware synthesis has precedents, including Chen et al. (2021),
[WebQA](https://arxiv.org/abs/2104.07162), and Lee et al. (2025),
[Program Synthesis via Test-Time Transduction](https://arxiv.org/abs/2509.17393).
The latter obtains additional predicted labels from an LLM; 023 does not.
No novelty is claimed for transductive selection or consensus in general.

## Verification and disposition

Fifteen new synthetic tests passed before the run, alongside all 15 original
018 tests and all 12 default/exception tests. The controls include preservation
of complete predictions, exact rational ties, consensus disagreement, an explicit
wrong complete answer, batch sensitivity, duplicate grouping and outer-label
isolation.

An independent set-based evaluator reconstructs training action intersections,
all tied default choices, internal predictive scores and candidate predictions.
A separate rank/filter implementation checks EVERY new full-data and outer
selection, not merely the outputs of models already chosen by the learner.
Across ARC1 and controlled modes it checks 118,024 table constructions,
3,498,856 context action sets, 133,504 candidate-model occurrences, 34,448
policy choices, 13,896 outer-grid scores, 6,592 raw policy/task scores and
6,400 hybrid policy/task scores. These counts repeat related candidates/folds;
they are not independent statistical observations. Input feature/action
extraction is reused and feature-subset search is not independently reimplemented.

All 800 base ARC1 tasks, 5,614 full candidate models and 1,689 baseline outer folds
match 019. Both controlled modes reproduce all 24 original problems, 4,188 full
models and 48 outer folds per mode from 022. A repeated 12-worker run gives
byte-identical predictions for both ARC1 modes and both controlled modes.

Primary SHA-256:
- ARC1/base: `de6cbb72e0728899190a7e185f07f26c7aef7f680adc03b464453e10581f95f6`
- ARC1/default: `c15515956f88b17319b738a94715210bb1960dcfe633aa04c50a5383af006ed4`
- Controlled/base: `7f30268c7c6f14952b180ac17aee7172cef81591ba400e530bdd4cd6d5d5a4a5`
- Controlled/default: `6296f9b7a7ba9cc0ec18865fbc9d7508bbabcabd202306020f68c6b90646c63d`

Successful ARC1 batches sum to 71.021s/base and 70.535s/default; controlled runs
18.985s and 18.906s. These are provenance only, excluding interrupted attempts,
not speed comparisons with older learners. No scientific code was changed after
these scores. The source/test hashes match the registered protocol.

Retain tie-complete as a small selection candidate and the complete-first failures
as regression controls; no core, earlier wrapper, accepted-math or ARC2-score
change. The archive contains all new frozen predictions, inputs, separately
stored answers, stable baseline predictions, full outer candidate pools, scores,
source dependencies, independent audits and a reproduction script. Historical
comparison counts and hashes point back to the existing 019 and 022 archives.
