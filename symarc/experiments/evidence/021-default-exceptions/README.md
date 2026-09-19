# Defaults resolve operation ambiguity, not the scope of a rule

## Outcome

Experiment 021 changes only operation selection inside the existing 018
learner. A learned default plus exceptions raises the measured 019 ARC1
composition from **78 to 80 of 400 training tasks**, while evaluation stays
**27/400**. No previously solved task is lost. The relational component itself
goes from 27 to 29 training solutions, but from one to **four complete wrong
answers**. These are test-query scores on reused public data, not independent
holdout validation. The existing 019 composition is not modified or integrated.

On the controlled endpoint teachers, the original operation-identification
failure is partly repaired: the correct guard now identifies copying at the
endpoints. It still is not selected. Its validation disadvantage becomes a tie,
and the unchanged feature-count tie-break prefers the accidental guard. Literal
output controls also expose guard failures; this was not a hidden preference
for copying. No primary policy was changed after the results.

## Registered comparison and exact prior

Baseline: `11466555c84f0b8096f2f8fcf41c4122385a4c2e`.
[Protocol](../../021-default-exceptions/README.md) and source hashes committed at
`b9227afb`; active ledger registered at `41304763`, before the scientific runs.
Runner published unchanged at `07a853b0`. Twelve synthetic tests passed first.
Scientific computation is local; temporary workflows only register/finalize
these records and remove themselves. Stable code and accepted math are unchanged.

Reuse 018's features, operations, three-feature limit, exact candidate search,
internal ranking, and nested outer validation. Let V_z be the operations fitting
all observed labels with context key z. For every possible default d, compute

$$c(d)=\#\{z:d\notin V_z\}.$$

Choose all defaults minimizing c. A key compatible with d uses d; an incompatible
observed key is an exception and may use any operation in V_z. All tied defaults
and all exception operations survive. There is no privileged cost for copying.
A default is valid ONLY on the observed key domain: unseen contexts still
abstain, as do undefined operations. This is a partial default/exception model,
not an unconditional fallback on all grids.

This preference is equivalent to minimizing the number of exception entries for
a fixed table. It is NOT a complete minimum-description-length objective over
programs: guard/key encoding costs are not optimized, and the unchanged outer
ranking still prefers fewer features. No novelty is claimed for defaults or
conditional synthesis; Alur, Radhakrishna and Udupa's TACAS 2017 divide-and-conquer
synthesis is a primary precedent for separating predicates and operations.

The retained per-key action set is the projection of all optimal programs:

$$W_z=\bigcup_{d\in D_{\min}}\begin{cases}\{d\},&d\in V_z,\\V_z,&d\notin V_z.\end{cases}$$

Consequently, W_z is a nonempty subset of V_z on every fitting observed key.
For a fixed representation/table, any already determined prediction is
preserved. Some abstentions can become predictions, and these can be wrong.
Model selection can also change, so monotonicity of a fixed model is not
monotonicity of the complete learner. Candidate fit existence and feature pools
are unchanged. The implementation is a subclass of 018 and reuses its driver;
the process-local class substitution is not thread-safe, so workers are separate
processes.

## ARC1 results

Every arm uses the same projected 400 training + 400 evaluation problems from
019. The saved stable predictions are unchanged. Both sets of relational
predictions are written and hashed before scoring the separate query answers.
The existing no-fit composition rule is frozen: keep stable if it fits every
demonstration; otherwise use a complete relational answer, else keep stable.
Selection happens per task, never by query correctness.

| Policy | Training exact tasks | Training complete wrong | Evaluation exact tasks |
|---|---:|---:|---:|
| Stable alone, retained 019 baseline | 57 | 343 | 23 |
| Original relational learner | 27 | 1 | 5 |
| Default/exception relational learner | 29 | 4 | 5 |
| Existing stable + relational composition | 78 | 322 | 27 |
| Stable + default/exception composition | **80** | **320** | **27** |

The stable baseline always emits a complete guess here; its unsuccessful tasks
are complete errors. The relational learners usually abstain. Those different
coverage regimes must not be confused by comparing error counts alone.

The two additional successes are `3bdb4ada` and `97999447`, with no solved-task
losses. Combined correct query grids rise from 83/416 to 85/416 in training and
stay 31/419 in evaluation. Across all 800 tasks, the composition is 107/800,
versus 105/800 for 019 and 80/800 for the stable solver. Excluding the three prior
development fixtures gives **78/398 training and 27/399 evaluation**, versus
76/398 and 27/399. These comparisons do not establish hidden benchmark gains.

The relational learner's complete errors are `321b1fc6`, `3aa6fb7a`,
`6e82a1ae`, and `a699fb00`. Only `6e82a1ae` was already a complete error;
the other three were previously incomplete. Both methods fail these tasks, but the new method is more committed.
The strict outer filter retains nine correct training answers, down from ten,
and three evaluation answers, unchanged; no filtered complete errors occur.
It loses `a5313dff` through changed outer selection. The unfiltered learner also
loses one previously correct individual query, `d5d6de2d` query 1, while neither
method solved that entire task. Whole-task and grid counts are both retained.

### What the two gains actually do

Both new successes retain their previous selected feature keys. There is no new
feature discovery or newly covered key in these examples.

- `3bdb4ada`: eastward run length plus comparison of north/south lengths. CopyCentre
  fits 75 of 89 observed keys, leaving 14 exception keys. The default resolves
  16 previously ambiguous query cells, all unchanged and now correct. At query
  cell (1,0), centre/south/east copying return 0 but west-copy exits the canvas;
  the default preference removes that undefined alternative.
- `97999447`: whether a same-colour peer touches the canvas border, westward run
  length, and west-endpoint validity. CopyCentre fits 13 of 23 keys. Just one
  query cell, (4,3), was unresolved: constant 0 and centre/west copies return 0,
  but east-copy returns 8. The new preference selects centre-copy, completing
  the correct output. The input and expected value there are both 0.

These are 17 correctly completed unchanged cells, not evidence that the learner
invented a new structural rule. The larger scope/generalisation questions remain.
On `a699fb00`, by contrast, the new selected rule predicts all three internal
held-out demonstrations yet makes a complete wrong query answer. Its fully
relearned outer check still fails one fold. A good fixed-rule validation score
and a successful outer learner are not the same property.

## Controlled identification improves, but selection still fails

Six controlled teachers were frozen before execution. Two are the old nearer/
farther endpoint rules. Two keep those geometric distinctions but use literal
interior colours 6 and 8. Two retain input-dependent interior filling but make
all nonactive rows literal colour 6. Each teacher has the same two training
widths (7,11), height 5, and the same fourteen query width/palette/height variants.
These are researcher-defined examples, not independent ARC annotations.

| Controlled pair | Correct grids, original | Correct grids, defaults | Complete wrong / incomplete, either method |
|---|---:|---:|---:|
| Input-dependent endpoint colours | 14/28 | 14/28 | 2 / 12 |
| Literal interior colours | 14/28 | 14/28 | 14 / 0 |
| Literal background | 14/28 | 14/28 | 2 / 12 |
| Total | 42/84 | 42/84 | 18 / 24 |

All selected predictions are unchanged. The supplied endpoint-validity candidate
in the same pool predicts all 84 grids correctly in both methods, but is never
selected. No cost or feature was changed to break that tie after seeing outcomes.

For the two original teachers, the endpoint-validity candidate's one-demo table
now has unique default CopyCentre: five of eight contexts use it and three are
exceptions. The literal/copy ambiguity at both endpoints disappears. The
internal validation fraction sum rises from 634/385 to 134/77, exactly eliminating
018's gap of 36/385. But the border-based guard already scores 134/77. Neither
gets a complete withheld demonstration correct, and the existing tie-break
prefers its two features to the endpoint model's three.

For the literal-fill teachers, both models now predict both internally withheld
demonstrations exactly. The fewer-feature wrong guard still wins and fails at
height 7. They share the same confounding training height. The fully relearned
outer check rejects the selected learners; these are not counterexamples to
passing that strict check. They are concrete limits of validating one chosen
representation and resolving remaining uncertainty by its feature count.

For the literal-background teachers, the learned default is instead constant 6,
not copying. That is the appropriate dominant operation, but endpoint copying
remains an exception with the original ambiguity. This confirms that the prior
is symmetric while showing that it does not magically identify every exception.

## A structural limitation of the new prior

Counting exception contexts is representation-dependent. With two contexts whose
allowed sets are {a} and {b}, both defaults tie. Split the first context into two
keys with the same allowed set {a}, without adding observations or changing their
labels, and a becomes the unique default. Duplicate observations do not change
the result, but splitting a context can. A synthetic control explicitly checks
this limitation; it is not called a bug or an invariance guarantee.

The experiment separates three outcomes: selecting a useful shared operation,
selecting its correct domain, and producing a correct whole-grid answer. The
first can improve while the second remains unresolved. More committed output
is not additional observed evidence. The operation prior resolves some useful
ambiguity and also commits to wrong alternatives.

Do not repair this result by silently changing the feature-count tie-break or
always preferring copying. A next scope-learning experiment should compare
confounded and genuinely discriminating demonstrations at a fixed label budget,
or explicitly retain the competing scope explanations. Renaming table entries
as branches alone does not create evidence about which scope is intended.

## Verification, artifacts and disposition

Twelve synthetic tests include 300 exhaustive tiny-program comparisons, all tied
defaults, literal and copy defaults, exceptions, unknown keys, undefined actions,
duplicate invariance, label-subset isolation and context-splitting sensitivity.
The complete base-mode ARC1 run reproduces 019's prediction file byte-for-byte.
A second twelve-worker default run reproduces all eight 100-task batches exactly.

An independent set-based audit reconstructs action sets from actual training
labels using the previously validated 018 feature/action evaluator. It checks
24,353 ARC1 table records, including 21,483 nonempty fitting tables; all 408,177
default choices; 1,100,428 context projections; and all 2,128 eligible feature
pools against base. It replays 5,866 candidate query grids, 17,152 internal fold
grids, 8,445 outer grids, and verifies 2,660 full-data selections. A separate
scoring audit checks 8,800 standalone/combined policy-task decisions. Counts
repeat related candidates/folds and are not independent experimental samples.
The controlled audit additionally checks 4,772 table records. The auditor does
not independently implement primitive extraction, search, or outer selection.

ARC1 default prediction SHA-256:
`9ef4d2a4add18023bbb13bfd6fd75bb5c86c03f04a933391b0949dbba858993f`.
Base replay SHA-256:
`e47927459066e61bc19b1b1d3e6d2678d62a26aebf4da7f5bc9cc359c8f1ed18`.
First default prediction batches total 54.456 seconds on this environment,
provenance only. All runs use twelve workers and fixed task order; no search
bound or task is dropped to fit a tool window. Primary sources match the
pre-registered hashes. The complete archive includes source, projected problems,
separate answers, predictions, default certificates, scores and replay metadata.

Close the targeted guard-repair study without changing the existing composition.
Retain this exact implementation and its positive/negative fixtures for future
comparisons. The observed two-task gain is a separate experimental composition,
not a silently promoted production heuristic. Stable core and accepted math stay
unchanged.

Primary precedent: https://www.microsoft.com/en-us/research/publication/scaling-enumerative-program-synthesis-via-divide-and-conquer/
