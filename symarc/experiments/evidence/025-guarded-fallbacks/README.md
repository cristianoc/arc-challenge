# A domain guard is not evidence for its fallback

## Outcome

Experiment 025 completes the guarded-repair investigation registered before the
interruption. Adding conditional fallback operations yields no new ARC1 solution.
Requiring both branches to have labelled training witnesses loses one correct
training-task answer and makes one old complete error incomplete. Unrestricted
fallbacks lose four correct answers. Keeping the primitive models alongside the
branch-supported ones preserves the old results but adds nothing.

| Policy | Relational training solutions | Complete wrong answers | Stable + relational training solutions | Hybrid evaluation |
|---|---:|---:|---:|---:|
| Primitive baseline, 024 total-ranked | 30/400 | 2 | 80/400 | 27/400 |
| All fitting guarded operations | 26/400 | 1 | 77/400 | 27/400 |
| Guards with both branches observed | 29/400 | 1 | 79/400 | 27/400 |
| Primitive + observed-guard model selection | 30/400 | 2 | 80/400 | 27/400 |

These are exact answers to the supplied test queries, not demonstration fits.
The prior default/tie-completion composition at 81/400 training and 27/400
evaluation is unchanged. Nothing here is integrated into the stable solver.
All public tasks and controlled teachers were used previously; this is not an
untouched holdout, and no new ARC2 run was made.

The constructive finding is narrower. A correct concrete program for the known
failure `f76d97a5` now exists in the unrestricted guarded family, but it uses an
unobserved fallback. Training does not identify it. The branch-support condition
excludes that correct possibility. Conversely, a guard with both branches
observed introduces ambiguity on previously solved `a699fb00`. Branch coverage
is neither necessary nor sufficient for the intended extrapolation.

## Recovery, registration and scope

Scientific baseline: `cf09c2e633ff76f2389140a3540301e522d0e332`.
The interrupted step published the [protocol](../../025-guarded-fallbacks/README.md)
at `42c75add`, hashes and an active ledger entry, but not its Python sources or
outcomes. Resume baseline was `2421c4bf`. The unpublished source files were not
recoverable. The implementation was reconstructed from the unchanged protocol;
its replacement hashes and successful synthetic checks were committed at
`e68100e9` before any resumed corpus computation. Source and tests were then
published at `b5fe81fe` and `81a70191`, also before the resumed scientific runs.
The original preregistration is retained as history, not represented as a
reproduction of unavailable bytes. See `resume-registration.json`.

Reuse all 800 projected ARC1 problems, archived stable predictions, and all 24
controlled specifications from 023/024. The latter cross six known teacher rules
with four palette/row demonstration designs. Each has two demonstrations and
36 crossed plus 14 overlapping legacy queries. Report the banks separately.
The previous three development fixtures are also excluded in secondary scores.

The 29 input features and the <=3-feature tuples are frozen from the BASE 023
full-data and outer-fold pools. Old predictions and scores are stripped from
the cache; only tuples, old arm identities and excluded-index groups are used.
Equivalent tuples across old arms are aliases, not votes. No new feature search
is performed under the larger action grammar. This is repair of old fitting
representations, NOT a complete conditional-program synthesis comparison.

## Guarded operations and branch evidence

For primitive operations a and b, define

$$
\operatorname{OrElse}(a,b)(u)=
\begin{cases}
a(u),&a(u)\text{ is defined},\\
b(u),&\text{otherwise}.
\end{cases}
$$

The guard is exactly the primary operation's domain. Four adjacent copies and
four copies beyond a same-colour directional run can be partial. Combining each
with any different one of the 19 original primitives gives 144 new expressions,
163 operations in total. No nesting, new input feature, new literal colour,
learned coordinate threshold or privileged copy fallback is added. Zero is a
colour; only -1 denotes undefinedness.

Three libraries retain every consistent operation:

- Base retains the original 19 primitives.
- All retains primitives and every fitting guarded expression.
- Observed retains primitives plus a fitting expression only if its primary and
  fallback branches each execute at a labelled occurrence of the SAME context
  key. Witnesses may lie in one grid; this is branch coverage, not independent
  or causal evidence.

Action intersections and branch support are rebuilt inside every internal and
outer fold. The internal score remains exact withheld grids followed by the
rational sum of correct-known-cell fractions. Feature count, constructor costs
and old syntax ordering break ties. Query inputs impose joint totality as in
024: one operation must execute at every occurrence of a key, unknown keys
reject the family, and nonempty but disagreeing families abstain.

Each library selects its best-ranked feasible model. The fourth policy retains
base and observed models, preferring base after equal evidence/count/cost. This
preserves candidate availability, not correctness by theorem. All predictions
for both corpora were persisted and hashed before separate answer scoring.
The no-fit hybrid uses the unchanged 019 rule and saved stable predictions.

## Why a previously unseen fallback stays unknown

Suppose a partial primitive a fits every labelled occurrence of context z. It
must have been defined at each occurrence. Then, for every colour c,

$$\operatorname{OrElse}(a,\operatorname{Constant}(c))$$

fits exactly the same observations and is total on new occurrences of z. Where
a is undefined, these ten hypotheses return ten different colours. Totality
therefore cannot identify the fallback output. This obstruction was stated in
the protocol before the run; it is not an unexpected empirical discovery or a
claim about all possible synthesis methods.

Because every retained context tuple already has a primitive fit, unrestricted
fallbacks make a family query-total whenever every query key is observed. For a
key with a total fitting primitive there is nothing to do; for a partial one,
any constant fallback totalizes it. Missing keys still reject the family. The
independent audit verifies this equivalence on every full and outer model.
Existence of a total program has become much less informative than agreement.

## Both observed branches can still support the wrong extrapolation

On `a699fb00`, the selected context remains
(colour, west-endpoint-defined, east-endpoint-defined). At key `(0,1,0)`, consider

```text
OrElse(copy south, constant 0)
```

It has both required labelled witnesses:

| Occurrence | South value | Executed branch | Required output |
|---|---:|---|---:|
| Training 0, cell (0,3) | 0 | Primary south copy | 0 |
| Training 2, cell (9,4) | Undefined | Fallback constant | 0 |
| Query 0, cell (4,5) | 1 | Primary south copy | 0 |

The expression predicts 1 at the query while primitive alternatives still predict
0. The conservative predictor abstains; `(4,7)` is similarly unresolved. The
base predictor had solved the entire task. No query fallback is even needed for
this key: all 19 query occurrences execute the primary branch.

Thus verifying that a branch has occurred does not establish that its behaviour
or scope extrapolates to new input values. The policy loses this correct task
in both standalone and hybrid scores. Coordinates and actual operation values
are retained in the case certificate; the official query label is used only
for the after-score explanation, not for fitting.

## f76d97a5: expressibility improves, identification does not

All libraries select the same feature tuple `[5,6,14]`: local minimum degree,
local maximum degree and global same-colour-peer maximum degree. The primitive
baseline makes a complete wrong answer. The unrestricted and observed-branch
variants instead produce incomplete answers; neither solves the task.

A separate AFTER-SCORE diagnostic intersects each concrete family's operations
with all official query labels, sharing one operation across each key. It finds
some correct concrete table for 34 training tasks in base, 35 in All, and 34 in
Observed; evaluation is five for each. The only additional unrestricted task
is `f76d97a5`. This is an oracle expressivity check, not a deployed prediction.

At its problematic key `(0,0,2)`, one suitable expression is

```text
OrElse(copy east, copy north)
```

The labelled occurrence uses east-copy. At query cell `(2,4)`, east is undefined
and north supplies colour 3. No labelled occurrence of this key exercises that
fallback. The complete correct-program diagnostic has 48 possible guarded
operations at this key, none with both branches observed. Its choices are
therefore excluded by the Observed restriction.

The unrestricted family also contains wrong constant fallbacks. It cannot choose
the correct program merely because that program now exists. Under Observed,
three determined cells still have the wrong value 6 and six others abstain.
The old complete error becoming incomplete is not a repaired answer. Full
query grids and the unsupported-operation certificates are in the archive;
compact selected witnesses are retained beside this report.

## Other task results and controls

All loses `a699fb00`, `d511f180`, `d5d6de2d`, and `e8593010` relative to the
primitive baseline, with no gain. Only three are hybrid losses: the frozen
stable-first rule already declines to use the relational answer on `d5d6de2d`
because its stable program fits the demonstrations. Observed loses only
`a699fb00`. Both retain the old complete error `6e82a1ae`.

Every evaluation policy has five correct standalone answers, no complete errors,
and 27/400 hybrid solutions. The hybrid still emits stable guesses on most
unsolved tasks; low relational complete-error counts are not a reliable hybrid.
Excluding the three prior fixtures gives hybrid training scores 78/398 base,
75/398 All, 77/398 Observed and 78/398 union; evaluation is 27/399 throughout.
The strict outer filter retains 15/8/10/15 correct training answers respectively,
and 3/1/2/3 evaluation answers, with no complete errors. These are coverage/error
tradeoffs, not proofs that the domain or branch restrictions imply correctness.

On the crossed controlled bank the results are correct / complete-wrong /
incomplete grids, out of 216 in each design:

| Palette varies | Row varies | Base | All | Observed | Union |
|---|---|---|---|---|---|
| No | No | 12 / 204 / 0 | 12 / 12 / 192 | 12 / 108 / 96 | 12 / 204 / 0 |
| Yes | No | 156 / 60 / 0 | 36 / 20 / 160 | 156 / 12 / 48 | 156 / 60 / 0 |
| No | Yes | 0 / 0 / 216 | 0 / 0 / 216 | 0 / 0 / 216 | 0 / 0 / 216 |
| Yes | Yes | 216 / 0 / 0 | 216 / 0 / 0 | 216 / 0 / 0 | 216 / 0 / 0 |

More cautious outputs reduce some errors in confounded designs, principally by
abstaining. They do not add correct answers. The previously informative combined
contrasts preserve 216/216. These dependent teacher-labelled constructions are
not new ARC ground truth or a valid augmentation scheme for unknown ARC rules.

## Why the union remains unchanged, and what this design cannot test

At every fixed representation and labelled subset, both extended libraries retain
all the fitting primitive operations. Their program families are supersets.
Unconditioned predictions can therefore retain a primitive model's determined
colour or become less determined, but cannot supply a different uniquely
identified colour. Their fixed-tuple internal predictive evidence cannot improve.
The same inclusion holds after totality whenever the primitive total family is
nonempty. These are exact set-inclusion facts, not claims of general safety.

Accordingly, a guarded model can beat its primitive counterpart in the union
only when the primitive family fails the query-domain requirement. There are
five newly feasible Observed full-data models, on `25ff71a9` and `7f4411dc`.
None wins the final ranking. The union selects a base model and preserves its
output on every one of the 824 problems, not merely its aggregate score.

This explains an important design limit: freezing old primitive-fitting tuples
prevents us from testing a COARSER context that becomes sufficient only when a
conditional operation is available. The positive synthetic test has exactly
that form: no primitive serves its class, but a guarded expression does. Such a
class cannot enter this experiment's inherited pool.

The appropriate next construction question is joint context-and-operation
synthesis, with conditional expressions allowed while conflicts generate the
context pool. Any extension should include primitive-incompatible training
classes and explicit controls against unobserved fallbacks. It should not claim
that branch coverage uniquely identifies the rule or that adding guards to
already fitting tables necessarily increases determined predictions.
Conditional synthesis has established precedents, including Alur, Cerny and
Radhakrishna, [Synthesis through Unification](https://arxiv.org/abs/1505.05868)
(CAV 2015). No novelty is claimed for guards, hypothesis elimination or this
finite-library construction.

## Verification, provenance and disposition

Twelve new synthetic tests passed: 1,800 independent explicit-fit comparisons,
200 concrete-program/domain checks, both-branch and unobserved-branch controls,
zero/undefined handling, shared-key semantics, duplicates and fold-label
isolation. All 58 inherited tests also pass. The matched primitive baseline
reproduces 024 on all 824 problems, 66,752 old model occurrences and 1,737 outer
choices, accounting for syntax aliases.

The independent audit uses explicit sets and a separately written guarded
interpreter, conditioner and rank/filter selector. It reconstructs 59,012 tables,
5,248,284 context action sets, 89,916 model occurrences, 436,557 candidate query
grids, 139,587 internal grids, 6,948 outer-grid scores, 10,244 policy choices,
3,296 raw policy/task scores and 3,200 hybrid scores. It verifies 2,145 grouped
branch witnesses covering 16,699 retained conditional-operation instances,
59,944 nesting checks, and 12,906 after-score oracle model checks. Counts repeat
related candidates and folds; they are not independent experimental samples.
Primitive feature/action extraction is reused; original feature search is not
independently reimplemented.

All fourteen scientific batches, covering 800 ARC1 and 24 controlled problems,
repeat byte-for-byte. Successful first-run computation intervals sum to 19.990s
for ARC1 and 19.364s for controls; they exclude loading cached JSON and are
provenance only. They must not be compared with complete search runtimes.
Oversized verification/repetition invocations were interrupted; the unchanged
work was completed in smaller recorded batches with process isolation. No task,
search bound, grammar or selection rule changed in response to a score.

Prediction SHA-256:
- ARC1: `10b5d4e374029b314acb6ee83689dfaf1e9fc2220b517f4e7e21fc33a269b250`
- Controlled: `a0c1151e87b5fbb2277765b29cf8ef98aeb4421117d9e8ae662273d5e04efe2c`

Close without integration. Retain the exact guarded interpreter, independent
auditor, and the two branch-evidence counterexamples as controls for joint
context/conditional construction. Stable core, previous wrappers and accepted
mathematics remain unchanged. The companion archive includes all needed source
dependencies, projected inputs, separately scored answers, frozen pools,
compressed predictions, audits and reproducible commands.
