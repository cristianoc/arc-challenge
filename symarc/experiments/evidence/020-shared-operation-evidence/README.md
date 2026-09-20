# Sharing an operation palette does not yet resolve guard selection

## Registered test

Following 018 and alongside the ARC1 measurement in 019, this bounded study tests
whether separate guarded contexts can share evidence for operations without
being merged. Protocol `4146e1427a620a07ecc33148860a8b5b2acb3754` was published
before the controlled run. Runner source was frozen locally and hashed before
execution, then published unchanged at `e2317014`; no claim is made that the
runner was in GitHub before execution. Eight synthetic tests passed first.

Keep 018's complete feature/action vocabularies, search bound, selection ranking
and two known endpoint teachers unchanged. For each consistent context table,
let V_z be its surviving operations at key z. Choose all minimum-cardinality
palettes B satisfying

$$B\cap V_z\ne\varnothing\quad\text{for every observed key }z.$$

At each key retain every operation belonging to some optimal palette and V_z.
All optimal palettes survive; no arbitrary tie-break chooses copying. Keys
remain distinct, unseen keys abstain, and undefined survivors are not dropped.
The palette is a prior favoring fewer distinct operations, not evidence supplied
by an extra observation or a prior-free theorem. It can favor constants too.

The modified table procedure is used inside internal validation and inside every
outer reconstruction. It is not fitted once using held-out labels. Feature
consistency is unchanged; sharing only reduces action ambiguity and can change
selection through predictions. This is finite minimum hitting set, not a new
general synthesis architecture.

## Result: no change to the targeted failure

The two previously declared teachers have the same two demonstrations and 28
constructed query grids used in 018. These are not new ARC task labels or an
untouched holdout. Predictions were persisted before teacher query scoring.

| Method | Correct grids | Complete wrong | Incomplete |
|---|---:|---:|---:|
| Original 018 selected learner | 14/28 | 2 | 12 |
| Minimum-palette sharing, selected learner | 14/28 | 2 | 12 |
| Available endpoint-validity candidate, either procedure | 28/28 | 0 | 0 |

Both teachers still select `[near.any_border, compare(W,E)]`. Neither selected
learner passes its outer holdout. The correct endpoint-validity candidate still
has internal score 634/385 versus the selected guard's 134/77; the same endpoint
abstentions remain. No costs, teachers, guard rules or tie-breaks were changed
after this result. No additional ARC corpus scoring was undertaken for 020.

The implementation is not inert: it narrows 2,278 distinct action-set constraints
across 412 of the 1,070 recorded tables. These counts repeat related folds and
models; they are not independent observations. But it fails to remove the
specific alternatives responsible for the guard-selection error.

## Why minimum operation count does not identify copying here

With only one demonstration available inside a validation fold, the correct
endpoint-validity model has **five optimal palettes, each containing four
operations**. Among them are:

```text
{constant 0, constant 1, constant 2, constant 5}
{CopyCentre, constant 5, CopyBeyondRunWest, CopyBeyondRunEast}
```

Both explain all observed contexts. Mixed literal/directional palettes are
optimal too. The literal explanation and the reusable directional explanation
therefore tie under this declared objective. Keeping all minima leaves literal
endpoint outputs alive; they disagree with copying on unseen endpoint colours.

The accidental guard, meanwhile, puts differently coloured unchanged examples
in the same context, logically eliminating a single literal there. Its four
optimal palettes all include CopyCentre. The extra global sharing preference
does not overcome that asymmetric local evidence. More agreement under a chosen
abstraction still does not establish that abstraction's applicability domain.

This is a sharper negative result than saying that sharing is impossible.
Distinct guarded contexts CAN share an operation in the model, and the synthetic
control with two different unchanged colours uniquely identifies copying under
the minimum-palette prior. But this particular operation-count prior does not
resolve the controlled ARC-like ambiguity. Another operation prior, changed
representation of conditional programs, or additional discriminating evidence
would be a different experiment, not an interpretation of these unchanged scores.

## What survives

019's complementary ARC1 gain is independent of this experiment and uses the
unchanged learner. It is reasonable to retain that executable combination while
keeping research on operation/guard identification separate. There is no reason
to make a failed sharing heuristic a prerequisite for using the verified gains.

A next conditional model should make its assumptions about reusable operations
explicit. Simply renaming exact tables as branches or minimizing the number of
operation names will not by itself distinguish these two explanations. A new
criterion must also be tested on examples where literal outputs are genuinely
intended, so that a universal preference for copying cannot masquerade as
learned evidence.

## Verification and disposition

Eight synthetic checks include 600 deterministic comparisons with exhaustive
minimum-cover enumeration, distinct-context sharing, preservation of one-colour
ambiguity, mandatory constants, unknown contexts, undefined actions and isolation
of labels outside the supplied fitting indices.

An independent audit enumerates candidate operation subsets by increasing size,
verifying **all 1,070 returned palette families** and **178,284 candidate subsets**.
This verifies global minimum cardinality and completeness of the returned optima,
not just that each palette happens to cover the observations. The original 018
controlled predictions are reproduced. A repeat produces identical scientific
outputs; timing fields are excluded.

Prediction SHA-256:
`7c08ff0e61c6b935a283f7fbff9f78baa6c7ee43b0bafbac8308e48d2cc85892`.
Runner SHA-256:
`8c906f78d58cd69a761301d98cd650f9b4406f9dfbf9d386c7a36636ea3d3321`.

Close without integration. Retain the small exact implementation, synthetic
controls and this failed-selection fixture as a regression control for future
conditional learners. No stable-core or accepted-mathematics change.
