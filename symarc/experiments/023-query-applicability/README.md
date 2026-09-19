# 023 — Can query-input applicability resolve remaining selection ties?

Baseline: PR #32 at `784f282d9b528afba86b003c264a8e185bea61f4`.
Continue 022 without changing its features, actions or examples. Keep both the
018 base learner and 021 default/exception learner. No new primitive, operation
prior, training-example transformation or cost is introduced.

## Frozen question and policies

A candidate's internal evidence score is its complete held-out-demonstration
count followed by its exact rational correct-known-cell fraction. Original
ranking then uses feature count, constructor cost, arm order and syntax.

A candidate is query-complete when its existing conservative predictor supplies
a valid colour for every cell of every supplied query grid. Unknown keys,
disagreeing actions and undefined surviving actions remain abstentions. This
checks model-level identification at the query inputs; it does NOT prove that
an incomplete model lacks a correct total program among its possible completions.

Compare four policies on the same candidate pool:

1. **Baseline:** unchanged training-based ranking.
2. **Tie complete:** among models tied for best internal evidence, prefer those
   complete on the query set; use the old cost/syntax order within that set.
   If none is complete, keep the original model.
3. **Complete first:** choose the old-ranked best among every query-complete
   model, even if its internal evidence is worse. If none, keep the original.
4. **Tie consensus:** consider the query-complete, best-evidence models. If they
   all predict the same joint output, return it; if their joint outputs differ,
   abstain on the entire task. If that set is empty, keep the original model.

Select once per task using its full supplied query-input set, not per answer.
No output values are treated as true merely because a candidate produced them.
All equally ranked syntax aliases count as hypotheses, but never as extra votes.
An input-only supplementary diagnostic applies the same policies independently
to each query and records differences from joint selection; it is not the
primary policy and does not contribute extra primary solutions.

## A useful exact boundary

For fixed candidates and original rank, tie-complete and complete-first preserve
an already complete original prediction: the original remains the best-ranked
candidate in the relevant eligible set. Therefore no loss of a fully correct
original task is possible in this fixed-pool comparison. **That is a property of
the policy, not empirical evidence of safety.** New complete answers may be wrong.
Tie-consensus can reject previously complete answers when tied models disagree.
Consensus among a restricted set is not a correctness guarantee either.

More generally, the selected rule is now a function of demonstrations D AND
query inputs Q. It is a transductive selection procedure, not a single learned
function independent of Q. Adding a second query can change the first answer.
The explicit batch-sensitivity diagnostic and synthetic control check this.

## Data and evaluation frozen before execution

ARC1: all 400 training and 400 evaluation problems projected in 019. Both
learner modes run independently. Report raw relational and unchanged no-fit
composition with saved stable predictions: retain stable whenever it fits all
demonstrations; otherwise substitute a complete relational task answer. Preserve
all 400+400 denominators and separately report exclusion of 018's three known
fixtures. These public tasks were used earlier, including through ARC2 overlap;
this is NOT untouched-holdout evaluation. No new ARC2 score is claimed.

Controlled: all 24 two-demonstration specifications from 022, six teachers by
four palette/row conditions. Reuse the identical 36-grid crossed and 14-grid
legacy banks per specification. Choice sees all 50 query inputs, just as the
previous learner predicted that supplied set; overlapping banks are scored
separately, not counted as independent tasks. Opaque IDs and inputs are passed
to the learner, not teacher names or design metadata. These researcher-defined
teachers do not supply new ARC ground truth.

For every outer fold, search and internal ranking are recomputed on the remaining
labelled demonstrations. The query-aware policy then sees only the withheld
INPUTS. Freeze its candidate pool and decisions before comparing with withheld
outputs. Duplicate inputs are withheld together. Never select an outer model
using original test-query inputs, full-data scores or outer output labels.

The prediction command reads projected problems only and writes all candidate
predictions, choices, outer pools and an SHA-256 manifest. A separate score
command reads query answers after predictions are complete. Twelve processes
per batch, deterministic order, serial learner modes. Fixed contiguous batches
are permitted solely for tool execution limits; check exhaustive reassembly
before scoring. No score-dependent stopping, task exclusion or tuning.

## Outcomes and controls

Primary: exact tasks/grids, complete wrong answers, abstentions, paired gains
and losses, no-fit combination, and strict outer-gate retention/error tradeoffs.
Report validation tiedness, disagreements among complete candidates, changes of
selected features, and finite-pool oracle separately. Count newly wrong complete
answers, not merely newly solved tasks. Controlled results are descriptive grid
counts under six dependent teachers, not 216 independent ARC tasks.

Required checks: reproduce the base ARC1 candidates/choices against 019 and the
base/default controlled candidates against 022; audit changed decisions and all
outer pools with an independent rank/filter implementation; reconstruct action
sets and selected predictions from labelled training subsets without calling
learner fit/apply/compress. Preserve explicit limitations where the feature
extractor or search implementation is reused. Repeat predictions byte-for-byte.

Fifteen synthetic checks passed before corpus execution, including exact rational
ties, unchanged complete predictions, disagreements, syntax aliases, query-batch
sensitivity, outer-label isolation, duplicate grouping and an explicit wrong
complete prediction. Existing 018 (15 tests) and 021 (12 tests) also pass.

Pre-run SHA-256:
- run.py: `38d287293884489543cc6f8934f17526cd3fc06c24f77b7e11794f199be80461`
- test_run.py: `474afd1c699f7e16c29b168fc10419b3783570cccdcf531fdb38d3fc2fa861bd`

Retain the stable core, previous compositions and accepted mathematics unchanged.
Results, including a failure to select the desired guard, must be recorded
without silently modifying the old rank or evaluating additional policies.

Related context: Chen et al., *Web Question Answering with Neurosymbolic Program
Synthesis* (2021), explicitly uses transductive selection; Lee et al., *Program
Synthesis via Test-Time Transduction* (2025), uses predicted labels from an LLM
and hypothesis elimination. Our check neither queries an oracle nor invents
query labels. These precedents do not establish the safety of completeness-based
selection. No novelty is claimed for using test inputs during synthesis.
