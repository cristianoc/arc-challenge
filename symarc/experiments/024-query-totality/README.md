# 024 — Condition concrete programs on query totality

Baseline: `f2a926f30783de3f2ef985d582c748e57c603907` (PR #32).
Question: does requiring a concrete consistent program to be defined on the
supplied query inputs remove genuine operation ambiguity, without confusing
nonempty program families with agreement about the output?

## Fixed semantic interpretation

An ordinary fitted table represents all functions choosing one operation from
V_z at each observed key z. Keys absent from the table are outside every such
program's domain. This experiment does not synthesize new entries for them.
The operation at one key is shared by ALL query occurrences of that key.

Let T_z(Q) contain operations defined at every query point whose key is z.
For independent action choices the query-total family is nonempty iff every
query key is observed and V_z intersect T_z(Q) is nonempty at every used key.
The exact remaining projection is V_z intersect T_z(Q). Predict a colour only
when all remaining operations agree; nonempty does not imply determined.
If the whole family is empty, emit no predictions from that family, rather than
combining partial answers from programs that cannot jointly answer the query set.

021's default prior couples keys. Keep its ORIGINAL training-optimal defaults:
D_min minimizes the number of exception keys. For each d in D_min the key's
choices are {d} when d is in V_z, otherwise V_z. Retain a default branch only
when ALL its query-key choices intersect their T_z(Q). Project the union of
those globally surviving branches. Do not independently filter the already
projected table; that can invent combinations absent from every concrete
program. Do not reoptimize exception cost after imposing totality. Unknown
keys reject the whole table family in both modes; no unconditional copy fallback.

The domain requirement is an additional specification: the intended answer must
be defined on the given inputs. It does not provide output labels or establish
that the chosen finite representation/action family contains the intended rule.
Shared defaults remain an explicit prior, not a logical consequence of fitting.

## Frozen comparison

No changes to the 29 features, 19 actions, <=3-feature search, inclusion-minimal
candidate pools, costs, or internal predictive evidence from 018/021/023.
Compare four policies on the same retained candidate pools:

- baseline: original training-ranked model and unconditioned prediction;
- tie_complete: unchanged 023 evidence-tie query-completeness policy;
- total_ranked: discard empty query-total families, then take the original
  best-ranked remaining model and its exact conditioned prediction, even when
  it still abstains because total programs disagree;
- total_tie_complete: among query-total families with the best remaining internal
  evidence, prefer a determined prediction, using 023's unchanged cost/order
  tie-break. If none is determined keep the best feasible family. This adds the
  earlier heuristic after the logical domain restriction; report it separately.

If no family is query-total, both totality policies abstain on the whole task.
No votes, arbitrary action selection, new operation prior or threshold tuning.
For a fixed nonempty family, a previously determined colour is preserved under
conditioning. An already complete training-ranked prediction is preserved by
both totality policies. These are structural facts, not empirical correctness
guarantees. The tie-complete reference and the stable composition must have their
paired losses checked; they are not automatically preserved by every policy.

## Data, execution and isolation

Run both base and default modes on the same 800 projected ARC1 problems and
24 controlled specifications from 023. The six teachers, four demonstration
conditions, 36 crossed and 14 overlapping legacy queries remain unchanged.
All public/controlled tasks were used previously; no untouched-holdout claim.
No new ARC2 score. Keep the 019 stable predictions and no-fit composition rule.
Report all 400+400 tasks plus the prior development-exclusion analysis.

The primary run may reuse 023's frozen full-data and outer candidate pools and
internal ranks, since they are deliberately unchanged. Reconstruct each table
from its labelled training subset, not stored per-cell answer guesses. In every
outer pool the excluded labels are unavailable to table construction, default
choice and totality conditioning; its excluded INPUTS alone supply Q. Cache only
input-derived features. A fresh end-to-end replay is compared with the cached
execution on all controlled cases and a deterministic hash-selected ARC1 subset.

Persist and hash BOTH modes' predictions before scoring either query-answer
file. Twelve worker processes, serial modes, deterministic contiguous batches
if needed. No score-dependent filtering or budget changes. Runtime is provenance,
not a speed comparison with runs that also enumerate candidate pools.

## Required diagnostics and checks

Record exact family feasibility, original/surviving defaults, removed actions,
unknown keys, disagreements, empty families, and unconditioned/conditioned
predictions. Compare exact default conditioning with naive per-key projection
filtering as a diagnostic ONLY. Inspect any differences, including partial
predictions manufactured when no common default survives.

Independently enumerate all concrete programs for small synthetic tables,
including coupled defaults, multiple occurrences of a key, conflicting domain
requirements, zero-valued source colours, unknown keys, nonempty disagreement,
all-optima retention and conditioning before projection. Verify at the corpus
level using explicit sets and a separate rank/filter implementation; do not call
the scientific conditioner from the auditor. Reconstruct returned predictions
from actual input operations. Check unchanged baseline/023 decisions on every
full and outer pool. Report exact task/grid scores, complete errors, hybrid
paired gains/losses, controlled factorial outcomes and outer-gate tradeoffs.

This is finite hypothesis elimination by a domain constraint, not a novelty
claim for transduction or constraint-based synthesis. Unlike label-predicting
transductive methods, it uses no guessed query outputs. Stable core, previous
wrappers and accepted mathematics remain unchanged. Retain negative results.
