# 015 — Structural roles and output actions

Question: does explicit region structure improve on 014's local context tables,
and can demonstration-level prediction select useful structural explanations?
Separate the representation from the output-action language. The known rectangle
case e88171ec is a development fixture and is excluded from primary scores.

Baseline: f285542b672ee66587f5d7cdbac73716b2067842. Data: official ARC-AGI-2
f3283f727488ad98fe575ea6a5ac981e4a188e49, 1000 training + 120 evaluation tasks.
Public data used earlier in this project; no untouched-holdout claim.

## Frozen hypothesis family

Pointed-grid input u=(grid,row,column). Each representation supplies a context
key a(u) used to choose an output action. Representations are supplied, not
invented by the learner. There are exactly 38, in this order:

1. Raw-colour square neighbourhoods, radii 0,1,2,3 (014's definition).
2. Colour-canonical neighbourhoods, same radii. Centre colour is 0, subsequent
   colours receive first-encounter IDs; padding keeps sentinel 10.
3. For each literal colour 0..9: union of all monochromatic 4-connected
   components; unique largest such component; unique maximum-area monochromatic
   axis-aligned rectangle. The order is family first, then colour.

The 30 structural representations record only outside / boundary / strict
interior of the selected mask. Interior means all four neighbours belong to
that mask. Missing colours and nonunique maxima make the representation
undefined; there is no positional tie-break. The rectangle extractor reuses
014's checked routine on a binary colour mask. Connectivity, literal colour
selectors, maximum selection, neighbourhood bounds and family order are priors.

Compare two output languages for EVERY representation: (a) the ten literal
colour constants; (b) those same constants plus CopyCentre. No preference for
copy is hidden in the renderer. For each observed context retain ALL actions
consistent with all labelled occurrences. Reject a representation when any
intersection is empty. At prediction, output a colour only when every retained
action evaluates to that colour; otherwise abstain. An unobserved key retains
the full action universe and abstains. Undefined extraction also abstains.

This is a finite rule-table learner, not AST anti-unification, MDL, automatic
rectangle discovery or an extension of the stable Rust solver.

## The mathematical correction

For action language Theta and decoder E, fit

$$
V_z = \bigcap_{(u,y):a(u)=z}\{\theta\in\Theta:E(\theta,u)=y\}.
$$

Prediction is determined exactly when {E(theta,u): theta in V_a(u)} is a
singleton. With literals only this reduces to equality of output labels on
fibres. CopyCentre instead preserves the centre colour through the decoder:
this is a quotient for RULE SELECTION, not a claim that the complete predictor
factors through the role key alone. Different-coloured cells can share the
same operation without sharing their final output colour. The action language
is an explicit additional assumption.

## Selection and withheld demonstrations

Hold out each distinct demonstration input with all duplicates. For each
FIXED representation/action-language pair, refit its action sets on the other
demonstrations and score the withheld grid. The family is fixed before data;
its feature definition is never chosen using a withheld output inside its fit.

Among all full-training-fitting candidates compare:
- first in the declared fixed order;
- cross-validation selection: maximise number of completely exact held-out
  grids, then mean correct-cell fraction (unknown counts as incorrect), then
  fixed order;
- conservative consensus over all candidates tied on both validation scores.
  An undefined candidate blocks that query; it is not silently removed.

Apply each selector to local-only, structural-only, and their union, separately
for each output language: 18 prespecified policies. Fractions are exact rational
numbers. Cell-based tie-breaking can favour unchanged background, so changed-cell
metrics must be reported alongside primary task metrics. Cross-validation is a
model-selection signal, not an unbiased reported accuracy estimate after model
selection. Final query answers are scored separately.

No query-output-driven family selection, default copying, threshold tuning,
combined-model pixel stitching or second attempt is permitted. Query input is
not used to change rankings, and there is no fallback after a selected model
abstains. Preserve per-candidate results so oracle headroom can be distinguished
from achieved selection gains. A family chosen after examining known task
answers remains a development prior even with algorithmic label isolation.

## Staged execution

Reuse 014's preparation command to create projected problems and a separate
answer file. Run coverage first: workers receive demonstration inputs/outputs
only, not query inputs or answers. Report fitting structural tasks other than
e88171ec. If fewer than five fit, record a coverage failure rather than claiming
a general selector comparison. Otherwise run the frozen prediction command and
hash its complete outputs before executing the separate scoring command.

Eligibility is unchanged from 014: at least two distinct demonstration inputs,
all demonstrations have matching input/output shape. Query shape changes are
failures, not retrospective exclusions. Use 12 workers, deterministic order,
no seed and no timing comparison to older studies.

Primary metrics by original corpus split, EXCLUDING e88171ec: eligible tasks,
fits, complete query predictions, correct tasks, complete wrong tasks, and
abstentions. Report local/structural union oracle and actual selector wins/losses.
The known rectangle witness is reported separately. Compare the best-validated
local copy-language control against adding structure, rather than crediting
copying or colour canonicalisation to rectangles.

Report exact per-task predictions and candidate/fold results, partial known-cell
errors, changed-cell metrics, and parser-domain/tie failures. A positive
representation result needs at least five structurally fitting nondevelopment
tasks and at least two correct tasks unavailable to ANY fitted local candidate
under the same output language. Selector superiority requires at least five
recoverable wrong/incomplete first-choice answers; below that, selection
comparisons are descriptive only. These are exploratory criteria, not tests
of statistical significance. No automatic core integration.

## Checks and source freeze

14 synthetic controls passed before any corpus execution: literal/copy contrast,
ambiguous actions, unseen keys, undefined/tied masks, 4-connectivity, role erosion,
rectangle-vs-component scope, exhaustive 2x3 binary-grid rectangle oracle,
colour canonicalisation, duplicate withholding, test-answer read trap,
undefined-consensus handling, literal propagation and shape eligibility.

Pre-run SHA-256:
- run.py: 26db5874f736f287af29bad91076427900bcc2c217aa6e0d86937b54c484725b
- test_run.py: 1e331576ea2d6ac7b1d3c78ce64d95cb620573c2a1b76bab2a15c205d4f0ab74

The original 014 raw-local first policy should reproduce byte-for-byte. An
independent scorer must recompute whole-task results, validate action-set
intersection against brute action enumeration, and check the prediction hash.
Record retained-source and data hashes and repeat predictions for determinism.

```sh
E=symarc/experiments/015-structural-roles
B=symarc/experiments/014-cross-demonstration-transport
python3 "$E/test_run.py"
python3 "$B/run.py" prepare --data /path/to/ARC-AGI-2/data --out /tmp/arc015/input
python3 "$E/run.py" coverage --problems /tmp/arc015/input/problems.json --out /tmp/arc015/run --workers 12
python3 "$E/run.py" predict --problems /tmp/arc015/input/problems.json --out /tmp/arc015/run --workers 12
python3 "$E/run.py" score --predictions /tmp/arc015/run/predictions.json --answers /tmp/arc015/input/answers.json --out /tmp/arc015/run
```

Keep experimental code/evidence local to this experiment. Stable core and
accepted mathematics are unchanged. Status and disposition belong in RESULTS.md.
