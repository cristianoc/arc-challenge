# 003-objects: Does a compositional object language add useful coverage?

Baseline: `3c4b94a`, core hash
`e76a7031af456aeba4348a32b08e32017f44641eb17256e1b01e09a5985a7f5c`.
No core change. This standalone package imports core grids, task loading,
program enumeration and evaluation; it adds an experimental object AST.

## Frozen language and question

An object program is `segment → select → act → render`.

- Segmentation: background zero or most frequent input colour (lowest colour
  breaks frequency ties); 4/8 connectivity; connected foreground or connected
  same-colour foreground. Eight named interpretations, retained as distinct
  syntax even when they agree on an input. Background is computed per input.
- Selection: all; largest/smallest area; widest/tallest bounding box; unique or
  repeated occupancy shape; unique or repeated dominant colour; a specified
  dominant colour. Shape is the translation-normalized set of occupied cells,
  ignoring colour, without rotation/reflection normalization. Dominant-colour
  ties use the lowest colour. Extremal selectors keep ALL ties. Empty selection
  is undefined, not a blank answer. Uniqueness is relative to objects in scene.
- Actions: identity; erase; recolour to a palette colour; move each selected
  object to top/bottom/left/right grid edge preserving the other coordinate;
  reflect each selected object horizontally/vertically within its bounding box.
- Rendering: edit the original canvas; selected objects on a background canvas;
  crop the selected transformed objects to their joint bounding box. Erase is
  allowed only with original-canvas rendering. Out-of-grid cells and overlapping
  transformed objects are undefined; painting over unselected foreground is
  undefined in original-canvas rendering. Recolour-to-background can remove
  visible cells but retains the object's occupied geometry for crop bounds.
- Palette parameters: sorted colours from training inputs/outputs and test
  INPUTS, never test labels. Fixed candidate order follows the lists above,
  with zero background, 4 connectivity, same-colour segmentation first.

The language does not yet express arbitrary object relations, object creation,
loops, nested objects, nonuniform backgrounds or multistep action composition.
This tests one coherent object pipeline, not a universal object DSL.

## Does the puzzle admit a predictively useful object interpretation?

The representation question is central, alongside coverage. Before fitting the
full task, reserve its LAST training pair (tasks with at least two pairs).
Enumerate each language on the other pairs and predict the reserved input.
The reserved output is excluded from palette construction and inference;
its input and the test inputs are allowed, as in the core task protocol.
Use the same AST grammar, order and bounds for discovery and full refitting.

Compute reserved-answer probability, answer entropy, multiclass Brier loss,
first-program accuracy, and correct-answer support. Primary probabilities are
uniform over fitting syntax within each family. As a cheap prior sensitivity
check, also compute the grid prediction under q(length) ∝ 2^-length distributed
uniformly over K^length syntactic sequences. Empty pools abstain with probability
one on undefined: zero correct support, zero entropy, Brier 2. This prevents
mistaking absent hypotheses for confident successful prediction.

A fixed validation gate selects the object family only if it assigns strictly
more probability to the reserved answer than the grid family (tolerance 1e-12);
ties and unavailable validation prefer grid. Then refit the chosen language on
ALL training pairs, select its first fitting program, and score test answers.
Compare this gate with grid alone, object alone, and grid-first fallback.
Do not fall back between families inside the validation gate after refitting.
Report whether per-task predictive preference transfers to test correctness.
One reserved example is limited evidence, and palette size changes on refitting.

Within the object family, report entropy over the eight named segmentation
interpretations after full training fit (mass proportional to fitting AST count).
This measures uncertainty under the specified syntax prior, not semantic object
entropy: two named interpretations may coincide on these grids. Neither this
entropy nor family predictive advantage establishes that a task is intrinsically
"about objects". No representation labels are invented or used as ground truth.

## Comparison and metrics

Exhaustively enumerate the finite object grammar and the core grid DSL at depth
2. Search consumes training examples and test inputs; test outputs score only.
No symmetry, repair, hill climbing, fitted-program cap, or entropy-based ranking.
Both have a common 600s run limit; report separate wall times and candidate/check
counts. Their syntax sizes and operation costs differ, so this is a bounded
coverage comparison, not a matched-operation speed comparison.

Primary metric: tasks with any single training-fitting program that is correct
on ALL test outputs (oracle coverage), especially among core-depth-2 unfitted
training tasks. Also report training-fit coverage, task/grid exact match,
prediction coverage, fit-conditional accuracy, and oracle-minus-selected
headroom. The core chooses its first shortest program. The object arm chooses
its first fitting AST in the frozen order (not claimed MAP or shortest across
languages). The conservative union uses core if its pool is nonempty, otherwise
object; union oracle includes both pools. Report examples/witnesses of gains
and losses, per-task prediction entropy, counts of distinct joint predictions,
and viable segmentation interpretations; uniform syntax entropy is diagnostic
only. Undefined predictions are an explicit category. Keep entire fitting
object pools plus primitive core pools/predictions as JSONL for later work.

These representation diagnostics were registered before the first scientific run.

Run on public training only. First a 24-task pilot selected by FNV-1a 64-bit
hash of ID, irrespective of labels. Proceed to all 400 if pilot finishes within
120s and peak RSS <4GiB. Twelve workers; runs serial with `out/.bench-lock`.
No test-label-driven language changes after first scientific run.

Depth-3 audit: among tasks with object fits and no core-depth-2 fit, choose up
to 16 by the same hash order, BEFORE considering test correctness. Enumerate
the core at depth 3 on those tasks, with the same 600s process limit. Report
whether apparent gains are already reachable at depth 3; do not extrapolate
this targeted audit to full-corpus depth-3 performance. Save audited pools too.
Pilot omits this audit; full run includes it after primary comparison.

## Decision and checks

Retain as an integration candidate only if it adds at least one oracle-correct
solution absent at core depth 3 in the audit, with bounded cost and verified
object semantics. This warrants further testing, not automatic integration.
If no such evidence, close and remove the experiment unless a specific follow-up
needs it. Existing core remains the only production solver.

Tests: 4/8 and mono/multicolour object partitions, background ties, relational
shape selection, all-tie selection, crop geometry, edits/moves, collision and
empty-selection undefinedness, and end-to-end inference unchanged by test-label
replacement. Check labels are structurally separated from search. Verify known
core depth-2 full-corpus counts (55 fit, 53 selected-correct, 54 oracle).

Commands from `symarc/`:

```
cargo test --release --manifest-path experiments/003-objects/Cargo.toml
python3 experiments/003-objects/run.py pilot
python3 experiments/003-objects/run.py full
```

Reports are emitted directly by the Rust experiment; manifests retain exact
source/data/binary/artifact hashes and commands. Raw pools live in ignored
`out/experiments/003-objects/`; decisive reports and provenance are retained
under `experiments/evidence/003-objects/`.
