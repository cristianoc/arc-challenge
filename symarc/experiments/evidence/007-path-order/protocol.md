# 007-path-order: does preserving path order add useful expressivity?

## Hypothesis and decision (registered before scientific runs)

006's `7b5033c1` witness needs path order, which a one-run-per-colour histogram
cannot represent. Test a bounded path language, discovered solely by fitting
training pairs, against a matched histogram language and grid-depth-2 + 003
objects. This is representation change, not an anti-unification algorithm.
The known ARC-AGI-2 witness is development evidence, not a fresh holdout.

Retain a language candidate only if at least one public-training task gains an
oracle-correct path program over grid d2 + objects + histogram and the audited
grid d3 class. Report deterministic selection separately from oracle coverage.
Otherwise close this language extension, retaining compact evidence and its
runnable implementation in Git history. No core or accepted-math changes.

## Frozen language

Reuse 003's eight segmentations and selectors. Select a nonempty collection of
objects, then view the union of selected cells as an induced graph. There are
no new segmentation or selection rules. Enumerate, in fixed order:

1. segment order from 003;
2. selector order from 003 with sorted observed palette;
3. graph adjacency: four-neighbour, then eight-neighbour;
4. endpoint: lexicographically smaller, then larger (row, column);
5. output layout: column, then row.

Require the graph to be a single simple path: one vertex, or exactly two
vertices of degree one and all others degree two; traversal must reach every
vertex. Emit each visited cell's original colour. Reject more than 30 cells
because this renderer is a single ARC row/column. Cycles, branches, disconnected
selections and larger paths are undefined. No colour canonicalisation, moves,
turn counts, loops, per-task constants or post-run grammar changes.

Matched ablation: same segment/select and output layout; count each colour and
emit one run per colour, ordering colours by their first raster occurrence,
forward or reversed. This has no graph adjacency parameter. Compare distinct
predictions as well as syntax counts; no uniform-prior or entropy claim.

Both languages enumerate all fitting ASTs. Selection is first fitting, even if
undefined at a query. Combined control selects first grid fit, else first 003
object fit, else first histogram fit; treatment appends path fallback. Also
report path-first fallback, as a declared secondary selection diagnostic.
Program-pool oracle requires ONE fitting program correct on ALL test outputs.

## Data and controls

SymArc source baseline: `3553c009fda0210ca5e43b9bf3f9339afa4b98e8`.
Use the unchanged stable library and 003 shared API. Exact hashes in each run.
Enumerate on training pairs and construct palettes from those pairs plus test
INPUTS. The discovery API has no test outputs; those are read only for scoring.

- Development check: the already inspected ARC-AGI-2 `7b5033c1` task, pinned by
  006's input manifest. Fit its two training pairs; score its test afterwards.
- Scientific corpus: all 400 `data/training` tasks in this repository, already
  used in earlier SymArc studies. Exclude the ARC-AGI-2 evaluation corpus.
  This is prospective protocol registration on public development data, not
  an untouched benchmark claim.
- Pilot: 24 task IDs chosen by FNV-1a hash, independent of output correctness.
- Full run if pilot completes in <120 seconds and <4 GiB RSS. Process limit
  600 seconds; 12 workers; timing runs serial, one source revision for all arms.
- Grid d3 audit: up to 16 tasks with a path training fit but no grid d2, 003
  object or histogram training fit. Select by FNV-1a before scoring. This
  bounded audit cannot establish an all-depth expressivity separation.

## Checks and evidence

Test repeated colours along a bent path, reverse endpoints, row/column layout,
4-vs-8 adjacency, cycles, branches, disconnected selections, singleton and size
limits. A synthetic case should fit both path and histogram on one example but
disagree on a query with repeated colours; a second training pair should remove
the histogram. This is an expected mechanism check, not ARC effectiveness.
Check that changing test labels leaves fitting pools and predictions unchanged.
Record source/data/binary hashes, command, environment, 12 workers, wall time,
peak RSS, task results and exact witnesses. Reproduce known d2/003 counts in
full run. Distinguish fit coverage, selection accuracy and oracle headroom.

## Commands (from symarc)

```sh
cargo test --release --manifest-path experiments/007-path-order/Cargo.toml
python3 experiments/007-path-order/run.py pilot
python3 experiments/007-path-order/run.py full
```

Results: not run at registration. Disposition is in `../RESULTS.md`.
