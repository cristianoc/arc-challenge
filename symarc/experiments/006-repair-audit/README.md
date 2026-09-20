# 006-repair-audit: what mechanisms repair training-perfect programs?

Status is recorded in [the experiment ledger](../RESULTS.md). Start with the
[case catalogue](repair_patterns.md) for twelve analyses, ten verified repair
witnesses, and the distinction between anti-unification and additional inference.

## Question and decision

Which hand-written changes make training-perfect, test-failing external ARC
programs also match their known test answers, and what mechanisms recur?
This supplies explicit witnesses for the expressivity audit requested by 005.
It does not yet test whether SymArc can express or discover these repairs.

The initial audit happened **before registration**, with official test outputs
visible. It must not be described as preregistered, blind discovery, or a
production benchmark gain. All twelve examined tasks are development data.

## Control, treatment and evidence

- External control: 120 Python programs from
  `cristianoc/arc-agi-2-abstraction-dataset@ddf6a3e7e2db6e0e0cf24dc99368e8386e78e63c`.
- Data: ARC-AGI-2 evaluation examples at
  `arcprize/ARC-AGI-2@f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Treatment: ten hand-written repaired Python functions, reusing external
  helpers where appropriate. No SymArc core copy or modification.
- Primary measurement: exact grids, preserving training fit; distinguish task
  exact match from individual test-grid exact match.
- Whole-corpus control: 116/120 training-perfect tasks, 27/120 test-perfect
  tasks, 38/167 correct test grids. Selected treatment: 25/25 training pairs,
  14/14 test pairs, versus 0/14 tests for their controls.
- Checks: 390 seeded colour permutations and 13 geometry transformations.
  Applicable transformations and fixed colours were manually chosen per task
  after inspecting answers. Passing checks supports consistency with these
  supplied assumptions, not their validity or automatic discovery.
- Serial functional checks; three-second per-example timeout for the control
  census. No search-depth, seed or compute-budget comparison with SymArc.
  Historical audit timings/toolchain were not systematically retained.
- Source/data hashes are in `inputs.json`; preserved results and provenance
  are under [evidence/006-repair-audit](../evidence/006-repair-audit/README.md).

## Reproduce

From `symarc/`, using Python 3.10 or later:

```sh
python3 experiments/006-repair-audit/fetch_inputs.py
python3 experiments/006-repair-audit/run.py
# Optional full original-solver census: requires NumPy and POSIX SIGALRM.
python3 experiments/006-repair-audit/run.py --baseline
```

The first command fetches revision-pinned files and checks hashes. Subsequent
checks can run offline. Only fetched Python source with the recorded hash is
executed. Inputs and new results live in ignored
`out/experiments/006-repair-audit/`; `ARC_REPAIR_CORPUS` can select another
cache file. Its parent is also the output directory. The runner asserts agreement
with committed evidence and writes `run.json` with environment and hashes.

No Rust or Lean build is required: this does not modify either implementation.
The partial symmetry function deliberately rejects the eight unresolved test
cells and is excluded from the ten successful repair witnesses.

## Follow-up and retention

The mechanism-selection step is complete. [007](../evidence/007-path-order/README.md)
implemented a manually designed path language: it expressed the known repair
but added no coverage on 400 public training tasks.
[008](../evidence/008-repair-constraints/README.md) tested manually selected
symmetry requirements against the original/repair pairs. Seven originals were
rejected, conditional on accepting the task-specific requirements; three survived.
Neither study discovers or justifies the intended abstraction automatically.

This audit is closed. The catalogue and executable witnesses remain as reusable
inputs for ongoing research, as recorded in the ledger. Inspected test answers
remain development evidence. Accepted mathematics and the stable core are
unchanged.
