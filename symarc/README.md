# SymArc experiments

One Rust solver and a Lean mathematical model for exploring abstraction,
program entropy, and functional closure under symmetry. Start with [MATH.md](MATH.md) for the
current definitions, proved statements, and approximation boundaries.

## Build and check

Rust and Cargo are sufficient for the solver; it has no external crate dependencies.
Lean is needed only to check the mathematics (toolchain pinned in `lean-toolchain`).

```sh
cargo build --release
cargo test --release
python3 tests/check.py
lake build
```

`tests/check.py` uses the ARC data in `../data`. It checks seeded hill climbing,
realizability repair, ablation, cap behavior, the demo, and worker independence.
Fixtures specify stable behavior; update them deliberately when an accepted
change is integrated. There is no second solver implementation.

## Run the stable baseline

```sh
./target/release/symarc --demo
./target/release/symarc --data ../data/training --task 67a3c6ac --show
./bench.sh quick
./bench.sh quick --seed 17 --norealize
THREADS=12 ./bench.sh full
```

The harness runs the **complete pipeline** by default and creates a fresh
output directory containing the command, configuration arguments, source/data
hashes, toolchain, and timing. Its `report.md` is the solver's direct Markdown
output, with predictive metrics first and task details below; `run.json` stores
provenance and the report hash. It holds a lock to serialize harness
runs. Use `--bench` to time only functionality, closure, and enumeration.
Run the quick set while iterating; confirm useful changes on the full set.
See [subsets/README.md](subsets/README.md) for weighting and limitations.

The harness defaults to **12 workers** for both quick and full runs. Keep this
worker count fixed across baseline and treatment runs; change `THREADS` only
when resource allocation is itself the experiment. Time runs serially.

The committed [full run report](results/baseline/full/report.md) and
[quick run report](results/baseline/quick/report.md) are the solver's direct
output. [Baseline provenance](results/baseline/README.md) identifies the source
revision and run manifests. `out/` is only scratch space.

## Develop an experiment

Keep the current solver as the control. Add trial code under
`experiments/<id>/`, using the `symarc` library rather than copying the solver.
Experiment packages are independent of the default core build.

Follow [the experiment workflow](experiments/README.md) and register the
question in [the compact experiment record](experiments/RESULTS.md). Keep
active work and runnable integration candidates; delete completed experiment
code once its finding and disposition are recorded. Integration is a separate
change into the stable core, with tests and any current-math updates.

## Configuration

```text
--data DIR [--task ID ...]         load a sorted directory, optionally select IDs
--root DIR --tasks-file FILE      load id/split lines from a task list
--limit N --threads N --show      truncate tasks, choose workers, print predictions
--cap N                          final closure cap (20000)
--capgreedy N                    functionality closure cap (1000)
--samplecap N                    sample closure cap (300)
--fit N                          extra sample attempts beyond one-step images (64)
--enumlen N                      enumeration depth (2); 0 uses hill climbing
--restarts N --steps N            hill climbing (16 restarts, 200 steps)
--repair N --maxlen N             repair steps (150), mutation length threshold (3)
--seed N                         per-task seed (0), independent of worker scheduling
--norealize                      functionality-only ablation
--bench                          three deterministic phase timings only
```

Enumeration depth and hill-climbing length are separate controls. A starting
program is not truncated by `--maxlen`. Test outputs never affect search.
Task outputs are printed in input order for any worker count.

## Read the output

Normal solver stdout is a ready-to-read Markdown report: split-specific task
and test-grid exact-match accuracy, prediction coverage, training-fit metrics,
symmetry diagnostics, configuration, and task details. The split labels are
reporting metadata only. No converter or manually maintained score table is
needed. `--bench` instead prints phase timings in `bench.txt`.

The per-task detail lines retain the format consumed by the subset tools:

```text
id func[...] real[...] |C|=n[+] gain=bits outs=m progsD=a progs=b fitD=f fitC=f evals=e sym=s/t(ok k) equiv=... det=... STATUS :: program
```

`func`/`real` count generators by family. `real[n/a]` means no program fit the
training data. `+` means closure completion was not established; gain is then
only explored coverage. `outs` counts distinct outputs. `progsD`/`progs` count
programs before/after realizability. `fitD`/`fitC` are fractions fitted on data
and the final closure sample. `sym` counts test inputs reached by symmetry;
`ok` scores those answers. `equiv` checks the chosen program at test inputs;
`det` checks agreement among defined surviving programs (`-` if fewer than two).
`SOLVED` scores all test predictions; `fit-only` fits training but misses a test.

## Files

- `src/lib.rs` and modules: stable grids/closure, DSL, search, RNG, tasks and reporting.
- `src/main.rs`: stable CLI, using that same library.
- `experiments/`: isolated research code, protocols, and a compact result ledger.
- `SymArc/Theory.lean`, `SymArc.lean`: mathematical definitions and proofs only.
- `MATH.md`: the current mathematical account; no chronological theory history.
- `subsets/`: task lists and estimation/rebuilding tools.
- `tests/`: behavioral checks and fixtures.
- `bench.sh`: reproducible serial run harness.
- `results/baseline/`: committed current baseline results and provenance.
- `out/`: generated, ignored run artifacts; decisive findings live in the experiment ledger.

The [object abstraction report](experiments/evidence/003-objects/full/report.md)
compares object and grid languages, including prediction on a reserved training
example. The [object language](experiments/003-objects/README.md) is retained as
an integration candidate; the stable solver remains the control. Current
research findings and dispositions live in the experiment record above.

The [interpretation-selection report](experiments/evidence/004-competition/full/report.md)
measures whether choosing among surviving interpretations could improve on the
baseline, before comparing predictive evidence and minimum answer entropy.
The [relational-language and information-gain report](experiments/evidence/005-relations/full/report.md)
separates real-task coverage from a controlled test of choosing discriminating
examples; the unsuccessful relational extension was retired.
