# Experiments

The stable solver lives in `../src/`. This directory holds temporary research
code and clear candidates for integration. An experiment does not become part
of the stable solver merely because it works or finishes.

## Layout and dependencies

Each experiment gets an immutable ID and one directory, for example
`001-pruning/`, containing its `README.md`, code, and small experiment-specific
checks. Use [TEMPLATE.md](TEMPLATE.md) for the short protocol. Keep large or
regenerable outputs in `../out/experiments/<id>/` (ignored by Git).

Rust experiments are standalone Cargo packages depending on the stable library:

```toml
[package]
name = "symarc-exp-001-pruning"
version = "0.1.0"
edition = "2021"

[dependencies]
symarc = { path = "../.." }
```

Put the entry point in `<id>/src/main.rs`. From `symarc/`, run:

```sh
cargo run --release --manifest-path experiments/001-pruning/Cargo.toml -- <arguments>
```

Keep the experiment's Cargo.lock. Standalone packages are deliberately outside
the core build; deleting an experiment cannot break `cargo build` or core tests.
Scripts may invoke the stable CLI instead. Reuse `symarc::task::load`,
`Task::candidates_pool`, `dsl`, `grid`, `random`, and `search` as needed.
Do not copy the solver or add trial behavior to the stable CLI. Share a minimal,
behavior-preserving core API refactor only when necessary, with core checks.

## Run and record

1. Create the protocol and register the ID in [RESULTS.md](RESULTS.md) as
   **active** before the first scientific run. State the question, baseline,
   controls, primary metric, and what would justify integration.
2. Identify the baseline by Git revision and source hash. Core runs through
   `../bench.sh` record `core_sha256`, command, source/data hashes, and timing.
   Use the same core revision for both arms of a comparison. For experimental
   runs, record the command, seed, dataset, caps/depth, source revision/hash,
   outcome, and runtime in the protocol or a run manifest. Match compute
   budgets and report approximation limits. Keep timing runs serial.
3. After each meaningful batch, update the ledger's compact result. Keep the
   decisive numbers, configuration, and conclusion there—not only a link to
   disposable logs. Do not log every seed or repeat as a separate experiment.
4. At completion, choose a disposition below. A negative or inconclusive
   result is useful; record why and delete code that no longer has a purpose.

| Status | Meaning | Code policy |
|---|---|---|
| active | A bounded question is being tested | Keep only what is needed to finish |
| candidate | Evidence warrants later integration | Retain runnable code/tests; record remaining work |
| integrated | Accepted into the stable core | Move useful code/tests into core; delete the experiment directory |
| closed | Rejected, inconclusive, or completed without a retained candidate | Delete the experiment directory unless a concrete reason is recorded |

Status is recorded in one place: the ledger. A candidate stays at its original
path; do not create a second copy in a candidates folder. IDs are never reused.

## Integration and pruning

Integration is a separate, deliberate change supported by the recorded result.
Move the accepted implementation and useful tests into the core, verify the
stable suite and relevant full-set comparison, update the current `MATH.md`
only if the accepted model changes, and record the integration revision.
Remove superseded implementations and experimental switches.

Before deleting an experiment, make its ledger entry self-contained: baseline,
question, decisive setup/results, conclusion, and disposition. Preserve a small
non-regenerable result artifact only if needed to support the conclusion; put
it under `evidence/<id>/` and link it. Raw run directories can be pruned freely
once the result is recorded. Rejected theory belongs only in the experiment's
short finding, never as a growing history inside `MATH.md`.
