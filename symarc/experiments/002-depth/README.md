# 002-depth: Does depth 3 create useful headroom for program entropy?

Baseline: `3b8cf66`, stable baseline core hash
`0633d22fb7d1c9d4699fecf4ff851b59cb97c638bcbd6be17db0176865331146`.
The only core change skips storing the unused final enumeration frontier;
program contents and ordering must be preserved.

Compare exhaustive length <=2 and <=3 pools over identical task-specific DSLs.
No symmetry, repair, hill climbing, or sampling. Keep shortest-program selection
and enumeration tie order fixed. Uniform prior over syntactic programs.

First census all 400 public training tasks at depth 2. Select all tasks with
nonempty pools lacking unanimous defined test predictions, plus 24 empty-pool
tasks ordered by FNV-1a 64-bit hash of their ASCII IDs (ties by ID). Selection
uses training fit and test inputs, never test labels. Save this full selection
before any depth-3 search. Pilot takes the first six hash-ranked tasks in each
cohort; proceed to the full selected cohort if pilot finishes within 120 seconds
and process peak RSS is below 4 GiB. Each run has a 600-second subprocess timeout.
12 workers, serial runs, same harness lock as baseline. No evaluation tasks.

Primary measures: tasks with a correct program in the pool (oracle), shortest
program accuracy, and oracle-minus-selected headroom. Also report fitting-pool
size, distinct joint test prediction vectors, uniform program/answer entropy,
undefined predictions, new distinct predictions and runtime/peak process RSS.
Answer entropy is averaged over test inputs, then eligible tasks. Joint vectors
are observational equivalence on these inputs only, not semantic equivalence.
Undefined is an explicit category; determination requires a defined answer at
every test input. Empty pools have no entropy (display n/a).

Depth-2 census checks the prior 55 fitting / 32 determined result. Verify the
core enumeration against independent tiny brute-force enumeration through depth
3 including exact ordering; run core regression tests. Assert every depth-2 pool
is exactly the <=2 prefix of its depth-3 pool. Save all selected task programs,
primitive indices, and predictions as JSONL for subsequent entropy experiments;
labels are consulted only after predictions are constructed.

This is a depth sensitivity study, not an entropy policy comparison. Cohort
selection deliberately targets ambiguity/missing coverage, so selected-cohort
rates are not full-corpus estimates. Extra syntax with identical predictions
is measured separately from new answer diversity. No solver integration is
justified by larger counts alone. Retain the pool-generation experiment only if
needed for a concrete next entropy comparison.

Commands from `symarc/`:

```
cargo test --release --manifest-path experiments/002-depth/Cargo.toml
python3 experiments/002-depth/run.py pilot
python3 experiments/002-depth/run.py selected
```

Reports are emitted directly by the experiment. Manifests retain source/data/
binary/output hashes, exact command, wall time, and `/usr/bin/time -l` memory.


## Outcome and retained purpose

Completed; disposition and compact findings are in
[the ledger](../RESULTS.md#002-depth--does-depth-3-create-useful-headroom-for-program-entropy).
Read the direct [47-task report](../evidence/002-depth/selected/report.md).
This runner is retained to regenerate the frozen program pools for the next
prior/answer-entropy experiment; it is not a candidate solver policy.
Successful pilot and selected runs used implementation `2114781`. The pilot
passed the prespecified cost gate. The initial sandboxed attempt failed only
at macOS resource collection and was rerun with resource access.

Frozen pools: `out/experiments/002-depth/20260918T054850.184033Z-selected/pools/`
(relative to `symarc/`). Each JSONL line contains program length, primitive
indices in the task's core pool, readable program, and all test predictions.
Filter `length <= 2` to recover the control. The retained run manifest records
every pool hash. These files are disposable and reproducible by the runner.
