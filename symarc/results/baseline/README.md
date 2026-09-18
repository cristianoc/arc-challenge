# Current stable baseline

- **[Full run report](full/report.md)** — all 800 public tasks, with separate
  training/evaluation metrics and individual task results.
- **[Quick run report](quick/report.md)** — the selected 40-task iteration set.

These are the solver's Markdown outputs, retained byte-for-byte from fresh
12-worker runs. The metrics, definitions, configuration, and measured search
time are printed by the solver itself; there is no manually maintained score
table or conversion step.

The corresponding [full manifest](full/run.json) and
[quick manifest](quick/run.json) record the exact source revision, command,
source/data/binary hashes, report hash, machine/toolchain, and end-to-end timing.
The quick set contains tasks from both public splits; it is a development
subset, not an untouched holdout.

From `symarc/`, reproduce with:

```sh
./bench.sh quick
./bench.sh full
```

Each command writes `report.md` and `run.json` into its printed output
directory. Retain those files directly when refreshing this baseline. The
harness defaults to 12 workers and runs must be timed serially.
