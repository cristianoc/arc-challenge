# NNN-short-name: question

Ledger entry: `../RESULTS.md#NNN-short-name` (link to the actual heading).

## Hypothesis and decision

One question. State the control, treatment, primary metric, and what evidence
would justify integration. Specify how correctness and cost will be assessed,
not just the quantity being optimized.

## Protocol

- Baseline: core revision/hash.
- Data and controls: task set, depth, caps, seeds, sampling, repair policy, workers (normally 12).
- Budget: wall time/evaluation budget and stopping rule.
- Checks: invariant or small-case checks needed to trust the measurement.
- Limitations: approximations and what the comparison cannot establish.

## Commands

Exact commands from `symarc/`, including building/running the control and
treatment. Put raw outputs in `out/experiments/NNN-short-name/` and identify
source/configuration/data for each meaningful run.

## Findings

Brief observations and links to raw runs while active. Move decisive numbers
and the final disposition to the ledger before pruning this directory.

## Integration work

Only for a retained candidate: remaining changes, tests, evidence, and known
risks. Keep the code runnable against the recorded core revision.
