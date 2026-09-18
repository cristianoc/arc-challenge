# Experiment record

Compact findings and decisions, retained even when code and raw outputs are
deleted. This is an experiment ledger, not a code changelog or a theory history.
Status meanings and code-retention rules are in [README.md](README.md).

No scientific experiments have been registered under this structure yet.
The existing solver is the stable starting point; migration and regression
checks are validation, not evidence for an experimental hypothesis.

## Entry format

Use one short entry per experiment; update it in place. Keep decisive facts
here so the entry remains useful after its directory is deleted.

```text
### NNN-short-name — short question
Status: active | candidate | integrated | closed
Baseline: revision + core hash; date
Comparison: control vs treatment; tasks, depth/caps, seeds, compute budget
Result: decisive numbers, runtime, uncertainty/limitations; or "not run yet"
Decision: conclusion and integration criterion met/missed; next step if retained
Code/evidence: path while retained, or "deleted"; integration revision if applicable
```
