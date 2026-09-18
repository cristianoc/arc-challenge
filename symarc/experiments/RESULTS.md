# Experiment record

Compact findings and decisions, retained even when code and raw outputs are
deleted. This is an experiment ledger, not a code changelog or a theory history.
Status meanings and code-retention rules are in [README.md](README.md).

### 001-pruning — Does aggressive hypothesis pruning improve predictions?
Status: active
Baseline: `be25423adee3d2214987c677aa3a1762132d5469`; core hash `0633d22fb7d1c9d4699fecf4ff851b59cb97c638bcbd6be17db0176865331146`; 2026-09-18.
Comparison: no symmetry, fixed order, random order, marginal pruning; public training tasks only; fixed depth-2 program set, no repair/fallback; seeds 0–4; 12 workers; common checking budget.
Result: not run yet.
Decision: measure before considering integration; increased pruning alone is not success.
Code/evidence: [protocol and code](001-pruning/README.md); raw outputs under `out/experiments/001-pruning/`.

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
